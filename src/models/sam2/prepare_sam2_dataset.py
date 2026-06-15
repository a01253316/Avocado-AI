"""
Prepare SAM2 fine-tuning data from AguaVerde Sentinel-2 patches.

The project does not yet have hand-drawn orchard masks. This exporter creates
initial pseudo-labels from the NDMI pixel classifier used by the dashboard:

  image: RGB PNG built from recent NDVI/NDMI/NDWI means
  mask:  binary PNG where stress pixels are foreground

These masks are suitable for bootstrapping SAM2 fine-tuning and can later be
replaced by manually reviewed masks without changing the trainer contract.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.features import NDMI_CH, WINDOW_SIZE, build_ndmi_mask, load_thresholds

NDVI_CH = 0
NDWI_CH = 1
RGB_CHANNELS = (NDVI_CH, NDMI_CH, NDWI_CH)


def patch_to_rgb(
    patch_data: np.ndarray,
    window_size: int = WINDOW_SIZE,
    channels: tuple[int, int, int] = RGB_CHANNELS,
) -> np.ndarray:
    """Build a uint8 RGB image from recent spectral index means."""
    if patch_data.ndim != 4:
        raise ValueError("patch_data must have shape (T, C, H, W)")
    if patch_data.shape[0] < window_size:
        raise ValueError(f"Series too short: {patch_data.shape[0]} < {window_size}")
    if max(channels) >= patch_data.shape[1]:
        raise ValueError("patch_data does not contain the requested RGB channels")

    recent = patch_data[-window_size:, list(channels), :, :]
    with np.errstate(invalid="ignore"):
        img = np.nanmean(recent, axis=0)
    img = np.transpose(img, (1, 2, 0))
    return _scale_to_uint8(img)


def mask_to_binary(mask: np.ndarray, min_class: int = 1) -> np.ndarray:
    """Convert {-1,0,1,2} stress classes to a binary SAM2 mask."""
    if min_class not in (1, 2):
        raise ValueError("min_class must be 1 (moderate+) or 2 (severe only)")
    out = np.zeros(mask.shape, dtype=np.uint8)
    out[mask >= min_class] = 255
    return out


def export_dataset(
    patches_dir: Path,
    norm_path: Path,
    output_dir: Path,
    min_class: int = 1,
    limit: int | None = None,
    val_ratio: float = 0.2,
) -> list[dict[str, str | int]]:
    """Export images, masks, and manifest rows for SAM2 fine-tuning."""
    from PIL import Image

    t_mod, t_sev = load_thresholds(norm_path)
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    npz_files = sorted(patches_dir.glob("*.npz"))
    if limit is not None:
        npz_files = npz_files[:limit]

    split_at = int(round(len(npz_files) * (1.0 - val_ratio)))

    for i, npz_path in enumerate(npz_files):
        npz = np.load(npz_path, allow_pickle=True)
        patch_data = npz["data"].astype(np.float32)
        rgb = patch_to_rgb(patch_data)
        stress_mask = build_ndmi_mask(patch_data, t_mod, t_sev)
        binary_mask = mask_to_binary(stress_mask, min_class=min_class)

        image_path = image_dir / f"{npz_path.stem}.png"
        mask_path = mask_dir / f"{npz_path.stem}.png"
        Image.fromarray(rgb, mode="RGB").save(image_path)
        Image.fromarray(binary_mask, mode="L").save(mask_path)

        latest_date = ""
        if "dates" in npz and len(npz["dates"]):
            latest_date = str(npz["dates"][-1])

        rows.append({
            "parcel_id": npz_path.stem,
            "image": image_path.as_posix(),
            "mask": mask_path.as_posix(),
            "split": "train" if i < split_at else "val",
            "foreground_pixels": int((binary_mask > 0).sum()),
            "total_pixels": int(binary_mask.size),
            "latest_date": latest_date,
        })

    _write_manifest(output_dir / "manifest.csv", rows)
    return rows


def _scale_to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        channel = img[:, :, c]
        lo = float(np.percentile(channel, 2))
        hi = float(np.percentile(channel, 98))
        if hi <= lo:
            hi = lo + 1.0
        out[:, :, c] = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255).round().astype(np.uint8)


def _write_manifest(path: Path, rows: Iterable[dict[str, str | int]]) -> None:
    rows = list(rows)
    fieldnames = [
        "parcel_id",
        "image",
        "mask",
        "split",
        "foreground_pixels",
        "total_pixels",
        "latest_date",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SAM2 fine-tuning dataset.")
    parser.add_argument("--patches-dir", type=Path, default=Path("data/datasets/patches"))
    parser.add_argument("--norm-path", type=Path, default=Path("data/datasets/normalizer_stats.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/sam2"))
    parser.add_argument("--min-class", type=int, choices=[1, 2], default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = export_dataset(
        patches_dir=args.patches_dir,
        norm_path=args.norm_path,
        output_dir=args.output_dir,
        min_class=args.min_class,
        limit=args.limit,
        val_ratio=args.val_ratio,
    )
    print(f"Exported {len(rows)} SAM2 samples to {args.output_dir}")


if __name__ == "__main__":
    main()
