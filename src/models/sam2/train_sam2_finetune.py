"""
Fine-tune SAM2 mask decoder on AguaVerde stress masks.

Expected input is the dataset exported by prepare_sam2_dataset.py:

  data/sam2/images/<parcel_id>.png
  data/sam2/masks/<parcel_id>.png
  data/sam2/manifest.csv

This script intentionally keeps checkpoints outside git. Download a SAM2
checkpoint separately and pass it with --checkpoint.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Sample:
    image_path: Path
    mask_path: Path
    parcel_id: str


def read_manifest(manifest: Path, split: str = "train") -> list[Sample]:
    rows: list[Sample] = []
    with manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") != split:
                continue
            if int(row.get("foreground_pixels", 0)) <= 0:
                continue
            rows.append(
                Sample(
                    image_path=Path(row["image"]),
                    mask_path=Path(row["mask"]),
                    parcel_id=row["parcel_id"],
                )
            )
    if not rows:
        raise ValueError(f"No training samples found in {manifest} for split={split}")
    return rows


def positive_point(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Mask has no positive pixels")
    mid = len(xs) // 2
    return np.array([[xs[mid], ys[mid]]], dtype=np.float32)


def load_sample(sample: Sample):
    from PIL import Image

    image = np.array(Image.open(sample.image_path).convert("RGB"))
    mask = np.array(Image.open(sample.mask_path).convert("L"))
    mask = (mask > 0).astype(np.float32)
    point = positive_point(mask)
    labels = np.array([1], dtype=np.int32)
    return image, mask, point, labels


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but this PyTorch build has no CUDA. Falling back to CPU.")
        return "cpu"
    return requested


def build_predictor(model_cfg: str, checkpoint: Path, device: str):
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {checkpoint}\n"
            "Run: make download-sam2-checkpoint\n"
            "Or pass an existing file with SAM2_CHECKPOINT=path/to/checkpoint.pt"
        )

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SAM2 is not installed. Install Meta's SAM2 package before training."
        ) from exc

    model = build_sam2(model_cfg, str(checkpoint), device=device)
    return SAM2ImagePredictor(model)


def train(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F

    args.device = resolve_device(args.device)
    print(f"Using device: {args.device}")

    samples = read_manifest(args.manifest, split="train")
    predictor = build_predictor(args.model_cfg, args.checkpoint, args.device)
    model = predictor.model

    model.sam_mask_decoder.train(True)
    model.sam_prompt_encoder.train(False)
    model.image_encoder.train(False)

    optimizer = torch.optim.AdamW(
        model.sam_mask_decoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    running_iou = 0.0
    step = 0
    for epoch in range(args.epochs):
        for sample in samples:
            image, gt_mask, point, point_labels = load_sample(sample)

            predictor.set_image(image)
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                point,
                point_labels,
                box=None,
                mask_logits=None,
                normalize_coords=True,
            )

            sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
                points=(unnorm_coords, labels),
                boxes=unnorm_box,
                masks=mask_input,
            )

            high_res_features = [
                feat_level[-1].unsqueeze(0)
                for feat_level in predictor._features.get("high_res_feats", [])
            ]
            low_res_masks, pred_scores, _, _ = model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=False,
                high_res_features=high_res_features,
            )

            pred_masks = predictor._transforms.postprocess_masks(
                low_res_masks,
                predictor._orig_hw[-1],
            )
            pred_mask = pred_masks[:, 0]

            gt = torch.as_tensor(gt_mask, device=args.device).unsqueeze(0)
            loss_mask = F.binary_cross_entropy_with_logits(pred_mask, gt)
            pred_binary = (torch.sigmoid(pred_mask) > 0.5).float()
            inter = (pred_binary * gt).sum(dim=(1, 2))
            union = pred_binary.sum(dim=(1, 2)) + gt.sum(dim=(1, 2)) - inter
            iou = inter / (union + 1e-6)
            loss_score = torch.abs(pred_scores[:, 0] - iou).mean()
            loss = loss_mask + args.score_weight * loss_score

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_iou = 0.95 * running_iou + 0.05 * float(iou.mean().detach().cpu())
            step += 1
            if step % args.log_every == 0:
                print(
                    f"epoch={epoch + 1} step={step} "
                    f"loss={float(loss.detach().cpu()):.4f} "
                    f"ema_iou={running_iou:.4f}"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Saved fine-tuned SAM2 weights to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SAM2 on AguaVerde masks.")
    parser.add_argument("--manifest", type=Path, default=Path("data/sam2/manifest.csv"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-cfg", default="sam2_hiera_t.yaml")
    parser.add_argument("--output", type=Path, default=Path("models/sam2_avocado_finetuned.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--score-weight", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
