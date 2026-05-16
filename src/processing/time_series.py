"""
processing/time_series.py
Lee los GeoTIFFs descargados y construye series temporales por parcela.
Salida: data/processed/time_series/<nombre>.parquet
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml

logger = logging.getLogger(__name__)

BAND_INDICES = {
    "NDVI": 0,
    "NDWI": 1,
    "NDMI": 2,
    "EVI2": 3,
    "MSI":  4,
}


def tif_to_stats(tif_path: Path) -> dict | None:
    """
    Lee un GeoTIFF y devuelve estadísticos por banda (mean, std, p10, p90).
    Solo usa píxeles con valid_mask == 1.
    """
    with rasterio.open(tif_path) as src:
        arr = src.read().astype(np.float32)  # (6, H, W)

    valid = arr[5] > 0.5  # banda 5 = valid_mask

    if valid.sum() < 10:  # parcela casi completamente nublada
        return None

    stats = {"date": tif_path.stem, "valid_pixels": int(valid.sum())}
    for name, idx in BAND_INDICES.items():
        band = arr[idx][valid]
        band = band[band > -9000]  # quita nodata residual
        if len(band) == 0:
            stats.update({f"{name}_mean": np.nan, f"{name}_std": np.nan,
                           f"{name}_p10": np.nan, f"{name}_p90": np.nan})
        else:
            stats.update({
                f"{name}_mean": float(np.nanmean(band)),
                f"{name}_std":  float(np.nanstd(band)),
                f"{name}_p10":  float(np.nanpercentile(band, 10)),
                f"{name}_p90":  float(np.nanpercentile(band, 90)),
            })
    return stats


def build_time_series(parcel_dir: Path, out_dir: Path,
                      min_obs: int = 12) -> pd.DataFrame | None:
    """
    Construye la serie temporal de una parcela.
    Devuelve None si hay menos de min_obs imágenes válidas.
    """
    tifs = sorted(parcel_dir.glob("*.tif"))
    rows = []
    for tif in tifs:
        s = tif_to_stats(tif)
        if s:
            rows.append(s)

    if len(rows) < min_obs:
        logger.warning(f"  {parcel_dir.name}: solo {len(rows)} obs válidas, omitiendo")
        return None

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["parcel"] = parcel_dir.name

    out_path = out_dir / f"{parcel_dir.name}.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"  ✓ {parcel_dir.name}: {len(df)} observaciones guardadas")
    return df


def regularize(df: pd.DataFrame, resample_days: int = 8) -> pd.DataFrame:
    """
    Regulariza la serie temporal a intervalos fijos con interpolación lineal.
    """
    df = df.set_index("date")
    freq = f"{resample_days}D"
    df_reg = df.select_dtypes(include="number").resample(freq).mean()
    df_reg = df_reg.interpolate(method="time", limit_direction="both")
    df_reg = df_reg.fillna(method="ffill").fillna(method="bfill")
    df_reg["parcel"] = df["parcel"].iloc[0]
    return df_reg.reset_index()


def run_time_series(cfg_path: str = "configs/base.yaml"):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    cfg     = yaml.safe_load(Path(cfg_path).read_text())
    raw_dir = Path(cfg["paths"]["raw_sentinel"])
    out_dir = Path(cfg["paths"]["time_series"])
    out_dir.mkdir(parents=True, exist_ok=True)

    parcels = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    logger.info(f"Procesando {len(parcels)} parcelas")

    for parcel_dir in parcels:
        build_time_series(
            parcel_dir, out_dir,
            min_obs=cfg["time_series"]["min_valid_observations"]
        )


if __name__ == "__main__":
    run_time_series()
