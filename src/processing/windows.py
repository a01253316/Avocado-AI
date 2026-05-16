"""
processing/windows.py
Convierte las series temporales regularizadas en ventanas deslizantes para el LSTM.
Salida:
    data/processed/windows/X.npy          (N, window_size, n_features)
    data/processed/windows/meta.parquet   (N filas con parcel, date_start, date_end)
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "NDVI_mean", "NDWI_mean", "NDMI_mean", "EVI2_mean", "MSI_mean",
    "NDVI_std",  "NDWI_std",  "NDMI_std",  "EVI2_std",  "MSI_std",
    "NDVI_p10",  "NDWI_p10",  "NDMI_p10",
    "NDVI_p90",  "NDWI_p90",  "NDMI_p90",
    "valid_pixels",
]


def parquet_to_windows(ts_path: Path, window_size: int, step: int = 1) -> tuple:
    """
    Lee un parquet de serie temporal y devuelve (windows, meta).
    windows: ndarray (N, window_size, n_features)
    meta: lista de dicts con parcel, date_start, date_end
    """
    df = pd.read_parquet(ts_path)
    parcel = df["parcel"].iloc[0]

    # Solo columnas de features que existan en este parquet
    cols = [c for c in FEATURE_COLS if c in df.columns]

    # ─── TRATAMIENTO DE NaNs (Red de seguridad de 3 pasos) ──────────────
    # 1. Interpolar: Traza una línea entre el último valor válido y el siguiente
    df[cols] = df[cols].interpolate(method='linear', limit_direction='both')
    # 2. Rellenar bordes: Si los NaNs están al principio o al final de la serie, 
    # la interpolación no los cubre. Usamos backward-fill y forward-fill.
    df[cols] = df[cols].bfill().ffill()
    # 3. Último recurso: Si toda una columna es NaN (ej. un sensor falló por completo),
    # rellenamos con 0.0 para que el código no colapse.
    df[cols] = df[cols].fillna(0.0)
    # ────────────────────────────────────────────────────────────────────
    
    arr = df[cols].values.astype(np.float32)

    windows, meta = [], []
    for start in range(0, len(arr) - window_size + 1, step):
        end = start + window_size
        windows.append(arr[start:end])
        meta.append({
            "parcel":     parcel,
            "date_start": str(df["date"].iloc[start])[:10],
            "date_end":   str(df["date"].iloc[end - 1])[:10],
            "n_features": len(cols),
        })

    return np.stack(windows, axis=0), meta, cols


def run_windows(cfg_path: str = "configs/base.yaml"):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    cfg      = yaml.safe_load(Path(cfg_path).read_text())
    ts_dir   = Path(cfg["paths"]["time_series"])
    win_dir  = Path(cfg["paths"]["windows"])
    win_dir.mkdir(parents=True, exist_ok=True)

    window_size = cfg["time_series"]["window_size"]
    all_X, all_meta = [], []

    for ts_path in sorted(ts_dir.glob("*.parquet")):
        try:
            X, meta, cols = parquet_to_windows(ts_path, window_size)
            all_X.append(X)
            all_meta.extend(meta)
            logger.info(f"  ✓ {ts_path.stem}: {X.shape[0]} ventanas")
        except Exception as e:
            logger.warning(f"  ✗ {ts_path.stem}: {e}")

    if not all_X:
        logger.error("No se generaron ventanas.")
        return

    X = np.concatenate(all_X, axis=0)
    meta_df = pd.DataFrame(all_meta)

    # Normalización: fit sobre train, save scaler
    N, T, F = X.shape
    X_flat = X.reshape(-1, F)
    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)
    X_scaled = X_flat_scaled.reshape(N, T, F)

    np.save(win_dir / "X.npy", X_scaled)
    np.save(win_dir / "X_raw.npy", X)
    meta_df.to_parquet(win_dir / "meta.parquet", index=False)
    joblib.dump(scaler, win_dir / "scaler.pkl")

    logger.info(f"Ventanas: X.shape={X_scaled.shape}, meta={len(meta_df)} filas")
    logger.info(f"Features: {cols}")


if __name__ == "__main__":
    run_windows()
