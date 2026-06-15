"""
Extraccion de caracteristicas SITS desde parches .npz.
Misma logica que el notebook Avance5 para consistencia con el modelo entrenado.
"""
from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np

CHANNEL_NAMES = ["NDVI", "NDWI", "NDMI", "NDRE", "EVI"]
STAT_NAMES    = ["mean", "std", "min", "max", "p25", "p75", "trend"]
NDMI_CH       = 2
WINDOW_SIZE   = 24
WINDOW_STEP   = 4
N_FEATURES    = len(CHANNEL_NAMES) * len(STAT_NAMES)  # 35


def load_thresholds(norm_path: pathlib.Path) -> tuple[float, float]:
    """Carga umbrales NDMI normalizados desde metadata o stats minmax."""
    with open(norm_path) as f:
        ns = json.load(f)
    if "t_mod" in ns and "t_sev" in ns:
        return float(ns["t_mod"]), float(ns["t_sev"])
    stats = ns["stats"]["NDMI"]
    denom = stats["max"] - stats["min"]
    t_mod = (-0.10 - stats["min"]) / denom
    t_sev = (-0.20 - stats["min"]) / denom
    return t_mod, t_sev


def extract_window_features(window: np.ndarray) -> np.ndarray:
    """
    Devuelve 35 caracteristicas estadisticas de una ventana (W, C):
    [canal0_mean, canal0_std, ..., canal4_trend]
    """
    feats = []
    t = np.arange(window.shape[0], dtype=np.float32)
    for c in range(window.shape[1]):
        col = window[:, c]
        slope_raw = np.polyfit(t, col, 1)[0]
        feats += [
            float(np.mean(col)),
            float(np.std(col)),
            float(np.min(col)),
            float(np.max(col)),
            float(np.percentile(col, 25)),
            float(np.percentile(col, 75)),
            float(slope_raw) if np.isfinite(slope_raw) else 0.0,
        ]
    return np.array(feats, dtype=np.float32)


def label_window(window: np.ndarray, t_mod: float, t_sev: float) -> int:
    m = float(np.mean(window[:, NDMI_CH]))
    if m < t_sev:
        return 2
    if m < t_mod:
        return 1
    return 0


def build_ndmi_mask(
    patch_data: np.ndarray,
    t_mod: float,
    t_sev: float,
    window_size: int = WINDOW_SIZE,
) -> np.ndarray:
    """
    Clasifica cada pixel del patch con NDMI promedio de la ventana reciente.

    Devuelve una mascara (H, W):
      0 = sin estres
      1 = estres moderado
      2 = estres severo
     -1 = sin dato
    """
    if patch_data.ndim != 4:
        raise ValueError("patch_data debe tener forma (T, C, H, W)")
    if patch_data.shape[0] < window_size:
        raise ValueError(f"Serie muy corta: {patch_data.shape[0]} < {window_size}")
    if patch_data.shape[1] <= NDMI_CH:
        raise ValueError("patch_data no contiene el canal NDMI")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ndmi = np.nanmean(patch_data[-window_size:, NDMI_CH, :, :], axis=0)
    mask = np.zeros(ndmi.shape, dtype=np.int8)
    mask[ndmi < t_mod] = 1
    mask[ndmi < t_sev] = 2
    mask[~np.isfinite(ndmi)] = -1
    return mask


def patch_to_timeseries(npz_path: pathlib.Path) -> np.ndarray:
    """Carga un .npz y devuelve la serie temporal espacialmente promediada (T, C)."""
    p    = np.load(npz_path)
    data = np.nan_to_num(p["data"].astype(np.float32), nan=0.0)
    return data.mean(axis=(2, 3))          # (T, C)


def extract_last_window(
    ts: np.ndarray,
    t_mod: float,
    t_sev: float,
    window_size: int = WINDOW_SIZE,
) -> dict:
    """
    Extrae la ventana temporal mas reciente de la serie (T, C) y devuelve
    features + label + indices espectrales crudos para el prompt LLM.
    """
    if len(ts) < window_size:
        raise ValueError(f"Serie muy corta: {len(ts)} < {window_size}")

    win = ts[-window_size:]          # ultimas 24 fechas
    feats = extract_window_features(win)
    lbl   = label_window(win, t_mod, t_sev)

    # Indices medios en espacio normalizado (para el prompt)
    means = win.mean(axis=0)
    return {
        "features": feats,
        "label":    lbl,
        "indices":  {ch: float(means[i]) for i, ch in enumerate(CHANNEL_NAMES)},
        "n_dates":  len(ts),
    }


def extract_trend_windows(
    ts: np.ndarray,
    t_mod: float,
    t_sev: float,
    n_windows: int = 4,
    window_size: int = WINDOW_SIZE,
    step: int = WINDOW_STEP,
) -> list[dict]:
    """
    Extrae las ultimas n_windows ventanas para mostrar tendencia temporal al LLM.
    Devuelve lista de {label, ndmi_mean, ndvi_mean} de mas antigua a mas reciente.
    """
    T = len(ts)
    starts = list(range(0, T - window_size + 1, step))
    selected = starts[-n_windows:]
    result = []
    for s in selected:
        win = ts[s:s + window_size]
        result.append({
            "label":     label_window(win, t_mod, t_sev),
            "ndmi_mean": float(np.mean(win[:, NDMI_CH])),
            "ndvi_mean": float(np.mean(win[:, 0])),
        })
    return result
