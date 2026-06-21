"""
trend.py - Tendencia por parcela via Gaussian Process (Experimento D).

Sugerido por los profesores asesores (reunion 2026-06-18): en vez de un
umbral/percentil global fijo sobre NDMI, ajustar un Gaussian Process por
parcela que aprenda la media y desviacion estandar esperadas para cada
fecha condicionadas al historial de esa parcela, y clasificar nuevas
observaciones por desviaciones estandar respecto a esa media.

Reutiliza src/models/gp/parcel_gp.py y src/models/gp/group_gp.py (mismo
codigo usado en el prototipo offline) para no duplicar la logica del GP.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.gp.group_gp import group_trend_curve  # noqa: E402
from src.models.gp.parcel_gp import (  # noqa: E402
    DEFAULT_HORIZON,
    ParcelGaussianProcess,
    load_parcel_series,
)


def _compute_group_block(
    parcel_id: str,
    groups_json: Path,
    index_name: str,
    horizon_days: int,
    n_grid: int,
    signals_dir: Path,
    normalizer_path: Path,
    to_date,
) -> dict | None:
    """None si no hay agrupacion por terreno disponible para esta parcela
    (terrain_groups.json no existe, o no se ha re-ejecutado el clustering)
    - la tendencia individual sigue funcionando igual."""
    if not groups_json.exists():
        return None
    try:
        block = group_trend_curve(
            parcel_id, groups_json, index_name, horizon_days, n_grid,
            signals_dir, normalizer_path,
        )
    except KeyError:
        return None

    block["forecast"]["date"] = to_date(block["forecast"]["day"])
    block["last_observation"]["date"] = to_date(block["last_observation"]["day"])
    return block


def compute_trend(
    parcel_id: str,
    signals_dir: Path,
    normalizer_path: Path,
    index_name: str = "NDMI",
    horizon_days: int = DEFAULT_HORIZON,
    n_grid: int = 150,
    groups_json: Path | None = None,
) -> dict:
    series = load_parcel_series(parcel_id, index_name, signals_dir, normalizer_path)
    gp = ParcelGaussianProcess().fit(series.days, series.values)

    date0 = np.datetime64(series.dates[0], "D")

    def to_date(day_offset: float) -> str:
        return str(date0 + np.timedelta64(int(round(day_offset)), "D"))

    grid_days = np.linspace(series.days.min(), series.days.max() + horizon_days, n_grid)
    grid_mean, grid_std = gp.predict(grid_days)

    next_day = float(series.days[-1] + horizon_days)
    f_mean, f_std = gp.predict(np.array([next_day]))

    last_day   = float(series.days[-1])
    last_value = float(series.values[-1])
    z, label   = gp.anomaly_score(last_day, last_value)

    group_block = None
    if groups_json is not None:
        group_block = _compute_group_block(
            parcel_id, groups_json, index_name, horizon_days, n_grid,
            signals_dir, normalizer_path, to_date,
        )

    return {
        "parcel_id": parcel_id,
        "index":     index_name,
        "date0":     str(date0),
        "history": {
            "days":   [round(float(d), 1) for d in series.days],
            "dates":  [str(d) for d in series.dates],
            "values": [round(float(v), 4) for v in series.values],
        },
        "gp_curve": {
            "days": [round(float(d), 1) for d in grid_days],
            "mean": [round(float(m), 4) for m in grid_mean],
            "std":  [round(float(s), 4) for s in grid_std],
        },
        "forecast": {
            "day":  round(next_day, 1),
            "date": to_date(next_day),
            "mean": round(float(f_mean[0]), 4),
            "std":  round(float(f_std[0]), 4),
        },
        "last_observation": {
            "day":   round(last_day, 1),
            "date":  str(series.dates[-1]),
            "value": round(last_value, 4),
            "z":     round(z, 2),
            "label": label,
        },
        "group": group_block,
    }
