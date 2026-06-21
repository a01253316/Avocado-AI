"""
group_gp.py
===========
GP de tendencia por GRUPO de parcelas con terreno similar (Experimento D.2
- ver terrain_groups.py para como se forman los grupos).

Para cada parcela del grupo se calcula su propio promedio historico
("baseline" - nivel tipico de ese suelo/dosel) y se centra su serie
restando ese baseline. Las desviaciones de TODAS las parcelas del grupo
se promedian ENTRE parcelas en cada fecha (no concatenadas crudas). Asi,
una parcela con pocos eventos de estres severo en su propio historial
"toma prestada" la dinamica estacional (cuanto y cuando se desvia del
baseline) de las demas parcelas con terreno similar - exactamente lo que
pidieron los profesores: "agarras doscientas parcelas que son igualitas...
y ya tienes mas muestras por distribucion".

Para una parcela y fecha concretas:
    pronostico_raw(t)    = baseline_parcela + GP_grupo.media(t)
    desviacion_observada = valor_raw_observado - baseline_parcela
    z = (GP_grupo.media(t) - desviacion_observada) / GP_grupo.std(t)

Esto reutiliza ParcelGaussianProcess sin cambios - el GP no distingue si
lo que ajusta son valores crudos (caso individual) o desviaciones pooled
(caso grupal), el kernel es el mismo.

Uso:
    python -m src.models.gp.group_gp --parcel-id H16 --groups-json data/datasets/terrain_groups.json --plot
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .parcel_gp import (
    DEFAULT_HORIZON,
    DEFAULT_INDEX,
    ParcelGaussianProcess,
    load_parcel_series,
)

logger = logging.getLogger("group_gp")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@dataclass
class GroupSeries:
    group_id  : str
    index_name: str
    parcel_ids: list[str]
    baselines : dict[str, float]    # parcel_id -> promedio historico (raw)
    date0     : np.datetime64       # origen comun del eje de tiempo
    days      : np.ndarray          # (T,) ordenado
    values    : np.ndarray          # (T,) DESVIACION promediada entre parcelas


def load_groups(groups_json: Path) -> dict:
    return json.loads(Path(groups_json).read_text(encoding="utf-8"))


def group_id_for_parcel(groups: dict, parcel_id: str) -> str:
    gid = groups["parcel_to_group"].get(parcel_id)
    if gid is None:
        raise KeyError(f"Parcela {parcel_id} no esta en ningun grupo de terreno (reejecuta el clustering)")
    return gid


DEFAULT_BIN_DAYS = 10   # solo se usa si las parcelas NO comparten fechas exactas (ver fallback abajo)


def load_group_series(
    group_id: str,
    parcel_ids: list[str],
    index_name: str = DEFAULT_INDEX,
    signals_dir: Path = Path("data/datasets/signals"),
    normalizer_path: Path = Path("data/datasets/normalizer_stats.json"),
    bin_days: int = DEFAULT_BIN_DAYS,
) -> GroupSeries:
    """
    Junta las desviaciones (valor - baseline propio) de todas las parcelas
    del grupo, promediadas ENTRE parcelas en cada fecha (no concatenadas
    crudas). Promediar entre parcelas, en vez de solo concatenar, es justo
    el mecanismo estadistico que pidieron los profesores: reduce el ruido
    idiosincratico de una sola parcela aprovechando que hay mas parcelas
    con terreno similar, en vez de solo inflar N.

    Las 100 parcelas del catalogo vienen del mismo tile Sentinel-2 con el
    mismo filtro de nubosidad, asi que comparten exactamente las mismas
    277 fechas - se verifica y se promedia directo por fecha (rapido y
    exacto). Si algun dia el catalogo deja de tener fechas identicas entre
    parcelas, se cae a un fallback de bins de `bin_days` dias.

    Promediar (en vez de concatenar) tambien evita un problema de
    rendimiento: GaussianProcessRegressor ajusta de forma exacta (Cholesky,
    O(N^3)) - concatenar ~12,000 observaciones crudas de un grupo de 45
    parcelas vuelve el ajuste inviable.
    """
    per_parcel = {
        pid: load_parcel_series(pid, index_name, signals_dir, normalizer_path)
        for pid in parcel_ids
    }
    baselines = {pid: float(np.mean(s.values)) for pid, s in per_parcel.items()}

    series_list = list(per_parcel.values())
    ref_dates   = series_list[0].dates
    same_dates  = all(np.array_equal(s.dates, ref_dates) for s in series_list[1:])

    if same_dates:
        date0 = np.datetime64(ref_dates[0], "D")
        days  = series_list[0].days.copy()
        devs_matrix = np.stack(
            [per_parcel[pid].values - baselines[pid] for pid in parcel_ids], axis=0,
        )  # (n_parcels, T)
        values = devs_matrix.mean(axis=0)
        order  = np.arange(len(days))
    else:
        date0 = min(np.datetime64(s.dates[0], "D") for s in series_list)
        bins: dict[int, list[float]] = {}
        for pid, s in per_parcel.items():
            offset = (np.datetime64(s.dates[0], "D") - date0).astype(np.float64)
            abs_days = s.days + offset
            devs     = s.values - baselines[pid]
            for d, v in zip(abs_days, devs):
                bin_id = int(d // bin_days)
                bins.setdefault(bin_id, []).append(float(v))
        bin_ids = sorted(bins)
        days    = np.array([b * bin_days + bin_days / 2 for b in bin_ids], dtype=np.float64)
        values  = np.array([float(np.mean(bins[b])) for b in bin_ids], dtype=np.float64)
        order   = np.argsort(days)

    return GroupSeries(
        group_id=group_id, index_name=index_name, parcel_ids=parcel_ids,
        baselines=baselines, date0=date0, days=days[order], values=values[order],
    )


def fit_group_gp(group: GroupSeries) -> ParcelGaussianProcess:
    return ParcelGaussianProcess().fit(group.days, group.values)


def group_trend_curve(
    parcel_id: str,
    groups_json: Path,
    index_name: str = DEFAULT_INDEX,
    horizon_days: int = DEFAULT_HORIZON,
    n_grid: int = 150,
    signals_dir: Path = Path("data/datasets/signals"),
    normalizer_path: Path = Path("data/datasets/normalizer_stats.json"),
) -> dict:
    """
    Curva del GP de grupo (traducida a escala raw de `parcel_id` via su
    baseline) sobre una rejilla de dias, mas pronostico y z-score de la
    ultima observacion. Pensado para que api/trend.py lo combine con el
    bloque individual y lo sirva en un solo endpoint (dias numericos -
    la conversion a fecha calendario la hace el caller, que ya conoce
    `date0`).
    """
    groups   = load_groups(groups_json)
    group_id = group_id_for_parcel(groups, parcel_id)
    members  = groups["groups"][group_id]

    series = load_parcel_series(parcel_id, index_name, signals_dir, normalizer_path)
    group  = load_group_series(group_id, members, index_name, signals_dir, normalizer_path)
    gp_grp = fit_group_gp(group)
    baseline = group.baselines[parcel_id]

    grid_days = np.linspace(series.days.min(), series.days.max() + horizon_days, n_grid)
    dev_mean, dev_std = gp_grp.predict(grid_days)
    grid_mean = baseline + dev_mean

    next_day = float(series.days[-1] + horizon_days)
    f_dev_mean, f_std = gp_grp.predict(np.array([next_day]))

    last_day, last_value = float(series.days[-1]), float(series.values[-1])
    z, label = gp_grp.anomaly_score(last_day, last_value - baseline)

    return {
        "group_id":  group_id,
        "n_members": len(members),
        "members":   members,
        "baseline":  round(baseline, 4),
        "gp_curve": {
            "days": [round(float(d), 1) for d in grid_days],
            "mean": [round(float(m), 4) for m in grid_mean],
            "std":  [round(float(s), 4) for s in dev_std],
        },
        "forecast": {
            "day":  round(next_day, 1),
            "mean": round(baseline + float(f_dev_mean[0]), 4),
            "std":  round(float(f_std[0]), 4),
        },
        "last_observation": {
            "day":   round(last_day, 1),
            "value": round(last_value, 4),
            "z":     round(z, 2),
            "label": label,
        },
    }


def compare_individual_vs_group(
    parcel_id: str,
    groups_json: Path = Path("data/datasets/terrain_groups.json"),
    index_name: str = DEFAULT_INDEX,
    horizon_days: int = DEFAULT_HORIZON,
    signals_dir: Path = Path("data/datasets/signals"),
    normalizer_path: Path = Path("data/datasets/normalizer_stats.json"),
) -> dict:
    """Calcula el z-score / pronostico de la ultima observacion con el GP
    individual (una parcela) y con el GP de grupo (terreno similar), para
    comparar lado a lado."""
    groups   = load_groups(groups_json)
    group_id = group_id_for_parcel(groups, parcel_id)
    members  = groups["groups"][group_id]

    # -- Individual --
    series = load_parcel_series(parcel_id, index_name, signals_dir, normalizer_path)
    gp_ind = ParcelGaussianProcess().fit(series.days, series.values)
    last_day, last_value = float(series.days[-1]), float(series.values[-1])
    z_ind, label_ind = gp_ind.anomaly_score(last_day, last_value)
    mean_ind, std_ind = gp_ind.predict(np.array([last_day + horizon_days]))

    # -- Grupo --
    group = load_group_series(group_id, members, index_name, signals_dir, normalizer_path)
    gp_grp = fit_group_gp(group)
    baseline = group.baselines[parcel_id]
    last_day_grp = last_day + float((np.datetime64(series.dates[0], "D") - group.date0).astype(np.float64))
    deviation = last_value - baseline
    z_grp, label_grp = gp_grp.anomaly_score(last_day_grp, deviation)
    mean_dev, std_grp = gp_grp.predict(np.array([last_day_grp + horizon_days]))
    mean_grp = baseline + mean_dev[0]

    return {
        "parcel_id":  parcel_id,
        "group_id":   group_id,
        "n_members":  len(members),
        "members":    members,
        "individual": {
            "z": z_ind, "label": label_ind,
            "forecast_mean": float(mean_ind[0]), "forecast_std": float(std_ind[0]),
        },
        "group": {
            "z": z_grp, "label": label_grp,
            "baseline": baseline,
            "forecast_mean": float(mean_grp), "forecast_std": float(std_grp[0]),
        },
    }


def plot_comparison(
    parcel_id: str,
    groups_json: Path,
    index_name: str,
    output_dir: Path,
    signals_dir: Path = Path("data/datasets/signals"),
    normalizer_path: Path = Path("data/datasets/normalizer_stats.json"),
) -> Path:
    """GP individual vs. GP de grupo (traducido a escala raw de esta parcela), superpuestos."""
    import matplotlib.pyplot as plt

    groups   = load_groups(groups_json)
    group_id = group_id_for_parcel(groups, parcel_id)
    members  = groups["groups"][group_id]

    series = load_parcel_series(parcel_id, index_name, signals_dir, normalizer_path)
    gp_ind = ParcelGaussianProcess().fit(series.days, series.values)

    group = load_group_series(group_id, members, index_name, signals_dir, normalizer_path)
    gp_grp = fit_group_gp(group)
    baseline = group.baselines[parcel_id]

    days_grid = np.linspace(series.days.min(), series.days.max() + 30, 400)
    mean_ind, std_ind = gp_ind.predict(days_grid)
    mean_dev, std_grp = gp_grp.predict(days_grid)
    mean_grp = baseline + mean_dev

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(days_grid, mean_ind - std_ind, mean_ind + std_ind, alpha=0.18, color="C0", label="Individual +-1 sigma")
    ax.plot(days_grid, mean_ind, color="C0", lw=1.5, label="GP individual")
    ax.fill_between(days_grid, mean_grp - std_grp, mean_grp + std_grp, alpha=0.18, color="C1", label="Grupo +-1 sigma")
    ax.plot(days_grid, mean_grp, color="C1", lw=1.5, ls="--", label=f"GP grupo (n={len(members)} parcelas)")
    ax.scatter(series.days, series.values, color="black", s=8, label=f"{index_name} observado ({parcel_id})", zorder=5)
    ax.set_title(f"Parcela {parcel_id} (grupo {group_id}) - Individual vs. Grupo de terreno")
    ax.set_xlabel(f"Dias desde {series.dates[0]}")
    ax.set_ylabel(index_name)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    out_path = output_dir / f"{parcel_id}_group{group_id}_{index_name}_comparison.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="GP de grupo de terreno vs. GP individual (Experimento D.2)")
    parser.add_argument("--parcel-id", required=True)
    parser.add_argument("--groups-json", type=Path, default=Path("data/datasets/terrain_groups.json"))
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gp_prototype"))
    args = parser.parse_args()

    result = compare_individual_vs_group(
        args.parcel_id, args.groups_json, args.index, args.horizon,
    )

    logger.info(
        "Parcela %s | grupo %s (%d parcelas: %s)",
        result["parcel_id"], result["group_id"], result["n_members"],
        ", ".join(result["members"][:8]) + ("..." if result["n_members"] > 8 else ""),
    )
    ind, grp = result["individual"], result["group"]
    logger.info(
        "  Individual -> z=%+.2f (%s) | pronostico +%dd = %.4f +- %.4f",
        ind["z"], ind["label"], args.horizon, ind["forecast_mean"], ind["forecast_std"],
    )
    logger.info(
        "  Grupo      -> z=%+.2f (%s) | baseline=%.4f | pronostico +%dd = %.4f +- %.4f",
        grp["z"], grp["label"], grp["baseline"], args.horizon, grp["forecast_mean"], grp["forecast_std"],
    )

    if args.plot:
        path = plot_comparison(args.parcel_id, args.groups_json, args.index, args.output_dir)
        logger.info("  Plot guardado en %s", path)


if __name__ == "__main__":
    main()
