"""
parcel_gp.py
============
Experimento D - Gaussian Process por parcela (sugerido por los profesores
asesores, reunion 2026-06-18).

Idea: en vez de clasificar estres hidrico con un percentil/decil fijo sobre
NDMI (umbral arbitrario, solo valido si todas las parcelas comparten el
mismo tipo de terreno), se ajusta un Gaussian Process por parcela sobre la
serie temporal de un indice espectral (NDMI por defecto). El GP aprende la
media y desviacion estandar esperadas para cada fecha condicionadas al
historial propio de esa parcela (estacionalidad + autocorrelacion
temporal), via maximizacion del likelihood marginal - eso ya lo resuelve
GaussianProcessRegressor internamente.

Una observacion se clasifica por cuantas desviaciones estandar cae por
DEBAJO de la media predicha por el GP para esa fecha - no por un percentil
global fijo:
    z < 1          -> sin estres
    1 <= z < 2      -> estres moderado
    z >= 2          -> estres severo

Este es un experimento adicional, documentado aparte del E3 Stacking ya
productivizado (no lo reemplaza).

Uso:
    python -m src.models.gp.parcel_gp --parcel-ids H1 H37 H88
    python -m src.models.gp.parcel_gp --parcel-ids H1 --holdout 6 --plot
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    ExpSineSquared,
    RBF,
    WhiteKernel,
)

logger = logging.getLogger("parcel_gp")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

CHANNEL_NAMES   = ["NDVI", "NDWI", "NDMI", "NDRE", "EVI"]
DEFAULT_INDEX   = "NDMI"
DEFAULT_HORIZON = 5          # dias - cadencia de revisita Sentinel-2
Z_MODERADO      = 1.0
Z_SEVERO        = 2.0


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
@dataclass
class ParcelSeries:
    parcel_id : str
    index_name: str
    days      : np.ndarray   # (T,) float - dias transcurridos desde la 1a fecha
    dates     : np.ndarray   # (T,) "YYYY-MM-DD"
    values    : np.ndarray   # (T,) float - valores RAW (espacio fisico, no normalizado)


def _denormalize(values_norm: np.ndarray, index_name: str, norm_stats: dict) -> np.ndarray:
    """Invierte la normalizacion minmax para volver a unidades fisicas del indice."""
    stats = norm_stats["stats"][index_name]
    vmin, vmax = stats["min"], stats["max"]
    return values_norm * (vmax - vmin) + vmin


def load_parcel_series(
    parcel_id: str,
    index_name: str = DEFAULT_INDEX,
    signals_dir: Path = Path("data/datasets/signals"),
    normalizer_path: Path = Path("data/datasets/normalizer_stats.json"),
) -> ParcelSeries:
    npz_path = signals_dir / f"{parcel_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"No hay senal para la parcela {parcel_id}: {npz_path}")

    d = np.load(npz_path, allow_pickle=True)
    ch_idx = CHANNEL_NAMES.index(index_name)
    values_norm = d["data"][:, ch_idx].astype(np.float64)
    dates = d["dates"]

    with open(normalizer_path) as f:
        norm_stats = json.load(f)
    values_raw = _denormalize(values_norm, index_name, norm_stats)

    dates_dt = np.array(dates, dtype="datetime64[D]")
    days = (dates_dt - dates_dt[0]).astype(np.float64)

    return ParcelSeries(
        parcel_id=parcel_id, index_name=index_name,
        days=days, dates=dates, values=values_raw,
    )


# ---------------------------------------------------------------------------
# Gaussian Process por parcela
# ---------------------------------------------------------------------------
class ParcelGaussianProcess:
    """
    GP de 1 indice espectral vs. tiempo, para una sola parcela.

    Kernel = tendencia suave (RBF) + estacionalidad anual (ExpSineSquared,
    periodo=365.25 dias) + ruido de medicion (White). El ajuste de
    hiperparametros (length_scale, amplitud, ruido) se hace maximizando el
    likelihood marginal - la pieza de "maximum likelihood" que pidieron
    los profesores, resuelta por GaussianProcessRegressor.
    """

    def __init__(self, n_restarts: int = 5, random_state: int = 42):
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=60.0, length_scale_bounds=(10.0, 365.0))
            + ConstantKernel(1.0, (1e-3, 1e3))
            * ExpSineSquared(length_scale=1.0, periodicity=365.25,
                              length_scale_bounds=(0.2, 10.0),
                              periodicity_bounds="fixed")
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1.0))
        )
        self._gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=n_restarts,
            normalize_y=True, random_state=random_state,
        )

    def fit(self, days: np.ndarray, values: np.ndarray) -> "ParcelGaussianProcess":
        self._gp.fit(days.reshape(-1, 1), values)
        return self

    def predict(self, days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (media, std) predichas para cada dia solicitado."""
        mean, std = self._gp.predict(days.reshape(-1, 1), return_std=True)
        return mean, std

    def anomaly_score(self, day: float, value: float) -> tuple[float, str]:
        """
        z = cuantas desviaciones estandar cae `value` por DEBAJO de la media
        predicha por el GP para `day`. z negativo = indice por ENCIMA de lo
        esperado (sin estres / mejora).
        """
        mean, std = self.predict(np.array([day]))
        std = max(float(std[0]), 1e-6)
        z = (float(mean[0]) - value) / std
        if z < Z_MODERADO:
            label = "sin_estres"
        elif z < Z_SEVERO:
            label = "moderado"
        else:
            label = "severo"
        return z, label


# ---------------------------------------------------------------------------
# Validacion holdout (sanity check de una sola parcela)
# ---------------------------------------------------------------------------
def evaluate_holdout(series: ParcelSeries, holdout: int) -> None:
    """
    Entrena el GP sin las ultimas `holdout` observaciones y compara la
    prediccion contra el valor real para esas fechas - chequeo rapido de
    que el GP no esta prediciendo a ciegas.
    """
    if holdout <= 0 or holdout >= len(series.days):
        return

    train_days, train_vals = series.days[:-holdout], series.values[:-holdout]
    test_days, test_vals   = series.days[-holdout:], series.values[-holdout:]
    test_dates              = series.dates[-holdout:]

    gp = ParcelGaussianProcess().fit(train_days, train_vals)
    mean, std = gp.predict(test_days)

    logger.info("-- Holdout (%d fechas) - %s / %s --", holdout, series.parcel_id, series.index_name)
    for date, real, mu, sigma in zip(test_dates, test_vals, mean, std):
        z = (mu - real) / max(sigma, 1e-6)
        logger.info("  %s  real=%.4f  gp_media=%.4f+-%.4f  z=%.2f", date, real, mu, sigma, z)


# ---------------------------------------------------------------------------
# Pronostico a horizonte (siguiente revisita)
# ---------------------------------------------------------------------------
def forecast_next(series: ParcelSeries, horizon_days: int = DEFAULT_HORIZON) -> dict:
    gp = ParcelGaussianProcess().fit(series.days, series.values)
    next_day = series.days[-1] + horizon_days
    mean, std = gp.predict(np.array([next_day]))

    last_value = series.values[-1]
    z, label = gp.anomaly_score(series.days[-1], last_value)

    return {
        "parcel_id":       series.parcel_id,
        "index":           series.index_name,
        "last_date":       str(series.dates[-1]),
        "last_value":      float(last_value),
        "next_day_offset": next_day,
        "forecast_mean":   float(mean[0]),
        "forecast_std":    float(std[0]),
        "last_obs_z":      z,
        "last_obs_label":  label,
        "gp":              gp,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_parcel(series: ParcelSeries, gp: ParcelGaussianProcess, output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    days_grid = np.linspace(series.days.min(), series.days.max() + 30, 400)
    mean, std = gp.predict(days_grid)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(days_grid, mean - 2 * std, mean + 2 * std, alpha=0.15, color="C0", label="+-2 sigma")
    ax.fill_between(days_grid, mean - std, mean + std, alpha=0.25, color="C0", label="+-1 sigma")
    ax.plot(days_grid, mean, color="C0", label="GP media")
    ax.scatter(series.days, series.values, color="black", s=8, label=f"{series.index_name} observado")
    ax.set_title(f"Parcela {series.parcel_id} - {series.index_name} vs. tiempo (Gaussian Process)")
    ax.set_xlabel(f"Dias desde {series.dates[0]}")
    ax.set_ylabel(series.index_name)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    out_path = output_dir / f"{series.parcel_id}_{series.index_name}_gp.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prototipo de Gaussian Process por parcela (Experimento D)",
    )
    parser.add_argument("--parcel-ids", nargs="+", required=True, metavar="ID")
    parser.add_argument("--index", default=DEFAULT_INDEX, choices=CHANNEL_NAMES)
    parser.add_argument("--signals-dir", type=Path, default=Path("data/datasets/signals"))
    parser.add_argument("--normalizer", type=Path, default=Path("data/datasets/normalizer_stats.json"))
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Dias a pronosticar")
    parser.add_argument("--holdout", type=int, default=0, help="Fechas a ocultar para validar el GP")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gp_prototype"))
    args = parser.parse_args()

    for pid in args.parcel_ids:
        series = load_parcel_series(pid, args.index, args.signals_dir, args.normalizer)
        logger.info("Parcela %s | %s | %d fechas (%s -> %s)",
                    pid, args.index, len(series.days), series.dates[0], series.dates[-1])

        if args.holdout:
            evaluate_holdout(series, args.holdout)

        result = forecast_next(series, args.horizon)
        logger.info(
            "  Ultima obs (%s)=%.4f | z=%.2f -> %s | pronostico +%dd = %.4f +- %.4f",
            result["last_date"], result["last_value"], result["last_obs_z"],
            result["last_obs_label"], args.horizon,
            result["forecast_mean"], result["forecast_std"],
        )

        if args.plot:
            path = plot_parcel(series, result["gp"], args.output_dir)
            logger.info("  Plot guardado en %s", path)


if __name__ == "__main__":
    main()
