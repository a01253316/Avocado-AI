"""
likelihood_predictor.py — Inferencia con StressLikelihoodNet (Experimento E).

Carga el modelo entrenado y expone predict_with_doy() para el endpoint REST.

Flujo (el que describió el profe):
  1. La red predice (μ, σ) por índice — parámetros de la PDF multivariada.
  2. Se calcula el z-score de la observación actual respecto a esa distribución.
  3. Se combina en una señal de estrés ponderada (NDMI domina con 40%).
  4. Se obtiene el percentil bajo la normal estándar (equivalente a la CDF
     multivariada) y se clasifica el nivel de estrés.

Requiere:
    models/stress_likelihood_net.pt  (generado por make train-likelihood)
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.likelihood_nn.stress_likelihood_net import StressLikelihoodNet  # noqa: E402

CHANNEL_NAMES = ["NDVI", "NDWI", "NDMI", "NDRE", "EVI"]

# NDMI (humedad de hoja/dosel) recibe el mayor peso; NDWI (agua en vegetación) segundo.
# Orden coincide con CHANNEL_NAMES.
_STRESS_WEIGHTS = np.array([0.15, 0.20, 0.40, 0.15, 0.10], dtype=np.float32)

_STRESS_LABELS = {0: "Sin estrés", 1: "Estrés moderado", 2: "Estrés severo"}
_STRESS_COLORS = {0: "green",      1: "yellow",          2: "red"}

# Umbrales de percentil para clasificación (calibrados con la lógica GP existente)
_P_MODERATE = 25.0   # percentil mínimo para "sin estrés"
_P_SEVERE   = 60.0   # percentil mínimo para "moderado"; por encima → severo


class LikelihoodPredictor:
    """Wrapper de inferencia para el StressLikelihoodNet."""

    def __init__(self, model_path: str | Path, device: str = "auto"):
        if device == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)

        ckpt = torch.load(model_path, map_location=self._device, weights_only=False)
        cfg  = ckpt["config"]
        self._model = StressLikelihoodNet(**cfg).to(self._device)
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()
        self._val_nll = ckpt.get("best_val_nll")

    def predict_with_doy(
        self,
        x_hist:   np.ndarray,  # (T=24, 5) — ventana de historial normalizado
        x_static: np.ndarray,  # (5,)       — huella espectral de la parcela
        y_obs:    np.ndarray,  # (5,)       — observación actual (la que se evalúa)
        doy:      int,          # día del año de la observación actual [1, 365]
    ) -> dict:
        """
        Evalúa si la observación actual es anómala respecto a la distribución
        esperada por el modelo para ese terreno, historial y fecha.
        """
        x_hist_t   = torch.from_numpy(x_hist.astype(np.float32)).unsqueeze(0).to(self._device)
        x_static_t = torch.from_numpy(x_static.astype(np.float32)).unsqueeze(0).to(self._device)
        doy_t      = torch.tensor([float(doy)], dtype=torch.float32).to(self._device)

        with torch.no_grad():
            mu_t, sigma_t = self._model(x_hist_t, x_static_t, doy_t)

        mu    = mu_t.cpu().numpy()[0]      # (5,)
        sigma = sigma_t.cpu().numpy()[0]   # (5,)

        return self._classify(y_obs, mu, sigma)

    def _classify(
        self, y_obs: np.ndarray, mu: np.ndarray, sigma: np.ndarray
    ) -> dict:
        # z-score positivo = por encima de lo esperado (más húmedo = sin estrés)
        z_scores = (y_obs - mu) / sigma             # (5,)

        # Señal de estrés: positiva cuando la observación está por DEBAJO de lo esperado
        stress_signal = float(np.dot(-z_scores, _STRESS_WEIGHTS))

        # Percentil bajo la CDF de la normal estándar — "CDF multivariada" del profe
        percentile = float(stats.norm.cdf(stress_signal) * 100)

        if percentile < _P_MODERATE:
            stress_class = 0
        elif percentile < _P_SEVERE:
            stress_class = 1
        else:
            stress_class = 2

        return {
            "stress_class":          stress_class,
            "stress_label":          _STRESS_LABELS[stress_class],
            "stress_color":          _STRESS_COLORS[stress_class],
            "likelihood_percentile": round(percentile, 1),
            "combined_z_score":      round(stress_signal, 3),
            "distribution": {
                ch: {
                    "mu":       round(float(mu[i]),      4),
                    "sigma":    round(float(sigma[i]),   4),
                    "observed": round(float(y_obs[i]),   4),
                    "z_score":  round(float(z_scores[i]), 3),
                }
                for i, ch in enumerate(CHANNEL_NAMES)
            },
            "model_val_nll": round(float(self._val_nll), 4) if self._val_nll else None,
        }


@lru_cache(maxsize=1)
def get_likelihood_predictor(model_path: str) -> LikelihoodPredictor:
    """Singleton cacheado — el modelo se carga una sola vez por proceso."""
    return LikelihoodPredictor(model_path)
