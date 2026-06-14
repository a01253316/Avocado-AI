"""
Carga el modelo E3 Stacking serializado y ejecuta predicciones.
Si el modelo no existe, lanza un error claro con instrucciones.
"""
from __future__ import annotations

import json
import pathlib
from functools import lru_cache

import joblib
import numpy as np

CLASS_NAMES   = ["Sin estrés", "Moderado", "Severo"]
CLASS_COLORS  = ["green", "yellow", "red"]
CLASS_EMOJI   = ["🟢", "🟡", "🔴"]


class EnsemblePredictor:
    def __init__(self, model_path: pathlib.Path, scaler_path: pathlib.Path, meta_path: pathlib.Path):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado: {model_path}\n"
                "Ejecuta primero:  python scripts/train_save_model.py"
            )
        self._model  = joblib.load(model_path)
        self._scaler = joblib.load(scaler_path)
        with open(meta_path) as f:
            self._meta = json.load(f)

    def predict(self, features: np.ndarray) -> dict:
        """
        features: array (35,) con las estadísticas de la ventana temporal.
        Devuelve clase, probabilidades y metadatos de confianza.
        """
        X      = features.reshape(1, -1)
        X_s    = self._scaler.transform(X)
        proba  = self._model.predict_proba(X_s)[0]
        cls    = int(np.argmax(proba))
        confidence = float(proba[cls])

        return {
            "class":       cls,
            "label":       CLASS_NAMES[cls],
            "color":       CLASS_COLORS[cls],
            "emoji":       CLASS_EMOJI[cls],
            "confidence":  round(confidence, 4),
            "probabilities": {
                CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(proba)
            },
            "f1_macro_train": self._meta.get("f1_macro_test"),
        }


@lru_cache(maxsize=1)
def get_predictor(
    model_path:  str,
    scaler_path: str,
    meta_path:   str,
) -> EnsemblePredictor:
    return EnsemblePredictor(
        pathlib.Path(model_path),
        pathlib.Path(scaler_path),
        pathlib.Path(meta_path),
    )
