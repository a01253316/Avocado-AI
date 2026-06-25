"""
likelihood_dataset.py — Dataset para StressLikelihoodNet (Experimento E).

Carga las series temporales de índices espectrales (data/datasets/signals/*.npz)
y construye ventanas deslizantes para entrenar el modelo probabilístico.

Cada muestra:
  x_hist   : (T=24, 5)  historial normalizado de índices (ventana de entrada)
  x_static : (5,)       media histórica completa de la parcela (huella del terreno)
  doy      : scalar     día del año de la fecha objetivo [1, 365]
  y        : (5,)       índices en la fecha objetivo (lo que el modelo predice)

La separación train/val/test se hace por parcela (usando el split.json existente)
para evitar data-leakage entre entrenamiento y evaluación.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class LikelihoodWindowDataset(Dataset):
    """
    Dataset de ventanas deslizantes sobre series temporales de índices.

    El stride por defecto (1) genera el mayor número de muestras y es
    apropiado para entrenamiento. Para validación/test también stride=1 ya
    que la separación es por parcela, no por timestep.
    """

    def __init__(
        self,
        signals_dir: str | Path,
        parcel_ids:  list[str],
        window_size: int = 24,
        stride:      int = 1,
    ):
        self._samples: list[tuple[np.ndarray, np.ndarray, float, np.ndarray]] = []

        for pid in parcel_ids:
            path = Path(signals_dir) / f"{pid}.npz"
            if not path.exists():
                continue
            npz    = np.load(path, allow_pickle=True)
            data   = npz["data"].astype(np.float32)   # (T, 5)
            doy    = npz["doy"].astype(np.float32)    # (T,)
            static = data.mean(axis=0)                # (5,) — huella estática de la parcela

            T = len(data)
            for t in range(0, T - window_size, stride):
                x_hist     = data[t : t + window_size]     # (window_size, 5)
                target_doy = float(doy[t + window_size])   # DOY de la fecha a predecir
                y          = data[t + window_size]         # (5,) — valores reales
                self._samples.append((x_hist, static, target_doy, y))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        x_hist, x_static, doy, y = self._samples[idx]
        return (
            torch.from_numpy(x_hist),
            torch.from_numpy(x_static),
            torch.tensor(doy, dtype=torch.float32),
            torch.from_numpy(y),
        )


def load_split(
    signals_dir: str | Path,
    split_json:  str | Path,
    window_size: int = 24,
    stride:      int = 1,
) -> tuple[LikelihoodWindowDataset, LikelihoodWindowDataset, LikelihoodWindowDataset]:
    """Carga train/val/test usando la partición existente por parcela."""
    with open(split_json) as f:
        split = json.load(f)

    train_ds = LikelihoodWindowDataset(signals_dir, split["train"], window_size, stride)
    val_ds   = LikelihoodWindowDataset(signals_dir, split["val"],   window_size, stride)
    test_ds  = LikelihoodWindowDataset(signals_dir, split["test"],  window_size, stride)
    return train_ds, val_ds, test_ds
