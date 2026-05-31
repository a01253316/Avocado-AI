"""
sits_vit_dataset.py
===================
Dataset PyTorch para el ViT for SITS.

A diferencia del SITSPixelDataset (CNN), este devuelve:
    x   : (T, C) float32  — serie temporal completa de la parcela
    doy : (T,)   int64    — día del año por timestep
    y   : (1,)   float32  — etiqueta de estrés hídrico

La longitud T varía por parcela (series irregulares). El padding y la
máscara se generan en el collate_fn (sits_collate_fn en sits_vit.py)
para no desperdiciar memoria pre-padding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Índice de NDMI en el vector de canales (NDVI, NDWI, NDMI, NDRE, EVI)
NDMI_CHANNEL = 2
N_CHANNELS   = 5


class SITSViTDataset(Dataset):
    """
    Dataset para el ViT for SITS.

    Lee los .npz de señales (T, C) generados por time_series_builder.py
    y expone (x, doy, label) por parcela.

    Parámetros
    ----------
    signals_dir    : directorio con .npz de señales
    parcel_ids     : lista de IDs a cargar (None = todos)
    ndmi_threshold : umbral NDMI para pseudo-etiqueta de estrés (default -0.1)
    """

    def __init__(
        self,
        signals_dir    : Path,
        parcel_ids     : list[str] | None = None,
        ndmi_threshold : float = -0.1,
    ):
        self.signals_dir    = Path(signals_dir)
        self.ndmi_threshold = ndmi_threshold
        self._samples: list[tuple[np.ndarray, np.ndarray, float]] = []
        self._load(parcel_ids)

    def _load(self, parcel_ids: list[str] | None) -> None:
        npz_files = sorted(self.signals_dir.glob("*.npz"))
        if parcel_ids:
            npz_files = [f for f in npz_files if f.stem in parcel_ids]

        if not npz_files:
            raise FileNotFoundError(
                f"No se encontraron .npz en {self.signals_dir}. "
                "¿Ya corriste make build-dataset?"
            )

        for npz_path in npz_files:
            npz  = np.load(npz_path, allow_pickle=True)
            data = npz["data"].astype(np.float32)      # (T, C)
            doy  = npz["doy"].astype(np.int64)         # (T,)

            # Reemplazar NaN con 0.0 (valor neutro para los índices normalizados)
            data = np.nan_to_num(data, nan=0.0)

            # Pseudo-etiqueta: media temporal de NDMI
            mean_ndmi = float(np.mean(data[:, NDMI_CHANNEL]))
            label     = 1.0 if mean_ndmi < self.ndmi_threshold else 0.0

            self._samples.append((data, doy, label))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data, doy, label = self._samples[idx]
        x = torch.from_numpy(data)                       # (T, C)
        d = torch.from_numpy(doy)                        # (T,)
        y = torch.tensor([label], dtype=torch.float32)  # (1,)
        return x, d, y


def load_vit_datasets(
    dataset_dir    : Path,
    ndmi_threshold : float = -0.1,
) -> dict[str, SITSViTDataset]:
    """
    Carga train/val/test usando split.json.
    Si no existe, todo va a 'train'.
    """
    split_path = dataset_dir / "split.json"
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
    else:
        npz_dir = dataset_dir / "signals"
        all_ids = [f.stem for f in sorted(npz_dir.glob("*.npz"))]
        split   = {"train": all_ids, "val": [], "test": []}

    datasets: dict[str, SITSViTDataset] = {}
    for subset, ids in split.items():
        if not ids:
            continue
        datasets[subset] = SITSViTDataset(
            signals_dir    = dataset_dir / "signals",
            parcel_ids     = ids or None,
            ndmi_threshold = ndmi_threshold,
        )
    return datasets
