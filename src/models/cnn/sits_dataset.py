"""
sits_dataset.py
===============
Datasets PyTorch para cargar los .npz generados por time_series_builder.py.

  SITSPixelDataset
  ----------------
  Para PixelCNN. Cada muestra es UN PÍXEL de UNA FECHA de UNA parcela.
  Entrada : (T, C)  — serie temporal del píxel
  Etiqueta: 0/1     — sano / estresado (NDMI < umbral)

  SITSPatchDataset
  ----------------
  Para PatchCNN. Cada muestra es UN CHIP completo de UNA PARCELA.
  Entrada : (T, C, H, W) — serie temporal espacial
  Etiqueta: (H, W) float — mapa de estrés por píxel (0.0-1.0)

Etiquetado automático por umbral (pseudo-labels):
  Como aún no tenemos etiquetas manuales, se usa NDMI como proxy:
    NDMI < -0.1  → estrés hídrico (1)
    NDMI >= -0.1 → sano (0)
  Este umbral es ajustable en el YAML y será reemplazado por etiquetas
  reales cuando estén disponibles.

  Referencia umbral NDMI para aguacate:
    < -0.2  estrés severo
    -0.2 a -0.1  estrés moderado
    > -0.1  sin estrés significativo
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

# Índice de NDMI en el vector de canales (orden: NDVI, NDWI, NDMI, NDRE, EVI)
NDMI_CHANNEL = 2
N_CHANNELS   = 5


# ---------------------------------------------------------------------------
# Dataset de píxeles (para PixelCNN)
# ---------------------------------------------------------------------------
class SITSPixelDataset(Dataset):
    """
    Genera muestras a nivel de píxel desde los archivos .npz de señales.

    Cada .npz tiene:
        data  : (T, C) float32  — medias por índice y fecha
        dates : (T,)  str
        doy   : (T,)  int16

    Pseudo-etiqueta: 1 si la media de NDMI a lo largo del tiempo < umbral.

    Parámetros
    ----------
    signals_dir   : directorio con los .npz de señales
    parcel_ids    : lista de IDs a cargar (None = todos)
    ndmi_threshold: umbral NDMI para etiquetar estrés (default -0.1)
    transform     : función opcional aplicada a cada tensor de entrada
    """

    def __init__(
        self,
        signals_dir   : Path,
        parcel_ids    : list[str] | None = None,
        ndmi_threshold: float = -0.1,
        transform     : Callable | None = None,
    ):
        self.signals_dir    = Path(signals_dir)
        self.ndmi_threshold = ndmi_threshold
        self.transform      = transform

        self._samples: list[tuple[np.ndarray, float]] = []
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
            npz  = np.load(npz_path)
            data = npz["data"].astype(np.float32)   # (T, C)

            # Rellenar NaN con 0 (índice neutro)
            data = np.nan_to_num(data, nan=0.0)

            # Pseudo-etiqueta: media de NDMI a lo largo del tiempo
            mean_ndmi = float(np.mean(data[:, NDMI_CHANNEL]))
            label     = 1.0 if mean_ndmi < self.ndmi_threshold else 0.0

            self._samples.append((data, label))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data, label = self._samples[idx]
        x = torch.from_numpy(data)          # (T, C)
        y = torch.tensor([label], dtype=torch.float32)
        if self.transform:
            x = self.transform(x)
        return x, y


# ---------------------------------------------------------------------------
# Dataset de patches (para PatchCNN)
# ---------------------------------------------------------------------------
class SITSPatchDataset(Dataset):
    """
    Genera muestras a nivel de chip completo desde los .npz de patches.

    Cada .npz tiene:
        data  : (T, C, H, W) float32
        dates : (T,) str
        doy   : (T,) int16

    Pseudo-etiqueta: mapa (H, W) donde cada píxel es 1.0 si su NDMI
    promedio temporal < ndmi_threshold.

    Parámetros
    ----------
    patches_dir   : directorio con los .npz de patches
    parcel_ids    : lista de IDs a cargar (None = todos)
    ndmi_threshold: umbral NDMI para estrés (default -0.1)
    max_t         : recorta la serie a las primeras max_t fechas (None = todas)
    transform     : función opcional sobre el tensor de entrada
    """

    def __init__(
        self,
        patches_dir   : Path,
        parcel_ids    : list[str] | None = None,
        ndmi_threshold: float = -0.1,
        max_t         : int | None = None,
        transform     : Callable | None = None,
    ):
        self.patches_dir    = Path(patches_dir)
        self.ndmi_threshold = ndmi_threshold
        self.max_t          = max_t
        self.transform      = transform

        self._samples: list[tuple[np.ndarray, np.ndarray]] = []
        self._load(parcel_ids)

    def _load(self, parcel_ids: list[str] | None) -> None:
        npz_files = sorted(self.patches_dir.glob("*.npz"))
        if parcel_ids:
            npz_files = [f for f in npz_files if f.stem in parcel_ids]

        if not npz_files:
            raise FileNotFoundError(
                f"No se encontraron .npz en {self.patches_dir}. "
                "¿Ya corriste make build-dataset?"
            )

        for npz_path in npz_files:
            npz  = np.load(npz_path)
            data = npz["data"].astype(np.float32)   # (T, C, H, W)

            # Recortar serie temporal si max_t especificado
            if self.max_t is not None:
                data = data[: self.max_t]

            # Rellenar NaN
            data = np.nan_to_num(data, nan=0.0)

            # Mapa de etiquetas: media temporal del canal NDMI → (H, W)
            mean_ndmi = data[:, NDMI_CHANNEL, :, :].mean(axis=0)   # (H, W)
            label_map = (mean_ndmi < self.ndmi_threshold).astype(np.float32)

            self._samples.append((data, label_map))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data, label_map = self._samples[idx]
        x = torch.from_numpy(data)              # (T, C, H, W)
        y = torch.from_numpy(label_map)         # (H, W)
        if self.transform:
            x = self.transform(x)
        return x, y


# ---------------------------------------------------------------------------
# Factory con split.json
# ---------------------------------------------------------------------------
def load_datasets(
    dataset_dir   : Path,
    mode          : str = "pixel",
    ndmi_threshold: float = -0.1,
    max_t         : int | None = None,
) -> dict[str, Dataset]:
    """
    Carga los datasets de train/val/test usando el split.json generado
    por time_series_builder.

    Retorna un dict {"train": ds, "val": ds, "test": ds}.
    Si no existe split.json, carga todo en "train".

    Parámetros
    ----------
    dataset_dir    : directorio raíz del dataset (data/datasets/)
    mode           : "pixel" → SITSPixelDataset | "patch" → SITSPatchDataset
    ndmi_threshold : umbral NDMI para pseudo-etiquetas
    max_t          : límite de fechas (solo PatchDataset)
    """
    split_path = dataset_dir / "split.json"
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
    else:
        # Sin split: todo va a train
        npz_dir = dataset_dir / ("signals" if mode == "pixel" else "patches")
        all_ids = [f.stem for f in npz_dir.glob("*.npz")]
        split   = {"train": all_ids, "val": [], "test": []}

    datasets: dict[str, Dataset] = {}
    for subset, ids in split.items():
        if not ids:
            continue
        if mode == "pixel":
            datasets[subset] = SITSPixelDataset(
                signals_dir    = dataset_dir / "signals",
                parcel_ids     = ids or None,
                ndmi_threshold = ndmi_threshold,
            )
        else:
            datasets[subset] = SITSPatchDataset(
                patches_dir    = dataset_dir / "patches",
                parcel_ids     = ids or None,
                ndmi_threshold = ndmi_threshold,
                max_t          = max_t,
            )

    return datasets
