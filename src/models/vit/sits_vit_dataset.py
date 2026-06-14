"""
sits_dataset.py
===============
Datasets PyTorch para cargar los .npz generados por time_series_builder.py.

  SITSPixelDataset
  ----------------
  Para PixelCNN / ViT-SITS. Genera muestras de ventanas temporales deslizantes.
  Cada muestra = una ventana de W fechas consecutivas de UNA parcela.
  Entrada : (W, C)   — W fechas × C índices
  Etiqueta: 0/1 (binario) o 0/1/2 (triclase)

  SITSPatchDataset
  ----------------
  Para PatchCNN. Cada muestra es UN CHIP completo de UNA PARCELA.
  Entrada : (T, C, H, W) — serie temporal espacial completa
  Etiqueta: (H, W) float — mapa de estrés por píxel

Etiquetado automático (pseudo-labels con umbral NDMI):

  Binario (num_classes=2):
    stressed_frac >= stress_frac_min → estrés (1)
    de lo contrario → sano (0)

  Triclase (num_classes=3):
    mean(NDMI) >= ndmi_threshold         → sin estrés         (0)
    ndmi_severe_threshold <= mean < ndmi_threshold → moderado (1)
    mean(NDMI) < ndmi_severe_threshold   → severo             (2)

  Referencia umbral NDMI para aguacate:
    < -0.2  estrés severo
    -0.2 a -0.1  estrés moderado
    > -0.1  sin estrés significativo
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("sits_dataset")

# Índice de NDMI en el vector de canales (orden: NDVI, NDWI, NDMI, NDRE, EVI)
NDMI_CHANNEL  = 2
N_CHANNELS    = 5
CHANNEL_NAMES = ["NDVI", "NDWI", "NDMI", "NDRE", "EVI"]


# ---------------------------------------------------------------------------
# Helper: convierte umbral raw → espacio normalizado
# ---------------------------------------------------------------------------
def convert_threshold(
    raw_threshold   : float,
    normalizer_path : Path | None,
    index_name      : str = "NDMI",
) -> float:
    """
    Convierte un umbral en espacio original (reflectancia) al espacio
    normalizado en que están guardados los datos en los .npz.

    Si no se encuentra el archivo de estadísticas, devuelve el umbral sin cambio
    (asumiendo que los datos NO están normalizados).
    """
    if normalizer_path is None or not normalizer_path.exists():
        logger.warning(
            "normalizer_stats.json no encontrado en %s. "
            "Usando umbral raw=%.3f (¿los datos están sin normalizar?)",
            normalizer_path, raw_threshold,
        )
        return raw_threshold

    body = json.loads(normalizer_path.read_text(encoding="utf-8"))
    mode = body.get("mode", "none")
    idx_stats = body.get("stats", {}).get(index_name, {})

    if not idx_stats:
        logger.warning(
            "No hay stats para '%s' en normalizer_stats.json. Umbral sin convertir.",
            index_name,
        )
        return raw_threshold

    if mode == "minmax":
        vmin = idx_stats["min"]
        vmax = idx_stats["max"]
        denom = vmax - vmin
        if denom == 0:
            return 0.5
        t_norm = (raw_threshold - vmin) / denom
        logger.info(
            "Umbral NDMI: raw=%.3f → normalizado=%.4f  (minmax [%.4f, %.4f])",
            raw_threshold, t_norm, vmin, vmax,
        )
        return float(np.clip(t_norm, 0.0, 1.0))

    elif mode == "zscore":
        mu    = idx_stats["mean"]
        sigma = idx_stats["std"] or 1.0
        t_norm = (raw_threshold - mu) / sigma
        logger.info(
            "Umbral NDMI: raw=%.3f → normalizado=%.4f  (zscore μ=%.4f σ=%.4f)",
            raw_threshold, t_norm, mu, sigma,
        )
        return float(t_norm)

    return raw_threshold


# ---------------------------------------------------------------------------
# Dataset de ventanas temporales (para PixelCNN / ViT-SITS)
# ---------------------------------------------------------------------------
class SITSPixelDataset(Dataset):
    """
    Genera muestras de ventanas temporales deslizantes desde señales .npz.

    Parámetros
    ----------
    signals_dir          : directorio con los .npz de señales
    parcel_ids           : lista de IDs a cargar (None = todos)
    ndmi_threshold       : umbral NDMI raw (default -0.1) — límite no_stress/moderate
    ndmi_severe_threshold: umbral NDMI raw (default -0.2) — límite moderate/severe
                           Solo aplica cuando num_classes=3
    normalizer_path      : ruta a normalizer_stats.json para convertir umbrales
    window_size          : número de fechas por ventana (default 24 ≈ 3 meses)
    window_step          : paso entre ventanas (default 4 fechas ≈ 20 días)
    stress_frac_min      : fracción mínima estresada → label=1 (solo binario)
    num_classes          : 2=binario, 3=triclase (default 3)
    exclude_channels     : canales a excluir del input al modelo
    transform            : función opcional aplicada a cada tensor de entrada
    """

    def __init__(
        self,
        signals_dir           : Path,
        parcel_ids            : list[str] | None = None,
        ndmi_threshold        : float = -0.1,
        ndmi_severe_threshold : float = -0.2,   # umbral severo
        normalizer_path       : Path | None = None,
        window_size           : int   = 24,
        window_step           : int   = 4,
        stress_frac_min       : float = 0.25,
        num_classes           : int   = 3,       # 2=binario, 3=triclase
        exclude_channels      : list[str] | None = None,
        transform             : Callable | None = None,
    ):
        self.signals_dir     = Path(signals_dir)
        self.ndmi_threshold  = ndmi_threshold
        self.window_size     = window_size
        self.window_step     = window_step
        self.stress_frac_min = stress_frac_min
        self.num_classes     = num_classes
        self.transform       = transform

        excluded = set(exclude_channels or [])
        self._keep_channels = [
            i for i, name in enumerate(CHANNEL_NAMES) if name not in excluded
        ]
        self._channel_names_used = [CHANNEL_NAMES[i] for i in self._keep_channels]
        self.n_channels = len(self._keep_channels)

        if excluded:
            logger.info(
                "Canales excluidos del input: %s  |  Usando: %s  (n=%d)",
                sorted(excluded), self._channel_names_used, self.n_channels,
            )

        # Convertir ambos umbrales al espacio normalizado
        self.threshold_norm = convert_threshold(
            ndmi_threshold, normalizer_path, index_name="NDMI"
        )
        # NUEVO: segundo umbral para separar moderado/severo
        self.threshold_severe_norm = convert_threshold(
            ndmi_severe_threshold, normalizer_path, index_name="NDMI"
        )

        self._samples: list[tuple[np.ndarray, np.ndarray, int]] = []  # CAMBIO: int label
        self._load(parcel_ids)

        # Reporte de balance de clases
        if self._samples:
            labels = [s[2] for s in self._samples]
            if num_classes == 2:
                n_pos = int(sum(labels))
                n_neg = len(labels) - n_pos
                logger.info(
                    "Dataset cargado: %d ventanas | sanas=%d estresadas=%d "
                    "(threshold_norm=%.4f  stress_frac≥%.2f)",
                    len(labels), n_neg, n_pos,
                    self.threshold_norm, self.stress_frac_min,
                )
                if n_pos == 0:
                    logger.warning("⚠️  No hay muestras estresadas (label=1).")
                elif n_neg == 0:
                    logger.warning("⚠️  No hay muestras sanas (label=0).")
            else:
                from collections import Counter
                dist = Counter(labels)
                logger.info(
                    "Dataset cargado: %d ventanas | "
                    "no_stress=%d  moderate=%d  severe=%d",
                    len(labels),
                    dist.get(0, 0), dist.get(1, 0), dist.get(2, 0),
                )

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
            data = npz["data"].astype(np.float32)   # (T, C) normalizado [0, 1]
            data = np.nan_to_num(data, nan=0.0)
            T    = len(data)

            doy_full = npz["doy"].astype(np.int32) if "doy" in npz else np.arange(1, T + 1, dtype=np.int32)

            if T < self.window_size:
                logger.warning(
                    "Parcela %s: solo %d fechas (< window_size=%d), skip",
                    npz_path.stem, T, self.window_size,
                )
                continue

            for start in range(0, T - self.window_size + 1, self.window_step):
                end    = start + self.window_size
                window = data[start:end]             # (W, C_full)
                doy_w  = doy_full[start:end]         # (W,) DOY

                ndmi_col = window[:, NDMI_CHANNEL]

                # CAMBIO: etiquetado según num_classes
                if self.num_classes == 3:
                    # Triclase: usar media NDMI de la ventana
                    mean_ndmi = float(np.mean(ndmi_col))
                    if mean_ndmi < self.threshold_severe_norm:
                        label = 2   # estrés severo
                    elif mean_ndmi < self.threshold_norm:
                        label = 1   # estrés moderado
                    else:
                        label = 0   # sin estrés
                else:
                    # Binario: lógica original con stressed_frac
                    stressed_frac = float(np.mean(ndmi_col < self.threshold_norm))
                    label = 1 if stressed_frac >= self.stress_frac_min else 0

                window_input = window[:, self._keep_channels]   # (W, n_channels)
                self._samples.append((window_input, doy_w, label))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data, doy_w, label = self._samples[idx]
        x   = torch.from_numpy(data)                              # (W, C)
        doy = torch.from_numpy(doy_w).long()                     # (W,)
        # CAMBIO: escalar long en lugar de (1,) float
        y   = torch.tensor(int(label), dtype=torch.long)         # escalar
        if self.transform:
            x = self.transform(x)
        return x, doy, y


# ---------------------------------------------------------------------------
# Dataset de patches (para PatchCNN) — sin cambios
# ---------------------------------------------------------------------------
class SITSPatchDataset(Dataset):
    """
    Genera muestras a nivel de chip completo desde los .npz de patches.

    Cada .npz tiene:
        data  : (T, C, H, W) float32  — datos normalizados [0, 1]
        dates : (T,) unicode
        doy   : (T,) int16

    Pseudo-etiqueta: mapa (H, W) donde cada píxel es 1.0 si su NDMI
    promedio temporal < threshold (en espacio normalizado).

    Parámetros
    ----------
    patches_dir     : directorio con los .npz de patches
    parcel_ids      : lista de IDs a cargar (None = todos)
    ndmi_threshold  : umbral NDMI en espacio RAW (se convierte internamente)
    normalizer_path : ruta a normalizer_stats.json
    max_t           : recorta la serie a las primeras max_t fechas (None = todas)
    transform       : función opcional sobre el tensor de entrada
    """

    def __init__(
        self,
        patches_dir     : Path,
        parcel_ids      : list[str] | None = None,
        ndmi_threshold  : float = -0.1,
        normalizer_path : Path | None = None,
        max_t           : int | None = None,
        exclude_channels: list[str] | None = None,
        transform       : Callable | None = None,
    ):
        self.patches_dir    = Path(patches_dir)
        self.ndmi_threshold = ndmi_threshold
        self.max_t          = max_t
        self.transform      = transform

        excluded = set(exclude_channels or [])
        self._keep_channels = [
            i for i, name in enumerate(CHANNEL_NAMES) if name not in excluded
        ]
        self.n_channels = len(self._keep_channels)

        self.threshold_norm = convert_threshold(
            ndmi_threshold, normalizer_path, index_name="NDMI"
        )

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

            if self.max_t is not None:
                data = data[: self.max_t]

            data = np.nan_to_num(data, nan=0.0)

            mean_ndmi = data[:, NDMI_CHANNEL, :, :].mean(axis=0)   # (H, W)
            label_map = (mean_ndmi < self.threshold_norm).astype(np.float32)

            data_input = data[:, self._keep_channels, :, :]
            self._samples.append((data_input, label_map))

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
    dataset_dir           : Path,
    mode                  : str   = "pixel",
    ndmi_threshold        : float = -0.1,
    ndmi_severe_threshold : float = -0.2,   # NUEVO
    window_size           : int   = 24,
    window_step           : int   = 4,
    stress_frac_min       : float = 0.25,
    num_classes           : int   = 3,       # NUEVO
    exclude_channels      : list[str] | None = None,
    max_t                 : int | None = None,
) -> dict[str, Dataset]:
    """
    Carga los datasets de train/val/test usando el split.json generado
    por time_series_builder.

    Retorna un dict con las claves presentes (subsets vacíos se omiten).
    """
    split_path      = dataset_dir / "split.json"
    normalizer_path = dataset_dir / "normalizer_stats.json"

    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
    else:
        npz_dir = dataset_dir / ("signals" if mode == "pixel" else "patches")
        all_ids = [f.stem for f in sorted(npz_dir.glob("*.npz"))]
        split   = {"train": all_ids, "val": [], "test": []}
        logger.warning(
            "split.json no encontrado — usando todas las parcelas en train: %s",
            all_ids,
        )

    if not split.get("val"):
        logger.warning(
            "⚠️  Conjunto de validación vacío (split.json val=[]).\n"
            "   Con pocas parcelas (n=%d), el split 70/15/15 no tiene parcelas para val.\n"
            "   Early stopping usará train_f1 como proxy.\n"
            "   Considera --no-split y validación cruzada cuando tengas más parcelas.",
            sum(len(v) for v in split.values()),
        )

    datasets: dict[str, Dataset] = {}
    for subset, ids in split.items():
        if not ids:
            continue
        if mode == "pixel":
            datasets[subset] = SITSPixelDataset(
                signals_dir           = dataset_dir / "signals",
                parcel_ids            = ids,
                ndmi_threshold        = ndmi_threshold,
                ndmi_severe_threshold = ndmi_severe_threshold,  # NUEVO
                normalizer_path       = normalizer_path,
                window_size           = window_size,
                window_step           = window_step,
                stress_frac_min       = stress_frac_min,
                num_classes           = num_classes,             # NUEVO
                exclude_channels      = exclude_channels,
            )
        else:
            datasets[subset] = SITSPatchDataset(
                patches_dir     = dataset_dir / "patches",
                parcel_ids      = ids,
                ndmi_threshold  = ndmi_threshold,
                normalizer_path = normalizer_path,
                max_t           = max_t,
                exclude_channels= exclude_channels,
            )

    return datasets


# ---------------------------------------------------------------------------
# Factory específica para el ViT (alias con nombre que train_vit.py importa)
# ---------------------------------------------------------------------------
def load_vit_datasets(
    dataset_dir           : Path,
    ndmi_threshold        : float = -0.1,
    ndmi_severe_threshold : float = -0.2,   # NUEVO
    window_size           : int   = 24,
    window_step           : int   = 4,
    stress_frac_min       : float = 0.25,
    num_classes           : int   = 3,       # NUEVO
    exclude_channels      : list[str] | None = None,
) -> dict[str, "SITSPixelDataset"]:
    """
    Wrapper de load_datasets fijado a mode='pixel' para el SITSViT.
    Devuelve datasets cuyo __getitem__ retorna (x, doy, y) — 3-tupla
    compatible con sits_collate_fn.
    """
    return load_datasets(
        dataset_dir           = dataset_dir,
        mode                  = "pixel",
        ndmi_threshold        = ndmi_threshold,
        ndmi_severe_threshold = ndmi_severe_threshold,  # NUEVO
        window_size           = window_size,
        window_step           = window_step,
        stress_frac_min       = stress_frac_min,
        num_classes           = num_classes,             # NUEVO
        exclude_channels      = exclude_channels,
    )