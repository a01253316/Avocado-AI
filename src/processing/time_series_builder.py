"""
time_series_builder.py
======================
Construye series de tiempo (SITS — Satellite Image Time Series) por parcela
a partir de los índices espectrales calculados por spectral_indices.py.

Genera dos tipos de dataset complementarios:

  1. PATCH dataset  (para CNN y ViT espaciotemporal)
     Shape por muestra: (T, C, H, W)
       T = timesteps   (fechas disponibles, variable por parcela)
       C = canales     (índices: NDVI, NDWI, NDMI, NDRE, EVI → 5)
       H, W = píxeles del chip (ej. 50×50 a 250m buffer / 10m resolución)
     Archivo: data/datasets/patches/{parcel_id}.npz
       → 'data'  : float32 (T, C, H, W)
       → 'dates' : str     (T,)  "YYYY-MM-DD"
       → 'doy'   : int16   (T,)   día del año [1-365]

  2. SIGNAL dataset (para ViT ligero y análisis temporal)
     Shape por muestra: (T, C)  — valor medio del chip por índice y fecha
     Archivo: data/datasets/signals/{parcel_id}.npz
       → 'data'  : float32 (T, C)
       → 'dates' : str     (T,)
       → 'doy'   : int16   (T,)

  3. MANIFEST CSV  data/datasets/manifest.csv
     Una fila por parcela con: parcel_id, n_dates, date_min, date_max,
     mean_ndvi, mean_ndmi, mean_ndre, has_patch, has_signal

Uso:
    python time_series_builder.py
    python time_series_builder.py --mode both --parcel-ids H1 H2 H3
    python time_series_builder.py --mode signal   # solo signals (rápido)
    python time_series_builder.py --split         # genera train/val/test split
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import rasterio

logger = logging.getLogger("time_series_builder")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
INDICES       = ["NDVI", "NDWI", "NDMI", "NDRE", "EVI"]   # orden fijo → canales 0–4
N_CHANNELS    = len(INDICES)
MIN_DATES     = 6        # Mínimo de fechas para incluir una parcela en el dataset
NODATA_VALUE  = np.nan
SPLIT_RATIOS  = (0.70, 0.15, 0.15)   # train / val / test


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ParcelTimeSeries:
    """
    Contiene la serie de tiempo completa de una parcela.
    """
    parcel_id : str
    dates     : list[str]               = field(default_factory=list)
    doy       : list[int]               = field(default_factory=list)
    patches   : list[np.ndarray]        = field(default_factory=list)
    signals   : np.ndarray | None       = None

    @property
    def n_dates(self) -> int:
        return len(self.dates)

    @property
    def is_valid(self) -> bool:
        return self.n_dates >= MIN_DATES


# ---------------------------------------------------------------------------
# 1. Lector de índices por parcela
# ---------------------------------------------------------------------------
class IndexReader:
    def __init__(self, indices_dir: Path):
        self._indices_dir = indices_dir

    def read_parcel(self, parcel_id: str) -> ParcelTimeSeries:
        parcel_dir = self._indices_dir / parcel_id
        if not parcel_dir.exists():
            logger.warning("Parcela %s no tiene directorio de índices", parcel_id)
            return ParcelTimeSeries(parcel_id=parcel_id)

        date_dirs = sorted(
            [d for d in parcel_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        if not date_dirs:
            return ParcelTimeSeries(parcel_id=parcel_id)

        ts = ParcelTimeSeries(parcel_id=parcel_id)
        ref_shape: tuple[int, int] | None = None

        for date_dir in date_dirs:
            date_str = date_dir.name
            patch, shape = self._load_patch(date_dir, ref_shape)

            if patch is None:
                logger.debug("Fecha %s/%s sin datos usables, skip", parcel_id, date_str)
                continue

            if ref_shape is None:
                ref_shape = shape

            ts.dates.append(date_str)
            ts.doy.append(self._date_to_doy(date_str))
            ts.patches.append(patch)

        logger.debug("Parcela %s: %d fechas leídas", parcel_id, ts.n_dates)
        return ts

    def _load_patch(
        self,
        date_dir: Path,
        ref_shape: tuple[int, int] | None,
    ) -> tuple[np.ndarray | None, tuple[int, int] | None]:
        channels: list[np.ndarray] = []
        shape: tuple[int, int] | None = None

        for idx in INDICES:
            tif_path = date_dir / f"{idx}.tif"
            if not tif_path.exists():
                if shape is not None:
                    channels.append(np.full(shape, np.nan, dtype=np.float32))
                elif ref_shape is not None:
                    channels.append(np.full(ref_shape, np.nan, dtype=np.float32))
                else:
                    channels.append(None)  # type: ignore[arg-type]
                continue

            with rasterio.open(tif_path) as src:
                arr   = src.read(1).astype(np.float32)
                shape = arr.shape

            channels.append(arr)

        if all(c is None for c in channels):
            return None, None

        h, w   = shape or ref_shape or (0, 0)
        channels = [
            np.full((h, w), np.nan, dtype=np.float32) if c is None else c
            for c in channels
        ]

        patch = np.stack(channels, axis=0)
        return patch, (h, w)

    @staticmethod
    def _date_to_doy(date_str: str) -> int:
        return datetime.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday


# ---------------------------------------------------------------------------
# 2. Generador de señales (medias espaciales)
# ---------------------------------------------------------------------------
class SignalBuilder:
    @staticmethod
    def build(ts: ParcelTimeSeries) -> np.ndarray:
        if not ts.patches:
            return np.empty((0, N_CHANNELS), dtype=np.float32)

        signals = []
        for patch in ts.patches:
            means = np.nanmean(patch.reshape(N_CHANNELS, -1), axis=1)
            signals.append(means)

        return np.stack(signals, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Normalización
# ---------------------------------------------------------------------------
class TSNormalizer:
    def __init__(self, mode: Literal["minmax", "zscore", "none"] = "minmax"):
        self.mode  = mode
        self.stats : dict = {}

    def fit(self, signals_list: list[np.ndarray]) -> "TSNormalizer":
        if self.mode == "none":
            return self
        all_data = np.concatenate(signals_list, axis=0)
        for i, idx_name in enumerate(INDICES):
            col = all_data[:, i]
            col = col[~np.isnan(col)]
            if self.mode == "minmax":
                self.stats[idx_name] = {
                    "min": float(np.percentile(col, 1)),
                    "max": float(np.percentile(col, 99)),
                }
            elif self.mode == "zscore":
                self.stats[idx_name] = {
                    "mean": float(np.mean(col)),
                    "std":  float(np.std(col)) or 1.0,
                }
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mode == "none" or not self.stats:
            return data
        out = data.copy()
        for i, idx_name in enumerate(INDICES):
            s = self.stats.get(idx_name, {})
            if self.mode == "minmax":
                vmin, vmax = s.get("min", -1), s.get("max", 1)
                denom = (vmax - vmin) or 1.0
                if data.ndim == 2:    # (T, C)
                    out[:, i] = np.clip((out[:, i] - vmin) / denom, 0, 1)
                else:                 # (T, C, H, W)
                    out[:, i] = np.clip((out[:, i] - vmin) / denom, 0, 1)
            elif self.mode == "zscore":
                mu, sigma = s.get("mean", 0), s.get("std", 1)
                if data.ndim == 2:
                    out[:, i] = (out[:, i] - mu) / sigma
                else:
                    out[:, i] = (out[:, i] - mu) / sigma
        return out

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"mode": self.mode, "stats": self.stats}, indent=2),
            encoding="utf-8",
        )
        logger.info("Normalizer stats guardadas en %s", path)

    @classmethod
    def load(cls, path: Path) -> "TSNormalizer":
        body = json.loads(path.read_text(encoding="utf-8"))
        obj  = cls(mode=body["mode"])
        obj.stats = body["stats"]
        return obj


# ---------------------------------------------------------------------------
# 4. Escritor de datasets
# ---------------------------------------------------------------------------
class DatasetWriter:
    def __init__(self, output_dir: Path):
        self._patches_dir = output_dir / "patches"
        self._signals_dir = output_dir / "signals"

    def write_patch(self, ts: ParcelTimeSeries) -> Path:
        self._patches_dir.mkdir(parents=True, exist_ok=True)
        out = self._patches_dir / f"{ts.parcel_id}.npz"
        np.savez_compressed(
            out,
            data  = np.stack(ts.patches, axis=0),   # (T, C, H, W)
            dates = np.array(ts.dates, dtype="U10"),
            doy   = np.array(ts.doy,   dtype=np.int16),
        )
        return out

    def write_signal(self, ts: ParcelTimeSeries, signals: np.ndarray) -> Path:
        self._signals_dir.mkdir(parents=True, exist_ok=True)
        out = self._signals_dir / f"{ts.parcel_id}.npz"
        np.savez_compressed(
            out,
            data  = signals,
            dates = np.array(ts.dates, dtype="U10"),
            doy   = np.array(ts.doy,   dtype=np.int16),
        )
        return out


# ---------------------------------------------------------------------------
# 5. Manifest + Split
# ---------------------------------------------------------------------------
class ManifestBuilder:
    def __init__(self, output_dir: Path):
        self._output_dir = output_dir

    def build(
        self,
        records: list[dict],
        parcel_ids: list[str],
        split: bool = False,
    ) -> None:
        manifest_path = self._output_dir / "manifest.csv"
        fieldnames    = [
            "parcel_id", "n_dates", "date_min", "date_max",
            "mean_ndvi", "mean_ndwi", "mean_ndmi", "mean_ndre", "mean_evi",
            "has_patch", "has_signal",
        ]
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        logger.info("Manifest guardado: %s (%d parcelas)", manifest_path, len(records))

        if split and parcel_ids:
            self._write_split(parcel_ids)

    def _write_split(self, parcel_ids: list[str]) -> None:
        rng    = np.random.default_rng(seed=42)
        ids    = np.array(parcel_ids)
        rng.shuffle(ids)
        n      = len(ids)
        n_tr   = int(n * SPLIT_RATIOS[0])
        n_val  = int(n * SPLIT_RATIOS[1])
        split  = {
            "train": ids[:n_tr].tolist(),
            "val":   ids[n_tr:n_tr + n_val].tolist(),
            "test":  ids[n_tr + n_val:].tolist(),
        }
        split_path = self._output_dir / "split.json"
        split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
        logger.info(
            "Split guardado: train=%d val=%d test=%d",
            len(split["train"]), len(split["val"]), len(split["test"])
        )


# ---------------------------------------------------------------------------
# 6. Orquestador principal
# ---------------------------------------------------------------------------
class TimeSeriesBuilder:
    def __init__(
        self,
        indices_dir : Path,
        output_dir  : Path,
        mode        : Literal["patch", "signal", "both"] = "both",
        normalize   : Literal["minmax", "zscore", "none"] = "minmax",
        dry_run     : bool = False,
    ):
        self._reader    = IndexReader(indices_dir)
        self._sig_builder = SignalBuilder()
        self._normalizer  = TSNormalizer(mode=normalize)
        self._writer    = DatasetWriter(output_dir)
        self._manifest  = ManifestBuilder(output_dir)
        self._mode      = mode
        self._dry_run   = dry_run
        self._output_dir = output_dir

    def run(
        self,
        parcels_csv : Path,
        parcel_ids  : list[str] | None = None,
        split       : bool = False,
    ) -> None:
        df = pd.read_csv(parcels_csv)
        if parcel_ids:
            df = df[df["parcel_id"].isin(parcel_ids)]

        all_ids = df["parcel_id"].tolist()
        logger.info("=" * 60)
        logger.info("Construyendo SITS — %d parcelas | modo=%s | norm=%s",
                    len(all_ids), self._mode, self._normalizer.mode)
        if self._dry_run:
            logger.info("[DRY-RUN] Sin escritura en disco")
        logger.info("=" * 60)

        # ── Paso 1: leer todas las series ──────────────────────
        all_ts: dict[str, ParcelTimeSeries] = {}
        for pid in all_ids:
            ts = self._reader.read_parcel(pid)
            if ts.is_valid:
                all_ts[pid] = ts
            else:
                logger.warning(
                    "Parcela %s omitida: solo %d fechas (mín. %d)",
                    pid, ts.n_dates, MIN_DATES
                )

        if not all_ts:
            logger.error("Ninguna parcela superó el mínimo de %d fechas.", MIN_DATES)
            sys.exit(1)

        # ── Paso 2: construir signals ───────────────────────────
        all_signals: dict[str, np.ndarray] = {}
        for pid, ts in all_ts.items():
            all_signals[pid] = self._sig_builder.build(ts)

        # ── Paso 3: fit normalizer sobre TODOS ──────────────────
        if self._normalizer.mode != "none":
            self._normalizer.fit(list(all_signals.values()))
            if not self._dry_run:
                self._output_dir.mkdir(parents=True, exist_ok=True)
                self._normalizer.save(self._output_dir / "normalizer_stats.json")

        # ── Paso 4: guardar datasets + construir manifest ───────
        records: list[dict] = []
        valid_ids: list[str] = []

        for pid, ts in all_ts.items():
            
            # NORMALIZACIÓN CORREGIDA 
            # 1. Normalizar las señales (T, C)
            signals_norm = self._normalizer.transform(all_signals[pid])
            
            # 2. Normalizar los parches (T, C, H, W) si el modo lo requiere
            if self._mode in ("patch", "both") and ts.patches:
                # Convertimos la lista de parches a un array 4D para normalizar
                patches_array = np.stack(ts.patches, axis=0)
                patches_norm = self._normalizer.transform(patches_array)
                # Reasignamos a la dataclass como lista para mantener compatibilidad
                ts.patches = list(patches_norm)

            has_patch = has_signal = False

            if not self._dry_run:
                if self._mode in ("patch", "both"):
                    self._writer.write_patch(ts)
                    has_patch = True
                if self._mode in ("signal", "both"):
                    self._writer.write_signal(ts, signals_norm)
                    has_signal = True

            # Stats para el manifest (desde signals sin normalizar)
            raw_sig = all_signals[pid]       
            col_means = {
                f"mean_{idx.lower()}": (
                    float(np.nanmean(raw_sig[:, i]))
                    if raw_sig.size > 0 else None
                )
                for i, idx in enumerate(INDICES)
            }

            records.append({
                "parcel_id": pid,
                "n_dates":   ts.n_dates,
                "date_min":  ts.dates[0]  if ts.dates else "",
                "date_max":  ts.dates[-1] if ts.dates else "",
                **col_means,
                "has_patch":  has_patch,
                "has_signal": has_signal,
            })
            valid_ids.append(pid)

            logger.info(
                "✓ %s | T=%d | NDVI=%.3f NDMI=%.3f NDRE=%.3f",
                pid, ts.n_dates,
                col_means.get("mean_ndvi") or float("nan"),
                col_means.get("mean_ndmi") or float("nan"),
                col_means.get("mean_ndre") or float("nan"),
            )

        # ── Paso 5: manifest + split ────────────────────────────
        if not self._dry_run:
            self._manifest.build(records, valid_ids, split=split)

        logger.info("=" * 60)
        logger.info("Dataset listo: %d parcelas válidas de %d totales",
                    len(valid_ids), len(all_ids))
        logger.info("Salida en: %s", self._output_dir)
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construye SITS (series de tiempo) desde índices espectrales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--indices-dir", type=Path,
        default=Path("data/processed/indices/"),
        help="Directorio con índices calculados (default: data/processed/indices/)",
    )
    parser.add_argument(
        "--parcels", type=Path,
        default=Path("data/raw/parcels/parcelas.csv"),
        help="CSV de parcelas (default: data/raw/parcels/parcelas.csv)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/datasets/"),
        help="Directorio de salida del dataset (default: data/datasets/)",
    )
    parser.add_argument(
        "--mode", choices=["patch", "signal", "both"], default="both",
        help="Tipo de dataset: patch=(T,C,H,W), signal=(T,C), both=ambos (default: both)",
    )
    parser.add_argument(
        "--normalize", choices=["minmax", "zscore", "none"], default="minmax",
        help="Estrategia de normalización (default: minmax)",
    )
    parser.add_argument(
        "--parcel-ids", nargs="+", metavar="ID",
        help="Parcelas específicas (ej: H1 H2). Sin esto: todas.",
    )
    parser.add_argument(
        "--split", action="store_true",
        help="Genera split.json con train/val/test (70/15/15)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Muestra qué construiría sin escribir en disco",
    )
    args = parser.parse_args()

    if not args.parcels.exists():
        logger.error("CSV de parcelas no encontrado: %s", args.parcels)
        sys.exit(1)

    builder = TimeSeriesBuilder(
        indices_dir = args.indices_dir,
        output_dir  = args.output,
        mode        = args.mode,
        normalize   = args.normalize,
        dry_run     = args.dry_run,
    )
    builder.run(
        parcels_csv = args.parcels,
        parcel_ids  = args.parcel_ids,
        split       = args.split,
    )


if __name__ == "__main__":
    main()