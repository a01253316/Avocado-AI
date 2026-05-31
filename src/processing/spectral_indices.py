"""
spectral_indices.py
===================
Calcula índices de vegetación / estrés hídrico a partir de las bandas
Sentinel-2. Soporta tanto archivos individuales (B02.tif) como
archivos multibanda (parcel_multiband.tif).

Índices implementados:
    NDVI  = (B08 - B04) / (B08 + B04)          Vigor vegetal general
    NDWI  = (B03 - B08) / (B03 + B08)          Contenido de agua en vegetación
    NDMI  = (B08 - B11) / (B08 + B11)          Humedad en hoja/dosel          
    NDRE  = (B08 - B05) / (B08 + B05)          Estrés temprano (clorofila)    
    EVI   = 2.5*(B08-B04)/(B08+6*B04-7.5*B02+1)  Vegetación en zonas densas
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio
from rasterio.enums import Resampling

logger = logging.getLogger("spectral_indices")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constantes y Configuración
# ---------------------------------------------------------------------------
DN_SCALE      = 10_000.0   # Factor de escala Sentinel-2 L2A
NODATA_DN     = 0          # Valor NoData en DN (antes de escalar)
NODATA_FLOAT  = np.nan     # Valor NoData en el raster de salida
EVI_G         = 2.5        # Ganancia EVI
EVI_C1        = 6.0        # Coeficiente aerosol rojo
EVI_C2        = 7.5        # Coeficiente aerosol azul
EVI_L         = 1.0        # Factor de ajuste suelo EVI

# Bandas requeridas por cada índice
INDEX_BANDS: dict[str, list[str]] = {
    "NDVI": ["B08", "B04"],
    "NDWI": ["B03", "B08"],
    "NDMI": ["B08", "B11"],
    "NDRE": ["B08", "B05"],
    "EVI":  ["B08", "B04", "B02"],
}

# IMPORTANTE: Si usas parcel_multiband.tif, este es el orden esperado de las capas.
# Cambia los números si tu metadata.json indica un orden distinto.
MULTIBAND_MAPPING = {
    "B02": 1,  # Capa 1: Azul
    "B03": 2,  # Capa 2: Verde
    "B04": 3,  # Capa 3: Rojo
    "B05": 4,  # Capa 4: Red Edge
    "B08": 5,  # Capa 5: NIR
    "B11": 6,  # Capa 6: SWIR
}

BANDS_20M = {"B05", "B11"}

# ---------------------------------------------------------------------------
# 1. Carga y preprocesado de bandas (Archivos Individuales)
# ---------------------------------------------------------------------------
class BandLoader:
    """Lee un TIFF individual, normaliza DN→reflectancia y resamplea si es necesario."""

    def __init__(self, target_shape: tuple[int, int] | None = None):
        self._target = target_shape

    def load(self, tif_path: Path, band_name: str) -> tuple[np.ndarray, dict]:
        if not tif_path.exists():
            raise FileNotFoundError(f"Banda {band_name} no encontrada: {tif_path}")

        with rasterio.open(tif_path) as src:
            data    = src.read(1).astype(np.float32)
            profile = src.profile.copy()

            if band_name in BANDS_20M and self._target is not None:
                th, tw = self._target
                if data.shape != (th, tw):
                    data = self._resample(src, th, tw)
                    profile.update(height=th, width=tw)

        data = np.where(data == NODATA_DN, np.nan, data)
        data /= DN_SCALE
        data = np.clip(data, 0.0, 1.0)
        return data, profile

    @staticmethod
    def _resample(src: rasterio.DatasetReader, th: int, tw: int) -> np.ndarray:
        return src.read(
            1,
            out_shape=(th, tw),
            resampling=Resampling.bilinear,
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Calculador de índices
# ---------------------------------------------------------------------------
class SpectralIndexCalculator:
    _REGISTRY: dict[str, Callable[[dict[str, np.ndarray]], np.ndarray]] = {}

    @classmethod
    def _register(cls, name: str):
        def decorator(fn):
            cls._REGISTRY[name] = fn
            return fn
        return decorator

    def compute(self, index_name: str, bands: dict[str, np.ndarray]) -> np.ndarray:
        if index_name not in self._REGISTRY:
            raise ValueError(f"Índice '{index_name}' no implementado.")
        return self._REGISTRY[index_name](bands)

    def compute_all(self, bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        results: dict[str, np.ndarray] = {}
        for name, required in INDEX_BANDS.items():
            missing = [b for b in required if b not in bands]
            if missing:
                logger.warning("Índice %s omitido: faltan bandas %s", name, missing)
                continue
            try:
                results[name] = self.compute(name, bands)
            except Exception as exc:
                logger.error("Error calculando %s: %s", name, exc)
        return results

    @staticmethod
    def _safe_norm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = np.where((a + b) == 0, np.nan, (a - b) / (a + b))
        return result.astype(np.float32)

@SpectralIndexCalculator._register("NDVI")
def _ndvi(bands: dict) -> np.ndarray:
    return SpectralIndexCalculator._safe_norm_diff(bands["B08"], bands["B04"])

@SpectralIndexCalculator._register("NDWI")
def _ndwi(bands: dict) -> np.ndarray:
    return SpectralIndexCalculator._safe_norm_diff(bands["B03"], bands["B08"])

@SpectralIndexCalculator._register("NDMI")
def _ndmi(bands: dict) -> np.ndarray:
    return SpectralIndexCalculator._safe_norm_diff(bands["B08"], bands["B11"])

@SpectralIndexCalculator._register("NDRE")
def _ndre(bands: dict) -> np.ndarray:
    return SpectralIndexCalculator._safe_norm_diff(bands["B08"], bands["B05"])

@SpectralIndexCalculator._register("EVI")
def _evi(bands: dict) -> np.ndarray:
    B08, B04, B02 = bands["B08"], bands["B04"], bands["B02"]
    denom = B08 + EVI_C1 * B04 - EVI_C2 * B02 + EVI_L
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        evi = np.where(denom == 0, np.nan, EVI_G * (B08 - B04) / denom)
    return np.clip(evi, -1.0, 1.0).astype(np.float32)

# ---------------------------------------------------------------------------
# 3. Escritura de TIFFs de índices y Estadísticas
# ---------------------------------------------------------------------------
class IndexWriter:
    @staticmethod
    def write(data: np.ndarray, ref_profile: dict, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        profile = ref_profile.copy()
        profile.update(
            dtype=rasterio.float32,
            count=1,
            nodata=np.nan,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data.astype(np.float32), 1)

def compute_summary(indices: dict[str, np.ndarray]) -> dict:
    summary: dict[str, dict] = {}
    for name, arr in indices.items():
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            summary[name] = {"mean": None, "std": None, "min": None, "max": None, "pct_valid": 0.0}
        else:
            summary[name] = {
                "mean": float(np.mean(valid)),
                "std": float(np.std(valid)),
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
                "pct_valid": float(100.0 * valid.size / arr.size),
            }
    return summary

# ---------------------------------------------------------------------------
# 4. Runner por fecha (Extrae de Multibanda o Individuales)
# ---------------------------------------------------------------------------
class DateProcessor:
    def __init__(self, dry_run: bool = False):
        self._calculator = SpectralIndexCalculator()
        self._writer     = IndexWriter()
        self._dry_run    = dry_run

    def process(self, bands_dir: Path, output_dir: Path) -> dict | None:
        if self._already_done(output_dir):
            logger.debug("  ↳ %s ya procesado, skip", bands_dir.name)
            return None

        bands: dict[str, np.ndarray] = {}
        ref_profile = None

        multiband_file = bands_dir / "parcel_multiband.tif"

        # CASO 1: Existe el archivo parcel_multiband.tif (tu caso actual)
        if multiband_file.exists():
            if self._dry_run:
                logger.info("  [DRY-RUN] Procesaría multibanda %s", multiband_file)
                return None
            
            logger.info("  Leyendo archivo multibanda: %s", multiband_file.name)
            with rasterio.open(multiband_file) as src:
                ref_profile = src.profile.copy()
                for band_name, band_idx in MULTIBAND_MAPPING.items():
                    if band_idx <= src.count:
                        data = src.read(band_idx).astype(np.float32)
                        data = np.where(data == NODATA_DN, np.nan, data)
                        data /= DN_SCALE
                        bands[band_name] = np.clip(data, 0.0, 1.0)
                    else:
                        logger.warning("  Falta capa %d (%s) en %s", band_idx, band_name, multiband_file.name)
        
        # CASO 2: Son archivos individuales (B02.tif, B08.tif, etc.)
        else:
            band_names = {"B02", "B03", "B04", "B05", "B08", "B11"}
            available = {}
            for ext in ("*.jp2", "*.tif", "*.JP2", "*.TIF"):
                for p in bands_dir.glob(ext):
                    for b in band_names:
                        if b in p.name.upper() and b not in available:
                            available[b] = p
                            break
                            
            if not available:
                logger.warning("  No hay bandas ni multibanda en %s, skip", bands_dir)
                return None

            if self._dry_run:
                logger.info("  [DRY-RUN] Procesaría %d bandas en %s", len(available), bands_dir)
                return None

            ref_name = "B08" if "B08" in available else next(iter(available))
            ref_arr, ref_profile = BandLoader().load(available[ref_name], ref_name)
            
            loader = BandLoader(target_shape=ref_arr.shape)
            bands = {ref_name: ref_arr}
            for band_name, tif_path in available.items():
                if band_name == ref_name: continue
                try:
                    bands[band_name], _ = loader.load(tif_path, band_name)
                except Exception as exc:
                    logger.warning("  Error cargando %s: %s", band_name, exc)

        # ── Calcular índices ──
        indices = self._calculator.compute_all(bands)
        if not indices:
            logger.warning("  No se calculó ningún índice en %s", bands_dir)
            return None

        # ── Guardar TIFFs ──
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in indices.items():
            out_path = output_dir / f"{name}.tif"
            self._writer.write(arr, ref_profile, out_path)

        # ── Summary JSON ──
        summary = compute_summary(indices)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info(
            "  ✓ %s → %s | NDVI=%.3f NDMI=%.3f NDRE=%.3f",
            bands_dir.parent.name,
            bands_dir.name,
            summary.get("NDVI", {}).get("mean") or float("nan"),
            summary.get("NDMI", {}).get("mean") or float("nan"),
            summary.get("NDRE", {}).get("mean") or float("nan"),
        )
        return summary

    @staticmethod
    def _already_done(output_dir: Path) -> bool:
        expected = [f"{idx}.tif" for idx in INDEX_BANDS] + ["summary.json"]
        return all((output_dir / f).exists() for f in expected)

# ---------------------------------------------------------------------------
# 5. Orquestador principal
# ---------------------------------------------------------------------------
class IndexPipelineRunner:
    def __init__(self, dry_run: bool = False):
        self._processor = DateProcessor(dry_run=dry_run)
        self._dry_run   = dry_run

    def run(self, sentinel2_dir: Path, output_dir: Path, parcel_ids: list[str] | None = None) -> dict:
        if not sentinel2_dir.exists():
            logger.error("Directorio de entrada no existe: %s", sentinel2_dir)
            sys.exit(1)

        parcel_dirs = sorted([d for d in sentinel2_dir.iterdir() if d.is_dir()], key=lambda p: p.name)
        if parcel_ids:
            parcel_dirs = [d for d in parcel_dirs if d.name in parcel_ids]

        if not parcel_dirs:
            logger.warning("No se encontraron parcelas en %s", sentinel2_dir)
            return {}

        total_parcels, total_dates = len(parcel_dirs), 0
        global_report: dict = {}

        logger.info("=" * 60)
        logger.info("Calculando índices — %d parcelas", total_parcels)
        if self._dry_run: logger.info("[DRY-RUN] No se escribirá nada en disco")
        logger.info("=" * 60)

        for parcel_dir in parcel_dirs:
            parcel_id = parcel_dir.name
            date_dirs = sorted([d for d in parcel_dir.iterdir() if d.is_dir()])

            if not date_dirs:
                continue

            logger.info("Parcela %s — %d fechas", parcel_id, len(date_dirs))
            global_report[parcel_id] = {}

            for date_dir in date_dirs:
                date_str = date_dir.name
                out_dir  = output_dir / parcel_id / date_str
                summary  = self._processor.process(date_dir, out_dir)
                if summary:
                    global_report[parcel_id][date_str] = summary
                    total_dates += 1

        report_path = output_dir / "pipeline_report.json"
        if not self._dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(global_report, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info("=" * 60)
        logger.info("Completado: %d parcelas × %d fechas procesadas", total_parcels, total_dates)
        logger.info("=" * 60)
        return global_report

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Calcula índices espectrales desde Sentinel-2")
    parser.add_argument("--input", type=Path, default=Path("data/raw/sentinel2/"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/indices/"))
    parser.add_argument("--parcel-ids", nargs="+", metavar="ID")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runner = IndexPipelineRunner(dry_run=args.dry_run)
    runner.run(sentinel2_dir=args.input, output_dir=args.output, parcel_ids=args.parcel_ids)

if __name__ == "__main__":
    main()