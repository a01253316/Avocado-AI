"""
spectral_indices.py
===================
Calcula índices de vegetación / estrés hídrico a partir de las bandas
Sentinel-2 descargadas mediante la Process API de CDSE (Sentinel Hub).

Fuente primaria: parcel_multiband.tif  → GeoTIFF multibanda FLOAT32
  generado por sentinel2_downloader.py usando la Process API.
  Los valores están ya en reflectancia superficial [0.0, 1.0];
  NO se aplica ningún factor de escala DN.

Fuente legada: archivos individuales (B02.tif, B04.jp2, etc.) en DN uint16
  (descarga directa de CDSE sin Process API).
  Se detecta por dtype del raster y se divide por DN_SCALE=10000.

Índices implementados:
    NDVI  = (B08 - B04) / (B08 + B04)             Vigor vegetal general
    NDWI  = (B03 - B08) / (B03 + B08)             Contenido de agua en vegetación
    NDMI  = (B08 - B11) / (B08 + B11)             Humedad en hoja/dosel ★
    NDRE  = (B08 - B05) / (B08 + B05)             Estrés temprano (clorofila) ★
    EVI   = 2.5*(B08-B04)/(B08+6*B04-7.5*B02+1)  Vegetación en zonas densas

Salida por fecha:
    data/processed/indices/<parcel_id>/<date>/
        NDVI.tif, NDWI.tif, NDMI.tif, NDRE.tif, EVI.tif  — FLOAT32 [-1, 1]
        summary.json  — estadísticas por índice (mean, std, min, max, pct_valid)
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
# Constantes
# ---------------------------------------------------------------------------
DN_SCALE             = 10_000.0        # Factor de escala DN → reflectancia (solo para datos uint16)
NODATA_DN            = 0               # Nodata en valores DN (uint16, descarga directa)
NODATA_FLOAT_THRESH  = 0.0             # Sentinel Hub devuelve 0.0 exacto para píxeles sin datos en FLOAT32

# dtypes que indican que el raster YA está en reflectancia (Process API)
FLOAT_DTYPES = frozenset({"float32", "float64"})

# Bandas requeridas por cada índice (deben estar presentes en el multiband TIF)
INDEX_BANDS: dict[str, list[str]] = {
    "NDVI": ["B08", "B04"],
    "NDWI": ["B03", "B08"],
    "NDMI": ["B08", "B11"],
    "NDRE": ["B08", "B05"],
    "EVI":  ["B08", "B04", "B02"],
}

# Orden de bandas en parcel_multiband.tif — debe coincidir con el evalscript
# del sentinel2_downloader.py (orden: B02, B03, B04, B05, B08, B11)
MULTIBAND_MAPPING: dict[str, int] = {
    "B02": 1,   # Azul        490 nm   → EVI
    "B03": 2,   # Verde       560 nm   → NDWI
    "B04": 3,   # Rojo        665 nm   → NDVI, EVI
    "B05": 4,   # Red Edge    705 nm   → NDRE
    "B08": 5,   # NIR         842 nm   → todos
    "B11": 6,   # SWIR1      1610 nm   → NDMI
}

# Bandas nativas a 20 m (solo relevante para archivos individuales .jp2 legados)
BANDS_20M = {"B05", "B11"}

EVI_G  = 2.5   # Ganancia EVI
EVI_C1 = 6.0   # Coeficiente aerosol rojo
EVI_C2 = 7.5   # Coeficiente aerosol azul
EVI_L  = 1.0   # Factor ajuste suelo EVI


# ---------------------------------------------------------------------------
# Helpers de escala
# ---------------------------------------------------------------------------
def to_reflectance(data: np.ndarray, src_dtype: str) -> np.ndarray:
    """
    Convierte data a reflectancia [0, 1] según el dtype del raster fuente.

    - FLOAT32 (Process API): los valores ya son reflectancia; solo enmascara nodata (== 0.0)
    - uint16 / uint8 (DN legado): divide por DN_SCALE=10000 y enmascara nodata (== 0)
    """
    if src_dtype in FLOAT_DTYPES:
        # Process API: Sentinel Hub usa 0.0 exacto para píxeles fuera de área / sin datos
        data = np.where(data <= NODATA_FLOAT_THRESH, np.nan, data)
    else:
        # DN uint16: 0 es nodata, escalar a reflectancia
        data = np.where(data == NODATA_DN, np.nan, data)
        data = data / DN_SCALE

    return np.clip(data, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 1. Carga de bandas — archivos individuales (modo legado .jp2 / .tif DN)
# ---------------------------------------------------------------------------
class BandLoader:
    """
    Lee un TIFF/JP2 individual, convierte DN→reflectancia (si aplica)
    y resamplea bandas de 20 m al shape de referencia.
    Solo se usa cuando NO existe parcel_multiband.tif.
    """

    def __init__(self, target_shape: tuple[int, int] | None = None):
        self._target = target_shape

    def load(self, tif_path: Path, band_name: str) -> tuple[np.ndarray, dict]:
        if not tif_path.exists():
            raise FileNotFoundError(f"Banda {band_name} no encontrada: {tif_path}")

        with rasterio.open(tif_path) as src:
            src_dtype = src.dtypes[0]
            data      = src.read(1).astype(np.float32)
            profile   = src.profile.copy()

            if band_name in BANDS_20M and self._target is not None:
                th, tw = self._target
                if data.shape != (th, tw):
                    data = self._resample(src, th, tw)
                    profile.update(height=th, width=tw)

        data = to_reflectance(data, src_dtype)
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
# 3. Escritura de TIFFs de índices y estadísticas
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
            summary[name] = {
                "mean": None, "std": None,
                "min": None,  "max": None, "pct_valid": 0.0,
            }
        else:
            summary[name] = {
                "mean":      float(np.mean(valid)),
                "std":       float(np.std(valid)),
                "min":       float(np.min(valid)),
                "max":       float(np.max(valid)),
                "pct_valid": float(100.0 * valid.size / arr.size),
            }
    return summary


# ---------------------------------------------------------------------------
# 4. Runner por fecha — extrae de multibanda (Process API) o individuales
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

        multiband_file = bands_dir / "parcel_multiband.tif"

        # ── CASO 1: parcel_multiband.tif (Process API — flujo principal) ──
        if multiband_file.exists():
            return self._process_multiband(multiband_file, bands_dir, output_dir)

        # ── CASO 2: archivos individuales (DN legado, .jp2 / .tif) ──────
        return self._process_individual(bands_dir, output_dir)

    # ------------------------------------------------------------------
    def _process_multiband(
        self, multiband_file: Path, bands_dir: Path, output_dir: Path
    ) -> dict | None:
        if self._dry_run:
            logger.info("  [DRY-RUN] Procesaría multibanda %s", multiband_file)
            return None

        bands: dict[str, np.ndarray] = {}
        ref_profile: dict | None = None

        with rasterio.open(multiband_file) as src:
            src_dtype   = src.dtypes[0]      # 'float32' si viene de Process API
            ref_profile = src.profile.copy()
            n_bands     = src.count

            logger.debug(
                "  Multibanda: %d capas, dtype=%s, size=%dx%d",
                n_bands, src_dtype, src.width, src.height,
            )

            for band_name, band_idx in MULTIBAND_MAPPING.items():
                if band_idx > n_bands:
                    logger.warning(
                        "  Falta capa %d (%s) en %s — solo hay %d bandas",
                        band_idx, band_name, multiband_file.name, n_bands,
                    )
                    continue

                raw  = src.read(band_idx).astype(np.float32)
                data = to_reflectance(raw, src_dtype)
                bands[band_name] = data

        if ref_profile is None or not bands:
            logger.warning("  Sin bandas utilizables en %s", multiband_file)
            return None

        return self._compute_and_save(bands, ref_profile, bands_dir, output_dir)

    # ------------------------------------------------------------------
    def _process_individual(
        self, bands_dir: Path, output_dir: Path
    ) -> dict | None:
        """Modo legado: archivos individuales .jp2 / .tif con valores DN."""
        band_names = {"B02", "B03", "B04", "B05", "B08", "B11"}
        available: dict[str, Path] = {}

        for ext in ("*.jp2", "*.tif", "*.JP2", "*.TIF"):
            for p in bands_dir.glob(ext):
                for b in band_names:
                    if b in p.name.upper() and b not in available:
                        available[b] = p
                        break

        if not available:
            logger.warning("  Sin archivos de banda en %s, skip", bands_dir)
            return None

        if self._dry_run:
            logger.info(
                "  [DRY-RUN] Procesaría %d bandas individuales en %s",
                len(available), bands_dir,
            )
            return None

        ref_name     = "B08" if "B08" in available else next(iter(available))
        ref_arr, ref_profile = BandLoader().load(available[ref_name], ref_name)

        loader = BandLoader(target_shape=ref_arr.shape)
        bands: dict[str, np.ndarray] = {ref_name: ref_arr}

        for band_name, tif_path in available.items():
            if band_name == ref_name:
                continue
            try:
                bands[band_name], _ = loader.load(tif_path, band_name)
            except Exception as exc:
                logger.warning("  Error cargando %s: %s", band_name, exc)

        return self._compute_and_save(bands, ref_profile, bands_dir, output_dir)

    # ------------------------------------------------------------------
    def _compute_and_save(
        self,
        bands: dict[str, np.ndarray],
        ref_profile: dict,
        bands_dir: Path,
        output_dir: Path,
    ) -> dict | None:
        indices = self._calculator.compute_all(bands)
        if not indices:
            logger.warning("  No se calculó ningún índice en %s", bands_dir)
            return None

        output_dir.mkdir(parents=True, exist_ok=True)

        for name, arr in indices.items():
            self._writer.write(arr, ref_profile, output_dir / f"{name}.tif")

        summary = compute_summary(indices)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        logger.info(
            "  ✓ %s/%s | NDVI=%.3f NDWI=%.3f NDMI=%.3f NDRE=%.3f EVI=%.3f",
            bands_dir.parent.name,
            bands_dir.name,
            summary.get("NDVI", {}).get("mean") or float("nan"),
            summary.get("NDWI", {}).get("mean") or float("nan"),
            summary.get("NDMI", {}).get("mean") or float("nan"),
            summary.get("NDRE", {}).get("mean") or float("nan"),
            summary.get("EVI",  {}).get("mean") or float("nan"),
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

    def run(
        self,
        sentinel2_dir: Path,
        output_dir: Path,
        parcel_ids: list[str] | None = None,
    ) -> dict:
        if not sentinel2_dir.exists():
            logger.error("Directorio de entrada no existe: %s", sentinel2_dir)
            sys.exit(1)

        parcel_dirs = sorted(
            [d for d in sentinel2_dir.iterdir() if d.is_dir()],
            key=lambda p: p.name,
        )
        if parcel_ids:
            parcel_dirs = [d for d in parcel_dirs if d.name in parcel_ids]

        if not parcel_dirs:
            logger.warning("No se encontraron parcelas en %s", sentinel2_dir)
            return {}

        total_parcels = len(parcel_dirs)
        total_dates   = 0
        global_report: dict = {}

        logger.info("=" * 60)
        logger.info("Calculando índices espectrales — %d parcelas", total_parcels)
        if self._dry_run:
            logger.info("[DRY-RUN] No se escribirá nada en disco")
        logger.info("=" * 60)

        for parcel_dir in parcel_dirs:
            parcel_id = parcel_dir.name
            date_dirs = sorted([d for d in parcel_dir.iterdir() if d.is_dir()])

            if not date_dirs:
                logger.warning("Parcela %s sin fechas, skip", parcel_id)
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
            report_path.write_text(
                json.dumps(global_report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        logger.info("=" * 60)
        logger.info(
            "Completado: %d parcelas | %d fechas procesadas",
            total_parcels, total_dates,
        )
        if not self._dry_run:
            logger.info("Reporte guardado en: %s", report_path)
        logger.info("=" * 60)
        return global_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcula índices espectrales (NDVI/NDWI/NDMI/NDRE/EVI) desde Sentinel-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Todas las parcelas
  python spectral_indices.py --input data/raw/sentinel2/ --output data/processed/indices/

  # Solo dos parcelas específicas
  python spectral_indices.py --parcel-ids H1 H2

  # Dry-run para verificar qué se procesaría
  python spectral_indices.py --dry-run
        """,
    )
    parser.add_argument(
        "--input",  type=Path, default=Path("data/raw/sentinel2/"),
        help="Directorio raíz de los TIFFs descargados (default: data/raw/sentinel2/)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/indices/"),
        help="Directorio de salida de índices (default: data/processed/indices/)",
    )
    parser.add_argument(
        "--parcel-ids", nargs="+", metavar="ID",
        help="IDs de parcelas a procesar (ej: H1 H2). Sin esto: todas.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Muestra qué procesaría sin escribir nada en disco",
    )
    args = parser.parse_args()

    runner = IndexPipelineRunner(dry_run=args.dry_run)
    runner.run(
        sentinel2_dir=args.input,
        output_dir=args.output,
        parcel_ids=args.parcel_ids,
    )


if __name__ == "__main__":
    main()
