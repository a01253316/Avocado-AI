"""
spectral_indices.py
===================
Calcula índices de vegetación / estrés hídrico a partir de las bandas
Sentinel-2 descargadas por sentinel2_downloader.py.

Índices implementados:
    NDVI  = (B08 - B04) / (B08 + B04)          Vigor vegetal general
    NDWI  = (B03 - B08) / (B03 + B08)          Contenido de agua en vegetación
    NDMI  = (B08 - B11) / (B08 + B11)          Humedad en hoja/dosel          ★
    NDRE  = (B08 - B05) / (B08 + B05)          Estrés temprano (clorofila)    ★
    EVI   = 2.5*(B08-B04)/(B08+6*B04-7.5*B02+1)  Vegetación en zonas densas

Consideraciones técnicas críticas:
    1. B05 (Red Edge, 705 nm) y B11 (SWIR1, 1610 nm) están nativamente a 20 m/px.
       Se resamplearán a 10 m/px (bilineal) para alinearlas con el resto.
    2. Los DN de Sentinel-2 L2A están en [0, 10000]. Se normalizan a [0.0, 1.0]
       dividiendo entre 10 000 antes de calcular cualquier índice.
    3. La división por cero se maneja con np.errstate + np.where → NaN.
    4. Píxeles con valor NoData (0 en DN, o NaN tras normalizar) se propagan
       y quedan como NaN en el índice final.
    5. Los TIFFs de salida usan float32 con nodata=NaN para ahorrar espacio
       vs float64, sin perder precisión significativa en índices [−1, 1].

Estructura de entrada (salida del downloader):
    data/raw/sentinel2/
    └── {parcel_id}/
        └── {YYYY-MM-DD}/
            ├── B02.tif  B03.tif  B04.tif
            ├── B05.tif  B08.tif  B11.tif
            └── metadata.json

Estructura de salida:
    data/processed/indices/
    └── {parcel_id}/
        └── {YYYY-MM-DD}/
            ├── NDVI.tif  NDWI.tif  NDMI.tif
            ├── NDRE.tif  EVI.tif
            └── summary.json   ← estadísticas por índice (mean, std, min, max, pct_valid)

Uso:
    python spectral_indices.py
    python spectral_indices.py --input data/raw/sentinel2/ --output data/processed/indices/
    python spectral_indices.py --parcel-ids H1 H2 H3
    python spectral_indices.py --parcel-ids H1 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds

logger = logging.getLogger("spectral_indices")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
DN_SCALE      = 10_000.0   # Factor de escala Sentinel-2 L2A
NODATA_DN     = 0          # Valor NoData en DN (antes de escalar)
NODATA_FLOAT  = np.nan     # Valor NoData en el raster de salida
EVI_G         = 2.5        # Ganancia EVI
EVI_C1        = 6.0        # Coeficiente aerosol rojo
EVI_C2        = 7.5        # Coeficiente aerosol azul
EVI_L         = 1.0        # Factor de ajuste suelo EVI

# Bandas que necesita cada índice → usadas para detectar cuáles faltan
INDEX_BANDS: dict[str, list[str]] = {
    "NDVI": ["B08", "B04"],
    "NDWI": ["B03", "B08"],
    "NDMI": ["B08", "B11"],
    "NDRE": ["B08", "B05"],
    "EVI":  ["B08", "B04", "B02"],
}

# Bandas nativas a 20 m → se resamplearán a 10 m
BANDS_20M = {"B05", "B11"}


# ---------------------------------------------------------------------------
# 1. Carga y preprocesado de bandas
# ---------------------------------------------------------------------------
class BandLoader:
    """
    Lee un TIFF de banda, normaliza DN→reflectancia y resamplea si es de 20 m.

    Parámetros
    ----------
    target_shape : (height, width) en píxeles a 10 m de resolución.
                   Las bandas a 20 m se redimensionan con interpolación bilineal.
    """

    def __init__(self, target_shape: tuple[int, int] | None = None):
        self._target = target_shape

    def load(self, tif_path: Path, band_name: str) -> tuple[np.ndarray, dict]:
        """
        Carga una banda y retorna (array float32, perfil rasterio).

        El array tiene shape (H, W) y valores en [0.0, 1.0].
        Los píxeles NoData (DN == 0) se convierten en NaN.
        """
        if not tif_path.exists():
            raise FileNotFoundError(f"Banda {band_name} no encontrada: {tif_path}")

        with rasterio.open(tif_path) as src:
            data    = src.read(1).astype(np.float32)
            profile = src.profile.copy()

            # Si la banda es de 20 m y tenemos target shape, resamplear
            if band_name in BANDS_20M and self._target is not None:
                th, tw = self._target
                if data.shape != (th, tw):
                    data = self._resample(src, th, tw)
                    profile.update(height=th, width=tw)

        # NoData: DN == 0 → NaN
        data = np.where(data == NODATA_DN, np.nan, data)

        # Normalizar DN → reflectancia [0, 1]
        data /= DN_SCALE

        # Clip por seguridad: reflectancias fuera de [0, 1] son artefactos
        data = np.clip(data, 0.0, 1.0)

        return data, profile

    @staticmethod
    def _resample(src: rasterio.DatasetReader, th: int, tw: int) -> np.ndarray:
        """Resamplea la primera banda a (th, tw) usando interpolación bilineal."""
        return src.read(
            1,
            out_shape=(th, tw),
            resampling=Resampling.bilinear,
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Calculador de índices
# ---------------------------------------------------------------------------
class SpectralIndexCalculator:
    """
    Calcula los 5 índices de estrés hídrico a partir de un dict de arrays.

    Uso:
        calc  = SpectralIndexCalculator()
        ndvi  = calc.compute("NDVI", bands)
        all_  = calc.compute_all(bands)
    """

    # Mapa de nombre → función de cálculo
    _REGISTRY: dict[str, Callable[[dict[str, np.ndarray]], np.ndarray]] = {}

    @classmethod
    def _register(cls, name: str):
        """Decorador para registrar funciones de índice."""
        def decorator(fn):
            cls._REGISTRY[name] = fn
            return fn
        return decorator

    def compute(self, index_name: str, bands: dict[str, np.ndarray]) -> np.ndarray:
        """Calcula un índice por nombre."""
        if index_name not in self._REGISTRY:
            raise ValueError(
                f"Índice '{index_name}' no implementado. "
                f"Disponibles: {list(self._REGISTRY)}"
            )
        return self._REGISTRY[index_name](bands)

    def compute_all(
        self,
        bands: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """
        Calcula todos los índices disponibles según las bandas presentes.
        Si faltan bandas para un índice, lo omite con un warning.
        """
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

    # ── Fórmulas ──────────────────────────────────────────────

    @staticmethod
    def _safe_norm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Diferencia normalizada: (a - b) / (a + b)
        La división por cero produce NaN (no RuntimeWarning).
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = np.where(
                (a + b) == 0,
                np.nan,
                (a - b) / (a + b),
            )
        return result.astype(np.float32)


# Registramos cada índice como función estática del calculador
@SpectralIndexCalculator._register("NDVI")
def _ndvi(bands: dict) -> np.ndarray:
    """(B08 - B04) / (B08 + B04) — Vigor vegetal general."""
    return SpectralIndexCalculator._safe_norm_diff(bands["B08"], bands["B04"])


@SpectralIndexCalculator._register("NDWI")
def _ndwi(bands: dict) -> np.ndarray:
    """(B03 - B08) / (B03 + B08) — Contenido de agua en vegetación."""
    return SpectralIndexCalculator._safe_norm_diff(bands["B03"], bands["B08"])


@SpectralIndexCalculator._register("NDMI")
def _ndmi(bands: dict) -> np.ndarray:
    """(B08 - B11) / (B08 + B11) — Humedad en hoja/dosel. ★ más directo para estrés."""
    return SpectralIndexCalculator._safe_norm_diff(bands["B08"], bands["B11"])


@SpectralIndexCalculator._register("NDRE")
def _ndre(bands: dict) -> np.ndarray:
    """(B08 - B05) / (B08 + B05) — Estrés temprano vía clorofila. ★ detección precoz."""
    return SpectralIndexCalculator._safe_norm_diff(bands["B08"], bands["B05"])


@SpectralIndexCalculator._register("EVI")
def _evi(bands: dict) -> np.ndarray:
    """2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + L) — Vegetación densa."""
    B08, B04, B02 = bands["B08"], bands["B04"], bands["B02"]
    denom = B08 + EVI_C1 * B04 - EVI_C2 * B02 + EVI_L
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        evi = np.where(denom == 0, np.nan, EVI_G * (B08 - B04) / denom)
    # EVI puede salir de [-1, 1] en zonas muy oscuras o con ruido
    return np.clip(evi, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Escritura de TIFFs de índices
# ---------------------------------------------------------------------------
class IndexWriter:
    """Guarda un array de índice como GeoTIFF float32 con nodata=NaN."""

    @staticmethod
    def write(
        data: np.ndarray,
        ref_profile: dict,
        out_path: Path,
    ) -> None:
        """
        Escribe el índice en out_path usando el perfil de la banda de referencia.

        El perfil se actualiza a float32 y nodata=NaN.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        profile = ref_profile.copy()
        profile.update(
            dtype    = rasterio.float32,
            count    = 1,
            nodata   = np.nan,
            compress = "lzw",      # Compresión sin pérdida → reduce ~40% el tamaño
            tiled    = True,
            blockxsize = 256,
            blockysize = 256,
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data.astype(np.float32), 1)


# ---------------------------------------------------------------------------
# 4. Estadísticas por índice
# ---------------------------------------------------------------------------
def compute_summary(indices: dict[str, np.ndarray]) -> dict:
    """
    Calcula estadísticas descriptivas por índice para el summary.json.

    Para cada índice retorna: mean, std, min, max, pct_valid (% de píxeles no-NaN).
    """
    summary: dict[str, dict] = {}
    for name, arr in indices.items():
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            summary[name] = {"mean": None, "std": None, "min": None, "max": None, "pct_valid": 0.0}
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
# 5. Runner por fecha (una carpeta parcela/fecha)
# ---------------------------------------------------------------------------
class DateProcessor:
    """
    Procesa una carpeta {parcel_id}/{YYYY-MM-DD}: carga bandas, calcula índices,
    guarda TIFFs y genera el summary.json.
    """

    def __init__(self, dry_run: bool = False):
        self._calculator = SpectralIndexCalculator()
        self._writer     = IndexWriter()
        self._dry_run    = dry_run

    def process(
        self,
        bands_dir: Path,
        output_dir: Path,
    ) -> dict | None:
        """
        Procesa todas las bandas de una fecha.

        Retorna el summary de estadísticas, o None si ya estaba procesado.
        """
        if self._already_done(output_dir):
            logger.debug("  ↳ %s ya procesado, skip", bands_dir.name)
            return None

        # ── Detectar qué bandas existen ──
        # Soporta .jp2 (descarga nativa CDSE) y .tif (conversión manual)
        # Los archivos se llaman B02.jp2, B08.jp2, etc. (stem = nombre de banda)
        band_names = {"B02", "B03", "B04", "B05", "B08", "B11"}
        available = {}
        for ext in ("*.jp2", "*.tif"):
            for p in bands_dir.glob(ext):
                if p.stem in band_names and p.stem not in available:
                    available[p.stem] = p
        if not available:
            logger.warning("  No hay bandas en %s, skip", bands_dir)
            return None

        if self._dry_run:
            logger.info("  [DRY-RUN] Procesaría %s (%d bandas)", bands_dir, len(available))
            return None

        # ── Cargar banda de referencia (B08 a 10 m) primero ──
        ref_name = "B08" if "B08" in available else next(iter(available))
        ref_arr, ref_profile = BandLoader().load(available[ref_name], ref_name)
        target_shape = ref_arr.shape   # (H, W) a 10 m

        # ── Cargar el resto con resampling si aplica ──
        loader = BandLoader(target_shape=target_shape)
        bands: dict[str, np.ndarray] = {ref_name: ref_arr}
        for band_name, tif_path in available.items():
            if band_name == ref_name:
                continue
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
        """True si todos los índices + summary ya existen."""
        expected = [f"{idx}.tif" for idx in INDEX_BANDS] + ["summary.json"]
        return all((output_dir / f).exists() for f in expected)


# ---------------------------------------------------------------------------
# 6. Orquestador principal
# ---------------------------------------------------------------------------
class IndexPipelineRunner:
    """
    Recorre la estructura sentinel2/ y procesa cada (parcela, fecha).

    Estructura esperada de entrada:
        {sentinel2_dir}/
        └── {parcel_id}/
            └── {YYYY-MM-DD}/
                └── *.tif

    Estructura de salida:
        {output_dir}/
        └── {parcel_id}/
            └── {YYYY-MM-DD}/
                └── *.tif + summary.json
    """

    def __init__(self, dry_run: bool = False):
        self._processor = DateProcessor(dry_run=dry_run)
        self._dry_run   = dry_run

    def run(
        self,
        sentinel2_dir: Path,
        output_dir: Path,
        parcel_ids: list[str] | None = None,
    ) -> dict:
        """
        Ejecuta el pipeline completo.

        Retorna un dict con el resumen global:
            {parcel_id: {date: summary_dict}}
        """
        if not sentinel2_dir.exists():
            logger.error("Directorio de entrada no existe: %s", sentinel2_dir)
            sys.exit(1)

        # Listar parcelas disponibles
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
        logger.info("Calculando índices — %d parcelas", total_parcels)
        if self._dry_run:
            logger.info("[DRY-RUN] No se escribirá nada en disco")
        logger.info("=" * 60)

        for parcel_dir in parcel_dirs:
            parcel_id   = parcel_dir.name
            date_dirs   = sorted([d for d in parcel_dir.iterdir() if d.is_dir()])

            if not date_dirs:
                logger.warning("Parcela %s sin fechas, skip", parcel_id)
                continue

            logger.info("Parcela %s — %d fechas", parcel_id, len(date_dirs))
            global_report[parcel_id] = {}

            for date_dir in date_dirs:
                date_str   = date_dir.name
                out_dir    = output_dir / parcel_id / date_str
                summary    = self._processor.process(date_dir, out_dir)
                if summary:
                    global_report[parcel_id][date_str] = summary
                    total_dates += 1

        # Guardar reporte global
        report_path = output_dir / "pipeline_report.json"
        if not self._dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(global_report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        logger.info("=" * 60)
        logger.info(
            "Completado: %d parcelas × %d fechas procesadas",
            total_parcels, total_dates
        )
        logger.info("Reporte global: %s", report_path)
        logger.info("=" * 60)

        return global_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcula índices espectrales (NDVI, NDWI, NDMI, NDRE, EVI) "
                    "desde bandas Sentinel-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/sentinel2/"),
        help="Directorio con bandas descargadas (default: data/raw/sentinel2/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/indices/"),
        help="Directorio de salida para TIFFs de índices (default: data/processed/indices/)",
    )
    parser.add_argument(
        "--parcel-ids",
        nargs="+",
        metavar="ID",
        help="Parcelas específicas a procesar (ej: H1 H2 H5). Sin esto: todas.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra qué procesaría sin escribir nada en disco",
    )
    args = parser.parse_args()

    runner = IndexPipelineRunner(dry_run=args.dry_run)
    runner.run(
        sentinel2_dir = args.input,
        output_dir    = args.output,
        parcel_ids    = args.parcel_ids,
    )


if __name__ == "__main__":
    main()
