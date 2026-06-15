"""
tests/test_spectral_indices.py
Tests unitarios del calculador de índices espectrales.

Estrategia: se generan TIFFs sintéticos de 10×10 px en tmp_path usando
rasterio, para probar el pipeline completo (carga → cálculo → escritura)
sin necesitar imágenes reales de Sentinel-2.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from processing.spectral_indices import (
    BandLoader,
    DateProcessor,
    IndexPipelineRunner,
    IndexWriter,
    SpectralIndexCalculator,
    compute_summary,
)

# ---------------------------------------------------------------------------
# Helpers para crear TIFFs sintéticos
# ---------------------------------------------------------------------------
SHAPE = (10, 10)   # píxeles de prueba
TRANSFORM = from_bounds(-103.49, 19.64, -103.48, 19.65, *SHAPE)


def _make_tif(path: Path, value: float, shape: tuple = SHAPE, dtype="float32") -> None:
    """Crea un TIFF de una banda con valor uniforme en DN (0-10000)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full(shape, value, dtype=dtype)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=shape[0], width=shape[1],
        count=1,
        dtype=dtype,
        crs="EPSG:32614",
        transform=TRANSFORM,
    ) as dst:
        dst.write(data, 1)


def _make_band_dir(base: Path, bands: dict[str, float]) -> Path:
    """
    Crea un directorio de fecha con TIFFs de bandas.
    bands = {"B02": 2000.0, "B08": 7000.0, ...}  (valores en DN)
    """
    for band, value in bands.items():
        _make_tif(base / f"{band}.tif", value)
    return base


# ---------------------------------------------------------------------------
# BandLoader
# ---------------------------------------------------------------------------
class TestBandLoader:
    def test_normalizes_dn_to_reflectance(self, tmp_path):
        tif = tmp_path / "B08.tif"
        _make_tif(tif, 8000.0, dtype="uint16")  # DN = 8000 -> reflectance = 0.8

        loader = BandLoader()
        arr, _ = loader.load(tif, "B08")

        assert arr.shape == SHAPE
        np.testing.assert_allclose(arr, 0.8, atol=1e-4)

    def test_nodata_dn_zero_becomes_nan(self, tmp_path):
        tif = tmp_path / "B04.tif"
        _make_tif(tif, 0.0)   # DN == 0 → NoData

        loader = BandLoader()
        arr, _ = loader.load(tif, "B04")

        assert np.all(np.isnan(arr))

    def test_clips_values_above_1(self, tmp_path):
        tif = tmp_path / "B03.tif"
        _make_tif(tif, 12000.0)   # DN > 10000 → reflectance > 1 → clip a 1

        loader = BandLoader()
        arr, _ = loader.load(tif, "B03")

        assert np.all(arr <= 1.0)

    def test_resamples_20m_band_to_target_shape(self, tmp_path):
        """B11 es 20m (5×5 px) → se resamplea a target_shape (10×10 px)."""
        tif = tmp_path / "B11.tif"
        small_shape = (5, 5)
        data  = np.full(small_shape, 4000.0, dtype=np.float32)
        t20m  = from_bounds(-103.49, 19.64, -103.48, 19.65, *small_shape)
        with rasterio.open(
            tif, "w", driver="GTiff",
            height=5, width=5, count=1,
            dtype=rasterio.float32,
            crs="EPSG:32614", transform=t20m,
        ) as dst:
            dst.write(data, 1)

        loader = BandLoader(target_shape=(10, 10))
        arr, _ = loader.load(tif, "B11")

        assert arr.shape == (10, 10)

    def test_raises_if_tif_missing(self, tmp_path):
        loader = BandLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "MISSING.tif", "B99")


# ---------------------------------------------------------------------------
# SpectralIndexCalculator — fórmulas
# ---------------------------------------------------------------------------
class TestSpectralIndexCalculator:
    """Prueba las fórmulas con arrays constantes donde el resultado es conocido."""

    @pytest.fixture
    def calc(self):
        return SpectralIndexCalculator()

    def _bands(self, **values) -> dict[str, np.ndarray]:
        """Crea un dict de bandas con arrays 3×3 de valor constante."""
        return {k: np.full((3, 3), v, dtype=np.float32) for k, v in values.items()}

    # ── NDVI ──
    def test_ndvi_formula(self, calc):
        # NDVI = (0.8 - 0.4) / (0.8 + 0.4) = 0.4 / 1.2 ≈ 0.3333
        bands  = self._bands(B08=0.8, B04=0.4)
        result = calc.compute("NDVI", bands)
        np.testing.assert_allclose(result, 0.3333, atol=1e-4)

    def test_ndvi_zero_denominator_is_nan(self, calc):
        bands  = self._bands(B08=0.0, B04=0.0)
        result = calc.compute("NDVI", bands)
        assert np.all(np.isnan(result))

    def test_ndvi_range_minus1_to_1(self, calc):
        bands  = self._bands(B08=0.9, B04=0.1)
        result = calc.compute("NDVI", bands)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)

    # ── NDWI ──
    def test_ndwi_high_water_content(self, calc):
        # Agua: B03 alto, B08 bajo → NDWI positivo
        bands  = self._bands(B03=0.7, B08=0.2)
        result = calc.compute("NDWI", bands)
        assert np.all(result > 0)

    def test_ndwi_low_water_content(self, calc):
        # Suelo seco: B03 bajo, B08 alto → NDWI negativo
        bands  = self._bands(B03=0.2, B08=0.7)
        result = calc.compute("NDWI", bands)
        assert np.all(result < 0)

    # ── NDMI ──
    def test_ndmi_stressed_plant(self, calc):
        # Estrés hídrico: B11 (SWIR) sube cuando la planta pierde agua
        bands       = self._bands(B08=0.5, B11=0.4)
        stressed    = calc.compute("NDMI", bands)
        bands_ok    = self._bands(B08=0.5, B11=0.1)
        not_stressed = calc.compute("NDMI", bands_ok)
        assert np.all(stressed < not_stressed)   # Menos humedad → NDMI más bajo

    # ── NDRE ──
    def test_ndre_formula(self, calc):
        # NDRE = (0.6 - 0.3) / (0.6 + 0.3) = 0.3 / 0.9 ≈ 0.333
        bands  = self._bands(B08=0.6, B05=0.3)
        result = calc.compute("NDRE", bands)
        np.testing.assert_allclose(result, 0.3333, atol=1e-4)

    # ── EVI ──
    def test_evi_formula_known_value(self, calc):
        # EVI = 2.5 * (0.5 - 0.1) / (0.5 + 6*0.1 - 7.5*0.02 + 1)
        # denom = 0.5 + 0.6 - 0.15 + 1 = 1.95
        # EVI = 2.5 * 0.4 / 1.95 ≈ 0.5128
        bands  = self._bands(B08=0.5, B04=0.1, B02=0.02)
        result = calc.compute("EVI", bands)
        np.testing.assert_allclose(result, 0.5128, atol=1e-3)

    def test_evi_clipped_to_minus1_plus1(self, calc):
        bands  = self._bands(B08=1.0, B04=0.0, B02=0.0)
        result = calc.compute("EVI", bands)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)

    # ── compute_all ──
    def test_compute_all_returns_all_5_when_bands_available(self, calc):
        bands = self._bands(B02=0.02, B03=0.05, B04=0.1, B05=0.15, B08=0.5, B11=0.2)
        all_  = calc.compute_all(bands)
        assert set(all_.keys()) == {"NDVI", "NDWI", "NDMI", "NDRE", "EVI"}

    def test_compute_all_skips_index_with_missing_band(self, calc):
        # Sin B11 → NDMI se omite
        bands = self._bands(B02=0.02, B03=0.05, B04=0.1, B05=0.15, B08=0.5)
        all_  = calc.compute_all(bands)
        assert "NDMI" not in all_
        assert "NDVI" in all_

    def test_compute_unknown_index_raises(self, calc):
        with pytest.raises(ValueError, match="no implementado"):
            calc.compute("FAKE_INDEX", {})


# ---------------------------------------------------------------------------
# IndexWriter
# ---------------------------------------------------------------------------
class TestIndexWriter:
    def test_writes_float32_tif(self, tmp_path):
        data     = np.full((10, 10), 0.42, dtype=np.float32)
        profile  = {
            "driver": "GTiff", "dtype": "float32", "count": 1,
            "height": 10, "width": 10, "crs": "EPSG:32614",
            "transform": TRANSFORM,
        }
        out_path = tmp_path / "NDVI.tif"
        IndexWriter.write(data, profile, out_path)

        assert out_path.exists()
        with rasterio.open(out_path) as src:
            result = src.read(1)
            assert result.dtype == np.float32
            np.testing.assert_allclose(result, 0.42, atol=1e-4)

    def test_creates_parent_dirs(self, tmp_path):
        data    = np.zeros((5, 5), dtype=np.float32)
        profile = {
            "driver": "GTiff", "dtype": "float32", "count": 1,
            "height": 5, "width": 5, "crs": "EPSG:32614",
            "transform": TRANSFORM,
        }
        deep = tmp_path / "H1" / "2024-01-15" / "NDVI.tif"
        IndexWriter.write(data, profile, deep)
        assert deep.exists()


# ---------------------------------------------------------------------------
# compute_summary
# ---------------------------------------------------------------------------
class TestComputeSummary:
    def test_returns_stats_per_index(self):
        indices = {
            "NDVI": np.array([0.2, 0.4, 0.6], dtype=np.float32),
            "NDMI": np.array([0.1, 0.3, 0.5], dtype=np.float32),
        }
        summary = compute_summary(indices)
        assert "NDVI" in summary and "NDMI" in summary
        np.testing.assert_allclose(summary["NDVI"]["mean"], 0.4, atol=1e-4)
        assert summary["NDVI"]["pct_valid"] == 100.0

    def test_nan_pixels_excluded_from_stats(self):
        arr     = np.array([0.5, np.nan, 0.5], dtype=np.float32)
        summary = compute_summary({"NDVI": arr})
        np.testing.assert_allclose(summary["NDVI"]["mean"], 0.5, atol=1e-4)
        np.testing.assert_allclose(summary["NDVI"]["pct_valid"], 66.666, atol=0.1)

    def test_all_nan_returns_none_values(self):
        arr     = np.full((5,), np.nan, dtype=np.float32)
        summary = compute_summary({"NDMI": arr})
        assert summary["NDMI"]["mean"] is None
        assert summary["NDMI"]["pct_valid"] == 0.0


# ---------------------------------------------------------------------------
# DateProcessor — integración con TIFFs reales en tmp_path
# ---------------------------------------------------------------------------
class TestDateProcessor:
    def _create_full_date(self, base: Path) -> Path:
        """Crea una carpeta de fecha completa con las 6 bandas."""
        return _make_band_dir(base, {
            "B02": 500.0,
            "B03": 600.0,
            "B04": 800.0,
            "B05": 1500.0,
            "B08": 5000.0,
            "B11": 2000.0,
        })

    def test_produces_5_index_tifs_and_summary(self, tmp_path):
        bands_dir  = self._create_full_date(tmp_path / "in" / "H1" / "2024-01-15")
        output_dir = tmp_path / "out" / "H1" / "2024-01-15"

        processor = DateProcessor()
        summary   = processor.process(bands_dir, output_dir)

        assert summary is not None
        for idx in ["NDVI", "NDWI", "NDMI", "NDRE", "EVI"]:
            assert (output_dir / f"{idx}.tif").exists(), f"{idx}.tif no generado"
        assert (output_dir / "summary.json").exists()

    def test_summary_json_is_valid(self, tmp_path):
        bands_dir  = self._create_full_date(tmp_path / "in" / "H1" / "2024-02-01")
        output_dir = tmp_path / "out" / "H1" / "2024-02-01"

        DateProcessor().process(bands_dir, output_dir)
        data = json.loads((output_dir / "summary.json").read_text())

        assert "NDVI" in data
        assert "mean" in data["NDVI"]
        assert data["NDVI"]["pct_valid"] == 100.0

    def test_skips_if_already_done(self, tmp_path):
        bands_dir  = self._create_full_date(tmp_path / "in" / "H1" / "2024-03-01")
        output_dir = tmp_path / "out" / "H1" / "2024-03-01"

        processor = DateProcessor()
        processor.process(bands_dir, output_dir)  # Primera vez
        result2 = processor.process(bands_dir, output_dir)  # Segunda: skip

        assert result2 is None

    def test_dry_run_produces_no_files(self, tmp_path):
        bands_dir  = self._create_full_date(tmp_path / "in" / "H1" / "2024-04-01")
        output_dir = tmp_path / "out" / "H1" / "2024-04-01"

        DateProcessor(dry_run=True).process(bands_dir, output_dir)
        assert not output_dir.exists()
