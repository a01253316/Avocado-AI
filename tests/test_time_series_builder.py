"""
tests/test_time_series_builder.py
Tests unitarios del constructor de series de tiempo.

Estrategia: se genera en tmp_path una estructura de índices sintéticos
(TIFFs float32 de 10×10 px con valores conocidos) que simula la salida
de spectral_indices.py, y se prueba todo el pipeline.
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

from processing.time_series_builder import (
    INDICES,
    MIN_DATES,
    N_CHANNELS,
    DatasetWriter,
    IndexReader,
    ManifestBuilder,
    ParcelTimeSeries,
    SignalBuilder,
    TSNormalizer,
    TimeSeriesBuilder,
)

# ---------------------------------------------------------------------------
# Helpers para datos sintéticos
# ---------------------------------------------------------------------------
SHAPE     = (10, 10)
TRANSFORM = from_bounds(-103.49, 19.64, -103.48, 19.65, *SHAPE)


def _write_index_tif(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full(SHAPE, value, dtype=np.float32)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=SHAPE[0], width=SHAPE[1], count=1,
        dtype=rasterio.float32, crs="EPSG:32614", transform=TRANSFORM,
    ) as dst:
        dst.write(data, 1)


def _make_parcel_dates(
    base: Path,
    parcel_id: str,
    dates: list[str],
    index_values: dict[str, float] | None = None,
) -> None:
    """
    Crea la estructura de índices para una parcela con varias fechas.
    index_values: {"NDVI": 0.6, "NDWI": -0.1, ...} — mismos valores en todas las fechas.
    Si no se pasa, usa valores por defecto.
    """
    defaults = {
        "NDVI": 0.60, "NDWI": -0.10, "NDMI": 0.30, "NDRE": 0.45, "EVI": 0.50
    }
    vals = {**defaults, **(index_values or {})}
    for date in dates:
        for idx, val in vals.items():
            _write_index_tif(base / parcel_id / date / f"{idx}.tif", val)


def _make_parcels_csv(path: Path, parcel_ids: list[str]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "parcel_id", "latitude", "longitude", "altitude_m", "state",
        "buffer_m", "bbox_west", "bbox_east", "bbox_south", "bbox_north",
        "sentinel2_tile", "download_ready", "processed", "indices_computed",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for pid in parcel_ids:
            w.writerow({
                "parcel_id": pid, "latitude": 19.66, "longitude": -103.49,
                "altitude_m": 0, "state": "Jalisco", "buffer_m": 250,
                "bbox_west": -103.492, "bbox_east": -103.487,
                "bbox_south": 19.658, "bbox_north": 19.663,
                "sentinel2_tile": "14QMF",
                "download_ready": True, "processed": True, "indices_computed": True,
            })


DATES_12 = [f"2023-{m:02d}-15" for m in range(1, 13)]   # 12 fechas mensuales


# ---------------------------------------------------------------------------
# ParcelTimeSeries
# ---------------------------------------------------------------------------
class TestParcelTimeSeries:
    def test_is_valid_with_enough_dates(self):
        ts = ParcelTimeSeries(parcel_id="H1", dates=["2023-01-01"] * MIN_DATES)
        assert ts.is_valid

    def test_is_invalid_below_min_dates(self):
        ts = ParcelTimeSeries(parcel_id="H1", dates=["2023-01-01"] * (MIN_DATES - 1))
        assert not ts.is_valid

    def test_n_dates_matches_dates_list(self):
        ts = ParcelTimeSeries(parcel_id="H1", dates=DATES_12)
        assert ts.n_dates == 12


# ---------------------------------------------------------------------------
# IndexReader
# ---------------------------------------------------------------------------
class TestIndexReader:
    def test_reads_correct_number_of_dates(self, tmp_path):
        _make_parcel_dates(tmp_path, "H1", DATES_12)
        reader = IndexReader(tmp_path)
        ts = reader.read_parcel("H1")
        assert ts.n_dates == 12

    def test_dates_are_sorted_ascending(self, tmp_path):
        dates = ["2023-06-15", "2023-01-15", "2023-03-15"]
        _make_parcel_dates(tmp_path, "H1", dates)
        reader = IndexReader(tmp_path)
        ts = reader.read_parcel("H1")
        assert ts.dates == sorted(dates)

    def test_patch_shape_is_C_H_W(self, tmp_path):
        _make_parcel_dates(tmp_path, "H1", DATES_12)
        reader = IndexReader(tmp_path)
        ts = reader.read_parcel("H1")
        assert ts.patches[0].shape == (N_CHANNELS, *SHAPE)

    def test_reads_wgs84_bounds_from_tiff_metadata(self, tmp_path):
        _make_parcel_dates(tmp_path, "H1", DATES_12)
        reader = IndexReader(tmp_path)
        ts = reader.read_parcel("H1")
        assert ts.bounds_wgs84 is not None
        assert np.array(ts.bounds_wgs84).shape == (2, 2)

    def test_missing_index_becomes_nan_channel(self, tmp_path):
        """Si falta EVI.tif, ese canal debe ser NaN."""
        dates = DATES_12[:8]
        _make_parcel_dates(tmp_path, "H1", dates)
        # Borrar EVI de la primera fecha
        (tmp_path / "H1" / dates[0] / "EVI.tif").unlink()

        reader = IndexReader(tmp_path)
        ts = reader.read_parcel("H1")

        evi_channel = INDICES.index("EVI")
        assert np.all(np.isnan(ts.patches[0][evi_channel]))
        # El resto de la primera fecha sí tiene datos
        assert not np.all(np.isnan(ts.patches[0][0]))

    def test_nonexistent_parcel_returns_empty(self, tmp_path):
        reader = IndexReader(tmp_path)
        ts = reader.read_parcel("H999")
        assert ts.n_dates == 0
        assert not ts.is_valid

    def test_doy_range_is_1_to_365(self, tmp_path):
        _make_parcel_dates(tmp_path, "H1", DATES_12)
        reader = IndexReader(tmp_path)
        ts = reader.read_parcel("H1")
        assert all(1 <= d <= 365 for d in ts.doy)

    def test_doy_january_15_is_15(self, tmp_path):
        _make_parcel_dates(tmp_path, "H1", ["2023-01-15"])
        reader = IndexReader(tmp_path)
        ts = reader.read_parcel("H1")
        if ts.dates:  # puede no tener fechas si solo hay 1 < MIN_DATES
            assert ts.doy[0] == 15


# ---------------------------------------------------------------------------
# SignalBuilder
# ---------------------------------------------------------------------------
class TestSignalBuilder:
    def _make_ts(self, n_dates: int, value: float = 0.5) -> ParcelTimeSeries:
        patches = [
            np.full((N_CHANNELS, *SHAPE), value, dtype=np.float32)
            for _ in range(n_dates)
        ]
        dates = [f"2023-{i+1:02d}-15" for i in range(n_dates)]
        return ParcelTimeSeries(parcel_id="H1", dates=dates, patches=patches)

    def test_output_shape_is_T_C(self):
        ts  = self._make_ts(12, value=0.5)
        sig = SignalBuilder.build(ts)
        assert sig.shape == (12, N_CHANNELS)

    def test_mean_matches_patch_value(self):
        ts  = self._make_ts(5, value=0.7)
        sig = SignalBuilder.build(ts)
        np.testing.assert_allclose(sig, 0.7, atol=1e-4)

    def test_nan_pixels_ignored_in_mean(self):
        patch = np.full((N_CHANNELS, *SHAPE), 0.6, dtype=np.float32)
        patch[0, :5, :5] = np.nan    # NaN en la mitad del canal NDVI
        ts = ParcelTimeSeries(
            parcel_id="H1", dates=["2023-01-15"], patches=[patch]
        )
        sig = SignalBuilder.build(ts)
        # El mean del canal NDVI debe ignorar NaN y ser 0.6
        np.testing.assert_allclose(sig[0, 0], 0.6, atol=1e-4)

    def test_empty_ts_returns_empty_array(self):
        ts  = ParcelTimeSeries(parcel_id="H1")
        sig = SignalBuilder.build(ts)
        assert sig.shape == (0, N_CHANNELS)


# ---------------------------------------------------------------------------
# TSNormalizer
# ---------------------------------------------------------------------------
class TestTSNormalizer:
    def _sample_signals(self, n: int = 50) -> list[np.ndarray]:
        rng = np.random.default_rng(0)
        return [rng.uniform(-0.5, 0.8, (10, N_CHANNELS)).astype(np.float32)
                for _ in range(n)]

    def test_minmax_output_in_0_1(self):
        signals = self._sample_signals()
        norm    = TSNormalizer(mode="minmax").fit(signals)
        out     = norm.transform(signals[0])
        assert np.nanmin(out) >= -0.01   # pequeña tolerancia por percentiles
        assert np.nanmax(out) <=  1.01

    def test_zscore_mean_near_zero(self):
        signals = self._sample_signals()
        norm    = TSNormalizer(mode="zscore").fit(signals)
        all_out = np.concatenate([norm.transform(s) for s in signals])
        np.testing.assert_allclose(np.nanmean(all_out), 0.0, atol=0.1)

    def test_none_mode_no_change(self):
        signals = self._sample_signals(5)
        norm    = TSNormalizer(mode="none")
        out     = norm.transform(signals[0])
        np.testing.assert_array_equal(out, signals[0])

    def test_nan_preserved_after_transform(self):
        signals = self._sample_signals(10)
        signals[0][0, 0] = np.nan
        norm = TSNormalizer(mode="minmax").fit(signals)
        out  = norm.transform(signals[0])
        assert np.isnan(out[0, 0])

    def test_save_and_load_roundtrip(self, tmp_path):
        signals = self._sample_signals()
        norm    = TSNormalizer(mode="minmax").fit(signals)
        path    = tmp_path / "stats.json"
        norm.save(path)

        loaded = TSNormalizer.load(path)
        assert loaded.mode == "minmax"
        assert set(loaded.stats.keys()) == set(INDICES)
        np.testing.assert_allclose(
            norm.transform(signals[0]),
            loaded.transform(signals[0]),
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# DatasetWriter
# ---------------------------------------------------------------------------
class TestDatasetWriter:
    def _make_ts(self, pid: str, n: int = 8) -> ParcelTimeSeries:
        patches = [np.random.rand(N_CHANNELS, *SHAPE).astype(np.float32)
                   for _ in range(n)]
        dates   = [f"2023-{i+1:02d}-15" for i in range(n)]
        doy     = [i * 30 + 15 for i in range(n)]
        return ParcelTimeSeries(parcel_id=pid, dates=dates, doy=doy, patches=patches)

    def test_patch_npz_has_correct_keys(self, tmp_path):
        ts     = self._make_ts("H1")
        writer = DatasetWriter(tmp_path)
        path   = writer.write_patch(ts, np.stack(ts.patches))
        npz    = np.load(path, allow_pickle=True)
        assert set(npz.files) >= {"data", "dates", "doy", "bounds_wgs84"}

    def test_patch_npz_stores_bounds_when_available(self, tmp_path):
        ts = self._make_ts("H1")
        ts.bounds_wgs84 = [[19.65, -103.50], [19.66, -103.49]]
        writer = DatasetWriter(tmp_path)
        path = writer.write_patch(ts, np.stack(ts.patches))
        npz = np.load(path)
        np.testing.assert_allclose(npz["bounds_wgs84"], ts.bounds_wgs84)

    def test_patch_npz_stores_empty_bounds_when_unavailable(self, tmp_path):
        ts = self._make_ts("H1")
        writer = DatasetWriter(tmp_path)
        path = writer.write_patch(ts, np.stack(ts.patches))
        npz = np.load(path)
        assert npz["bounds_wgs84"].shape == (0, 2)

    def test_patch_data_shape(self, tmp_path):
        ts     = self._make_ts("H1", n=8)
        writer = DatasetWriter(tmp_path)
        path   = writer.write_patch(ts, np.stack(ts.patches))
        npz    = np.load(path)
        assert npz["data"].shape == (8, N_CHANNELS, *SHAPE)

    def test_signal_npz_shape(self, tmp_path):
        ts      = self._make_ts("H1", n=10)
        signals = np.random.rand(10, N_CHANNELS).astype(np.float32)
        writer  = DatasetWriter(tmp_path)
        path    = writer.write_signal(ts, signals)
        npz     = np.load(path)
        assert npz["data"].shape == (10, N_CHANNELS)

    def test_dates_array_length_matches(self, tmp_path):
        ts     = self._make_ts("H1", n=6)
        writer = DatasetWriter(tmp_path)
        path   = writer.write_patch(ts, np.stack(ts.patches))
        npz    = np.load(path, allow_pickle=True)
        assert len(npz["dates"]) == 6


# ---------------------------------------------------------------------------
# ManifestBuilder
# ---------------------------------------------------------------------------
class TestManifestBuilder:
    def _sample_records(self, n: int) -> list[dict]:
        return [
            {
                "parcel_id": f"H{i}", "n_dates": 12,
                "date_min": "2023-01-15", "date_max": "2023-12-15",
                "mean_ndvi": 0.6, "mean_ndwi": -0.1,
                "mean_ndmi": 0.3, "mean_ndre": 0.45, "mean_evi": 0.5,
                "has_patch": True, "has_signal": True,
            }
            for i in range(1, n + 1)
        ]

    def test_manifest_csv_created(self, tmp_path):
        builder = ManifestBuilder(tmp_path)
        builder.build(self._sample_records(5), [f"H{i}" for i in range(1, 6)])
        assert (tmp_path / "manifest.csv").exists()

    def test_manifest_row_count(self, tmp_path):
        import csv as csv_mod
        builder = ManifestBuilder(tmp_path)
        builder.build(self._sample_records(10), [f"H{i}" for i in range(1, 11)])
        with (tmp_path / "manifest.csv").open() as f:
            rows = list(csv_mod.DictReader(f))
        assert len(rows) == 10

    def test_split_json_created_when_requested(self, tmp_path):
        ids = [f"H{i}" for i in range(1, 21)]
        ManifestBuilder(tmp_path).build(self._sample_records(20), ids, split=True)
        assert (tmp_path / "split.json").exists()

    def test_split_covers_all_ids(self, tmp_path):
        ids = [f"H{i}" for i in range(1, 21)]
        ManifestBuilder(tmp_path).build(self._sample_records(20), ids, split=True)
        split = json.loads((tmp_path / "split.json").read_text())
        all_split = split["train"] + split["val"] + split["test"]
        assert sorted(all_split) == sorted(ids)

    def test_split_no_overlap_between_sets(self, tmp_path):
        ids = [f"H{i}" for i in range(1, 21)]
        ManifestBuilder(tmp_path).build(self._sample_records(20), ids, split=True)
        split = json.loads((tmp_path / "split.json").read_text())
        assert not set(split["train"]) & set(split["val"])
        assert not set(split["train"]) & set(split["test"])
        assert not set(split["val"]) & set(split["test"])


# ---------------------------------------------------------------------------
# TimeSeriesBuilder — integración completa
# ---------------------------------------------------------------------------
class TestTimeSeriesBuilderIntegration:
    def _setup(self, tmp_path, n_parcels: int = 5, n_dates: int = 12):
        indices_dir = tmp_path / "indices"
        output_dir  = tmp_path / "datasets"
        parcels_csv = tmp_path / "parcelas.csv"

        ids = [f"H{i}" for i in range(1, n_parcels + 1)]
        for pid in ids:
            _make_parcel_dates(
                indices_dir, pid,
                [f"2023-{m:02d}-15" for m in range(1, n_dates + 1)],
            )
        _make_parcels_csv(parcels_csv, ids)
        return indices_dir, output_dir, parcels_csv, ids

    def test_signal_files_created(self, tmp_path):
        indices_dir, output_dir, csv_path, ids = self._setup(tmp_path)
        TimeSeriesBuilder(indices_dir, output_dir, mode="signal").run(csv_path)
        for pid in ids:
            assert (output_dir / "signals" / f"{pid}.npz").exists()

    def test_patch_files_created(self, tmp_path):
        indices_dir, output_dir, csv_path, ids = self._setup(tmp_path)
        TimeSeriesBuilder(indices_dir, output_dir, mode="patch").run(csv_path)
        for pid in ids:
            assert (output_dir / "patches" / f"{pid}.npz").exists()

    def test_manifest_created(self, tmp_path):
        indices_dir, output_dir, csv_path, _ = self._setup(tmp_path)
        TimeSeriesBuilder(indices_dir, output_dir, mode="signal").run(csv_path)
        assert (output_dir / "manifest.csv").exists()

    def test_normalizer_stats_saved(self, tmp_path):
        indices_dir, output_dir, csv_path, _ = self._setup(tmp_path)
        TimeSeriesBuilder(indices_dir, output_dir, normalize="minmax").run(csv_path)
        assert (output_dir / "normalizer_stats.json").exists()

    def test_split_json_when_requested(self, tmp_path):
        indices_dir, output_dir, csv_path, _ = self._setup(tmp_path, n_parcels=10)
        TimeSeriesBuilder(indices_dir, output_dir, mode="signal").run(
            csv_path, split=True
        )
        assert (output_dir / "split.json").exists()

    def test_dry_run_produces_no_files(self, tmp_path):
        indices_dir, output_dir, csv_path, _ = self._setup(tmp_path)
        TimeSeriesBuilder(
            indices_dir, output_dir, mode="both", dry_run=True
        ).run(csv_path)
        assert not output_dir.exists()

    def test_parcelas_below_min_dates_excluded(self, tmp_path):
        indices_dir = tmp_path / "indices"
        output_dir  = tmp_path / "datasets"
        csv_path    = tmp_path / "parcelas.csv"
        # H1: 12 fechas (válida), H2: 3 fechas (inválida)
        _make_parcel_dates(
            indices_dir, "H1", [f"2023-{m:02d}-15" for m in range(1, 13)]
        )
        _make_parcel_dates(
            indices_dir, "H2", ["2023-01-15", "2023-02-15", "2023-03-15"]
        )
        _make_parcels_csv(csv_path, ["H1", "H2"])

        TimeSeriesBuilder(indices_dir, output_dir, mode="signal").run(csv_path)
        assert (output_dir / "signals" / "H1.npz").exists()
        assert not (output_dir / "signals" / "H2.npz").exists()
