"""
tests/test_kml_to_csv.py
Tests unitarios para el extractor KML → CSV
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

# Agrega src al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion.kml_to_csv import enrich, parse_kml, _natural_sort_key


KML_PATH = Path("notebooks/data/csv/aguacates_jalisco_5_5_26.kml")


# ── parse_kml ──────────────────────────────────────────────
class TestParseKml:
    def test_returns_list(self):
        result = parse_kml(KML_PATH)
        assert isinstance(result, list)

    def test_extracts_100_parcels(self):
        result = parse_kml(KML_PATH)
        assert len(result) == 100

    def test_required_keys_present(self):
        result = parse_kml(KML_PATH)
        for record in result:
            assert "parcel_id" in record
            assert "latitude" in record
            assert "longitude" in record
            assert "altitude_m" in record

    def test_coordinates_in_jalisco_range(self):
        """Las coordenadas deben caer en el rango sur de Jalisco."""
        result = parse_kml(KML_PATH)
        for r in result:
            assert -106 < r["longitude"] < -100, f"Longitud fuera de rango: {r['longitude']}"
            assert 17 < r["latitude"] < 23, f"Latitud fuera de rango: {r['latitude']}"

    def test_parcel_ids_are_strings(self):
        result = parse_kml(KML_PATH)
        for r in result:
            assert isinstance(r["parcel_id"], str)
            assert len(r["parcel_id"]) > 0


# ── enrich ─────────────────────────────────────────────────
class TestEnrich:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            [
                {"parcel_id": "H1", "latitude": 19.66, "longitude": -103.49, "altitude_m": 0},
                {"parcel_id": "H2", "latitude": 19.50, "longitude": -102.00, "altitude_m": 0},
            ]
        )

    def test_adds_required_columns(self, sample_df):
        enriched = enrich(sample_df, buffer_m=250)
        for col in ["state", "buffer_m", "bbox_west", "bbox_east", "bbox_south", "bbox_north"]:
            assert col in enriched.columns

    def test_bbox_is_consistent(self, sample_df):
        enriched = enrich(sample_df, buffer_m=250)
        assert (enriched["bbox_east"] > enriched["bbox_west"]).all()
        assert (enriched["bbox_north"] > enriched["bbox_south"]).all()

    def test_jalisco_classification(self, sample_df):
        enriched = enrich(sample_df, buffer_m=250)
        # Primera fila: lon=-103.49 → Jalisco
        assert enriched.iloc[0]["state"] == "Jalisco"

    def test_buffer_stored_correctly(self, sample_df):
        enriched = enrich(sample_df, buffer_m=500)
        assert (enriched["buffer_m"] == 500).all()

    def test_download_flags_default_false(self, sample_df):
        enriched = enrich(sample_df)
        assert not enriched["download_ready"].any()
        assert not enriched["processed"].any()
        assert not enriched["indices_computed"].any()


# ── _natural_sort_key ──────────────────────────────────────
class TestNaturalSort:
    def test_sorts_h1_before_h10(self):
        ids = ["H10", "H2", "H1", "H20", "H3"]
        sorted_ids = sorted(ids, key=_natural_sort_key)
        assert sorted_ids == ["H1", "H2", "H3", "H10", "H20"]
