"""
tests/test_sentinel2_downloader.py
Tests unitarios del downloader usando mocks — no requiere credenciales.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion.sentinel2_downloader import (
    BandDownloader,
    CDSEAuth,
    DownloadConfig,
    Parcel,
    STACSearcher,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_parcel():
    return Parcel(
        parcel_id  = "H1",
        latitude   =  19.663164,
        longitude  = -103.487470,
        bbox_west  = -103.489855,
        bbox_east  = -103.485086,
        bbox_south =  19.660919,
        bbox_north =  19.665410,
    )


@pytest.fixture
def sample_config(tmp_path):
    return DownloadConfig(
        start_date      = "2024-01-01",
        end_date        = "2024-03-31",
        max_cloud_cover = 20,
        bands           = ["B02", "B04", "B08"],
        output_dir      = tmp_path / "sentinel2",
    )


@pytest.fixture
def mock_product():
    return {
        "id": "S2A_MSIL2A_20240115T163901_N0510_R069_T14QMF_20240115T221345",
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": []},
        "properties": {
            "datetime":       "2024-01-15T16:39:01Z",
            "eo:cloud_cover": 5.2,
            "s2:product_type": "S2MSI2A",
        },
        "assets": {
            "B02": {"href": "https://cdn.example.com/B02.tif", "type": "image/tiff"},
            "B04": {"href": "https://cdn.example.com/B04.tif", "type": "image/tiff"},
            "B08": {"href": "https://cdn.example.com/B08.tif", "type": "image/tiff"},
        },
    }


# ---------------------------------------------------------------------------
# CDSEAuth
# ---------------------------------------------------------------------------
class TestCDSEAuth:
    def test_token_refreshed_on_first_call(self):
        auth = CDSEAuth("user@example.com", "password")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"access_token": "tok123", "expires_in": 600}

        with patch("requests.post", return_value=mock_resp) as mock_post:
            token = auth.token()

        assert token == "tok123"
        mock_post.assert_called_once()

    def test_headers_include_bearer(self):
        auth = CDSEAuth("u", "p")
        auth._token      = "mytoken"
        auth._expires_at = 99_999_999_999  # No expira

        headers = auth.headers()
        assert headers["Authorization"] == "Bearer mytoken"

    def test_raises_on_auth_failure(self):
        auth = CDSEAuth("bad", "creds")
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"

        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="autenticación CDSE"):
                auth.token()


# ---------------------------------------------------------------------------
# Parcel
# ---------------------------------------------------------------------------
class TestParcel:
    def test_bbox_order_is_west_south_east_north(self, sample_parcel):
        """STAC espera [west, south, east, north]."""
        bbox = sample_parcel.bbox
        assert bbox[0] == sample_parcel.bbox_west
        assert bbox[1] == sample_parcel.bbox_south
        assert bbox[2] == sample_parcel.bbox_east
        assert bbox[3] == sample_parcel.bbox_north

    def test_bbox_is_valid(self, sample_parcel):
        bbox = sample_parcel.bbox
        assert bbox[0] < bbox[2], "west debe ser menor que east"
        assert bbox[1] < bbox[3], "south debe ser menor que north"

    def test_output_dir_uses_parcel_id(self, sample_parcel):
        assert "H1" in str(sample_parcel.output_dir)


# ---------------------------------------------------------------------------
# BandDownloader
# ---------------------------------------------------------------------------
class TestBandDownloader:
    def test_parse_date_iso_format(self, sample_parcel, sample_config):
        auth       = MagicMock()
        downloader = BandDownloader(auth)
        product    = {"properties": {"datetime": "2024-01-15T16:39:01Z"}}
        date_str   = downloader._parse_date(product)
        assert date_str == "2024-01-15"

    def test_find_band_asset_exact_match(self, sample_parcel, sample_product=None):
        auth       = MagicMock()
        downloader = BandDownloader(auth)
        assets     = {
            "B02": {"href": "https://example.com/B02.tif"},
            "B04": {"href": "https://example.com/B04.tif"},
        }
        result = downloader._find_band_asset(assets, "B02")
        assert result is not None
        assert "B02.tif" in result["href"]

    def test_find_band_asset_prefix_match(self):
        auth       = MagicMock()
        downloader = BandDownloader(auth)
        assets     = {"B08_10m": {"href": "https://example.com/B08_10m.tif"}}
        result     = downloader._find_band_asset(assets, "B08")
        assert result is not None

    def test_find_band_asset_missing(self):
        auth       = MagicMock()
        downloader = BandDownloader(auth)
        result     = downloader._find_band_asset({}, "B99")
        assert result is None

    def test_already_downloaded_false_when_dir_missing(self, tmp_path):
        auth       = MagicMock()
        downloader = BandDownloader(auth)
        result     = downloader._already_downloaded(tmp_path / "nonexistent", ["B02"])
        assert result is False

    def test_already_downloaded_true_when_all_bands_present(self, tmp_path):
        auth       = MagicMock()
        downloader = BandDownloader(auth)
        (tmp_path / "B02.tif").touch()
        (tmp_path / "B04.tif").touch()
        result = downloader._already_downloaded(tmp_path, ["B02", "B04"])
        assert result is True

    def test_already_downloaded_false_when_band_missing(self, tmp_path):
        auth       = MagicMock()
        downloader = BandDownloader(auth)
        (tmp_path / "B02.tif").touch()
        result = downloader._already_downloaded(tmp_path, ["B02", "B04"])
        assert result is False

    def test_dry_run_does_not_write_files(self, tmp_path, sample_parcel, mock_product):
        auth               = MagicMock()
        auth.headers.return_value = {"Authorization": "Bearer tok"}
        downloader         = BandDownloader(auth, dry_run=True)
        sample_parcel.output_dir = tmp_path / "H1"
        result = downloader.download_product(mock_product, sample_parcel, ["B02"])
        assert result is None
        assert not sample_parcel.output_dir.exists()
