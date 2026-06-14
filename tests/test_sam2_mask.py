from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.features import NDMI_CH, build_ndmi_mask


def _patch_with_ndmi(ndmi: np.ndarray, t: int = 24, c: int = 5) -> np.ndarray:
    data = np.zeros((t, c, *ndmi.shape), dtype=np.float32)
    data[:, NDMI_CH, :, :] = ndmi
    return data


def test_build_ndmi_mask_classifies_pixels_by_thresholds():
    ndmi = np.array(
        [
            [0.70, 0.35, 0.10],
            [0.50, 0.24, 0.05],
        ],
        dtype=np.float32,
    )

    mask = build_ndmi_mask(_patch_with_ndmi(ndmi), t_mod=0.40, t_sev=0.20)

    expected = np.array(
        [
            [0, 1, 2],
            [0, 1, 2],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(mask, expected)


def test_build_ndmi_mask_marks_nan_pixels_as_no_data():
    ndmi = np.array([[0.50, np.nan]], dtype=np.float32)

    mask = build_ndmi_mask(_patch_with_ndmi(ndmi), t_mod=0.40, t_sev=0.20)

    np.testing.assert_array_equal(mask, np.array([[0, -1]], dtype=np.int8))


def test_build_ndmi_mask_uses_recent_window_only():
    data = _patch_with_ndmi(np.array([[0.70]], dtype=np.float32), t=30)
    data[-24:, NDMI_CH, :, :] = 0.10

    mask = build_ndmi_mask(data, t_mod=0.40, t_sev=0.20)

    assert mask[0, 0] == 2


def test_build_ndmi_mask_rejects_short_series():
    data = _patch_with_ndmi(np.array([[0.70]], dtype=np.float32), t=4)

    with pytest.raises(ValueError, match="Serie muy corta"):
        build_ndmi_mask(data, t_mod=0.40, t_sev=0.20)


def test_build_ndmi_mask_rejects_non_patch_shape():
    with pytest.raises(ValueError, match="forma"):
        build_ndmi_mask(np.zeros((24, 5), dtype=np.float32), t_mod=0.40, t_sev=0.20)
