from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sam2.prepare_sam2_dataset import mask_to_binary, patch_to_rgb
from src.models.sam2.train_sam2_finetune import positive_point, resolve_device


def test_patch_to_rgb_returns_uint8_image():
    data = np.zeros((24, 5, 4, 3), dtype=np.float32)
    data[:, 0, :, :] = 0.7
    data[:, 1, :, :] = 0.2
    data[:, 2, :, :] = 0.4

    rgb = patch_to_rgb(data)

    assert rgb.shape == (4, 3, 3)
    assert rgb.dtype == np.uint8


def test_patch_to_rgb_rejects_short_series():
    data = np.zeros((4, 5, 4, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="Series too short"):
        patch_to_rgb(data)


def test_mask_to_binary_uses_moderate_and_severe_by_default():
    mask = np.array([[-1, 0, 1, 2]], dtype=np.int8)

    binary = mask_to_binary(mask)

    np.testing.assert_array_equal(binary, np.array([[0, 0, 255, 255]], dtype=np.uint8))


def test_mask_to_binary_can_export_severe_only():
    mask = np.array([[0, 1, 2]], dtype=np.int8)

    binary = mask_to_binary(mask, min_class=2)

    np.testing.assert_array_equal(binary, np.array([[0, 0, 255]], dtype=np.uint8))


def test_positive_point_returns_point_inside_mask():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[2, 3] = 255

    point = positive_point(mask)

    np.testing.assert_array_equal(point, np.array([[3, 2]], dtype=np.float32))


def test_resolve_device_auto_uses_cpu_without_cuda(monkeypatch):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert resolve_device("auto") == "cpu"


def test_resolve_device_cuda_falls_back_without_cuda(monkeypatch):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert resolve_device("cuda") == "cpu"
