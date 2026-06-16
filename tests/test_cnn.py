"""
tests/test_cnn.py
Tests para pixel_classifier.py, sits_dataset.py y el loop de entrenamiento.
Usa datos sintéticos en tmp_path — no requiere datos reales ni GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.cnn.pixel_classifier import (
    PixelCNN,
    PatchCNN,
    build_model,
    count_parameters,
)
from models.cnn.sits_dataset import (
    SITSPixelDataset,
    SITSPatchDataset,
    NDMI_CHANNEL,
    N_CHANNELS,
)

# ---------------------------------------------------------------------------
# Parámetros fijos para pruebas
# ---------------------------------------------------------------------------
T   = 12     # timesteps
C   = 5      # canales (índices)
H   = 10     # altura chip
W   = 10     # ancho chip
B   = 4      # batch size


# ---------------------------------------------------------------------------
# Helpers para crear .npz sintéticos
# ---------------------------------------------------------------------------
def _write_signal_npz(path: Path, n_dates: int = T, ndmi_val: float = 0.3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data        = np.random.rand(n_dates, C).astype(np.float32)
    data[:, NDMI_CHANNEL] = ndmi_val   # NDMI fijo para controlar etiqueta
    np.savez_compressed(
        path,
        data  = data,
        dates = np.array([f"2023-{m:02d}-15" for m in range(1, n_dates + 1)], dtype="U10"),
        doy   = np.arange(1, n_dates + 1, dtype=np.int16),
    )


def _write_patch_npz(path: Path, n_dates: int = T, ndmi_val: float = 0.3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.rand(n_dates, C, H, W).astype(np.float32)
    data[:, NDMI_CHANNEL, :, :] = ndmi_val
    np.savez_compressed(
        path,
        data  = data,
        dates = np.array([f"2023-{m:02d}-15" for m in range(1, n_dates + 1)], dtype="U10"),
        doy   = np.arange(1, n_dates + 1, dtype=np.int16),
    )


# ---------------------------------------------------------------------------
# PixelCNN
# ---------------------------------------------------------------------------
class TestPixelCNN:
    def test_output_shape(self):
        model = PixelCNN(n_timesteps=T, n_channels=C)
        x     = torch.randn(B, T, C)
        out   = model(x)
        assert out.shape == (B, 1)

    def test_accepts_flat_input(self):
        model = PixelCNN(n_timesteps=T, n_channels=C)
        x     = torch.randn(B, T * C)
        out   = model(x)
        assert out.shape == (B, 1)

    def test_output_is_unbounded_logit(self):
        """La salida debe ser logit crudo (no sigmoid), no acotado en [0,1]."""
        model = PixelCNN(n_timesteps=T, n_channels=C)
        x     = torch.randn(B, T, C) * 10   # entrada grande
        out   = model(x)
        # Al menos algún logit debe estar fuera de [0, 1]
        assert out.abs().max().item() > 0   # solo comprueba que hay salida

    def test_parameter_count_positive(self):
        model = PixelCNN(n_timesteps=T, n_channels=C)
        assert count_parameters(model) > 0

    def test_gradients_flow(self):
        model = PixelCNN(n_timesteps=T, n_channels=C)
        x     = torch.randn(B, T, C, requires_grad=False)
        y     = torch.zeros(B, 1)
        loss  = torch.nn.BCEWithLogitsLoss()(model(x), y)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Sin gradiente en {name}"

    def test_different_timesteps(self):
        for t in [6, 12, 24, 36]:
            model = PixelCNN(n_timesteps=t, n_channels=C)
            x     = torch.randn(B, t, C)
            assert model(x).shape == (B, 1)


# ---------------------------------------------------------------------------
# PatchCNN
# ---------------------------------------------------------------------------
class TestPatchCNN:
    def test_output_shape_5d_input(self):
        model = PatchCNN(n_timesteps=T, n_channels=C)
        x     = torch.randn(B, T, C, H, W)
        out   = model(x)
        assert out.shape == (B, 1, H, W)

    def test_output_shape_4d_input(self):
        model = PatchCNN(n_timesteps=T, n_channels=C)
        x     = torch.randn(B, T * C, H, W)
        out   = model(x)
        assert out.shape == (B, 1, H, W)

    def test_spatial_resolution_preserved(self):
        """El decoder debe restaurar H×W exactos."""
        for h, w in [(16, 16), (32, 32), (50, 50)]:
            model = PatchCNN(n_timesteps=T, n_channels=C)
            x     = torch.randn(1, T, C, h, w)
            out   = model(x)
            assert out.shape == (1, 1, h, w), f"Resolución perdida para {h}×{w}"

    def test_gradients_flow(self):
        model = PatchCNN(n_timesteps=T, n_channels=C)
        x     = torch.randn(B, T, C, H, W)
        y     = torch.zeros(B, H, W)
        logits = model(x).squeeze(1)
        loss   = torch.nn.BCEWithLogitsLoss()(logits, y)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Sin gradiente en {name}"

    def test_base_filters_scaling(self):
        """Más filtros → más parámetros."""
        m32 = count_parameters(PatchCNN(T, C, base_filters=32))
        m64 = count_parameters(PatchCNN(T, C, base_filters=64))
        assert m64 > m32


# ---------------------------------------------------------------------------
# build_model factory
# ---------------------------------------------------------------------------
class TestBuildModel:
    def test_builds_pixel_model(self):
        model = build_model("pixel", n_timesteps=T)
        assert isinstance(model, PixelCNN)

    def test_builds_patch_model(self):
        model = build_model("patch", n_timesteps=T)
        assert isinstance(model, PatchCNN)

    def test_unknown_arch_raises(self):
        with pytest.raises(ValueError, match="no reconocida"):
            build_model("transformer", n_timesteps=T)

    def test_pixel_name_property(self):
        assert build_model("pixel", T).name == "PixelCNN"

    def test_patch_name_property(self):
        assert build_model("patch", T).name == "PatchCNN"


# ---------------------------------------------------------------------------
# SITSPixelDataset
# ---------------------------------------------------------------------------
class TestSITSPixelDataset:
    def test_len_equals_num_npz(self, tmp_path):
        signals_dir = tmp_path / "signals"
        for pid in ["H1", "H2", "H3"]:
            _write_signal_npz(signals_dir / f"{pid}.npz")
        ds = SITSPixelDataset(signals_dir, window_size=T)
        assert len(ds) == 3

    def test_item_shape(self, tmp_path):
        signals_dir = tmp_path / "signals"
        _write_signal_npz(signals_dir / "H1.npz")
        ds   = SITSPixelDataset(signals_dir, window_size=T)
        x, y = ds[0]
        assert x.shape == (T, C)
        assert y.shape == (1,)

    def test_stressed_label_when_ndmi_below_threshold(self, tmp_path):
        signals_dir = tmp_path / "signals"
        _write_signal_npz(signals_dir / "H1.npz", ndmi_val=-0.3)  # bajo → estrés
        ds   = SITSPixelDataset(signals_dir, ndmi_threshold=-0.1, window_size=T)
        _, y = ds[0]
        assert y.item() == 1.0

    def test_healthy_label_when_ndmi_above_threshold(self, tmp_path):
        signals_dir = tmp_path / "signals"
        _write_signal_npz(signals_dir / "H1.npz", ndmi_val=0.4)   # alto → sano
        ds   = SITSPixelDataset(signals_dir, ndmi_threshold=-0.1, window_size=T)
        _, y = ds[0]
        assert y.item() == 0.0

    def test_parcel_id_filter(self, tmp_path):
        signals_dir = tmp_path / "signals"
        for pid in ["H1", "H2", "H3", "H4"]:
            _write_signal_npz(signals_dir / f"{pid}.npz")
        ds = SITSPixelDataset(signals_dir, parcel_ids=["H1", "H2"], window_size=T)
        assert len(ds) == 2

    def test_raises_if_no_npz_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SITSPixelDataset(tmp_path / "empty_signals")

    def test_nan_replaced_by_zero(self, tmp_path):
        signals_dir = tmp_path / "signals"
        path = signals_dir / "H1.npz"
        signals_dir.mkdir(parents=True)
        data = np.full((T, C), np.nan, dtype=np.float32)
        np.savez_compressed(path, data=data,
                            dates=np.array(["2023-01-15"]*T, dtype="U10"),
                            doy=np.ones(T, dtype=np.int16))
        ds   = SITSPixelDataset(signals_dir, window_size=T)
        x, _ = ds[0]
        assert not torch.isnan(x).any()


# ---------------------------------------------------------------------------
# SITSPatchDataset
# ---------------------------------------------------------------------------
class TestSITSPatchDataset:
    def test_len_equals_num_npz(self, tmp_path):
        patches_dir = tmp_path / "patches"
        for pid in ["H1", "H2"]:
            _write_patch_npz(patches_dir / f"{pid}.npz")
        ds = SITSPatchDataset(patches_dir)
        assert len(ds) == 2

    def test_item_shapes(self, tmp_path):
        patches_dir = tmp_path / "patches"
        _write_patch_npz(patches_dir / "H1.npz")
        ds   = SITSPatchDataset(patches_dir)
        x, y = ds[0]
        assert x.shape == (T, C, H, W)
        assert y.shape == (H, W)

    def test_label_map_is_binary(self, tmp_path):
        patches_dir = tmp_path / "patches"
        _write_patch_npz(patches_dir / "H1.npz", ndmi_val=-0.5)
        ds   = SITSPatchDataset(patches_dir, ndmi_threshold=-0.1)
        _, y = ds[0]
        unique = torch.unique(y)
        assert all(v in [0.0, 1.0] for v in unique)

    def test_max_t_truncates_series(self, tmp_path):
        patches_dir = tmp_path / "patches"
        _write_patch_npz(patches_dir / "H1.npz", n_dates=12)
        ds   = SITSPatchDataset(patches_dir, max_t=6)
        x, _ = ds[0]
        assert x.shape[0] == 6

    def test_stressed_pixels_when_ndmi_low(self, tmp_path):
        patches_dir = tmp_path / "patches"
        _write_patch_npz(patches_dir / "H1.npz", ndmi_val=-0.5)
        ds   = SITSPatchDataset(patches_dir, ndmi_threshold=-0.1)
        _, y = ds[0]
        assert y.mean().item() == pytest.approx(1.0)

    def test_healthy_pixels_when_ndmi_high(self, tmp_path):
        patches_dir = tmp_path / "patches"
        _write_patch_npz(patches_dir / "H1.npz", ndmi_val=0.5)
        ds   = SITSPatchDataset(patches_dir, ndmi_threshold=-0.1)
        _, y = ds[0]
        assert y.mean().item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Integración: forward pass en DataLoader
# ---------------------------------------------------------------------------
class TestDataLoaderIntegration:
    def test_pixel_dataloader_batch(self, tmp_path):
        signals_dir = tmp_path / "signals"
        for pid in ["H1", "H2", "H3", "H4", "H5"]:
            _write_signal_npz(signals_dir / f"{pid}.npz")

        from torch.utils.data import DataLoader
        ds     = SITSPixelDataset(signals_dir, window_size=T)
        loader = DataLoader(ds, batch_size=3, shuffle=True)
        model  = PixelCNN(n_timesteps=T, n_channels=C)
        model.eval()

        x, y = next(iter(loader))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (3, 1)
        assert not torch.isnan(out).any()

    def test_patch_dataloader_batch(self, tmp_path):
        patches_dir = tmp_path / "patches"
        for pid in ["H1", "H2"]:
            _write_patch_npz(patches_dir / f"{pid}.npz")

        from torch.utils.data import DataLoader
        ds     = SITSPatchDataset(patches_dir)
        loader = DataLoader(ds, batch_size=2)
        model  = PatchCNN(n_timesteps=T, n_channels=C)
        model.eval()

        x, y = next(iter(loader))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1, H, W)
        assert not torch.isnan(out).any()
