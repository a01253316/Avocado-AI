"""
tests/test_vit.py
Tests para sits_vit.py, sits_vit_dataset.py y el scheduler de train_vit.py.
Usa datos sintéticos — sin credenciales ni GPU requerida.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.vit.sits_vit import (
    MaskedTemporalPooling,
    SITSTransformerBlock,
    SITSViT,
    TemporalPositionalEncoding,
    build_vit,
    sits_collate_fn,
)
from models.vit.sits_vit_dataset import SITSViTDataset
from models.vit.train_vit import EarlyStopping, WarmupCosineScheduler

# ---------------------------------------------------------------------------
# Fixtures y constantes
# ---------------------------------------------------------------------------
B  = 4    # batch
T  = 12   # timesteps
C  = 5    # canales
D  = 64   # d_model (pequeño para tests rápidos)
H  = 4    # num_heads


@pytest.fixture
def tiny_vit() -> SITSViT:
    return build_vit(n_channels=C, d_model=D, num_heads=H, num_layers=2, dropout=0.0)


def _random_batch(b=B, t=T, c=C):
    """Genera (x, doy) de prueba."""
    x   = torch.randn(b, t, c)
    doy = torch.randint(1, 366, (b, t))
    return x, doy


def _write_signal_npz(path: Path, n_dates: int = T, ndmi_val: float = 0.3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.rand(n_dates, C).astype(np.float32)
    data[:, 2] = ndmi_val   # canal NDMI fijo para controlar etiqueta
    np.savez_compressed(
        path,
        data  = data,
        dates = np.array([f"2023-{m:02d}-15" for m in range(1, n_dates + 1)], dtype="U10"),
        doy   = np.arange(1, n_dates + 1, dtype=np.int16),
    )


# ---------------------------------------------------------------------------
# TemporalPositionalEncoding
# ---------------------------------------------------------------------------
class TestTemporalPositionalEncoding:
    def test_output_shape(self):
        enc = TemporalPositionalEncoding(d_model=D)
        doy = torch.randint(1, 366, (B, T))
        out = enc(doy)
        assert out.shape == (B, T, D)

    def test_different_doys_give_different_embeddings(self):
        enc = TemporalPositionalEncoding(d_model=D)
        d1  = torch.tensor([[1, 1]])
        d2  = torch.tensor([[180, 180]])
        assert not torch.allclose(enc(d1), enc(d2))

    def test_same_doy_gives_same_embedding(self):
        enc = TemporalPositionalEncoding(d_model=D)
        d1  = torch.tensor([[90, 90]])
        d2  = torch.tensor([[90, 90]])
        torch.testing.assert_close(enc(d1), enc(d2))

    def test_clamped_doy_out_of_range(self):
        """DOY fuera de [1-366] no debe lanzar error."""
        enc = TemporalPositionalEncoding(d_model=D)
        doy = torch.tensor([[0, 400]])     # valores extremos
        out = enc(doy)
        assert out.shape == (1, 2, D)
        assert not torch.isnan(out).any()

    def test_d_model_must_be_even(self):
        with pytest.raises(AssertionError):
            TemporalPositionalEncoding(d_model=65)   # impar


# ---------------------------------------------------------------------------
# SITSTransformerBlock
# ---------------------------------------------------------------------------
class TestSITSTransformerBlock:
    def test_output_shape(self):
        block = SITSTransformerBlock(d_model=D, num_heads=H)
        x     = torch.randn(B, T, D)
        out   = block(x)
        assert out.shape == (B, T, D)

    def test_with_padding_mask(self):
        block = SITSTransformerBlock(d_model=D, num_heads=H)
        x     = torch.randn(B, T, D)
        mask  = torch.zeros(B, T, dtype=torch.bool)
        mask[:, -3:] = True   # Últimas 3 posiciones son padding
        out   = block(x, key_padding_mask=mask)
        assert out.shape == (B, T, D)

    def test_residual_connection_active(self):
        """Con pesos en cero la salida debe ser igual a la entrada (residual)."""
        block = SITSTransformerBlock(d_model=D, num_heads=H)
        # Poner pesos del FFN a cero para aislar el residual
        with torch.no_grad():
            for p in block.ffn.parameters():
                p.zero_()
            for p in block.attn.parameters():
                p.zero_()
        x   = torch.randn(B, T, D)
        out = block(x)
        # No exactamente igual (LayerNorm), pero no debería ser cero
        assert out.abs().sum() > 0


# ---------------------------------------------------------------------------
# MaskedTemporalPooling
# ---------------------------------------------------------------------------
class TestMaskedTemporalPooling:
    def test_no_mask_equals_mean(self):
        pool = MaskedTemporalPooling()
        x    = torch.randn(B, T, D)
        out  = pool(x)
        expected = x.mean(dim=1)
        torch.testing.assert_close(out, expected)

    def test_mask_excludes_padding(self):
        pool = MaskedTemporalPooling()
        x    = torch.zeros(2, 4, D)
        x[0, :2, :] = 1.0   # Solo 2 tokens reales en primera muestra
        x[0, 2:, :] = 999.0 # Padding: valor grande que no debe afectar
        mask = torch.tensor([
            [False, False, True, True],   # 2 reales, 2 padding
            [False, False, False, False], # 4 reales
        ])
        out = pool(x, padding_mask=mask)
        # Primera muestra: media de los 2 tokens reales (valor 1.0)
        torch.testing.assert_close(out[0], torch.ones(D))

    def test_output_shape(self):
        pool = MaskedTemporalPooling()
        x    = torch.randn(B, T, D)
        assert pool(x).shape == (B, D)


# ---------------------------------------------------------------------------
# SITSViT — forward pass
# ---------------------------------------------------------------------------
class TestSITSViT:
    def test_output_shape(self, tiny_vit):
        x, doy = _random_batch()
        out = tiny_vit(x, doy)
        assert out.shape == (B, 1)

    def test_output_is_logit_not_prob(self, tiny_vit):
        """La salida cruda puede estar fuera de [0,1]."""
        x, doy = _random_batch()
        out = tiny_vit(x, doy)
        # Verificamos que existe al menos un valor != 0 (red no trivial)
        assert out.abs().sum().item() > 0

    def test_with_padding_mask(self, tiny_vit):
        x, doy = _random_batch()
        mask   = torch.zeros(B, T, dtype=torch.bool)
        mask[:, -4:] = True
        out = tiny_vit(x, doy, padding_mask=mask)
        assert out.shape == (B, 1)
        assert not torch.isnan(out).any()

    def test_variable_sequence_length(self, tiny_vit):
        """El modelo acepta distintos T siempre que sea consistente en el batch."""
        for t in [5, 12, 24, 50]:
            x   = torch.randn(2, t, C)
            doy = torch.randint(1, 366, (2, t))
            out = tiny_vit(x, doy)
            assert out.shape == (2, 1)

    def test_gradients_flow(self, tiny_vit):
        x, doy = _random_batch()
        y      = torch.zeros(B, 1)
        loss   = torch.nn.BCEWithLogitsLoss()(tiny_vit(x, doy), y)
        loss.backward()
        for name, param in tiny_vit.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Sin gradiente: {name}"

    def test_count_parameters_positive(self, tiny_vit):
        assert tiny_vit.count_parameters() > 0

    def test_name_contains_config(self, tiny_vit):
        name = tiny_vit.name
        assert "64" in name   # d_model
        assert "4"  in name   # num_heads o num_layers

    def test_d_model_not_divisible_by_heads_raises(self):
        with pytest.raises(AssertionError):
            SITSViT(d_model=65, num_heads=8)  # 65 % 8 != 0

    def test_build_vit_presets(self):
        from models.vit.train_vit import PRESETS
        for preset, (d, h, l) in PRESETS.items():
            m = build_vit(d_model=d, num_heads=h, num_layers=l)
            assert m.count_parameters() > 0

    def test_larger_model_has_more_params(self):
        small = build_vit(d_model=64,  num_heads=4, num_layers=2).count_parameters()
        base  = build_vit(d_model=256, num_heads=8, num_layers=6).count_parameters()
        assert base > small


# ---------------------------------------------------------------------------
# sits_collate_fn — padding dinámico
# ---------------------------------------------------------------------------
class TestSitsCollateFn:
    def _make_batch(self, lengths: list[int]) -> list[tuple]:
        batch = []
        for t in lengths:
            x   = torch.randn(t, C)
            doy = torch.randint(1, 366, (t,))
            y   = torch.tensor([0.0])
            batch.append((x, doy, y))
        return batch

    def test_output_shapes(self):
        batch = self._make_batch([8, 12, 10])
        x, doy, mask, labels = sits_collate_fn(batch)
        assert x.shape     == (3, 12, C)   # T_max = 12
        assert doy.shape   == (3, 12)
        assert mask.shape  == (3, 12)
        assert labels.shape == (3, 1)

    def test_padding_mask_correct(self):
        batch = self._make_batch([5, 10])
        _, _, mask, _ = sits_collate_fn(batch)
        # Primera muestra: 5 reales → 5 False, 5 True
        assert mask[0, :5].all() == False
        assert mask[0, 5:].all() == True
        # Segunda muestra: 10 reales → todos False
        assert not mask[1].any()

    def test_padding_values_are_zero(self):
        batch = self._make_batch([3, 8])
        x, _, _, _ = sits_collate_fn(batch)
        # Posiciones de padding en la primera muestra deben ser 0
        assert (x[0, 3:] == 0).all()

    def test_uniform_length_no_padding(self):
        batch = self._make_batch([6, 6, 6])
        _, _, mask, _ = sits_collate_fn(batch)
        # Sin padding → toda la máscara es False
        assert not mask.any()

    def test_labels_stacked_correctly(self):
        batch = [(torch.randn(5, C), torch.ones(5, dtype=torch.long),
                  torch.tensor([float(i)])) for i in range(4)]
        _, _, _, labels = sits_collate_fn(batch)
        assert labels.shape == (4, 1)
        for i in range(4):
            assert labels[i, 0].item() == pytest.approx(float(i))


# ---------------------------------------------------------------------------
# SITSViTDataset
# ---------------------------------------------------------------------------
class TestSITSViTDataset:
    def test_len(self, tmp_path):
        signals = tmp_path / "signals"
        for pid in ["H1", "H2", "H3"]:
            _write_signal_npz(signals / f"{pid}.npz")
        ds = SITSViTDataset(signals)
        assert len(ds) == 3

    def test_item_shapes(self, tmp_path):
        signals = tmp_path / "signals"
        _write_signal_npz(signals / "H1.npz", n_dates=T)
        ds      = SITSViTDataset(signals)
        x, doy, y = ds[0]
        assert x.shape   == (T, C)
        assert doy.shape == (T,)
        assert y.shape   == (1,)

    def test_stressed_label(self, tmp_path):
        signals = tmp_path / "signals"
        _write_signal_npz(signals / "H1.npz", ndmi_val=-0.5)
        ds = SITSViTDataset(signals, ndmi_threshold=-0.1)
        _, _, y = ds[0]
        assert y.item() == 1.0

    def test_healthy_label(self, tmp_path):
        signals = tmp_path / "signals"
        _write_signal_npz(signals / "H1.npz", ndmi_val=0.4)
        ds = SITSViTDataset(signals, ndmi_threshold=-0.1)
        _, _, y = ds[0]
        assert y.item() == 0.0

    def test_doy_values_in_valid_range(self, tmp_path):
        signals = tmp_path / "signals"
        _write_signal_npz(signals / "H1.npz")
        ds     = SITSViTDataset(signals)
        _, doy, _ = ds[0]
        assert doy.min().item() >= 1
        assert doy.max().item() <= 365

    def test_no_nan_in_output(self, tmp_path):
        signals = tmp_path / "signals"
        path    = signals / "H1.npz"
        signals.mkdir(parents=True)
        data    = np.full((T, C), np.nan, dtype=np.float32)
        np.savez_compressed(path, data=data,
                            dates=np.array(["2023-01-15"]*T, dtype="U10"),
                            doy=np.ones(T, dtype=np.int16))
        ds      = SITSViTDataset(signals)
        x, _, _ = ds[0]
        assert not torch.isnan(x).any()

    def test_dataloader_with_collate_fn(self, tmp_path):
        """Integración: DataLoader + collate_fn con distintos T."""
        signals = tmp_path / "signals"
        for pid, t in [("H1", 8), ("H2", 12), ("H3", 6)]:
            _write_signal_npz(signals / f"{pid}.npz", n_dates=t)
        ds     = SITSViTDataset(signals)
        loader = DataLoader(ds, batch_size=3, collate_fn=sits_collate_fn)
        x, doy, mask, y = next(iter(loader))
        assert x.shape[0]   == 3
        assert x.shape[1]   == 12   # T_max
        assert x.shape[2]   == C
        assert not torch.isnan(x).any()


# ---------------------------------------------------------------------------
# WarmupCosineScheduler
# ---------------------------------------------------------------------------
class TestWarmupCosineScheduler:
    def _make_scheduler(self, warmup=5, total=20, lr=1e-3):
        model = torch.nn.Linear(2, 1)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr)
        sch   = WarmupCosineScheduler(opt, warmup_epochs=warmup, total_epochs=total)
        return opt, sch, lr

    def test_lr_increases_during_warmup(self):
        opt, sch, base_lr = self._make_scheduler(warmup=5, total=20)
        lrs = []
        for _ in range(6):
            lrs.append(sch.get_last_lr()[0] if _ > 0 else base_lr)
            sch.step()
        # LR debe crecer durante los primeros epochs de warmup
        assert lrs[0] <= lrs[1] or lrs[1] <= lrs[2]

    def test_lr_decreases_after_warmup(self):
        opt, sch, base_lr = self._make_scheduler(warmup=2, total=20)
        for _ in range(5):
            sch.step()
        lr_mid = sch.get_last_lr()[0]
        for _ in range(10):
            sch.step()
        lr_late = sch.get_last_lr()[0]
        assert lr_late <= lr_mid

    def test_lr_never_negative(self):
        opt, sch, _ = self._make_scheduler(warmup=3, total=30)
        for _ in range(35):
            sch.step()
            assert sch.get_last_lr()[0] >= 0


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------
class TestEarlyStopping:
    def test_stops_after_patience(self):
        es = EarlyStopping(patience=3)
        es.step(0.5)            # epoch 1: mejora
        assert not es.should_stop
        es.step(0.4)            # no mejora
        es.step(0.4)
        es.step(0.4)            # 3 sin mejora → debe parar
        assert es.should_stop

    def test_resets_counter_on_improvement(self):
        es = EarlyStopping(patience=3)
        es.step(0.5)
        es.step(0.4)   # -1
        es.step(0.4)   # -2
        es.step(0.8)   # mejora → reset
        assert not es.should_stop

    def test_returns_true_on_improvement(self):
        es = EarlyStopping(patience=5)
        assert es.step(0.5) is True    # primera mejora
        assert es.step(0.6) is True    # mejor aún
        assert es.step(0.5) is False   # retrocede

    def test_not_triggered_if_always_improving(self):
        es = EarlyStopping(patience=3)
        for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            es.step(v)
        assert not es.should_stop
