"""
pixel_classifier.py
====================
Modelos CNN para detección de estrés hídrico en parcelas de aguacate.

Dos arquitecturas complementarias:

  PixelCNN
  --------
  Trata cada píxel como una muestra independiente.
  Entrada : (B, T*C)  — serie temporal aplanada por píxel
  Salida  : (B, 1)    — logit de estrés hídrico [0=sano, 1=estresado]
  Uso     : baseline rápido, interpretable, bajo costo computacional.
            Ignora la relación espacial entre píxeles vecinos.

  PatchCNN
  --------
  Trata el chip completo como una imagen multicanal.
  Entrada : (B, T*C, H, W)  — T snapshots × C índices como canales
  Salida  : (B, 1, H, W)    — mapa de estrés por píxel (segmentación)
  Uso     : captura contexto espacial. Mejor precisión que PixelCNN
            a costa de más memoria. Precursor natural del ViT.

Notas de diseño:
  - Ambos usan BatchNorm + Dropout para regularización.
  - PatchCNN usa arquitectura encoder-decoder estilo U-Net lite
    para preservar la resolución espacial en la predicción.
  - Los pesos se inicializan con Kaiming (relu) para convergencia estable.
  - La salida es un logit crudo (sin sigmoid) para usarse con
    BCEWithLogitsLoss, que es numéricamente más estable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Bloque reutilizable: Conv + BN + ReLU
# ---------------------------------------------------------------------------
class ConvBnRelu(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        padding: int = 1,
        stride: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# 1. PixelCNN  — baseline ligero (T*C → 1)
# ---------------------------------------------------------------------------
class PixelCNN(nn.Module):
    """
    MLP-CNN de 1-D sobre la serie temporal de un píxel.

    La entrada es el vector de índices de todas las fechas aplanado:
        (B, T*C) donde T=timesteps, C=5 índices

    Arquitectura:
        Linear(T*C → 256) → BN → ReLU → Dropout
        Linear(256  → 128) → BN → ReLU → Dropout
        Linear(128  → 64)  → BN → ReLU
        Linear(64   → 1)   → logit

    Parámetros
    ----------
    n_timesteps : número de fechas en la serie (T)
    n_channels  : número de índices espectrales (C, default 5)
    dropout     : tasa de dropout (default 0.3)
    """

    def __init__(
        self,
        n_timesteps: int,
        n_channels: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        in_features = n_timesteps * n_channels

        self.net = nn.Sequential(
            # Bloque 1
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # Bloque 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # Bloque 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # Clasificador
            nn.Linear(64, 1),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C) o (B, T*C)
        retorna: (B, 1)
        """
        if x.dim() == 3:
            x = x.flatten(start_dim=1)   # (B, T*C)
        return self.net(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def name(self) -> str:
        return "PixelCNN"


# ---------------------------------------------------------------------------
# 2. PatchCNN  — segmentación espacial (T*C, H, W → 1, H, W)
# ---------------------------------------------------------------------------
class PatchCNN(nn.Module):
    """
    Encoder-decoder CNN ligero estilo U-Net para segmentación de estrés.

    Entrada : (B, T*C, H, W)  — canales = timesteps × índices
    Salida  : (B, 1,   H, W)  — mapa de probabilidad de estrés por píxel

    Encoder: 3 niveles de downsampling (stride-2 convs)
    Decoder: 3 niveles de upsampling (bilinear + conv)
    Skip connections entre encoder y decoder (U-Net style)

    Parámetros
    ----------
    n_timesteps : número de fechas (T)
    n_channels  : número de índices (C, default 5)
    base_filters: filtros base del encoder (se duplican en cada nivel, default 32)
    dropout     : tasa de dropout (default 0.3)
    """

    def __init__(
        self,
        n_timesteps: int,
        n_channels: int = 5,
        base_filters: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        in_ch = n_timesteps * n_channels
        f     = base_filters   # alias corto

        # ── Encoder ─────────────────────────────────────────
        self.enc1 = nn.Sequential(
            ConvBnRelu(in_ch, f),
            ConvBnRelu(f, f),
        )                                               # out: (B, f,   H,   W)
        self.enc2 = nn.Sequential(
            ConvBnRelu(f,    f*2, stride=2),
            ConvBnRelu(f*2,  f*2),
        )                                               # out: (B, f*2, H/2, W/2)
        self.enc3 = nn.Sequential(
            ConvBnRelu(f*2,  f*4, stride=2),
            ConvBnRelu(f*4,  f*4),
        )                                               # out: (B, f*4, H/4, W/4)

        # ── Bottleneck ──────────────────────────────────────
        self.bottleneck = nn.Sequential(
            ConvBnRelu(f*4, f*8, stride=2),
            ConvBnRelu(f*8, f*8),
            nn.Dropout2d(dropout),
        )                                               # out: (B, f*8, H/8, W/8)

        # ── Decoder (con skip connections) ──────────────────
        self.dec3 = nn.Sequential(
            ConvBnRelu(f*8 + f*4, f*4),
            ConvBnRelu(f*4, f*4),
        )
        self.dec2 = nn.Sequential(
            ConvBnRelu(f*4 + f*2, f*2),
            ConvBnRelu(f*2, f*2),
        )
        self.dec1 = nn.Sequential(
            ConvBnRelu(f*2 + f, f),
            ConvBnRelu(f, f),
        )

        # ── Cabeza de clasificación ──────────────────────────
        self.head = nn.Conv2d(f, 1, kernel_size=1)    # logit por píxel

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W) o (B, T*C, H, W)
        retorna: (B, 1, H, W)
        """
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B, T * C, H, W)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        bn = self.bottleneck(e3)

        # Decoder con skip connections
        d3 = self._up_cat(bn, e3)
        d3 = self.dec3(d3)

        d2 = self._up_cat(d3, e2)
        d2 = self.dec2(d2)

        d1 = self._up_cat(d2, e1)
        d1 = self.dec1(d1)

        return self.head(d1)   # (B, 1, H, W) — logit crudo

    @staticmethod
    def _up_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample x al tamaño de skip y concatena."""
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def name(self) -> str:
        return "PatchCNN"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_model(
    architecture: str,
    n_timesteps: int,
    n_channels: int = 5,
    **kwargs,
) -> nn.Module:
    """
    Construye el modelo por nombre.

    architecture : "pixel" | "patch"
    n_timesteps  : número de fechas T en la serie
    n_channels   : número de índices C (default 5: NDVI,NDWI,NDMI,NDRE,EVI)
    **kwargs     : parámetros adicionales pasados al constructor del modelo

    Ejemplo:
        model = build_model("pixel", n_timesteps=24)
        model = build_model("patch", n_timesteps=24, base_filters=64)
    """
    arch = architecture.lower()
    if arch == "pixel":
        return PixelCNN(n_timesteps=n_timesteps, n_channels=n_channels, **kwargs)
    elif arch == "patch":
        return PatchCNN(n_timesteps=n_timesteps, n_channels=n_channels, **kwargs)
    else:
        raise ValueError(
            f"Arquitectura '{architecture}' no reconocida. "
            "Usa 'pixel' o 'patch'."
        )


def count_parameters(model: nn.Module) -> int:
    """Retorna el número de parámetros entrenables del modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
