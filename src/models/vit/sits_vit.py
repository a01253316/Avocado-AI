"""
sits_vit.py
===========
Vision Transformer para Series de Tiempo Satelitales (ViT for SITS).

Implementación basada en:
    Garnot & Landrieu (2021) — "Lightweight Temporal Self-Attention for
    Classifying Satellite Image Time Series"
    https://arxiv.org/abs/2007.00586

Diferencias clave vs un ViT estándar de imágenes:
    - No hay patches espaciales — cada timestep es un "token" espectral.
    - El positional encoding es temporal: usa el Día del Año (DOY 1-365)
      en lugar de la posición en una cuadrícula, lo que permite manejar
      series temporales irregulares (imágenes no equiespaciadas en el tiempo).
    - La longitud de secuencia T varía por parcela → se usa padding + máscara.
    - La salida es un vector de logits para num_classes clases.

Arquitectura:
    ┌─────────────────────────────────────────────────────────────┐
    │  Entrada: (B, T, C)  series temporales de índices           │
    │           + doy: (B, T) día del año por timestep            │
    └────────────────┬────────────────────────────────────────────┘
                     │
    ┌────────────────▼────────────────────────────────────────────┐
    │  InputProjection  Linear(C → d_model)                       │
    └────────────────┬────────────────────────────────────────────┘
                     │
    ┌────────────────▼────────────────────────────────────────────┐
    │  TemporalPositionalEncoding  DOY → sin/cos embedding        │
    │  Suma al token embedding: x = proj(x) + pos_enc(doy)        │
    └────────────────┬────────────────────────────────────────────┘
                     │
    ┌────────────────▼────────────────────────────────────────────┐
    │  TransformerEncoder  × num_layers                            │
    │   MultiHeadSelfAttention(d_model, num_heads)                 │
    │   + LayerNorm + Dropout + FFN                                │
    └────────────────┬────────────────────────────────────────────┘
                     │
    ┌────────────────▼────────────────────────────────────────────┐
    │  TemporalPooling  mean sobre T (ignorando padding)           │
    └────────────────┬────────────────────────────────────────────┘
                     │
    ┌────────────────▼────────────────────────────────────────────┐
    │  ClassificationHead  Linear(d_model → num_classes)          │
    └─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Temporal Positional Encoding basado en DOY
# ---------------------------------------------------------------------------
class TemporalPositionalEncoding(nn.Module):
    """
    Codifica el Día del Año (DOY 1-365) en un embedding sinusoidal de
    dimensión d_model.

    A diferencia del positional encoding estándar (índice 0, 1, 2...),
    este usa el DOY real de cada imagen, permitiendo que el modelo
    distinga "es una imagen de enero" de "es una imagen de julio"
    independientemente del orden en la secuencia.

    Fórmula (análoga a Vaswani et al. 2017 pero sobre DOY):
        PE(doy, 2i)   = sin(doy / 10000^(2i/d_model))
        PE(doy, 2i+1) = cos(doy / 10000^(2i/d_model))

    Parámetros
    ----------
    d_model : dimensión del embedding (debe ser par)
    max_doy : máximo valor de DOY (default 366 para años bisiestos)
    """

    def __init__(self, d_model: int, max_doy: int = 366):
        super().__init__()
        assert d_model % 2 == 0, "d_model debe ser par para sin/cos encoding"

        pe  = torch.zeros(max_doy + 1, d_model)
        pos = torch.arange(0, max_doy + 1, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe)

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        """
        doy : (B, T) — día del año por timestep, valores en [1, 366]
        retorna: (B, T, d_model) — embedding temporal
        """
        doy = doy.clamp(1, self.pe.size(0) - 1).long()  # type: ignore[attr-defined]
        return self.pe[doy]   # (B, T, d_model)


# ---------------------------------------------------------------------------
# 2. Bloque Transformer Encoder (un layer)
# ---------------------------------------------------------------------------
class SITSTransformerBlock(nn.Module):
    """
    Un bloque Transformer estándar adaptado para series temporales:
        x → LayerNorm → MultiHeadAttention → residual
          → LayerNorm → FFN → residual

    Pre-LayerNorm (antes de la atención) en lugar de post-LayerNorm:
    más estable en series temporales cortas con pocas épocas.

    Parámetros
    ----------
    d_model    : dimensión del embedding
    num_heads  : cabezas de atención (d_model debe ser divisible)
    d_ff       : dimensión interna del FFN (default 4×d_model)
    dropout    : tasa de dropout
    """

    def __init__(
        self,
        d_model  : int,
        num_heads: int,
        d_ff     : int | None = None,
        dropout  : float = 0.1,
    ):
        super().__init__()
        d_ff = d_ff or d_model * 4

        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        x          : torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x                : (B, T, d_model)
        key_padding_mask : (B, T) bool — True en posiciones de padding
        retorna          : (B, T, d_model)
        """
        x2, _ = self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            key_padding_mask = key_padding_mask,
        )
        x = x + self.drop(x2)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# 3. Pooling temporal (ignora padding)
# ---------------------------------------------------------------------------
class MaskedTemporalPooling(nn.Module):
    """
    Calcula la media de los embeddings a lo largo del tiempo T,
    ignorando las posiciones de padding (donde la máscara es True).

    Si todas las posiciones son padding (caso degenerado), retorna ceros.
    """

    def forward(
        self,
        x           : torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x            : (B, T, d_model)
        padding_mask : (B, T) bool — True = padding, False = dato real
        retorna      : (B, d_model)
        """
        if padding_mask is None:
            return x.mean(dim=1)

        valid = (~padding_mask).float().unsqueeze(-1)   # (B, T, 1)
        total = valid.sum(dim=1).clamp(min=1)            # (B, 1)
        return (x * valid).sum(dim=1) / total            # (B, d_model)


# ---------------------------------------------------------------------------
# 4. Modelo completo: SITSViT
# ---------------------------------------------------------------------------
class SITSViT(nn.Module):
    """
    ViT for SITS — Vision Transformer para Series de Tiempo Satelitales.

    Parámetros
    ----------
    n_channels  : número de índices espectrales de entrada (C, default 5)
    d_model     : dimensión del embedding interno (default 128)
    num_heads   : cabezas de atención (default 8, d_model % num_heads == 0)
    num_layers  : número de bloques Transformer (default 4)
    d_ff        : dimensión FFN (default None → 4×d_model)
    dropout     : dropout general (default 0.1)
    num_classes : número de clases de salida (default 3)
                  2 = binario (sano/estresado)
                  3 = triclase (no_stress/moderate/severe)
    max_seq_len : longitud máxima de secuencia T (default 200)

    Entrada en forward():
        x   : (B, T, C)  — índices espectrales por timestep
        doy : (B, T)      — día del año por timestep [1-365]
        padding_mask : (B, T) bool opcional — True en posiciones vacías

    Salida:
        logits : (B, num_classes)  — logits crudos (sin softmax)
    """

    def __init__(
        self,
        n_channels  : int   = 5,
        d_model     : int   = 128,
        num_heads   : int   = 8,
        num_layers  : int   = 4,
        d_ff        : int | None = None,
        dropout     : float = 0.1,
        num_classes : int   = 3,       # CAMBIO: antes fijo en 1 (binario)
        max_seq_len : int   = 200,
    ):
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) debe ser divisible entre num_heads ({num_heads})"
        )

        self.d_model     = d_model
        self.num_heads   = num_heads
        self.num_layers  = num_layers
        self.num_classes = num_classes  # CAMBIO: guardar num_classes

        # ── Proyección de entrada ─────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(n_channels, d_model),
            nn.LayerNorm(d_model),
        )

        # ── Encoding temporal (DOY) ───────────────────────────
        self.temporal_enc = TemporalPositionalEncoding(d_model)

        # ── Capas Transformer ─────────────────────────────────
        self.transformer_blocks = nn.ModuleList([
            SITSTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # ── Norm final y pooling ──────────────────────────────
        self.final_norm = nn.LayerNorm(d_model)
        self.pooling    = MaskedTemporalPooling()

        # ── Cabeza de clasificación ───────────────────────────
        # CAMBIO: última capa de 1 → num_classes
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),   # CAMBIO: 1 → num_classes
        )

        self._init_weights()

    def forward(
        self,
        x           : torch.Tensor,
        doy         : torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x            : (B, T, C)  valores float32, NaN ya reemplazado por 0
        doy          : (B, T)     día del año int, [1-365]
        padding_mask : (B, T)     bool, True = posición de padding
        retorna      : (B, num_classes)  logits crudos
        """
        # 1. Proyección espectral
        x = self.input_proj(x)                         # (B, T, d_model)

        # 2. Sumar encoding temporal
        x = x + self.temporal_enc(doy)                 # (B, T, d_model)

        # 3. Bloques Transformer con máscara de padding
        for block in self.transformer_blocks:
            x = block(x, key_padding_mask=padding_mask)  # (B, T, d_model)

        # 4. Norm + pooling temporal
        x = self.final_norm(x)                         # (B, T, d_model)
        x = self.pooling(x, padding_mask)              # (B, d_model)

        # 5. Clasificación
        return self.classifier(x)                      # (B, num_classes)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def name(self) -> str:
        return (
            f"SITSViT(d={self.d_model}, h={self.num_heads}, "
            f"L={self.num_layers}, C={self.num_classes})"   # CAMBIO: + C=
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 5. Dataset-aware collate_fn para series de longitud variable
# ---------------------------------------------------------------------------
def sits_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function para DataLoader que maneja series de longitud variable.

    Cada elemento del batch debe ser una tupla:
        (x: Tensor (T, C),  doy: Tensor (T,),  label: Tensor escalar long)

    Retorna:
        x_padded     : (B, T_max, C)
        doy_padded   : (B, T_max)        — DOY, 0 en padding
        padding_mask : (B, T_max) bool   — True en posiciones de padding
        labels       : (B,) long         — CAMBIO: antes (B, 1) float
    """
    xs, doys, labels = zip(*batch)

    t_max   = max(x.shape[0] for x in xs)
    c       = xs[0].shape[1]
    b       = len(xs)

    x_padded   = torch.zeros(b, t_max, c, dtype=torch.float32)
    doy_padded = torch.zeros(b, t_max, dtype=torch.long)
    mask       = torch.ones(b, t_max, dtype=torch.bool)   # True = padding

    for i, (x, doy) in enumerate(zip(xs, doys)):
        t = x.shape[0]
        x_padded[i, :t, :]  = x
        doy_padded[i, :t]   = doy
        mask[i, :t]         = False   # posiciones reales → False

    # CAMBIO: labels son escalares long → stack da (B,), no (B, 1)
    labels = torch.stack(labels, dim=0).long()   # (B,)
    return x_padded, doy_padded, mask, labels


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_vit(
    n_channels  : int   = 5,
    d_model     : int   = 128,
    num_heads   : int   = 8,
    num_layers  : int   = 4,
    dropout     : float = 0.1,
    num_classes : int   = 3,       # CAMBIO: antes no existía, asumía 1
    **kwargs,
) -> SITSViT:
    """
    Construye un SITSViT con los parámetros dados.

    Configuraciones de referencia:
        Tiny  : d_model=64,  num_heads=4, num_layers=2  (~102K params)
        Small : d_model=128, num_heads=8, num_layers=4  (~803K params) ← default
        Base  : d_model=256, num_heads=8, num_layers=6  (~4.8M params)
    """
    return SITSViT(
        n_channels  = n_channels,
        d_model     = d_model,
        num_heads   = num_heads,
        num_layers  = num_layers,
        dropout     = dropout,
        num_classes = num_classes,  # CAMBIO: pasar al constructor
        **kwargs,
    )