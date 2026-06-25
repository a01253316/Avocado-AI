"""
stress_likelihood_net.py — Experimento E: Red Neuronal Probabilística
======================================================================
Mapea [historial de índices espectrales + huella espectral del terreno + fecha]
→ parámetros (μ, σ) de una distribución Gaussiana multivariada (diagonal) para
cada uno de los 5 índices espectrales.

En vez de un umbral fijo (percentil/decil, que el profe calificó de "arbitrario"),
este modelo aprende la distribución esperada de los índices condicionada a:
  1. El historial reciente (últimas T=24 observaciones) — contexto temporal.
  2. La huella espectral estática de la parcela (media histórica de cada índice)
     — proxy del tipo de terreno visible vía satélite.
  3. El día del año (DOY) — estacionalidad.

La clasificación de estrés se hace en espacio de verosimilitud:
  z_i = (y_obs_i − μ_i) / σ_i         → z-score por índice
  stress_signal = −Σ w_i · z_i         → positivo cuando la observación está
                                           por debajo de lo esperado (estrés)
  p = Φ(stress_signal)                  → percentil bajo la normal estándar
  p < 25  → sin estrés
  p < 60  → moderado
  p ≥ 60  → severo

La pérdida de entrenamiento es la Gaussian NLL, que es equivalente a ajustar
los parámetros de la PDF por máxima verosimilitud (lo que propuso el profe).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DayOfYearEncoder(nn.Module):
    """Codificación sinusoidal del día del año → vector de dimensión d_out."""

    def __init__(self, d_out: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(4, d_out), nn.ReLU())

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        doy = doy.float()
        a_yr = 2 * math.pi * doy / 365.25   # ciclo anual
        a_mo = 2 * math.pi * doy / 30.44    # ciclo mensual
        feats = torch.stack(
            [torch.sin(a_yr), torch.cos(a_yr), torch.sin(a_mo), torch.cos(a_mo)],
            dim=-1,
        )
        return self.proj(feats)


class StressLikelihoodNet(nn.Module):
    """
    Red neuronal probabilística para estrés hídrico.

    Entradas
    --------
    x_hist   : (B, T, 5)  historial normalizado de índices espectrales (T=24)
    x_static : (B, 5)     huella espectral de la parcela (media histórica)
    doy      : (B,)       día del año de la fecha objetivo [1, 365]

    Salidas
    -------
    mu    : (B, 5)  media predicha por índice
    sigma : (B, 5)  desviación estándar predicha (siempre > 0)

    Arquitectura
    ------------
    1. Transformer encoder sobre la secuencia temporal → vector de contexto
    2. Proyección lineal de la huella estática del terreno
    3. Codificación sinusoidal del DOY
    4. Fusión (concat) → MLP compartido → cabezas μ y log-σ
    """

    def __init__(
        self,
        n_indices: int   = 5,
        seq_len:   int   = 24,
        d_model:   int   = 64,
        n_heads:   int   = 4,
        n_layers:  int   = 2,
        d_static:  int   = 32,
        d_date:    int   = 32,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.n_indices = n_indices
        self.seq_len   = seq_len

        # ── Encoder de la secuencia temporal (Transformer) ────────────────
        self.input_proj = nn.Linear(n_indices, d_model)
        self.pos_emb    = nn.Embedding(seq_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout,
            dim_feedforward=d_model * 4, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # ── Huella espectral estática del terreno ─────────────────────────
        self.static_proj = nn.Sequential(
            nn.Linear(n_indices, d_static), nn.ReLU(),
        )

        # ── Codificación de la fecha ──────────────────────────────────────
        self.date_enc = _DayOfYearEncoder(d_date)

        # ── Cabezas de salida ─────────────────────────────────────────────
        d_fused = d_model + d_static + d_date
        self.shared_mlp = nn.Sequential(
            nn.Linear(d_fused, d_fused // 2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.mu_head       = nn.Linear(d_fused // 2, n_indices)
        self.logsigma_head = nn.Linear(d_fused // 2, n_indices)

    def forward(
        self,
        x_hist:   torch.Tensor,   # (B, T, 5)
        x_static: torch.Tensor,   # (B, 5)
        doy:      torch.Tensor,   # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x_hist.shape

        # Encoder temporal: proyectar + añadir embedding posicional → Transformer
        pos   = torch.arange(T, device=x_hist.device).unsqueeze(0).expand(B, -1)
        h     = self.input_proj(x_hist) + self.pos_emb(pos)   # (B, T, d_model)
        h     = self.transformer(h)                             # (B, T, d_model)
        ctx   = h[:, -1, :]                                    # token final → (B, d_model)

        s     = self.static_proj(x_static)   # (B, d_static)
        d     = self.date_enc(doy)           # (B, d_date)

        fused = torch.cat([ctx, s, d], dim=-1)                 # (B, d_fused)
        fused = self.shared_mlp(fused)                         # (B, d_fused//2)

        mu    = self.mu_head(fused)                            # (B, 5)
        sigma = F.softplus(self.logsigma_head(fused)) + 1e-4  # (B, 5) — siempre > 0

        return mu, sigma


def gaussian_nll(
    mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """
    Negative log-likelihood Gaussiana (por muestra y por índice).

    Minimizar esta pérdida es equivalente a ajustar μ y σ por máxima
    verosimilitud — el método que propuso el profe para los umbrales.
    """
    return (torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2).mean()
