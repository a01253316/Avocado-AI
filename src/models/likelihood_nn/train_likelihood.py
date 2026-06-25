"""
train_likelihood.py — Entrenamiento del StressLikelihoodNet (Experimento E).

Entrena la red probabilística que mapea:
  [historial de índices + huella del terreno + fecha] → (μ, σ) por índice

La pérdida (Gaussian NLL) es equivalente a ajustar los parámetros de la
PDF por máxima verosimilitud — la técnica que propusieron los profesores para
fundamentar matemáticamente los umbrales de clasificación de estrés.

Uso:
    python -m src.models.likelihood_nn.train_likelihood
    python -m src.models.likelihood_nn.train_likelihood --epochs 80 --d-model 128
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .likelihood_dataset import load_split
from .stress_likelihood_net import StressLikelihoodNet, gaussian_nll

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("likelihood_nn")


def train(
    signals_dir:  str   = "data/datasets/signals",
    split_json:   str   = "data/datasets/split.json",
    output_dir:   str   = "models",
    window_size:  int   = 24,
    d_model:      int   = 64,
    n_heads:      int   = 4,
    n_layers:     int   = 2,
    dropout:      float = 0.1,
    epochs:       int   = 60,
    batch_size:   int   = 256,
    lr:           float = 1e-3,
    device_str:   str   = "auto",
) -> Path:
    # ── Dispositivo ──────────────────────────────────────────────────────────
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    logger.info("Dispositivo: %s", device)

    # ── Datos ────────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds = load_split(signals_dir, split_json, window_size)
    logger.info(
        "Muestras — train: %d | val: %d | test: %d",
        len(train_ds), len(val_ds), len(test_ds),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Modelo ───────────────────────────────────────────────────────────────
    model = StressLikelihoodNet(
        n_indices=5, seq_len=window_size,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parámetros entrenables: %s", f"{n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    # ── Loop de entrenamiento ────────────────────────────────────────────────
    best_val_nll = float("inf")
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)
    output_path  = output_dir_p / "stress_likelihood_net.pt"

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_losses: list[float] = []
        for x_hist, x_static, doy, y in train_loader:
            x_hist, x_static, doy, y = (
                x_hist.to(device), x_static.to(device), doy.to(device), y.to(device)
            )
            mu, sigma = model(x_hist, x_static, doy)
            loss = gaussian_nll(mu, sigma, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validación
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for x_hist, x_static, doy, y in val_loader:
                x_hist, x_static, doy, y = (
                    x_hist.to(device), x_static.to(device), doy.to(device), y.to(device)
                )
                mu, sigma = model(x_hist, x_static, doy)
                val_losses.append(gaussian_nll(mu, sigma, y).item())

        tr_nll = float(np.mean(train_losses))
        va_nll = float(np.mean(val_losses))
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            logger.info("Epoch %3d/%d | train NLL=%.4f | val NLL=%.4f", epoch, epochs, tr_nll, va_nll)

        if va_nll < best_val_nll:
            best_val_nll = va_nll
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "n_indices": 5,
                        "seq_len":   window_size,
                        "d_model":   d_model,
                        "n_heads":   n_heads,
                        "n_layers":  n_layers,
                        "dropout":   dropout,
                    },
                    "best_val_nll": best_val_nll,
                },
                output_path,
            )

    # ── Evaluación en test con el mejor checkpoint ───────────────────────────
    ckpt = torch.load(output_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    test_losses: list[float] = []
    with torch.no_grad():
        for x_hist, x_static, doy, y in test_loader:
            x_hist, x_static, doy, y = (
                x_hist.to(device), x_static.to(device), doy.to(device), y.to(device)
            )
            mu, sigma = model(x_hist, x_static, doy)
            test_losses.append(gaussian_nll(mu, sigma, y).item())

    test_nll = float(np.mean(test_losses))
    logger.info(
        "Entrenamiento completo | mejor val NLL=%.4f | test NLL=%.4f",
        best_val_nll, test_nll,
    )
    logger.info("Modelo guardado: %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena StressLikelihoodNet — Experimento E (Red Neuronal Probabilística)",
    )
    parser.add_argument("--signals-dir",  default="data/datasets/signals")
    parser.add_argument("--split-json",   default="data/datasets/split.json")
    parser.add_argument("--output-dir",   default="models")
    parser.add_argument("--window-size",  type=int,   default=24)
    parser.add_argument("--d-model",      type=int,   default=64)
    parser.add_argument("--n-heads",      type=int,   default=4)
    parser.add_argument("--n-layers",     type=int,   default=2)
    parser.add_argument("--dropout",      type=float, default=0.1)
    parser.add_argument("--epochs",       type=int,   default=60)
    parser.add_argument("--batch-size",   type=int,   default=256)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--device",       default="auto")
    args = parser.parse_args()

    train(
        signals_dir=args.signals_dir,
        split_json=args.split_json,
        output_dir=args.output_dir,
        window_size=args.window_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
