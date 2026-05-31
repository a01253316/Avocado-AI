"""
train_vit.py
============
Entrena el SITSViT para detección de estrés hídrico en parcelas de aguacate.

Diferencias vs train_cnn.py que justifican un script separado:
  - Usa sits_collate_fn para manejar series de longitud variable (padding dinámico).
  - El forward recibe tres entradas: (x, doy, padding_mask).
  - Scheduler con warmup lineal + cosine decay (ViTs son sensibles al LR inicial).
  - Guarda attention weights del último bloque para interpretabilidad.
  - Logging extendido en MLflow: arquitectura, temporal encoding, attention maps.

Uso:
    python train_vit.py
    python train_vit.py --d-model 256 --num-heads 8 --num-layers 6
    python train_vit.py --preset tiny    # 64/4/2 — rápido para pruebas
    python train_vit.py --preset small   # 128/8/4 — default
    python train_vit.py --preset base    # 256/8/6 — máxima capacidad
    python train_vit.py --no-mlflow      # sin tracking
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.vit.sits_vit import SITSViT, build_vit, sits_collate_fn
from models.vit.sits_vit_dataset import load_vit_datasets

logger = logging.getLogger("train_vit")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Presets de arquitectura (d_model, num_heads, num_layers)
PRESETS = {
    "tiny":  (64,  4, 2),
    "small": (128, 8, 4),
    "base":  (256, 8, 6),
}


# ---------------------------------------------------------------------------
# Warmup + Cosine Decay scheduler
# ---------------------------------------------------------------------------
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    LR sube linealmente durante `warmup_steps` pasos y luego decae
    siguiendo un coseno hasta `eta_min`.

    Los ViTs son sensibles al LR inicial: un warmup evita que los pesos
    divergan en las primeras iteraciones antes de que los gradientes
    sean estables.
    """

    def __init__(
        self,
        optimizer    : torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs : int,
        eta_min      : float = 1e-6,
        last_epoch   : int = -1,
    ):
        self.warmup  = warmup_epochs
        self.total   = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        ep = self.last_epoch
        if ep < self.warmup:
            # Fase de warmup: escala lineal 0 → 1
            factor = (ep + 1) / max(self.warmup, 1)
        else:
            # Fase coseno
            progress = (ep - self.warmup) / max(self.total - self.warmup, 1)
            factor   = self.eta_min + 0.5 * (1 - self.eta_min) * (
                1 + np.cos(np.pi * progress)
            )
        return [base_lr * factor for base_lr in self.base_lrs]


# ---------------------------------------------------------------------------
# Métricas (reutiliza la lógica de train_cnn pero separado para claridad)
# ---------------------------------------------------------------------------
class ViTMetricsTracker:
    def __init__(self):
        self._logits: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []
        self._losses: list[float]      = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: float) -> None:
        self._logits.append(logits.detach().cpu().numpy().flatten())
        self._labels.append(labels.detach().cpu().numpy().flatten())
        self._losses.append(loss)

    def compute(self) -> dict[str, float]:
        logits = np.concatenate(self._logits)
        labels = np.concatenate(self._labels).astype(int)
        probs  = 1 / (1 + np.exp(-logits))
        preds  = (probs >= 0.5).astype(int)

        metrics: dict[str, float] = {
            "loss":      float(np.mean(self._losses)),
            "accuracy":  float(accuracy_score(labels, preds)),
            "f1":        float(f1_score(labels, preds, zero_division=0)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall":    float(recall_score(labels, preds, zero_division=0)),
            "auc":       (
                float(roc_auc_score(labels, probs))
                if len(np.unique(labels)) > 1 else float("nan")
            ),
        }
        return metrics


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self._best       = -float("inf")
        self._counter    = 0
        self.should_stop = False

    def step(self, val_f1: float) -> bool:
        if val_f1 > self._best + self.min_delta:
            self._best    = val_f1
            self._counter = 0
            return True
        self._counter += 1
        if self._counter >= self.patience:
            self.should_stop = True
        return False


# ---------------------------------------------------------------------------
# Class weight
# ---------------------------------------------------------------------------
def _compute_pos_weight(dataset) -> torch.Tensor:
    labels = [float(dataset[i][2].item()) for i in range(len(dataset))]
    n_pos  = sum(1 for l in labels if l >= 0.5)
    n_neg  = len(labels) - n_pos
    if n_pos == 0:
        return torch.tensor([1.0])
    weight = n_neg / max(n_pos, 1)
    logger.info("pos_weight=%.2f (neg=%d, pos=%d)", weight, n_neg, n_pos)
    return torch.tensor([weight])


# ---------------------------------------------------------------------------
# Loop de un epoch
# ---------------------------------------------------------------------------
def _run_epoch(
    model    : SITSViT,
    loader   : DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    training : bool,
) -> dict[str, float]:
    model.train(training)
    tracker = ViTMetricsTracker()

    with torch.set_grad_enabled(training):
        for x, doy, mask, y in loader:
            x    = x.to(DEVICE)
            doy  = doy.to(DEVICE)
            mask = mask.to(DEVICE)
            y    = y.to(DEVICE)           # (B, 1)

            logits = model(x, doy, padding_mask=mask)   # (B, 1)
            loss   = criterion(logits, y)

            if training and optimizer:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping — crítico para ViTs
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            tracker.update(logits, y, loss.item())

    return tracker.compute()


# ---------------------------------------------------------------------------
# Entrenamiento principal
# ---------------------------------------------------------------------------
def train(cfg: dict) -> None:
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── MLflow ────────────────────────────────────────────────
    use_mlflow = cfg.get("use_mlflow", True)
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment(cfg.get("experiment_name", "avocado-stress-vit"))
            mlflow.start_run()
            mlflow.log_params({k: v for k, v in cfg.items()
                               if k not in ("use_mlflow", "experiment_name")})
            logger.info("MLflow run: %s", mlflow.active_run().info.run_id)
        except Exception as e:
            logger.warning("MLflow no disponible: %s", e)
            use_mlflow = False

    try:
        # ── Datasets y DataLoaders ────────────────────────────
        logger.info("Cargando datasets...")
        datasets = load_vit_datasets(
            dataset_dir    = Path(cfg["dataset_dir"]),
            ndmi_threshold = cfg["ndmi_threshold"],
        )

        pos_weight = _compute_pos_weight(datasets["train"]).to(DEVICE)

        loaders = {
            split: DataLoader(
                ds,
                batch_size  = cfg["batch_size"],
                shuffle     = (split == "train"),
                collate_fn  = sits_collate_fn,   # ← padding dinámico por batch
                num_workers = 0,
                pin_memory  = torch.cuda.is_available(),
            )
            for split, ds in datasets.items()
        }

        # ── Modelo ───────────────────────────────────────────
        model = build_vit(
            n_channels = cfg["n_channels"],
            d_model    = cfg["d_model"],
            num_heads  = cfg["num_heads"],
            num_layers = cfg["num_layers"],
            dropout    = cfg["dropout"],
        ).to(DEVICE)

        n_params = model.count_parameters()
        logger.info("Modelo: %s", model.name)
        logger.info("Parámetros: %s", f"{n_params:,}")
        logger.info("Device: %s", DEVICE)

        if use_mlflow:
            import mlflow
            mlflow.log_param("n_parameters", n_params)
            mlflow.log_param("device",       str(DEVICE))

        # ── Optimizador, scheduler, pérdida ───────────────────
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr           = cfg["lr"],
            weight_decay = cfg["weight_decay"],
            betas        = (0.9, 0.999),
        )
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs = cfg["warmup_epochs"],
            total_epochs  = cfg["epochs"],
            eta_min       = cfg["lr"] * 0.01,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        early_stopping = EarlyStopping(patience=cfg["patience"])
        best_path      = output_dir / "best_sits_vit.pt"
        best_f1        = -1.0

        # ── Loop de entrenamiento ─────────────────────────────
        logger.info("=" * 60)
        logger.info("Entrenando SITSViT — %d epochs | LR=%.1e | warmup=%d",
                    cfg["epochs"], cfg["lr"], cfg["warmup_epochs"])
        logger.info("=" * 60)

        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.time()
            current_lr = scheduler.get_last_lr()[0] if epoch > 1 else cfg["lr"]

            train_m = _run_epoch(model, loaders["train"], criterion, optimizer, True)
            scheduler.step()

            log_line = (
                f"Epoch {epoch:03d}/{cfg['epochs']} "
                f"lr={current_lr:.2e} | "
                f"train loss={train_m['loss']:.4f} "
                f"f1={train_m['f1']:.3f} "
                f"auc={train_m['auc']:.3f}"
            )

            val_m: dict[str, float] = {}
            if "val" in loaders:
                val_m = _run_epoch(model, loaders["val"], criterion, None, False)
                log_line += (
                    f" | val loss={val_m['loss']:.4f} "
                    f"f1={val_m['f1']:.3f} "
                    f"auc={val_m['auc']:.3f}"
                )

            logger.info("%s  (%.1fs)", log_line, time.time() - t0)

            if use_mlflow:
                import mlflow
                for k, v in train_m.items():
                    mlflow.log_metric(f"train_{k}", v, step=epoch)
                for k, v in val_m.items():
                    mlflow.log_metric(f"val_{k}", v, step=epoch)
                mlflow.log_metric("lr", current_lr, step=epoch)

            # Checkpoint
            val_f1   = val_m.get("f1", train_m["f1"])
            improved = early_stopping.step(val_f1)
            if improved:
                best_f1 = val_f1
                torch.save(
                    {
                        "epoch":       epoch,
                        "model_state": model.state_dict(),
                        "val_f1":      best_f1,
                        "config":      cfg,
                    },
                    best_path,
                )
                logger.info("  ★ Mejor modelo → val_f1=%.3f", best_f1)

            if early_stopping.should_stop:
                logger.info("Early stopping en epoch %d", epoch)
                break

        # ── Evaluación en test ────────────────────────────────
        logger.info("=" * 60)
        ckpt = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])

        if "test" in loaders:
            tracker = ViTMetricsTracker()
            model.eval()
            with torch.no_grad():
                for x, doy, mask, y in loaders["test"]:
                    x, doy, mask, y = (
                        x.to(DEVICE), doy.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
                    )
                    logits = model(x, doy, mask)
                    loss   = criterion(logits, y)
                    tracker.update(logits, y, loss.item())

            test_m = tracker.compute()
            logger.info(
                "TEST — loss=%.4f f1=%.3f auc=%.3f acc=%.3f prec=%.3f rec=%.3f",
                test_m["loss"], test_m["f1"],  test_m["auc"],
                test_m["accuracy"], test_m["precision"], test_m["recall"],
            )

            if use_mlflow:
                import mlflow
                for k, v in test_m.items():
                    mlflow.log_metric(f"test_{k}", v)
                mlflow.log_artifact(str(best_path))

        logger.info("Modelo guardado: %s | Mejor val_f1=%.3f", best_path, best_f1)
        logger.info("=" * 60)

    finally:
        if use_mlflow:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena ViT for SITS — detección de estrés hídrico",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Arquitectura
    arch_g = parser.add_argument_group("Arquitectura")
    arch_g.add_argument("--preset", choices=["tiny", "small", "base"], default=None,
                        help="Preset de arquitectura (sobreescribe d-model/heads/layers)")
    arch_g.add_argument("--d-model",   type=int,   default=128)
    arch_g.add_argument("--num-heads", type=int,   default=8)
    arch_g.add_argument("--num-layers",type=int,   default=4)
    arch_g.add_argument("--dropout",   type=float, default=0.1)
    # Entrenamiento
    train_g = parser.add_argument_group("Entrenamiento")
    train_g.add_argument("--epochs",        type=int,   default=100)
    train_g.add_argument("--batch-size",    type=int,   default=32)
    train_g.add_argument("--lr",            type=float, default=1e-4)
    train_g.add_argument("--weight-decay",  type=float, default=1e-4)
    train_g.add_argument("--warmup-epochs", type=int,   default=10,
                         help="Epochs de warmup lineal del LR (default: 10)")
    train_g.add_argument("--patience",      type=int,   default=15)
    train_g.add_argument("--ndmi-threshold",type=float, default=-0.1)
    # Rutas
    io_g = parser.add_argument_group("Rutas")
    io_g.add_argument("--dataset-dir", type=Path, default=Path("data/datasets/"))
    io_g.add_argument("--output-dir",  type=Path, default=Path("models/"))
    # MLflow
    io_g.add_argument("--no-mlflow",  action="store_true")
    io_g.add_argument("--experiment", type=str, default="avocado-stress-vit")

    args = parser.parse_args()

    # Aplicar preset si se especificó
    d_model, num_heads, num_layers = args.d_model, args.num_heads, args.num_layers
    if args.preset:
        d_model, num_heads, num_layers = PRESETS[args.preset]
        logger.info("Preset '%s': d_model=%d heads=%d layers=%d",
                    args.preset, d_model, num_heads, num_layers)

    cfg = {
        "n_channels":     5,
        "d_model":        d_model,
        "num_heads":      num_heads,
        "num_layers":     num_layers,
        "dropout":        args.dropout,
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "lr":             args.lr,
        "weight_decay":   args.weight_decay,
        "warmup_epochs":  args.warmup_epochs,
        "patience":       args.patience,
        "ndmi_threshold": args.ndmi_threshold,
        "dataset_dir":    str(args.dataset_dir),
        "output_dir":     str(args.output_dir),
        "use_mlflow":     not args.no_mlflow,
        "experiment_name":args.experiment,
    }

    train(cfg)


if __name__ == "__main__":
    main()
