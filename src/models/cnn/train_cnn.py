"""
train_cnn.py
============
Entrena PixelCNN o PatchCNN para detección de estrés hídrico en SITS.

Features:
  - Tracking completo con MLflow (métricas, parámetros, modelo, artefactos)
  - Early stopping con paciencia configurable
  - Class weighting automático para el desbalance estresado/sano
  - Métricas: Loss, Accuracy, F1, AUC-ROC, Precision, Recall
  - Checkpointing del mejor modelo (según F1 en validación)
  - Reporte final con matriz de confusión en texto

Uso:
    python train_cnn.py --arch pixel
    python train_cnn.py --arch patch --epochs 100 --lr 0.0005
    python train_cnn.py --arch pixel --ndmi-threshold -0.15 --no-mlflow
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.cnn.pixel_classifier import build_model, count_parameters
from models.cnn.sits_dataset import load_datasets

logger = logging.getLogger("train_cnn")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # Dataset
    "dataset_dir":      "data/datasets/",
    "ndmi_threshold":   -0.1,
    # Modelo
    "arch":             "pixel",       # "pixel" | "patch"
    "n_channels":       5,
    "dropout":          0.3,
    "base_filters":     32,            # solo PatchCNN
    # Entrenamiento
    "epochs":           50,
    "batch_size":       64,
    "lr":               1e-3,
    "weight_decay":     1e-4,
    "patience":         10,            # early stopping
    # Salida
    "output_dir":       "models/",
    "use_mlflow":       True,
    "experiment_name":  "avocado-stress-cnn",
}


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
class MetricsTracker:
    """Acumula logits y labels de un epoch para calcular métricas al final."""

    def __init__(self):
        self._logits: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []
        self._losses: list[float]      = []

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss: float,
    ) -> None:
        self._logits.append(logits.detach().cpu().numpy().flatten())
        self._labels.append(labels.detach().cpu().numpy().flatten())
        self._losses.append(loss)

    def compute(self) -> dict[str, float]:
        all_logits = np.concatenate(self._logits)
        all_labels = np.concatenate(self._labels).astype(int)
        all_probs  = 1 / (1 + np.exp(-all_logits))   # sigmoid
        all_preds  = (all_probs >= 0.5).astype(int)

        metrics: dict[str, float] = {
            "loss":      float(np.mean(self._losses)),
            "accuracy":  float(accuracy_score(all_labels, all_preds)),
            "f1":        float(f1_score(all_labels, all_preds, zero_division=0)),
            "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
            "recall":    float(recall_score(all_labels, all_preds, zero_division=0)),
        }
        # AUC solo si hay ambas clases en el batch
        if len(np.unique(all_labels)) > 1:
            metrics["auc"] = float(roc_auc_score(all_labels, all_probs))
        else:
            metrics["auc"] = float("nan")

        return metrics

    def confusion(self) -> np.ndarray:
        all_logits = np.concatenate(self._logits)
        all_labels = np.concatenate(self._labels).astype(int)
        all_preds  = (1 / (1 + np.exp(-all_logits)) >= 0.5).astype(int)
        return confusion_matrix(all_labels, all_preds)


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    """Detiene el entrenamiento si val_f1 no mejora en `patience` epochs."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self._best      = -float("inf")
        self._counter   = 0
        self.should_stop = False

    def step(self, val_f1: float) -> bool:
        """Retorna True si el modelo mejoró (para hacer checkpoint)."""
        if val_f1 > self._best + self.min_delta:
            self._best    = val_f1
            self._counter = 0
            return True   # mejoró
        self._counter += 1
        if self._counter >= self.patience:
            self.should_stop = True
        return False      # no mejoró


# ---------------------------------------------------------------------------
# Inferir n_timesteps desde el dataset
# ---------------------------------------------------------------------------
def _infer_n_timesteps(datasets: dict, arch: str) -> int:
    """Obtiene T a partir del primer sample del dataset de entrenamiento."""
    ds    = datasets.get("train")
    if ds is None or len(ds) == 0:
        raise RuntimeError("Dataset de entrenamiento vacío.")
    x, _ = ds[0]
    # x shape: (T, C) para pixel | (T, C, H, W) para patch
    return x.shape[0]


# ---------------------------------------------------------------------------
# Class weights para desbalance
# ---------------------------------------------------------------------------
def _compute_pos_weight(dataset) -> torch.Tensor:
    """
    Calcula pos_weight = n_neg / n_pos para BCEWithLogitsLoss.
    Compensa el desbalance entre píxeles sanos y estresados.
    """
    labels = [float(dataset[i][1].mean()) for i in range(len(dataset))]
    n_pos  = sum(1 for l in labels if l >= 0.5)
    n_neg  = len(labels) - n_pos
    if n_pos == 0:
        return torch.tensor([1.0])
    weight = n_neg / n_pos
    logger.info("Class weight pos_weight=%.2f (neg=%d pos=%d)", weight, n_neg, n_pos)
    return torch.tensor([weight])


# ---------------------------------------------------------------------------
# Loop de un epoch
# ---------------------------------------------------------------------------
def _run_epoch(
    model     : nn.Module,
    loader    : DataLoader,
    criterion : nn.Module,
    optimizer : torch.optim.Optimizer | None,
    arch      : str,
    training  : bool,
) -> dict[str, float]:
    model.train(training)
    tracker = MetricsTracker()

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            # Forward
            logits = model(x)                  # (B,1) o (B,1,H,W)

            # Alinear shapes para la pérdida
            if arch == "patch":
                # logits: (B,1,H,W) → (B,H,W) | y: (B,H,W)
                logits_loss = logits.squeeze(1)
            else:
                # logits: (B,1) | y: (B,1)
                logits_loss = logits

            loss = criterion(logits_loss, y)

            if training and optimizer:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            tracker.update(logits_loss, y, loss.item())

    return tracker.compute()


# ---------------------------------------------------------------------------
# Entrenamiento principal
# ---------------------------------------------------------------------------
def train(cfg: dict) -> None:
    arch        = cfg["arch"]
    output_dir  = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── MLflow setup ───────────────────────────────────────
    run_ctx = None
    if cfg["use_mlflow"]:
        try:
            import mlflow
            mlflow.set_experiment(cfg["experiment_name"])
            run_ctx = mlflow.start_run()
            mlflow.log_params({k: v for k, v in cfg.items() if k != "use_mlflow"})
            logger.info("MLflow run iniciado: %s", mlflow.active_run().info.run_id)
        except Exception as e:
            logger.warning("MLflow no disponible: %s. Continuando sin tracking.", e)
            cfg["use_mlflow"] = False

    try:
        # ── Datasets ───────────────────────────────────────
        logger.info("Cargando datasets (modo=%s)...", arch)
        datasets = load_datasets(
            dataset_dir    = Path(cfg["dataset_dir"]),
            mode           = arch,
            ndmi_threshold = cfg["ndmi_threshold"],
        )

        n_timesteps = _infer_n_timesteps(datasets, arch)
        logger.info("n_timesteps=%d | device=%s", n_timesteps, DEVICE)

        pos_weight = _compute_pos_weight(datasets["train"]).to(DEVICE)

        loaders = {
            split: DataLoader(
                ds,
                batch_size = cfg["batch_size"],
                shuffle    = (split == "train"),
                num_workers= 0,
                pin_memory = torch.cuda.is_available(),
                # Solo tira el último batch si estamos en train Y el dataset es mayor al batch_size
                drop_last  = (split == "train" and len(ds) > cfg["batch_size"]), 
            )
            for split, ds in datasets.items()
        }

        # ── Modelo ─────────────────────────────────────────
        model_kwargs = {"dropout": cfg["dropout"]}
        if arch == "patch":
            model_kwargs["base_filters"] = cfg["base_filters"]

        model = build_model(
            architecture = arch,
            n_timesteps  = n_timesteps,
            n_channels   = cfg["n_channels"],
            **model_kwargs,
        ).to(DEVICE)

        n_params = count_parameters(model)
        logger.info("Modelo: %s | Parámetros: %s", model.name, f"{n_params:,}")

        if cfg["use_mlflow"]:
            import mlflow
            mlflow.log_param("n_timesteps",   n_timesteps)
            mlflow.log_param("n_parameters",  n_params)

        # ── Optimizador y pérdida ───────────────────────────
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr           = cfg["lr"],
            weight_decay = cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        early_stopping = EarlyStopping(patience=cfg["patience"])
        best_path      = output_dir / f"best_{arch}_cnn.pt"
        best_f1        = -1.0

        # ── Loop de entrenamiento ───────────────────────────
        logger.info("=" * 60)
        logger.info("Entrenando %s por %d epochs", model.name, cfg["epochs"])
        logger.info("=" * 60)

        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.time()

            train_metrics = _run_epoch(
                model, loaders["train"], criterion, optimizer, arch, training=True
            )
            scheduler.step()

            log_line = (
                f"Epoch {epoch:03d}/{cfg['epochs']} "
                f"| train loss={train_metrics['loss']:.4f} "
                f"f1={train_metrics['f1']:.3f} "
                f"auc={train_metrics['auc']:.3f}"
            )

            val_metrics: dict[str, float] = {}
            if "val" in loaders:
                val_metrics = _run_epoch(
                    model, loaders["val"], criterion, None, arch, training=False
                )
                log_line += (
                    f" | val loss={val_metrics['loss']:.4f} "
                    f"f1={val_metrics['f1']:.3f} "
                    f"auc={val_metrics['auc']:.3f}"
                )

            elapsed = time.time() - t0
            logger.info("%s (%.1fs)", log_line, elapsed)

            # MLflow logging
            if cfg["use_mlflow"]:
                import mlflow
                for k, v in train_metrics.items():
                    mlflow.log_metric(f"train_{k}", v, step=epoch)
                for k, v in val_metrics.items():
                    mlflow.log_metric(f"val_{k}", v, step=epoch)

            # Checkpoint del mejor modelo
            val_f1 = val_metrics.get("f1", train_metrics["f1"])
            improved = early_stopping.step(val_f1)
            if improved:
                best_f1 = val_f1
                torch.save(
                    {
                        "epoch":      epoch,
                        "model_state": model.state_dict(),
                        "val_f1":     best_f1,
                        "config":     cfg,
                        "n_timesteps": n_timesteps,
                    },
                    best_path,
                )
                logger.info("  ★ Nuevo mejor modelo (val_f1=%.3f) → %s", best_f1, best_path)

            if early_stopping.should_stop:
                logger.info("Early stopping en epoch %d (paciencia=%d)", epoch, cfg["patience"])
                break

        # ── Evaluación final en test ────────────────────────
        logger.info("=" * 60)
        logger.info("Cargando mejor modelo para evaluación en test...")
        checkpoint = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])

        if "test" in loaders:
            test_tracker = MetricsTracker()
            model.eval()
            with torch.no_grad():
                for x, y in loaders["test"]:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    logits = model(x)
                    if arch == "patch":
                        logits = logits.squeeze(1)
                    loss = criterion(logits, y)
                    test_tracker.update(logits, y, loss.item())

            test_metrics = test_tracker.compute()
            cm           = test_tracker.confusion()

            logger.info("Test — loss=%.4f f1=%.3f auc=%.3f acc=%.3f",
                        test_metrics["loss"], test_metrics["f1"],
                        test_metrics["auc"],  test_metrics["accuracy"])
            logger.info("Confusion Matrix:\n%s", cm)
            logger.info("  TN=%d FP=%d FN=%d TP=%d",
                        cm[0,0], cm[0,1], cm[1,0], cm[1,1])

            if cfg["use_mlflow"]:
                import mlflow
                for k, v in test_metrics.items():
                    mlflow.log_metric(f"test_{k}", v)
                mlflow.log_artifact(str(best_path))

        logger.info("Entrenamiento completo. Mejor val_f1=%.3f", best_f1)
        logger.info("Modelo guardado en: %s", best_path)
        logger.info("=" * 60)

    finally:
        if cfg["use_mlflow"] and run_ctx:
            import mlflow
            mlflow.end_run()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena CNN para detección de estrés hídrico en SITS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arch",      choices=["pixel", "patch"], default="pixel",
                        help="Arquitectura del modelo (default: pixel)")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/datasets/"),
                        help="Directorio del dataset .npz")
    parser.add_argument("--epochs",    type=int,   default=50)
    parser.add_argument("--batch-size",type=int,   default=64)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--dropout",   type=float, default=0.3)
    parser.add_argument("--patience",  type=int,   default=10,
                        help="Paciencia de early stopping")
    parser.add_argument("--ndmi-threshold", type=float, default=-0.1,
                        help="Umbral NDMI para pseudo-etiquetas de estrés (default: -0.1)")
    parser.add_argument("--base-filters", type=int, default=32,
                        help="Filtros base de PatchCNN (default: 32)")
    parser.add_argument("--output-dir", type=Path, default=Path("models/"),
                        help="Directorio para guardar checkpoints")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Desactiva el tracking con MLflow")
    parser.add_argument("--experiment", type=str, default="avocado-stress-cnn",
                        help="Nombre del experimento en MLflow")

    args = parser.parse_args()

    cfg = {
        "arch":             args.arch,
        "dataset_dir":      str(args.dataset_dir),
        "epochs":           args.epochs,
        "batch_size":       args.batch_size,
        "lr":               args.lr,
        "weight_decay":     1e-4,
        "dropout":          args.dropout,
        "patience":         args.patience,
        "ndmi_threshold":   args.ndmi_threshold,
        "n_channels":       5,
        "base_filters":     args.base_filters,
        "output_dir":       str(args.output_dir),
        "use_mlflow":       not args.no_mlflow,
        "experiment_name":  args.experiment,
    }

    train(cfg)


if __name__ == "__main__":
    main()
