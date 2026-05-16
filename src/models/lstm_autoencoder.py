"""
models/lstm_autoencoder.py
LSTM Autoencoder para detección no supervisada de estrés hídrico.
Registra métricas, parámetros y artefactos en MLflow.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import joblib
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

logger = logging.getLogger(__name__)


# ── Arquitectura ──────────────────────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_size, output_size, seq_len, num_layers, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        h = self.fc(z).unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, T, H)
        out, _ = self.lstm(h)
        return self.out(out)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, seq_len,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, latent_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_size, input_size, seq_len, num_layers, dropout)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def train(cfg_path: str = "configs/base.yaml",
          cred_path: str = "configs/credentials.yaml"):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    m   = cfg["model"]
    win_dir = Path(cfg["paths"]["windows"])

    # Datos
    X = np.load(win_dir / "X.npy").astype(np.float32)  # (N, T, F)
    N, T, F = X.shape
    logger.info(f"Dataset: {N} ventanas, {T} timesteps, {F} features")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.tensor(X)
    ds = TensorDataset(tensor, tensor)

    val_size  = max(1, int(0.15 * N))
    train_size = N - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    train_dl = DataLoader(train_ds, batch_size=m["batch_size"], shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=m["batch_size"])

    # Modelo
    model = LSTMAutoencoder(
        input_size=F,
        hidden_size=m["lstm_hidden"],
        latent_dim=m["latent_dim"],
        seq_len=T,
        num_layers=m["lstm_layers"],
        dropout=m["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=m["lr"])
    criterion = nn.MSELoss()

    # MLflow
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    model_dir = win_dir.parent.parent / "models" / "lstm_ae"
    model_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="lstm_autoencoder"):
        mlflow.log_params({
            "lstm_hidden":  m["lstm_hidden"],
            "lstm_layers":  m["lstm_layers"],
            "latent_dim":   m["latent_dim"],
            "dropout":      m["dropout"],
            "epochs":       m["epochs"],
            "batch_size":   m["batch_size"],
            "lr":           m["lr"],
            "window_size":  T,
            "n_features":   F,
            "n_windows":    N,
        })

        best_val_loss = float("inf")
        for epoch in range(m["epochs"]):
            # Train
            model.train()
            train_loss = 0.0
            for xb, _ in train_dl:
                xb = xb.to(device)
                optimizer.zero_grad()
                out, _ = model(xb)
                loss = criterion(out, xb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= train_size

            # Val
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, _ in val_dl:
                    xb = xb.to(device)
                    out, _ = model(xb)
                    val_loss += criterion(out, xb).item() * len(xb)
            val_loss /= val_size

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_dir / "best_model.pt")

            if (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch+1:3d} | train={train_loss:.4f} val={val_loss:.4f}")

        # Calcula scores de anomalía en todo el dataset
        model.eval()

        model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
        scores = []
        with torch.no_grad():
            for i in range(0, N, m["batch_size"]):
                xb = tensor[i:i + m["batch_size"]].to(device)
                out, _ = model(xb)
                mae = (out - xb).abs().mean(dim=[1, 2]).cpu().numpy()
                scores.append(mae)
        scores = np.concatenate(scores)

        threshold = float(np.percentile(scores, m["anomaly_percentile"]))
        mlflow.log_metric("anomaly_threshold", threshold)
        mlflow.log_metric("best_val_loss", best_val_loss)

        # Guarda threshold y config del modelo
        import json
        (model_dir / "threshold.json").write_text(json.dumps({
            "threshold": threshold,
            "percentile": m["anomaly_percentile"],
        }))
        (model_dir / "model_config.json").write_text(json.dumps({
            "input_size":   F,
            "hidden_size":  m["lstm_hidden"],
            "latent_dim":   m["latent_dim"],
            "seq_len":      T,
            "num_layers":   m["lstm_layers"],
            "dropout":      m["dropout"],
        }))

        # Guarda scores con metadatos
        meta_df = pd.read_parquet(win_dir / "meta.parquet")
        meta_df["anomaly_score"] = scores
        meta_df["is_anomaly"] = scores > threshold
        meta_df.to_parquet(model_dir / "scores.parquet", index=False)

        mlflow.log_artifacts(str(model_dir), artifact_path="model")
        logger.info(f"Entrenamiento completo. Umbral de anomalía: {threshold:.4f}")

    return model, threshold


if __name__ == "__main__":
    train()
