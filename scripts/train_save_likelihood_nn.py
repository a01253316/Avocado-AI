"""
train_save_likelihood_nn.py
===========================
Punto de entrada para entrenar el StressLikelihoodNet (Experimento E).

Uso:
    python scripts/train_save_likelihood_nn.py
    python scripts/train_save_likelihood_nn.py --epochs 80 --device mps
    make train-likelihood
"""
import sys
from pathlib import Path

# Asegurar que src/ esté en el path cuando se ejecuta desde la raíz del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.likelihood_nn.train_likelihood import main  # noqa: E402

if __name__ == "__main__":
    main()
