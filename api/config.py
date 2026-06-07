"""Configuración centralizada via variables de entorno / .env"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Anthropic Claude ────────────────────────────────────────
    anthropic_api_key: str
    anthropic_model: str = "claude-opus-4-8"

    # ── Rutas del proyecto ──────────────────────────────────────
    project_root: Path = Path(__file__).parent.parent
    model_path:   str  = "models/ensemble_stacking.joblib"
    scaler_path:  str  = "models/ensemble_scaler.joblib"
    meta_path:    str  = "models/ensemble_meta.json"
    patches_dir:  str  = "data/datasets/patches"
    parcelas_csv: str  = "data/raw/parcels/parcelas.csv"
    norm_path:    str  = "data/datasets/normalizer_stats.json"

    # ── CDSE (Copernicus) — para futura integración real-time ───
    cdse_user:     str = ""
    cdse_password: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"

    def abs(self, rel: str) -> Path:
        return self.project_root / rel


@lru_cache
def get_settings() -> Settings:
    return Settings()
