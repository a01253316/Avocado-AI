"""
main.py — FastAPI Backend: Detección de Estrés Hídrico en Aguacate
==================================================================
Endpoints:
    GET  /health                    Liveness check
    POST /analyze                   Diagnóstico completo (GPS + foto opcional)
    GET  /parcels                   Lista de parcelas en el catálogo local
    GET  /parcels/{parcel_id}       Diagnóstico de una parcela específica
"""
from __future__ import annotations

import base64
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

import anthropic
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .config import Settings, get_settings
from .features import (
    CHANNEL_NAMES,
    extract_last_window,
    extract_trend_windows,
    load_thresholds,
)
from .llm import generate_report
from .predictor import get_predictor
from .sentinel import LocalCatalog

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Water Stress Detector API",
    description="Detección de estrés hídrico en aguacate (Jalisco) con Sentinel-2 + ML + Claude",
    version="1.0.0",
)

# ── Schemas de entrada ───────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90,   description="Latitud decimal (WGS84)")
    lon: float = Field(..., ge=-180, le=180, description="Longitud decimal (WGS84)")
    photo_b64: Optional[str] = Field(
        None,
        description="Foto del campo en base64 (JPEG/PNG). Activa análisis visual con GPT-4o.",
    )
    photo_mime: str = Field("image/jpeg", description="MIME type de la foto")
    skip_llm: bool = Field(False, description="Si True, devuelve solo la predicción ML sin Claude.")

    @field_validator("photo_b64")
    @classmethod
    def validate_base64(cls, v):
        if v is None:
            return v
        try:
            base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError("photo_b64 no es base64 válido")
        return v


class ParcelRequest(BaseModel):
    parcel_id: str
    skip_llm:  bool = False


# ── Dependencias ─────────────────────────────────────────────────────────────

def deps(settings: Settings = Depends(get_settings)):
    return settings


def get_catalog(s: Settings = Depends(get_settings)) -> LocalCatalog:
    return LocalCatalog(s.abs(s.parcelas_csv), s.abs(s.patches_dir))


def get_anthropic(s: Settings = Depends(get_settings)) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=s.anthropic_api_key)


def get_thresholds(s: Settings = Depends(get_settings)) -> tuple[float, float]:
    return load_thresholds(s.abs(s.norm_path))


# ── Lógica compartida ────────────────────────────────────────────────────────

def _run_pipeline(
    ts: np.ndarray,
    parcel_info: dict,
    request_skip_llm: bool,
    photo_b64: Optional[str],
    photo_mime: str,
    settings: Settings,
    claude_client: anthropic.Anthropic,
    t_mod: float,
    t_sev: float,
) -> dict:
    """Pipeline completo: features → modelo → LLM → respuesta."""
    t_start = time.perf_counter()

    # 1. Extraer features e índices de la ventana más reciente
    window_data = extract_last_window(ts, t_mod, t_sev)
    trend       = extract_trend_windows(ts, t_mod, t_sev, n_windows=4)

    # 2. Predicción del modelo
    predictor = get_predictor(
        settings.abs(settings.model_path).as_posix(),
        settings.abs(settings.scaler_path).as_posix(),
        settings.abs(settings.meta_path).as_posix(),
    )
    prediction = predictor.predict(window_data["features"])

    # 3. Reporte LLM (opcional)
    llm_report = None
    if not request_skip_llm:
        llm_report = generate_report(
            client=claude_client,
            model=settings.anthropic_model,
            prediction=prediction,
            indices=window_data["indices"],
            trend=trend,
            parcel_info=parcel_info,
            photo_b64=photo_b64,
            photo_mime=photo_mime,
        )

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

    return {
        "location": {
            "user_lat":   parcel_info.get("user_lat"),
            "user_lon":   parcel_info.get("user_lon"),
            "parcel_id":  parcel_info["parcel_id"],
            "parcel_lat": parcel_info["lat"],
            "parcel_lon": parcel_info["lon"],
            "dist_km":    parcel_info.get("dist_km"),
            "state":      parcel_info.get("state"),
            "data_source": parcel_info.get("data_source"),
            "note":        parcel_info.get("note"),
        },
        "stress": prediction,
        "indices": {
            ch: round(window_data["indices"][ch], 4) for ch in CHANNEL_NAMES
        },
        "trend": {
            "windows":         trend,
            "direction":       _trend_direction(trend),
            "worsening_alert": _is_worsening(trend),
        },
        "llm_report":   llm_report,
        "n_dates":       window_data["n_dates"],
        "processed_at":  datetime.now(timezone.utc).isoformat(),
        "elapsed_ms":    elapsed_ms,
    }


def _trend_direction(trend: list[dict]) -> str:
    if len(trend) < 2:
        return "sin_datos"
    slope = trend[-1]["ndmi_mean"] - trend[0]["ndmi_mean"]
    if slope < -0.02:
        return "descendente"
    if slope > 0.02:
        return "ascendente"
    return "estable"


def _is_worsening(trend: list[dict]) -> bool:
    if len(trend) < 2:
        return False
    return trend[-1]["ndmi_mean"] < trend[0]["ndmi_mean"] - 0.02


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Sistema"])
def health():
    return {"status": "ok", "ts": datetime.now(timezone.utc).isoformat()}


@app.get("/parcels", tags=["Catálogo"])
def list_parcels(
    catalog: LocalCatalog = Depends(get_catalog),
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
):
    """Lista las primeras `limit` parcelas del catálogo local."""
    df = catalog._df.head(limit)
    return {
        "total": len(catalog._df),
        "parcels": df[["parcel_id", "latitude", "longitude", "state"]].to_dict("records"),
    }


@app.post("/analyze", tags=["Diagnóstico"])
def analyze(
    req:     AnalyzeRequest,
    settings:  Settings            = Depends(get_settings),
    catalog:   LocalCatalog        = Depends(get_catalog),
    claude_c:  anthropic.Anthropic = Depends(get_anthropic),
    thresholds: tuple              = Depends(get_thresholds),
):
    """
    Diagnóstico de estrés hídrico para una coordenada GPS.

    - Localiza la parcela Sentinel-2 más cercana al punto del usuario.
    - Extrae las características de la serie temporal.
    - Ejecuta el modelo E3 Stacking.
    - Genera un reporte agronómico con Claude (texto + foto opcional).
    """
    try:
        parcel = catalog.find_nearest(req.lat, req.lon)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    parcel["user_lat"] = req.lat
    parcel["user_lon"] = req.lon
    t_mod, t_sev = thresholds

    try:
        return _run_pipeline(
            ts=parcel.pop("ts"),
            parcel_info=parcel,
            request_skip_llm=req.skip_llm,
            photo_b64=req.photo_b64,
            photo_mime=req.photo_mime,
            settings=settings,
            claude_client=claude_c,
            t_mod=t_mod,
            t_sev=t_sev,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/analyze/parcel", tags=["Diagnóstico"])
def analyze_parcel(
    req:      ParcelRequest,
    settings:  Settings            = Depends(get_settings),
    catalog:   LocalCatalog        = Depends(get_catalog),
    claude_c:  anthropic.Anthropic = Depends(get_anthropic),
    thresholds: tuple              = Depends(get_thresholds),
):
    """Diagnóstico de una parcela específica por ID (para el dashboard de cooperativas)."""
    row = catalog._df[catalog._df["parcel_id"] == req.parcel_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Parcela '{req.parcel_id}' no encontrada")

    r = row.iloc[0]
    try:
        parcel = catalog.find_nearest(float(r["latitude"]), float(r["longitude"]))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    t_mod, t_sev = thresholds
    try:
        return _run_pipeline(
            ts=parcel.pop("ts"),
            parcel_info=parcel,
            request_skip_llm=req.skip_llm,
            photo_b64=None,
            photo_mime="image/jpeg",
            settings=settings,
            claude_client=claude_c,
            t_mod=t_mod,
            t_sev=t_sev,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Frontend dashboard ────────────────────────────────────────────────────────
_FRONTEND = Path(__file__).parent.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/ui", StaticFiles(directory=str(_FRONTEND), html=True), name="ui")
