"""
main.py - FastAPI Backend: Deteccion de Estres Hidrico en Aguacate
==================================================================
Endpoints:
    GET  /health                    Liveness check
    POST /analyze                   Diagnostico completo (GPS + foto opcional)
    GET  /parcels                   Lista de parcelas en el catalogo local
    GET  /parcels/{parcel_id}       Diagnostico de una parcela especifica
"""
from __future__ import annotations

import base64
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .config import Settings, get_settings
from .features import (
    CHANNEL_NAMES,
    WINDOW_SIZE,
    build_ndmi_mask,
    extract_last_window,
    extract_trend_windows,
    load_thresholds,
)
from .llm import generate_report
from .predictor import get_predictor
from .sentinel import LocalCatalog

try:
    import anthropic
except ModuleNotFoundError:
    anthropic = None

# -- App ---------------------------------------------------------------------
app = FastAPI(
    title="Water Stress Detector API",
    description="Deteccion de estres hidrico en aguacate (Jalisco) con Sentinel-2 + ML + Ollama",
    version="1.0.0",
)


@app.exception_handler(Exception)
def unhandled_exception_handler(_, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc) or exc.__class__.__name__},
    )

# -- Schemas de entrada -------------------------------------------------------

class AnalyzeRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90,   description="Latitud decimal (WGS84)")
    lon: float = Field(..., ge=-180, le=180, description="Longitud decimal (WGS84)")
    photo_b64: Optional[str] = Field(
        None,
        description="Foto del campo en base64 (JPEG/PNG). Activa analisis visual con GPT-4o.",
    )
    photo_mime: str = Field("image/jpeg", description="MIME type de la foto")
    skip_llm: bool = Field(False, description="Si True, devuelve solo la prediccion ML sin LLM.")

    @field_validator("photo_b64")
    @classmethod
    def validate_base64(cls, v):
        if v is None:
            return v
        try:
            base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError("photo_b64 no es base64 valido")
        return v


class ParcelRequest(BaseModel):
    parcel_id: str
    skip_llm:  bool = False


class Sam2MaskResponse(BaseModel):
    parcel_id: str
    width: int
    height: int
    classes: list[list[int]]
    source: str
    window_size: int
    latest_date: Optional[str] = None


# -- Dependencias -------------------------------------------------------------

def deps(settings: Settings = Depends(get_settings)):
    return settings


def get_catalog(s: Settings = Depends(get_settings)) -> LocalCatalog:
    return LocalCatalog(s.abs(s.parcelas_csv), s.abs(s.patches_dir))


def get_anthropic(s: Settings = Depends(get_settings)) -> Optional[Any]:
    if s.llm_provider.lower() == "ollama" or not s.anthropic_api_key:
        return None
    if anthropic is None:
        raise RuntimeError("The 'anthropic' package is required when LLM_PROVIDER=anthropic")
    return anthropic.Anthropic(api_key=s.anthropic_api_key)


def get_thresholds(s: Settings = Depends(get_settings)) -> tuple[float, float]:
    return load_thresholds(s.abs(s.norm_path))


# -- Logica compartida --------------------------------------------------------

def _run_pipeline(
    ts: np.ndarray,
    parcel_info: dict,
    request_skip_llm: bool,
    photo_b64: Optional[str],
    photo_mime: str,
    settings: Settings,
    claude_client: Optional[Any],
    t_mod: float,
    t_sev: float,
) -> dict:
    """Pipeline completo: features -> modelo -> LLM -> respuesta."""
    t_start = time.perf_counter()

    # 1. Extraer features e indices de la ventana mas reciente
    window_data = extract_last_window(ts, t_mod, t_sev)
    trend       = extract_trend_windows(ts, t_mod, t_sev, n_windows=4)

    # 2. Prediccion del modelo
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
            provider=settings.llm_provider,
            ollama_base_url=settings.ollama_base_url,
            ollama_model=settings.ollama_model,
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


# -- Endpoints ----------------------------------------------------------------

@app.get("/health", tags=["Sistema"])
def health():
    return {"status": "ok", "ts": datetime.now(timezone.utc).isoformat()}


@app.get("/parcels", tags=["Catalogo"])
def list_parcels(
    catalog: LocalCatalog = Depends(get_catalog),
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
):
    """Lista las primeras `limit` parcelas del catalogo local."""
    df = catalog._df.head(limit)
    return {
        "total": len(catalog._df),
        "parcels": df[["parcel_id", "latitude", "longitude", "state"]].to_dict("records"),
    }


@app.post("/analyze", tags=["Diagnostico"])
def analyze(
    req:     AnalyzeRequest,
    settings:  Settings            = Depends(get_settings),
    catalog:   LocalCatalog        = Depends(get_catalog),
    claude_c:  Optional[Any] = Depends(get_anthropic),
    thresholds: tuple              = Depends(get_thresholds),
):
    """
    Diagnostico de estres hidrico para una coordenada GPS.

    - Localiza la parcela Sentinel-2 mas cercana al punto del usuario.
    - Extrae las caracteristicas de la serie temporal.
    - Ejecuta el modelo E3 Stacking.
    - Genera un reporte agronomico con Ollama local (texto + foto opcional).
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


@app.post("/analyze/parcel", tags=["Diagnostico"])
def analyze_parcel(
    req:      ParcelRequest,
    settings:  Settings            = Depends(get_settings),
    catalog:   LocalCatalog        = Depends(get_catalog),
    claude_c:  Optional[Any] = Depends(get_anthropic),
    thresholds: tuple              = Depends(get_thresholds),
):
    """Diagnostico de una parcela especifica por ID (para el dashboard de cooperativas)."""
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


@app.get("/sam2/mask/{parcel_id}", response_model=Sam2MaskResponse, tags=["Segmentacion"])
def sam2_mask(
    parcel_id: str,
    settings: Settings = Depends(get_settings),
    catalog: LocalCatalog = Depends(get_catalog),
    thresholds: tuple = Depends(get_thresholds),
):
    """Mascara pixel a pixel basada en NDMI para la vista SAM2 preview."""
    row = catalog._df[catalog._df["parcel_id"] == parcel_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Parcela '{parcel_id}' no encontrada")

    npz_path = settings.abs(settings.patches_dir) / f"{parcel_id}.npz"
    if not npz_path.exists():
        raise HTTPException(status_code=404, detail=f"No se encontro el parche .npz para {parcel_id}")

    t_mod, t_sev = thresholds
    try:
        npz = np.load(npz_path, allow_pickle=True)
        data = npz["data"].astype(np.float32)
        if data.shape[0] < WINDOW_SIZE:
            raise ValueError(f"Serie muy corta: {data.shape[0]} < {WINDOW_SIZE}")

        mask = build_ndmi_mask(data, t_mod, t_sev)

        latest_date = None
        if "dates" in npz and len(npz["dates"]):
            latest_date = str(npz["dates"][-1])

        height, width = mask.shape
        return {
            "parcel_id": parcel_id,
            "width": int(width),
            "height": int(height),
            "classes": mask.tolist(),
            "source": "ndmi_window_preview",
            "window_size": WINDOW_SIZE,
            "latest_date": latest_date,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -- Frontend dashboard --------------------------------------------------------
_FRONTEND = Path(__file__).parent.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/ui", StaticFiles(directory=str(_FRONTEND), html=True), name="ui")
