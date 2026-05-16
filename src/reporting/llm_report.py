"""
reporting/llm_report.py
Genera reportes explicativos por parcela usando OpenAI (GPT-4o) multimodal.
Envía visualizaciones (PNG) + estadísticas de la serie temporal al LLM.
"""
import base64
import io
import json
import logging
from pathlib import Path

from openai import OpenAI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ── Visualizaciones ───────────────────────────────────────────────────────────

def plot_time_series(ts_df: pd.DataFrame, scores_df: pd.DataFrame,
                     parcel: str, threshold: float) -> bytes:
    """
    Genera una figura con 3 subplots:
    1. NDVI y EVI2 en el tiempo
    2. NDMI y MSI (humedad/estrés)
    3. Score de anomalía con umbral
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"Parcela: {parcel} — Análisis de Estrés Hídrico", fontsize=13, y=0.98)

    dates = pd.to_datetime(ts_df["date"])

    # Subplot 1: Vegetación
    ax = axes[0]
    ax.plot(dates, ts_df["NDVI_mean"], color="#2d7a2d", label="NDVI", linewidth=1.5)
    ax.fill_between(dates, ts_df["NDVI_p10"], ts_df["NDVI_p90"],
                    alpha=0.15, color="#2d7a2d")
    if "EVI2_mean" in ts_df.columns:
        ax.plot(dates, ts_df["EVI2_mean"], color="#8bc34a", label="EVI2",
                linewidth=1.2, linestyle="--")
    ax.axhline(0.4, color="orange", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("Índice")
    ax.set_title("Índices de Vegetación", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)

    # Subplot 2: Humedad / Estrés
    ax = axes[1]
    ax.plot(dates, ts_df["NDMI_mean"], color="#1e88e5", label="NDMI", linewidth=1.5)
    if "MSI_mean" in ts_df.columns:
        ax.plot(dates, ts_df["MSI_mean"], color="#e53935", label="MSI (↑ = más estrés)",
                linewidth=1.2, linestyle="--")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Índice")
    ax.set_title("Índices de Humedad y Estrés Hídrico", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)

    # Subplot 3: Score de anomalía
    ax = axes[2]
    if len(scores_df) > 0:
        sdates = pd.to_datetime(scores_df["date_start"])
        ax.fill_between(sdates, 0, scores_df["anomaly_score"],
                        where=scores_df["is_anomaly"], alpha=0.4,
                        color="#e53935", label="Anomalía detectada")
        ax.plot(sdates, scores_df["anomaly_score"], color="#6a1a1a", linewidth=1.2)
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2,
                   label=f"Umbral p{95} = {threshold:.3f}")
    ax.set_ylabel("MAE de reconstrucción")
    ax.set_title("Score de Anomalía (LSTM Autoencoder)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ── Reporte LLM ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres un experto agrónomo con especialidad en estrés hídrico en cultivos de aguacate en el sur de Jalisco, México.
Recibirás una imagen con series temporales de índices espectrales (NDVI, NDMI, MSI, EVI2) y el score de anomalía de un LSTM Autoencoder entrenado de forma no supervisada, junto con estadísticos de la parcela.
Tu tarea es generar un reporte técnico conciso en Markdown con las siguientes secciones:
1. **Resumen ejecutivo** (2-3 oraciones)
2. **Estado de la vegetación** (tendencias NDVI/EVI2, comparación temporal)
3. **Análisis de estrés hídrico** (NDMI, MSI, períodos críticos identificados)
4. **Anomalías detectadas** (cuántas, en qué períodos, posible causa)
5. **Recomendaciones** (máximo 3, accionables para el productor)
Sé técnico pero claro. Menciona fechas específicas cuando identifiques patrones relevantes."""

def generate_report(parcel: str, ts_df: pd.DataFrame, scores_df: pd.DataFrame,
                    threshold: float, api_key: str, out_dir: Path) -> str:
    """Genera el reporte Markdown para una parcela."""

    # Estadísticos de resumen para el prompt de texto
    stats = {
        "n_observations": len(ts_df),
        "date_range": f"{ts_df['date'].min()} a {ts_df['date'].max()}",
        "NDVI_mean_overall": round(ts_df["NDVI_mean"].mean(), 3),
        "NDVI_min": round(ts_df["NDVI_mean"].min(), 3),
        "NDMI_mean_overall": round(ts_df["NDMI_mean"].mean(), 3),
        "anomaly_count": int(scores_df["is_anomaly"].sum()) if len(scores_df) > 0 else 0,
        "anomaly_pct": round(100 * scores_df["is_anomaly"].mean(), 1) if len(scores_df) > 0 else 0,
        "anomaly_threshold": round(threshold, 4),
    }
    if len(scores_df) > 0:
        anomaly_dates = scores_df[scores_df["is_anomaly"]]["date_start"].tolist()
        stats["anomaly_periods"] = anomaly_dates[:10]  # máx 10 fechas en prompt

    # Figura
    fig_bytes = plot_time_series(ts_df, scores_df, parcel, threshold)
    fig_b64   = base64.standard_b64encode(fig_bytes).decode("utf-8")

    # Llamada a OpenAI
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-4o",  # <-- Modelo optimizado para visión y texto
        max_tokens=1500,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT  # <-- OpenAI ubica el system prompt como un mensaje inicial
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Parcela: **{parcel}**\n\n"
                            f"Estadísticos: {json.dumps(stats, ensure_ascii=False, indent=2)}\n\n"
                            "Por favor genera el reporte de estrés hídrico para esta parcela."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{fig_b64}"  # <-- Formato de URI de datos para OpenAI
                        }
                    }
                ]
            }
        ]
    )

    # Extracción de la respuesta
    report_md = response.choices[0].message.content

    # Guarda figura y reporte
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{parcel}.png").write_bytes(fig_bytes)
    (out_dir / f"{parcel}.md").write_text(report_md, encoding="utf-8")

    logger.info(f"  ✓ {parcel}: reporte generado")
    return report_md


def run_reports(cfg_path: str = "configs/base.yaml",
                cred_path: str = "configs/credentials.yaml",
                parcel_ids: list[str] | None = None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg      = yaml.safe_load(Path(cfg_path).read_text())
    creds    = yaml.safe_load(Path(cred_path).read_text())
    
    # <-- Asegúrate de cambiar esto en tu archivo credentials.yaml
    api_key  = creds["openai"]["api_key"] 

    ts_dir    = Path(cfg["paths"]["time_series"])
    model_dir = Path("data/models/lstm_ae")
    out_dir   = Path(cfg["paths"]["reports"])

    scores_df  = pd.read_parquet(model_dir / "scores.parquet")
    threshold  = json.loads((model_dir / "threshold.json").read_text())["threshold"]

    ts_files = sorted(ts_dir.glob("*.parquet"))
    if parcel_ids:
        ts_files = [f for f in ts_files if f.stem in parcel_ids]

    for ts_path in ts_files:
        parcel = ts_path.stem
        ts_df  = pd.read_parquet(ts_path)
        p_scores = scores_df[scores_df["parcel"] == parcel].copy()
        try:
            generate_report(parcel, ts_df, p_scores, threshold, api_key, out_dir)
        except Exception as e:
            logger.error(f"  ✗ {parcel}: {e}")


if __name__ == "__main__":
    run_reports()