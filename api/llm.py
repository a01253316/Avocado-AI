"""
Integración con Anthropic Claude (multimodal).
Genera reportes agronómicos en español a partir de la predicción + índices espectrales.
"""
from __future__ import annotations

import textwrap
from typing import Optional

import anthropic

_SYSTEM_PROMPT = textwrap.dedent("""
    Eres un asesor agronómico especialista en cultivos de aguacate (Hass) en Jalisco,
    México. Interpretas datos satelitales Sentinel-2 para diagnosticar estrés hídrico.

    Cuando recibas un diagnóstico del sistema de detección satelital, debes:
    1. Explicar en lenguaje claro (no técnico) qué significa para el agricultor.
    2. Dar entre 3 y 5 recomendaciones concretas y accionables según la severidad.
    3. Indicar el nivel de urgencia: INMEDIATA / ESTA SEMANA / MONITOREO.
    4. Si hay foto del campo, comentar si lo que ves visualmente es consistente
       con los datos satelitales o si detectas algo adicional.
    5. Cerrar con un dato breve de contexto climático o agronómico relevante.

    Usa un tono directo, cálido y práctico. El agricultor trabaja en campo y necesita
    respuestas claras, no jerga científica.
""").strip()


def _build_user_content(
    prediction: dict,
    indices: dict,
    trend: list[dict],
    parcel_info: dict,
    photo_b64: Optional[str],
    photo_mime: str,
) -> list[dict]:
    """Construye el contenido del mensaje de usuario (texto + foto opcional)."""
    label_map = {0: "Sin estrés hídrico", 1: "Estrés moderado", 2: "Estrés severo"}
    trend_labels = [label_map.get(t["label"], "?") for t in trend]
    ndmi_trend   = [round(t["ndmi_mean"], 4) for t in trend]

    text = textwrap.dedent(f"""
        REPORTE DE DIAGNÓSTICO SATELITAL — {parcel_info.get("state", "Jalisco")}
        ========================================================================

        Ubicación del agricultor:
          • Coordenadas: {parcel_info.get("user_lat", "N/A"):.5f}, {parcel_info.get("user_lon", "N/A"):.5f}
          • Parcela de referencia más cercana: {parcel_info.get("parcel_id", "?")} ({parcel_info.get("dist_km", "?")} km)

        RESULTADO DEL MODELO DE ESTRÉS HÍDRICO:
          • Diagnóstico: {prediction["emoji"]} {prediction["label"].upper()}
          • Confianza del modelo: {prediction["confidence"]*100:.1f}%
          • Probabilidades: sin_estrés={prediction["probabilities"]["Sin estrés"]*100:.1f}% | moderado={prediction["probabilities"]["Moderado"]*100:.1f}% | severo={prediction["probabilities"]["Severo"]*100:.1f}%

        ÍNDICES ESPECTRALES ACTUALES (promedio últimas 24 imágenes):
          • NDMI (humedad hoja/dosel): {indices.get("NDMI", 0):.4f}  ← el más importante
          • NDVI (vigor vegetal):      {indices.get("NDVI", 0):.4f}
          • NDWI (agua vegetación):    {indices.get("NDWI", 0):.4f}
          • NDRE (estrés temprano):    {indices.get("NDRE", 0):.4f}
          • EVI  (zonas densas):       {indices.get("EVI", 0):.4f}

        TENDENCIA DE LAS ÚLTIMAS {len(trend)} VENTANAS TEMPORALES (aprox. 3 meses):
          • Estados: {" → ".join(trend_labels)}
          • NDMI medio: {" → ".join(str(v) for v in ndmi_trend)}
          {"⚠️  TENDENCIA DECRECIENTE — el estrés está empeorando." if len(ndmi_trend) >= 2 and ndmi_trend[-1] < ndmi_trend[0] else "✅ TENDENCIA ESTABLE o MEJORANDO."}

        {'[Se adjunta fotografía del campo tomada por el agricultor]' if photo_b64 else '[Sin fotografía adjunta]'}

        Por favor genera el reporte agronómico completo para este agricultor.
    """).strip()

    content: list[dict] = [{"type": "text", "text": text}]

    if photo_b64:
        # Anthropic vision format: type=image, source.type=base64
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": photo_mime,
                "data": photo_b64,
            },
        })

    return content


def generate_report(
    client: anthropic.Anthropic,
    model: str,
    prediction: dict,
    indices: dict,
    trend: list[dict],
    parcel_info: dict,
    photo_b64: Optional[str] = None,
    photo_mime: str = "image/jpeg",
) -> dict:
    """
    Llama a Claude y devuelve el reporte estructurado.
    Si la API falla, devuelve un reporte de fallback basado en reglas.
    """
    try:
        user_content = _build_user_content(
            prediction, indices, trend, parcel_info, photo_b64, photo_mime
        )

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )

        full_text = response.content[0].text
        usage = {
            "input_tokens":  response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return {
            "full_text":  full_text,
            "model_used": model,
            "usage":      usage,
            "fallback":   False,
        }

    except Exception as exc:
        return _fallback_report(prediction, str(exc))


def _fallback_report(prediction: dict, error: str) -> dict:
    """Reporte basado en reglas cuando la API LLM no está disponible."""
    cls = prediction["class"]
    messages = {
        0: (
            "✅ Su parcela muestra condiciones hídricas normales. "
            "Mantenga el programa de riego actual y continúe el monitoreo regular."
        ),
        1: (
            "⚠️ Se detecta estrés hídrico moderado en su parcela. "
            "Recomendamos aumentar la frecuencia de riego en los próximos 5-7 días "
            "y revisar el estado de las hojas en horas de mayor temperatura."
        ),
        2: (
            "🚨 Se detecta estrés hídrico severo. Acción inmediata requerida: "
            "aplique riego de recuperación (40-60 mm) en las próximas 24-48 horas. "
            "Inspeccione el sistema de riego y contacte a su asesor agronómico."
        ),
    }
    return {
        "full_text":  messages[cls],
        "model_used": "fallback_rules",
        "usage":      {},
        "fallback":   True,
        "error":      error,
    }
