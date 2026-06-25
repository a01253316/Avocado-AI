"""
Integracion LLM con Ollama local y Claude opcional.
Genera reportes agronómicos en español a partir de la predicción + índices espectrales.
"""
from __future__ import annotations

import textwrap
import json
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import anthropic
except ModuleNotFoundError:
    anthropic = None

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
        # Shared image format; Ollama uses data only, Anthropic uses the source object.
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
    client: Optional[Any],
    model: str,
    prediction: dict,
    indices: dict,
    trend: list[dict],
    parcel_info: dict,
    photo_b64: Optional[str] = None,
    photo_mime: str = "image/jpeg",
    provider: str = "ollama",
    ollama_base_url: str = "http://localhost:11434",
    ollama_model: str = "openllama",
) -> dict:
    """
    Llama al proveedor LLM configurado y devuelve el reporte estructurado.
    Si la API falla, devuelve un reporte de fallback basado en reglas.
    """
    try:
        user_content = _build_user_content(
            prediction, indices, trend, parcel_info, photo_b64, photo_mime
        )

        if provider.lower() == "ollama":
            return _generate_ollama_report(
                model=ollama_model,
                base_url=ollama_base_url,
                user_content=user_content,
                photo_b64=photo_b64,
            )

        if client is None:
            raise RuntimeError("ANTHROPIC_API_KEY no esta configurada")

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


def _generate_ollama_report(
    model: str,
    base_url: str,
    user_content: list[dict],
    photo_b64: Optional[str],
) -> dict:
    """Genera el reporte usando Ollama local via /api/chat."""
    user_text = "\n\n".join(
        part["text"] for part in user_content if part.get("type") == "text"
    )
    message: dict[str, Any] = {"role": "user", "content": user_text}
    if photo_b64:
        message["images"] = [photo_b64]

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            message,
        ],
    }
    url = f"{base_url.rstrip('/')}/api/chat"
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"No se pudo conectar a Ollama en {url}: {exc.reason}") from exc

    full_text = data.get("message", {}).get("content", "").strip()
    if not full_text:
        raise RuntimeError("Ollama devolvio una respuesta vacia")

    return {
        "full_text": full_text,
        "model_used": model,
        "usage": {
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
        },
        "fallback": False,
        "provider": "ollama",
    }


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


# -- Recomendacion de acciones a partir del pronostico GP (pestana Tendencias) --

_TREND_SYSTEM_PROMPT = textwrap.dedent("""
    Eres un asesor agronomico especialista en cultivos de aguacate (Hass) en Jalisco,
    Mexico. Interpretas el pronostico de un Gaussian Process (GP) ajustado sobre la
    serie historica de un indice espectral Sentinel-2 (NDVI/NDWI/NDMI/NDRE/EVI) de
    una parcela (Experimento D del proyecto).

    El GP modela la media y desviacion estandar esperadas para cada fecha con base
    en el historial propio de la parcela (y, si esta disponible, de parcelas con
    terreno similar). El z-score indica cuantas desviaciones estandar cae la ultima
    observacion respecto a lo que el GP esperaba para esa parcela.

    Cuando recibas estos datos, debes:
    1. Explicar en lenguaje claro que dice la tendencia y el pronostico.
    2. Dar entre 3 y 5 recomendaciones concretas y accionables segun el z-score.
    3. Indicar el nivel de urgencia: INMEDIATA / ESTA SEMANA / MONITOREO.
    4. Si hay foto del campo, comentar si lo que ves visualmente es consistente.
    5. Cerrar con un dato breve de contexto climatico o agronomico relevante.

    Usa un tono directo, calido y practico. Nunca inventes valores.
""").strip()


def _build_trend_content(
    trend_data: dict,
    parcel_id: str,
    photo_b64: Optional[str],
    photo_mime: str,
) -> list[dict]:
    index   = trend_data["index"]
    last    = trend_data["last_observation"]
    fc      = trend_data["forecast"]
    history = trend_data["history"]
    group   = trend_data.get("group")

    n_hist = len(history["values"])
    slope_txt = "sin datos"
    if n_hist >= 2:
        slope = history["values"][-1] - history["values"][0]
        slope_txt = "ascendente" if slope > 0.02 else "descendente" if slope < -0.02 else "estable"

    group_txt = (
        textwrap.dedent(f"""
            COMPARACION CON GRUPO DE TERRENO SIMILAR (grupo {group["group_id"]}, {group["n_members"]} parcelas):
              - Pronostico de grupo: {group["forecast"]["mean"]:.4f} +/- {group["forecast"]["std"]:.4f}
              - z-score respecto al grupo: {group["last_observation"]["z"]:.2f} ({group["last_observation"]["label"]})
        """).strip()
        if group else
        "[Sin agrupacion por terreno disponible para esta parcela - solo vista individual]"
    )

    text = textwrap.dedent(f"""
        PRONOSTICO POR GAUSSIAN PROCESS -- Parcela {parcel_id}

        Indice analizado: {index}
        Historial: {n_hist} observaciones, de {history["dates"][0]} a {history["dates"][-1]}
        Tendencia general del historial: {slope_txt}

        ULTIMA OBSERVACION:
          - Fecha: {last["date"]}
          - Valor: {last["value"]:.4f}
          - z-score (GP individual): {last["z"]:.2f}
          - Clasificacion: {last["label"]}

        PRONOSTICO A {fc["day"] - last["day"]:.0f} DIAS (GP individual):
          - Fecha: {fc["date"]}
          - {index} esperado: {fc["mean"]:.4f} +/- {fc["std"]:.4f} (1 sigma)

        {group_txt}

        {'[Se adjunta fotografia del campo tomada por el agricultor]' if photo_b64 else '[Sin fotografia adjunta]'}

        Por favor genera recomendaciones de manejo de riego para este agricultor.
    """).strip()

    content: list[dict] = [{"type": "text", "text": text}]
    if photo_b64:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": photo_mime, "data": photo_b64},
        })
    return content


def generate_trend_recommendation(
    client: Optional[Any],
    model: str,
    trend_data: dict,
    parcel_id: str,
    photo_b64: Optional[str] = None,
    photo_mime: str = "image/jpeg",
) -> dict:
    """Llama al LLM para traducir el pronostico GP en recomendaciones de manejo."""
    try:
        user_content = _build_trend_content(trend_data, parcel_id, photo_b64, photo_mime)

        if client is None or not hasattr(client, "messages"):
            raise RuntimeError("Cliente LLM no disponible")

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=_TREND_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        full_text = response.content[0].text
        return {
            "full_text":  full_text,
            "model_used": model,
            "usage":      {
                "input_tokens":  response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            "fallback":   False,
        }
    except Exception as exc:
        label = trend_data.get("last_observation", {}).get("label", "sin_estres")
        messages = {
            "sin_estres": "El pronostico del GP esta dentro de lo esperado. Mantenga el riego actual.",
            "moderado": "El GP detecta una desviacion moderada. Revise la frecuencia de riego en los proximos 5-7 dias.",
            "severo": "El GP detecta una desviacion severa. Revise el sistema de riego de inmediato.",
        }
        return {
            "full_text":  messages.get(label, "Sin datos suficientes para una recomendacion."),
            "model_used": "fallback_rules",
            "usage":      {},
            "fallback":   True,
            "error":      str(exc),
        }
