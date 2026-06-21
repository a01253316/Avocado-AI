# 🥑 AguaVerde — Detección de Estrés Hídrico en Aguacate

<p align="center">
  <a href="https://github.com/a01253316/Avocado-AI/actions/workflows/main.yml">
    <img src="https://github.com/a01253316/Avocado-AI/actions/workflows/main.yml/badge.svg?style=for-the-badge" alt="CI">
  </a>
</p>

Sistema de monitoreo satelital para cooperativas aguacateras en Jalisco, México.  
Combina imágenes Sentinel-2, modelos de ensemble, segmentación pixelada tipo SAM2 y un reporte agronómico generado con **Ollama local** por defecto, con **Claude** como proveedor opcional.

---

## 🎯 ¿Qué hace el sistema?

1. Descarga y procesa series temporales de imágenes Sentinel-2 por parcela
2. Extrae 5 índices espectrales (NDVI, NDWI, NDMI, NDRE, EVI) en ventanas deslizantes
3. Predice el nivel de estrés hídrico con un modelo **E3 Stacking** (F1-macro = **0.8868**)
4. Genera un reporte agronómico en español con Ollama/OpenLLaMA o Claude opcional
5. Resuelve coordenadas GPS contra el catálogo local de parcelas y calcula tendencia temporal NDMI/NDVI
6. Visualiza diagnósticos, tendencias y máscaras pixeladas en un **dashboard interactivo** con mapa Leaflet
7. Prepara fine-tuning SAM2 con pseudo-máscaras NDMI generadas desde los parches Sentinel-2

---

## 🏆 Modelo Final — E3 Stacking

| Componente       | Detalle                                    |
|------------------|--------------------------------------------|
| Base learners    | Random Forest · XGBoost · SVM (RBF)       |
| Meta-learner     | Logistic Regression                        |
| Features         | 35 estadísticos × ventana de 24 fechas     |
| Split            | GroupShuffleSplit por `parcel_id`          |
| **F1-macro**     | **0.8868**                                 |

Cada ventana temporal genera 35 features: `mean`, `std`, `min`, `max`, `p25`, `p75` y `trend` para cada uno de los 5 índices espectrales. El campo `trend` es la pendiente lineal de la ventana y permite capturar si el vigor/humedad de la parcela mejora o empeora antes del diagnóstico final.

**Clases de estrés:**

| Clase | Etiqueta       | NDMI normalizado |
|-------|----------------|------------------|
| 0     | Sin estrés     | > 0.2493         |
| 1     | Moderado       | 0.0571 – 0.2493  |
| 2     | Severo         | < 0.0571         |

---

## 📦 Stack

| Capa          | Tecnología                                              |
|---------------|---------------------------------------------------------|
| Datos         | Sentinel-2 (ESA / CDSE), parcelas en Jalisco           |
| Modelo        | scikit-learn · XGBoost · joblib                        |
| Segmentación  | SAM2.1 tiny opcional · pseudo-máscaras NDMI            |
| Backend       | FastAPI · Pydantic-Settings · Uvicorn                  |
| LLM           | Ollama (`openllama` por defecto) · Claude opcional     |
| Frontend      | Vanilla JS · Leaflet.js · Chart.js                     |
| Notebooks     | Jupyter · MLflow (tracking de experimentos)            |

---

## 🗂️ Estructura del Proyecto

```text
integrative-project/
|-- api/                          # Backend FastAPI
|   |-- main.py                   # Endpoints + montura del frontend
|   |-- config.py                 # Settings (pydantic-settings + .env)
|   |-- features.py               # Extracción de features y máscaras por parcela
|   |-- predictor.py              # E3 Stacking inference (lru_cache)
|   |-- sentinel.py               # LocalCatalog de parcelas
|   `-- llm.py                    # Reporte agronómico con Ollama/Claude
|
|-- frontend/                     # Dashboard web (servido en /ui)
|-- src/                          # Ingesta, procesamiento y modelos entrenables
|-- tests/                        # Tests unitarios
|-- configs/                      # Configuración YAML
|-- scripts/                      # Utilidades y scripts manuales
|-- notebooks/                    # Notebooks de exploración, avances y salidas asociadas
|
|-- docs/
|   |-- reports/                  # Reportes finales y documentación académica
|   |-- proceso_ejecucion_ollama.md
|   `-- sam2_plan.md
|
|-- reports/
|   `-- metrics/                  # Métricas exportadas para análisis y gráficas
|
|-- models/                       # Artefactos serializados y checkpoints externos
|-- data/                         # Datos crudos, intermedios y datasets generados
|
|-- requirements.txt              # Dependencias del notebook / ML
|-- requirements-api.txt          # Dependencias del backend
|-- requirements-sam2.txt         # Dependencias opcionales SAM2
|-- Makefile
|-- .env.example
`-- README.md
```

---

## 🚀 Inicio Rápido

### 1. Preparar entorno

```powershell
conda create -n avocado-ai python=3.12 -y
conda activate avocado-ai
python -m pip install -r requirements-api.txt
```

### 2. Variables de entorno

```powershell
copy .env.example .env
```

Configuración mínima con Ollama local:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=openllama
```

Instala Ollama y descarga el modelo:

```powershell
ollama pull openllama
```

También puedes usar un modelo ligero:

```powershell
ollama pull llama3.2:3b
```

Para usar Claude en lugar de Ollama:

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-opus-4-8
```

### 3. Levantar el backend + dashboard

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

O con Make:

```powershell
make dev
```

Abre **http://127.0.0.1:8000/ui** para el dashboard interactivo.  
La documentación automática de la API está en **http://127.0.0.1:8000/docs**.

---

## 🛰️ Pipeline de Datos y Modelos

Para generar los datos preprocesados desde cero:

```powershell
make extract-parcels
make download-sentinel2
make compute-indices
make build-dataset
```

Ese proceso crea los artefactos que usa el dashboard:

```text
data/datasets/
  patches/*.npz
  normalizer_stats.json

models/
  ensemble_stacking.joblib
  ensemble_scaler.joblib
  ensemble_meta.json
```

Después entrena el modelo principal:

```powershell
make train-ensemble
```

Flujo recomendado completo:

```powershell
conda activate avocado-ai
python -m pip install -r requirements-api.txt
ollama pull openllama
make extract-parcels
make download-sentinel2
make compute-indices
make build-dataset
make train-ensemble
make dev
```

---

## 📍 GPS y Tendencia Temporal

La rama `task/experiment-gp-terrain-trend` queda documentada como soporte de análisis por coordenadas GPS y tendencia temporal:

- `api/sentinel.py` carga el catálogo local de parcelas y resuelve una coordenada GPS a la parcela Sentinel-2 más cercana con distancia en kilómetros.
- `api/features.py` calcula ventanas recientes de tendencia con promedios NDMI/NDVI y estadísticos por ventana, incluyendo la pendiente lineal `trend`.
- `api/main.py` expone `direction` (`ascendente`, `estable`, `descendente`, `sin_datos`) y `worsening_alert` dentro de la respuesta `trend`.
- `api/llm.py` incorpora la secuencia de estados y NDMI medio en el prompt agronómico para explicar si el estrés está empeorando o estabilizándose.
- El dashboard muestra la dirección de tendencia, la gráfica temporal y el diagnóstico por parcela o por coordenadas ingresadas manualmente.

Campos principales en la respuesta:

```json
"trend": {
  "windows": [
    { "label": "Moderado", "ndmi_mean": 0.1823, "ndvi_mean": 0.6102 }
  ],
  "direction": "descendente",
  "worsening_alert": true
}
```

---

## 🗺️ Dashboard

El dashboard permite a la cooperativa:

- **Visualizar parcelas** en un mapa interactivo de Jalisco con marcadores coloreados por nivel de estrés
- **Escanear mapa** — analiza parcelas visibles en batch (solo ML, sin LLM) para colorear el mapa en tiempo real
- **Clic en cualquier parcela** → diagnóstico completo: índices, tendencia NDMI, barras de probabilidad y reporte agronómico
- **Nueva ubicación** → ingresar lat/lon manualmente + foto del campo opcional para análisis multimodal
- **Filtros** por nivel de estrés (sin estrés / moderado / severo)
- **Vista SAM2** → capa raster pixel por pixel sobre el mapa usando máscaras calibradas por el diagnóstico Sentinel

En la pestaña `SAM2`:

- `Analizar todo` genera máscaras pixeladas para las parcelas cargadas
- `Ajustar overlap` compacta más los bounds y resuelve píxeles solapados por mayoría entre máscaras calibradas
- Los filtros de `Píxeles visibles` muestran u ocultan clases del raster final
- Los filtros de `Diagnóstico Sentinel` controlan qué parcelas aportan máscaras al raster según su clase de análisis
- Los filtros de `Coordenadas visibles` controlan qué puntos del mapa se muestran, sin afectar el raster
- `Sin máscara` muestra u oculta parcelas pendientes sin raster generado
- Clic en un marcador o en `Parcelas activas` oculta o muestra la máscara de esa parcela
- Las máscaras se calculan con `data/datasets/patches/<parcel_id>.npz` y `data/datasets/normalizer_stats.json`
- Si los `.npz` incluyen `bounds_wgs84`, las máscaras se colocan con bounds reales de Sentinel-2
- Regenera el dataset con `make build-dataset` para obtener `bounds_wgs84`

---

## 🧪 Fine-tuning SAM2

La rama `sam2-finetuning` prepara SAM2 con nuestro dataset usando pseudo-máscaras NDMI. Los pesos base de SAM2 no se guardan en GitHub; deben descargarse por separado en `models/checkpoints/`.

Exporta imágenes y máscaras para SAM2:

```powershell
make prepare-sam2
```

Instala dependencias opcionales:

```powershell
python -m pip install -r requirements-sam2.txt
```

Descarga el checkpoint base SAM2.1 tiny:

```powershell
make download-sam2-checkpoint
```

Ejecuta fine-tuning:

```powershell
make train-sam2
```

El entrenamiento usa `SAM2_DEVICE=auto`: intenta CUDA si está disponible y, si no, usa CPU. Para forzar CPU:

```powershell
make train-sam2 SAM2_DEVICE=cpu
```

La salida queda en:

```text
models/sam2_avocado_finetuned.pt
```

Ese archivo queda fuera de GitHub.

---

## 🌐 API Endpoints

| Método | Ruta                  | Descripción                                           |
|--------|-----------------------|-------------------------------------------------------|
| GET    | `/health`             | Liveness check                                        |
| GET    | `/parcels`            | Lista parcelas (`?limit=N`, máx 200)                  |
| POST   | `/analyze`            | Diagnóstico por coordenadas GPS + foto opcional       |
| POST   | `/analyze/parcel`     | Diagnóstico por `parcel_id` directo                   |
| GET    | `/sam2/mask/{parcel_id}` | Devuelve máscara pixelada por parcela              |
| GET    | `/ui`                 | Dashboard web (Leaflet + Chart.js)                    |
| GET    | `/docs`               | Swagger UI autogenerado                               |

### Ejemplo — POST /analyze

```json
{
  "lat": 19.6630,
  "lon": -103.4870,
  "photo_b64": null,
  "skip_llm": false
}
```

Respuesta:

```json
{
  "location": { "parcel_id": "H1", "dist_km": 0.12, "state": "Jalisco" },
  "stress": { "class": 1, "label": "Moderado", "emoji": "🟡", "confidence": 0.94 },
  "indices": { "NDMI": 0.1823, "NDVI": 0.6102, "NDWI": 0.1045, "NDRE": 0.3421, "EVI": 0.4809 },
  "trend": { "direction": "descendente", "worsening_alert": true, "windows": [] },
  "llm_report": { "full_text": "...", "model_used": "openllama", "fallback": false }
}
```

---

## 📡 Índices Espectrales Sentinel-2

| Índice | Bandas        | Qué mide                          |
|--------|---------------|-----------------------------------|
| NDMI   | B08, B11      | Humedad en hoja/dosel ★ (clave)   |
| NDVI   | B08, B04      | Vigor vegetal general             |
| NDWI   | B03, B08      | Contenido de agua en vegetación   |
| NDRE   | B08, B05      | Estrés temprano — clorofila ★     |
| EVI    | B08, B04, B02 | Vegetación en zonas densas        |

Cada parche tiene shape aproximado `(T, C=5, H, W)`.  
Las features se extraen con una ventana deslizante de W=24 fechas: **7 estadísticos × 5 canales = 35 features**.

---

## 🧯 Problemas Comunes

### `No se encontro el parche .npz`

Falta el dataset preprocesado:

```text
data/datasets/patches/<parcel_id>.npz
```

Genera el dataset con el pipeline:

```powershell
make extract-parcels
make download-sentinel2
make compute-indices
make build-dataset
```

### `No se pudo conectar a Ollama`

Confirma que Ollama está instalado y que el modelo existe:

```powershell
ollama list
ollama pull openllama
```

### El dashboard carga mapa pero no analiza parcelas

El CSV de parcelas está disponible, pero faltan los `.npz` en `data/datasets/patches/`. Genera el dataset con `make build-dataset` después de preparar los índices.

---

## 📌 Referencias

- Garnot & Landrieu (2021) — *ViTs for SITS: Vision Transformers for Satellite Image Time Series*
- ESA — Sentinel-2 User Guide
- Ollama — https://ollama.com
- Anthropic — [Claude API](https://docs.anthropic.com)
- scikit-learn — [Model persistence](https://scikit-learn.org/stable/model_persistence.html)
