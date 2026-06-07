# 🥑 AguaVerde — Detección de Estrés Hídrico en Aguacate

Sistema de monitoreo satelital para cooperativas aguacateras en Jalisco, México.  
Combina imágenes Sentinel-2, modelos de ensemble y un reporte agronómico generado por **Claude** (Anthropic).

---

## 🎯 ¿Qué hace el sistema?

1. Descarga y procesa series temporales de imágenes Sentinel-2 por parcela
2. Extrae 5 índices espectrales (NDVI, NDWI, NDMI, NDRE, EVI) en ventanas deslizantes
3. Predice el nivel de estrés hídrico con un modelo **E3 Stacking** (F1-macro = **0.8868**)
4. Genera un reporte agronómico en español con **Claude** (multimodal — acepta foto del campo)
5. Visualiza todo en un **dashboard interactivo** con mapa Leaflet

---

## 🏆 Modelo Final — E3 Stacking

| Componente       | Detalle                                    |
|------------------|--------------------------------------------|
| Base learners    | Random Forest · XGBoost · SVM (RBF)       |
| Meta-learner     | Logistic Regression                        |
| Features         | 35 estadísticos × ventana de 24 fechas     |
| Split            | GroupShuffleSplit por `parcel_id`          |
| **F1-macro**     | **0.8868**                                 |

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
| Datos         | Sentinel-2 (ESA / CDSE), 100 parcelas en Jalisco       |
| Modelo        | scikit-learn · XGBoost · joblib                        |
| Backend       | FastAPI · Pydantic-Settings · Uvicorn                  |
| LLM           | Anthropic Claude (`claude-opus-4-8`) — multimodal      |
| Frontend      | Vanilla JS · Leaflet.js · Chart.js                     |
| Notebooks     | Jupyter · MLflow (tracking de experimentos)            |

---

## 🗂️ Estructura del Proyecto

```
integrative-project/
├── api/                          # Backend FastAPI
│   ├── main.py                   # Endpoints + montura del frontend
│   ├── config.py                 # Settings (pydantic-settings + .env)
│   ├── features.py               # Extracción de 35 features por ventana
│   ├── predictor.py              # E3 Stacking inference (lru_cache)
│   ├── sentinel.py               # LocalCatalog — haversine nearest parcel
│   └── llm.py                    # Reporte agronómico con Claude
│
├── frontend/                     # Dashboard web (servido en /ui)
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── models/                       # Artefactos serializados
│   ├── ensemble_stacking.joblib  # E3 Stacking (pipeline de 3 base learners)
│   ├── ensemble_scaler.joblib    # MinMaxScaler
│   └── ensemble_meta.json        # Umbrales NDMI normalizados
│
├── data/
│   └── raw/parcels/
│       ├── parcelas.csv          # 100 parcelas (lat, lon, tile, estado)
│       └── patches/              # Parches .npz — shape (277, 5, 50, 53)
│
├── notebooks/
│   ├── Avance5.equipo16.ipynb    # Notebook principal: features → modelos → evaluación
│   └── 01_eda_parcelas.ipynb     # EDA de parcelas y Sentinel-2
│
├── configs/                      # YAMLs de parcelas y Sentinel-2
├── src/                          # Scripts de ingesta y procesamiento
├── tests/
├── requirements.txt              # Dependencias del notebook / ML
├── requirements-api.txt          # Dependencias del backend
├── Makefile
├── .env.example                  # Plantilla de variables de entorno
└── README.md
```

---

## 🚀 Inicio Rápido

### 1. Variables de entorno

```bash
cp .env.example .env
# Edita .env con tus credenciales:
#   ANTHROPIC_API_KEY=sk-ant-...
#   CDSE_USER=tu_email@ejemplo.com
#   CDSE_PASSWORD=tu_contraseña
```

### 2. Instalar dependencias

```bash
pip install -r requirements-api.txt
```

### 3. Levantar el backend + dashboard

```bash
uvicorn api.main:app --reload
```

Abre **http://127.0.0.1:8000/ui** para el dashboard interactivo.  
La documentación automática de la API está en **http://127.0.0.1:8000/docs**.

---

## 🗺️ Dashboard

El dashboard permite a la cooperativa:

- **Visualizar las 100 parcelas** en un mapa interactivo de Jalisco con marcadores coloreados por nivel de estrés
- **⚡ Escanear mapa** — analiza las primeras 30 parcelas en batch (solo ML, sin LLM) para colorear el mapa en tiempo real
- **Clic en cualquier marcela** → diagnóstico completo: índices, gráfica de tendencia NDMI, barras de probabilidad y reporte Claude
- **Nueva ubicación** → ingresar lat/lon manualmente + foto del campo opcional para análisis multimodal
- **Filtros** por nivel de estrés (sin estrés / moderado / severo)

---

## 🌐 API Endpoints

| Método | Ruta                  | Descripción                                           |
|--------|-----------------------|-------------------------------------------------------|
| GET    | `/health`             | Liveness check                                        |
| GET    | `/parcels`            | Lista parcelas (`?limit=N`, máx 200)                  |
| POST   | `/analyze`            | Diagnóstico por coordenadas GPS + foto opcional       |
| POST   | `/analyze/parcel`     | Diagnóstico por `parcel_id` directo                   |
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
  "stress":   { "class": 1, "label": "Moderado", "emoji": "🟡", "confidence": 0.94 },
  "indices":  { "NDMI": 0.1823, "NDVI": 0.6102, "NDWI": 0.1045, "NDRE": 0.3421, "EVI": 0.4809 },
  "trend":    { "direction": "descendente", "worsening_alert": true, "windows": [...] },
  "llm_report": { "full_text": "...", "model_used": "claude-opus-4-8" }
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

Cada parche tiene shape `(T=277, C=5, H=50, W=53)`.  
Las features se extraen con una ventana deslizante de W=24 fechas (paso=4): **7 estadísticos × 5 canales = 35 features**.

---

## 📌 Referencias

- Garnot & Landrieu (2021) — *ViTs for SITS: Vision Transformers for Satellite Image Time Series*
- ESA — Sentinel-2 User Guide
- Anthropic — [Claude API](https://docs.anthropic.com)
