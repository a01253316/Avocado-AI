# Reporte Técnico — AguaVerde
## Sistema de Detección de Estrés Hídrico en Aguacate
**Equipo 16 · Proyecto Integrador · Jalisco, México**

---

## 1. Objetivo General

Desarrollar un sistema end-to-end que permita a una **cooperativa aguacatera** de Jalisco detectar de forma automática el nivel de estrés hídrico en sus parcelas usando imágenes satelitales Sentinel-2, un modelo de machine learning de ensamble, y un reporte agronómico generado por un LLM multimodal (Claude de Anthropic).

El sistema consta de tres partes principales:

| Parte             | Qué hace                                               |
|-------------------|--------------------------------------------------------|
| Pipeline de datos | Descarga y procesa imágenes Sentinel-2 por parcela     |
| Backend (API)     | Predice el nivel de estrés y genera reporte con Claude |
| Dashboard (UI)    | Visualiza el mapa de parcelas e interactúa con la API  |

---

## 2. Contexto del Problema

Las plantas de aguacate son muy sensibles al déficit hídrico. El estrés por falta de agua reduce el rendimiento, afecta la calidad del fruto y puede dañar de forma permanente la copa del árbol. Detectarlo temprano — antes de que sea visible a ojo desnudo — es clave para que el agricultor pueda actuar.

Las imágenes Sentinel-2 del programa Copernicus (ESA) ofrecen cobertura gratuita cada ~5 días con resolución de 10-20 m, lo que permite monitorear el estado hídrico de la vegetación a escala de parcela sin necesidad de sensores en campo.

---

## 3. Datos: Sentinel-2 y Parcelas

### 3.1 ¿Qué es Sentinel-2?

Sentinel-2 es un satélite europeo que captura imágenes multiespectrales de la superficie terrestre. Usamos las imágenes del nivel de procesamiento **L2A** (reflectancia en superficie, ya corregidas atmosféricamente), descargadas a través de **Copernicus Data Space Ecosystem (CDSE)**.

De cada imagen usamos **5 bandas** combinadas en índices espectrales:

| Índice | Bandas Sentinel-2 | Qué mide                                    |
|--------|-------------------|---------------------------------------------|
| NDVI   | B08, B04          | Vigor vegetal general                       |
| NDWI   | B03, B08          | Contenido de agua en la vegetación          |
| **NDMI** | **B08, B11**    | **Humedad en hoja/dosel ★ — índice clave**  |
| NDRE   | B08, B05          | Estrés temprano por déficit de clorofila    |
| EVI    | B08, B04, B02     | Vigor en zonas de vegetación densa          |

El **NDMI** (Normalized Difference Moisture Index) es el indicador principal porque responde directamente al contenido de agua en el dosel del árbol, antes de que el estrés se manifieste visualmente.

### 3.2 Parcelas del catálogo

El dataset incluye **100 parcelas** georeferenciadas en Jalisco (tile Sentinel-2 `14QMF`), definidas originalmente en un archivo KML y convertidas a CSV:

```
parcel_id, latitude, longitude, altitude_m, state, buffer_m, sentinel2_tile
H1, 19.663, -103.487, 0.0, Jalisco, 250, 14QMF
H2, 19.659, -103.491, 0.0, Jalisco, 250, 14QMF
...
```

Cada parcela tiene un **buffer de 250 m** alrededor de su centroide. El sistema descarga el recorte (chip/patch) de Sentinel-2 correspondiente a ese buffer.

### 3.3 Estructura de un parche (.npz)

Cada parcela se almacena como un archivo NumPy comprimido con shape:

```
(T=277, C=5, H=50, W=53)
 │      │    │     └── ancho del chip en píxeles
 │      │    └──────── alto del chip en píxeles
 │      └───────────── canales = [NDVI, NDWI, NDMI, NDRE, EVI]
 └──────────────────── fechas de observación (277 imágenes ~2020–2026)
```

---

## 4. Pipeline de Features: de Imágenes a Números

El modelo no recibe las imágenes directamente — las imágenes tienen demasiadas dimensiones. En su lugar, **extrae 35 características estadísticas** por ventana temporal.

### 4.1 Promedio espacial

Primero se colapsan las dimensiones espaciales (H, W) promediando todos los píxeles del chip:

```
(T=277, C=5, H=50, W=53)  →  media espacial  →  (T=277, C=5)
```

Resultado: una **serie temporal de 5 índices espectrales** con 277 observaciones.

### 4.2 Ventana deslizante

Se divide la serie temporal en **ventanas de 24 fechas** con paso de 4 fechas:

```
Serie: |──────────────────────────── 277 fechas ────────────────────────────|
       [   ventana 1: fechas 0–23   ]
           [   ventana 2: fechas 4–27   ]
               [   ventana 3: fechas 8–31   ]
                           ...
                                         [ ventana N: fechas 253–276 ]
```

Cada ventana genera **~6 400 muestras de entrenamiento** (100 parcelas × ~64 ventanas por parcela).

### 4.3 Extracción de features por ventana

Para cada ventana `(24 fechas × 5 canales)`, se calculan **7 estadísticos por canal**:

| Estadístico | Descripción              |
|-------------|--------------------------|
| mean        | Promedio temporal        |
| std         | Desviación estándar      |
| min         | Valor mínimo             |
| max         | Valor máximo             |
| p25         | Percentil 25             |
| p75         | Percentil 75             |
| trend       | Pendiente de regresión lineal (polyfit de grado 1) |

**7 estadísticos × 5 canales = 35 features** por ventana.

### 4.4 Etiquetado automático (NDMI)

Las etiquetas de estrés se generan automáticamente usando umbrales sobre el **NDMI promedio** de cada ventana:

```
NDMI normalizado > 0.2493  →  Clase 0: Sin estrés  🟢
NDMI entre 0.0571 y 0.2493 →  Clase 1: Moderado    🟡
NDMI < 0.0571              →  Clase 2: Severo       🔴
```

Los umbrales equivalen a NDMI en escala física de −0.10 (moderado) y −0.20 (severo), normalizados con MinMax al rango de los datos.

> ⚠️ **Importante**: no se usaron etiquetas manuales de campo. Las clases se derivaron directamente de los umbrales fisiológicos del NDMI, lo que hace que el dataset sea completamente autosuficiente con las imágenes satelitales.

---

## 5. Modelo: E3 Stacking

### 5.1 ¿Qué es un modelo de Stacking?

El stacking (apilamiento) es una técnica de ensamble donde varios modelos "base" hacen predicciones, y un modelo "meta" aprende a combinarlas para producir la predicción final.

```
                  ┌─────────────────────────────┐
  35 features ──► │  Random Forest              │ ─► prob. RF
                  ├─────────────────────────────┤
  35 features ──► │  XGBoost                    │ ─► prob. XGB  ──► Logistic ──► clase final
                  ├─────────────────────────────┤              Regression
  35 features ──► │  SVM (kernel RBF)           │ ─► prob. SVM
                  └─────────────────────────────┘
                    BASE LEARNERS (nivel 1)          META-LEARNER (nivel 2)
```

### 5.2 Base learners

| Modelo          | Fortaleza principal                                |
|-----------------|----------------------------------------------------|
| Random Forest   | Robusto a outliers, captura interacciones no lineales |
| XGBoost         | Gradient boosting — muy preciso en datos tabulares |
| SVM (RBF)       | Eficiente con datos de alta dimensión, buen margen |

Cada modelo genera un **vector de probabilidades** para las 3 clases. Estos 9 valores (3 modelos × 3 probabilidades) son la entrada del meta-learner.

### 5.3 Meta-learner

Una **Regresión Logística** aprende el peso óptimo de cada base learner para cada clase, produciendo la predicción final.

### 5.4 Validación sin fuga de datos

Se usó **GroupShuffleSplit por `parcel_id`** para que ninguna ventana de una parcela del conjunto de entrenamiento aparezca en el conjunto de validación. Esto evita el "data leakage" que ocurriría si ventanas del mismo campo estuvieran en train y test simultáneamente.

```
Split: 80% train / 20% test (a nivel de parcela, no de muestra)
```

### 5.5 Resultados

| Métrica                | Valor  |
|------------------------|--------|
| **F1-macro (test)**    | **0.8868** |
| Accuracy (test)        | ~0.89  |

El modelo supera los resultados de cada base learner individualmente, confirmando el beneficio del ensamble.

### 5.6 Artefactos generados

```
models/
├── ensemble_stacking.joblib   # Pipeline completo (3 base learners + meta-learner)
├── ensemble_scaler.joblib     # MinMaxScaler ajustado en train
└── ensemble_meta.json         # Umbrales NDMI normalizados + métricas de entrenamiento
```

---

## 6. Backend: FastAPI

El backend expone los modelos como una API REST construida con **FastAPI**. Está diseñado para ser el puente entre los datos satelitales, el modelo ML y el LLM.

### 6.1 Endpoints

| Método | Ruta               | Descripción                                         |
|--------|--------------------|-----------------------------------------------------|
| GET    | `/health`          | Verificación de que el servidor está en línea       |
| GET    | `/parcels`         | Lista las parcelas disponibles (máx. 200)           |
| POST   | `/analyze`         | Diagnóstico por coordenadas GPS + foto opcional     |
| POST   | `/analyze/parcel`  | Diagnóstico por ID de parcela directamente          |
| GET    | `/ui`              | Dashboard web (servido como archivos estáticos)     |

### 6.2 Flujo de una petición `/analyze`

```
Usuario envía: { lat, lon, photo_b64 (opcional), skip_llm }
       │
       ▼
1. LocalCatalog.find_nearest(lat, lon)
   └── Haversine sobre las 100 parcelas del CSV
   └── Carga el .npz de la parcela más cercana
       │
       ▼
2. patch_to_timeseries(npz_path)
   └── (T=277, C=5, H=50, W=53) → promedio espacial → (T=277, C=5)
       │
       ▼
3. extract_last_window(ts, t_mod, t_sev)
   └── Últimas 24 fechas → 35 features estadísticas
       │
       ▼
4. EnsemblePredictor.predict(features)
   └── MinMaxScaler → E3 Stacking → probabilidades → clase + confianza
       │
       ▼
5. extract_trend_windows(ts, ..., n_windows=4)
   └── 4 últimas ventanas → tendencia NDMI (ascendente / estable / descendente)
       │
       ▼
6. generate_report(client=Claude, prediction, indices, trend, photo_b64)
   └── Construye prompt con datos del diagnóstico + foto (si la hay)
   └── Claude devuelve reporte agronómico en español
       │
       ▼
Respuesta JSON: { location, stress, indices, trend, llm_report }
```

### 6.3 Configuración

Las credenciales y rutas se gestionan con **pydantic-settings** y un archivo `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...      # Clave de Claude
CDSE_USER=email@ejemplo.com       # Usuario CDSE para descargar Sentinel-2
CDSE_PASSWORD=...                 # Contraseña CDSE
```

> ⚠️ El archivo `.env` **nunca se sube al repositorio** (está en `.gitignore`). Cada integrante del equipo debe crear el suyo a partir de `.env.example`.

---

## 7. Integración LLM: Claude (Anthropic)

### 7.1 ¿Por qué un LLM?

El modelo ML produce un número de clase y una probabilidad — útiles técnicamente, pero difíciles de interpretar para un agricultor sin formación técnica. Claude convierte esos números en un **reporte agronómico en español**, accionable y claro.

### 7.2 Prompt enviado a Claude

El sistema construye automáticamente un mensaje que incluye:

- Coordenadas de la parcela analizada
- Diagnóstico del modelo (clase, confianza, probabilidades)
- Los 5 índices espectrales en su valor actual
- La tendencia de las últimas 4 ventanas (3 meses aproximados)
- La foto del campo en base64 (si el usuario la adjuntó)

### 7.3 Rol del sistema (system prompt)

```
Eres un asesor agronómico especialista en cultivos de aguacate (Hass) en Jalisco.
Interpretas datos satelitales Sentinel-2 para diagnosticar estrés hídrico.

Cuando recibas un diagnóstico debes:
1. Explicar en lenguaje claro qué significa para el agricultor.
2. Dar entre 3 y 5 recomendaciones concretas y accionables según la severidad.
3. Indicar el nivel de urgencia: INMEDIATA / ESTA SEMANA / MONITOREO.
4. Si hay foto, comentar si es consistente con los datos satelitales.
5. Cerrar con un dato breve de contexto climático o agronómico relevante.
```

### 7.4 Soporte multimodal

Si el agricultor adjunta una **foto del campo**, Claude la analiza visualmente junto con los datos satelitales, buscando señales de estrés visibles (hojas enrolladas, coloración, necrosis) y comparándolas con lo que muestran los índices.

### 7.5 Fallback sin LLM

Si Claude no está disponible (red, cuota agotada), el sistema devuelve un **reporte de reglas** basado solo en la clase predicha, sin llamar a la API. Esto garantiza que el sistema funcione siempre.

---

## 8. Frontend: Dashboard de la Cooperativa

El dashboard es una aplicación web de una sola página (SPA) construida con **Vanilla JS** (sin frameworks) + **Leaflet.js** para el mapa + **Chart.js** para las gráficas. Se sirve directamente desde FastAPI en `http://localhost:8000/ui`.

### 8.1 Componentes del dashboard

```
┌─────────────────────────────────────────────────────────────────┐
│ 🥑 AguaVerde  Coop. Aguacatera · Jalisco   [stats]  [⚡ Scan]  │
├─────────────────────────────┬───────────────────────────────────┤
│                             │ 📍 Diagnóstico │ ➕ Nueva ubic.  │
│      MAPA LEAFLET           │                                   │
│                             │ Parcela H1 · 0.12 km             │
│  ⚫⚫🟢🔴🟡🟢⚫⚫         │ 🟡 Estrés Moderado  94%          │
│       Jalisco, México       │ ─────────────────────────────    │
│                             │ NDMI: 0.1823  NDVI: 0.6102       │
│                             │ NDWI: 0.1045  NDRE: 0.3421       │
│  🟢 Sin estrés              │                                   │
│  🟡 Moderado                │ 📈 Tendencia NDMI [chart]        │
│  🔴 Severo                  │ ⬇ Tendencia descendente          │
│  ⚫ Sin analizar            │                                   │
│                             │ 🤖 Reporte Claude                │
│                             │ "Su parcela muestra estrés       │
│                             │ moderado. Recomendamos..."       │
└─────────────────────────────┴───────────────────────────────────┘
```

### 8.2 Funciones principales

**⚡ Escanear mapa**
Analiza las primeras 30 parcelas en secuencia usando solo el modelo ML (sin llamar a Claude), para colorear el mapa rápidamente. Muestra una barra de progreso durante el escaneo.

**Clic en marcador**
Al hacer clic en cualquier marcela del mapa, se lanza el análisis completo: modelo ML + reporte Claude. El panel lateral muestra en tiempo real los resultados, la gráfica de tendencia NDMI y el reporte agronómico. Si la parcela ya fue escaneada (solo ML), solicita el reporte Claude en ese momento.

**Nueva ubicación (pestaña)**
Permite ingresar cualquier par de coordenadas GPS manualmente, subir una foto del campo (opcional), y obtener el diagnóstico completo. El sistema localiza la parcela Sentinel-2 más cercana mediante Haversine y ejecuta el pipeline completo.

**Filtros por clase**
Botones en el header para mostrar solo parcelas sin estrés / moderado / severo.

---

## 9. Arquitectura General del Sistema

```
┌──────────────────────────────────────────────────────────────────┐
│                        USUARIO / AGRICULTOR                      │
│              (navegador web o app móvil futura)                  │
└──────────────────┬───────────────────────────────────────────────┘
                   │ HTTP (mismo origen)
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FastAPI  (api/main.py)                         │
│                                                                  │
│  GET /parcels ──► LocalCatalog → parcelas.csv                    │
│                                                                  │
│  POST /analyze                                                   │
│    │                                                             │
│    ├─ LocalCatalog.find_nearest()  ──► parcelas.csv              │
│    │         └── haversine nearest → carga .npz                  │
│    │                                                             │
│    ├─ extract_last_window()        ──► features (35,)            │
│    │                                                             │
│    ├─ EnsemblePredictor.predict()  ──► clase + confianza         │
│    │         └── MinMaxScaler → Stacking → LogReg                │
│    │                                                             │
│    └─ generate_report()            ──► Anthropic Claude API      │
│              └── texto agronómico + análisis foto                │
│                                                                  │
│  GET /ui ──► frontend/ (HTML/CSS/JS + Leaflet + Chart.js)        │
└──────────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
 data/raw/parcels/               api.anthropic.com
   parcelas.csv                  claude-opus-4-8
   patches/*.npz
         │
         ▼
 models/
   ensemble_stacking.joblib
   ensemble_scaler.joblib
   ensemble_meta.json
```

---

## 10. Flujo Completo de Extremo a Extremo

```
 [Imágenes Sentinel-2 crudas]
          │
          ▼  src/ingestion/sentinel2_downloader.py
 [GeoTIFFs por parcela y fecha]
          │
          ▼  src/processing/spectral_indices.py
 [Índices NDVI/NDWI/NDMI/NDRE/EVI por fecha]
          │
          ▼  src/processing/time_series_builder.py
 [Parches .npz: (T=277, C=5, H=50, W=53)]
          │
          ▼  notebooks/Avance5.equipo16.ipynb
 [Promedio espacial → (T=277, C=5)]
          │
          ▼  Ventana deslizante W=24, step=4
 [~6,400 muestras × 35 features + etiqueta NDMI]
          │
          ▼  E3 Stacking (RF + XGB + SVM → LogReg)
 [Modelo entrenado: F1-macro = 0.8868]
          │
          ▼  joblib.dump()
 [models/ensemble_stacking.joblib]
          │
          ▼  uvicorn api.main:app
 [API REST en http://localhost:8000]
          │
     ┌────┴────┐
     ▼         ▼
 [/docs]     [/ui]
 Swagger    Dashboard
            Leaflet
```

---

## 11. Cómo Ejecutar el Proyecto

### Requisitos previos
- Python 3.10+
- Credenciales CDSE (gratuitas en [dataspace.copernicus.eu](https://dataspace.copernicus.eu/))
- API key de Anthropic ([console.anthropic.com](https://console.anthropic.com/))

### Pasos

```bash
# 1. Clonar y preparar entorno
git clone <repo>
cd integrative-project
cp .env.example .env          # editar con tus credenciales

# 2. Instalar dependencias
make setup-all

# 3. (Si no tienes los .npz ya generados) Pipeline de datos
make extract-parcels
make download-sentinel2
make compute-indices
make build-dataset

# 4. (Si no tienes el modelo ya entrenado) Entrenamiento
make train-ensemble

# 5. Levantar el servidor
make dev

# 6. Abrir en el navegador
# Dashboard:  http://127.0.0.1:8000/ui
# Swagger:    http://127.0.0.1:8000/docs
```

> Para el demo de la cooperativa, los pasos 3 y 4 ya están completados — los `.npz` y el modelo están en el repositorio/servidor. Solo se necesita el paso 5.

---

## 12. Estructura de Archivos

```
integrative-project/
│
├── api/                          # Backend FastAPI
│   ├── main.py                   # Endpoints + mount del frontend
│   ├── config.py                 # Configuración vía .env
│   ├── sentinel.py               # LocalCatalog — haversine + carga .npz
│   ├── features.py               # Extracción de 35 features por ventana
│   ├── predictor.py              # EnsemblePredictor (lru_cache)
│   └── llm.py                    # Reporte agronómico con Claude
│
├── frontend/                     # Dashboard (sirve en GET /ui)
│   ├── index.html                # Estructura del dashboard
│   ├── style.css                 # Estilos (diseño cooperativa)
│   └── app.js                    # Lógica: mapa, análisis, gráficas
│
├── models/                       # Artefactos del modelo entrenado
│   ├── ensemble_stacking.joblib  # Modelo E3 Stacking
│   ├── ensemble_scaler.joblib    # Scaler MinMax
│   └── ensemble_meta.json        # Umbrales y métricas
│
├── data/
│   └── raw/parcels/
│       ├── parcelas.csv          # 100 parcelas georeferenciadas
│       ├── patches/              # .npz por parcela (T,C,H,W)
│       └── *.kml                 # Archivo original de parcelas
│
├── notebooks/
│   └── Avance5.equipo16.ipynb    # Pipeline ML completo: EDA → features → E3 Stacking
│
├── src/                          # Scripts de ingesta y procesamiento
│   ├── ingestion/
│   │   ├── kml_to_csv.py
│   │   └── sentinel2_downloader.py
│   └── processing/
│       ├── spectral_indices.py
│       └── time_series_builder.py
│
├── .env.example                  # Plantilla de variables de entorno
├── requirements.txt              # Dependencias ML / notebooks
├── requirements-api.txt          # Dependencias del backend
├── Makefile                      # Comandos del proyecto
└── README.md                     # Documentación principal
```

---

## 13. Decisiones de Diseño

### ¿Por qué NDMI como índice principal?

El NDMI usa las bandas B08 (NIR) y B11 (SWIR), donde la segunda es especialmente sensible al contenido de agua en la hoja. Es más específico que el NDVI (que mide biomasa en general) y más robusto al ruido atmosférico que el NDWI.

### ¿Por qué Stacking y no un solo modelo?

Cada base learner tiene fortalezas distintas: Random Forest maneja bien la no-linealidad, XGBoost es muy preciso en datos tabulares y el SVM con kernel RBF genera buenas fronteras de decisión en espacios de alta dimensión. El meta-learner aprende cuándo confiar en cada uno según el caso, superando a cualquiera individualmente.

### ¿Por qué ventanas de 24 fechas?

Con un paso de revisita de ~5 días, 24 imágenes equivalen a aproximadamente **4 meses** de observación. Esto captura la variación estacional del estrés mientras mantiene el vector de features manejable (35 números).

### ¿Por qué FastAPI + archivos estáticos en lugar de React/Vue?

Para un demo de cooperativa, la simplicidad de una SPA con Vanilla JS sirve directamente por FastAPI reduce las dependencias a cero (sin `npm`, sin build step). Un solo comando (`make dev`) levanta todo el sistema.

### ¿Por qué Claude y no otro LLM?

La integración multimodal de Claude permite adjuntar la foto del campo directamente en el mismo mensaje que los datos satelitales, sin infraestructura adicional de visión. Además, el sistema prompt personalizado genera reportes en español con el tono correcto para agricultores.

---

## 14. Posibles Mejoras Futuras

| Mejora | Descripción |
|--------|-------------|
| Datos en tiempo real | Integrar CDSE STAC API para descargar el tile Sentinel-2 más reciente al momento de la consulta |
| Alertas automáticas | Enviar WhatsApp/email al agricultor cuando se detecte estrés severo |
| App móvil | PWA que tome la foto con la cámara del celular directamente |
| Histórico por parcela | Guardar en base de datos el historial de diagnósticos por parcela |
| ViT for SITS | Sustituir el ensamble por el Vision Transformer entrenado en SITS para mayor precisión |
| Calibración con datos de campo | Validar los umbrales de NDMI con mediciones reales de humedad en suelo |

---

*Documento generado para uso interno del Equipo 16 — Proyecto Integrador 2025*
