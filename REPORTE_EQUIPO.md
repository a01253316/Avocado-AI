# Reporte TÃ©cnico â€” AguaVerde
## Sistema de DetecciÃ³n de EstrÃ©s HÃ­drico en Aguacate
**Equipo 16 Â· Proyecto Integrador Â· Jalisco, MÃ©xico**

---

## 1. Objetivo General

Desarrollar un sistema end-to-end que permita a una **cooperativa aguacatera** de Jalisco detectar de forma automÃ¡tica el nivel de estrÃ©s hÃ­drico en sus parcelas usando imÃ¡genes satelitales Sentinel-2, un modelo de machine learning de ensamble, y un reporte agronÃ³mico generado por un LLM multimodal (Ollama/OpenLLaMA local).

El sistema consta de tres partes principales:

| Parte             | QuÃ© hace                                               |
|-------------------|--------------------------------------------------------|
| Pipeline de datos | Descarga y procesa imÃ¡genes Sentinel-2 por parcela     |
| Backend (API)     | Predice el nivel de estrÃ©s y genera reporte con Ollama/OpenLLaMA |
| Dashboard (UI)    | Visualiza el mapa de parcelas e interactÃºa con la API  |

---

## 2. Contexto del Problema

Las plantas de aguacate son muy sensibles al dÃ©ficit hÃ­drico. El estrÃ©s por falta de agua reduce el rendimiento, afecta la calidad del fruto y puede daÃ±ar de forma permanente la copa del Ã¡rbol. Detectarlo temprano â€” antes de que sea visible a ojo desnudo â€” es clave para que el agricultor pueda actuar.

Las imÃ¡genes Sentinel-2 del programa Copernicus (ESA) ofrecen cobertura gratuita cada ~5 dÃ­as con resoluciÃ³n de 10-20 m, lo que permite monitorear el estado hÃ­drico de la vegetaciÃ³n a escala de parcela sin necesidad de sensores en campo.

---

## 3. Datos: Sentinel-2 y Parcelas

### 3.1 Â¿QuÃ© es Sentinel-2?

Sentinel-2 es un satÃ©lite europeo que captura imÃ¡genes multiespectrales de la superficie terrestre. Usamos las imÃ¡genes del nivel de procesamiento **L2A** (reflectancia en superficie, ya corregidas atmosfÃ©ricamente), descargadas a travÃ©s de **Copernicus Data Space Ecosystem (CDSE)**.

De cada imagen usamos **5 bandas** combinadas en Ã­ndices espectrales:

| Ãndice | Bandas Sentinel-2 | QuÃ© mide                                    |
|--------|-------------------|---------------------------------------------|
| NDVI   | B08, B04          | Vigor vegetal general                       |
| NDWI   | B03, B08          | Contenido de agua en la vegetaciÃ³n          |
| **NDMI** | **B08, B11**    | **Humedad en hoja/dosel â˜… â€” Ã­ndice clave**  |
| NDRE   | B08, B05          | EstrÃ©s temprano por dÃ©ficit de clorofila    |
| EVI    | B08, B04, B02     | Vigor en zonas de vegetaciÃ³n densa          |

El **NDMI** (Normalized Difference Moisture Index) es el indicador principal porque responde directamente al contenido de agua en el dosel del Ã¡rbol, antes de que el estrÃ©s se manifieste visualmente.

### 3.2 Parcelas del catÃ¡logo

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
 â”‚      â”‚    â”‚     â””â”€â”€ ancho del chip en pÃ­xeles
 â”‚      â”‚    â””â”€â”€â”€â”€â”€â”€â”€â”€ alto del chip en pÃ­xeles
 â”‚      â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ canales = [NDVI, NDWI, NDMI, NDRE, EVI]
 â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ fechas de observaciÃ³n (277 imÃ¡genes ~2020â€“2026)
```

---

## 4. Pipeline de Features: de ImÃ¡genes a NÃºmeros

El modelo no recibe las imÃ¡genes directamente â€” las imÃ¡genes tienen demasiadas dimensiones. En su lugar, **extrae 35 caracterÃ­sticas estadÃ­sticas** por ventana temporal.

### 4.1 Promedio espacial

Primero se colapsan las dimensiones espaciales (H, W) promediando todos los pÃ­xeles del chip:

```
(T=277, C=5, H=50, W=53)  â†’  media espacial  â†’  (T=277, C=5)
```

Resultado: una **serie temporal de 5 Ã­ndices espectrales** con 277 observaciones.

### 4.2 Ventana deslizante

Se divide la serie temporal en **ventanas de 24 fechas** con paso de 4 fechas:

```
Serie: |â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ 277 fechas â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€|
       [   ventana 1: fechas 0â€“23   ]
           [   ventana 2: fechas 4â€“27   ]
               [   ventana 3: fechas 8â€“31   ]
                           ...
                                         [ ventana N: fechas 253â€“276 ]
```

Cada ventana genera **~6 400 muestras de entrenamiento** (100 parcelas Ã— ~64 ventanas por parcela).

### 4.3 ExtracciÃ³n de features por ventana

Para cada ventana `(24 fechas Ã— 5 canales)`, se calculan **7 estadÃ­sticos por canal**:

| EstadÃ­stico | DescripciÃ³n              |
|-------------|--------------------------|
| mean        | Promedio temporal        |
| std         | DesviaciÃ³n estÃ¡ndar      |
| min         | Valor mÃ­nimo             |
| max         | Valor mÃ¡ximo             |
| p25         | Percentil 25             |
| p75         | Percentil 75             |
| trend       | Pendiente de regresiÃ³n lineal (polyfit de grado 1) |

**7 estadÃ­sticos Ã— 5 canales = 35 features** por ventana.

### 4.4 Etiquetado automÃ¡tico (NDMI)

Las etiquetas de estrÃ©s se generan automÃ¡ticamente usando umbrales sobre el **NDMI promedio** de cada ventana:

```
NDMI normalizado > 0.2493  â†’  Clase 0: Sin estrÃ©s  ðŸŸ¢
NDMI entre 0.0571 y 0.2493 â†’  Clase 1: Moderado    ðŸŸ¡
NDMI < 0.0571              â†’  Clase 2: Severo       ðŸ”´
```

Los umbrales equivalen a NDMI en escala fÃ­sica de âˆ’0.10 (moderado) y âˆ’0.20 (severo), normalizados con MinMax al rango de los datos.

> âš ï¸ **Importante**: no se usaron etiquetas manuales de campo. Las clases se derivaron directamente de los umbrales fisiolÃ³gicos del NDMI, lo que hace que el dataset sea completamente autosuficiente con las imÃ¡genes satelitales.

---

## 5. Modelo: E3 Stacking

### 5.1 Â¿QuÃ© es un modelo de Stacking?

El stacking (apilamiento) es una tÃ©cnica de ensamble donde varios modelos "base" hacen predicciones, y un modelo "meta" aprende a combinarlas para producir la predicciÃ³n final.

```
                  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
  35 features â”€â”€â–º â”‚  Random Forest              â”‚ â”€â–º prob. RF
                  â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
  35 features â”€â”€â–º â”‚  XGBoost                    â”‚ â”€â–º prob. XGB  â”€â”€â–º Logistic â”€â”€â–º clase final
                  â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤              Regression
  35 features â”€â”€â–º â”‚  SVM (kernel RBF)           â”‚ â”€â–º prob. SVM
                  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                    BASE LEARNERS (nivel 1)          META-LEARNER (nivel 2)
```

### 5.2 Base learners

| Modelo          | Fortaleza principal                                |
|-----------------|----------------------------------------------------|
| Random Forest   | Robusto a outliers, captura interacciones no lineales |
| XGBoost         | Gradient boosting â€” muy preciso en datos tabulares |
| SVM (RBF)       | Eficiente con datos de alta dimensiÃ³n, buen margen |

Cada modelo genera un **vector de probabilidades** para las 3 clases. Estos 9 valores (3 modelos Ã— 3 probabilidades) son la entrada del meta-learner.

### 5.3 Meta-learner

Una **RegresiÃ³n LogÃ­stica** aprende el peso Ã³ptimo de cada base learner para cada clase, produciendo la predicciÃ³n final.

### 5.4 ValidaciÃ³n sin fuga de datos

Se usÃ³ **GroupShuffleSplit por `parcel_id`** para que ninguna ventana de una parcela del conjunto de entrenamiento aparezca en el conjunto de validaciÃ³n. Esto evita el "data leakage" que ocurrirÃ­a si ventanas del mismo campo estuvieran en train y test simultÃ¡neamente.

```
Split: 80% train / 20% test (a nivel de parcela, no de muestra)
```

### 5.5 Resultados

| MÃ©trica                | Valor  |
|------------------------|--------|
| **F1-macro (test)**    | **0.8868** |
| Accuracy (test)        | ~0.89  |

El modelo supera los resultados de cada base learner individualmente, confirmando el beneficio del ensamble.

### 5.6 Artefactos generados

```
models/
â”œâ”€â”€ ensemble_stacking.joblib   # Pipeline completo (3 base learners + meta-learner)
â”œâ”€â”€ ensemble_scaler.joblib     # MinMaxScaler ajustado en train
â””â”€â”€ ensemble_meta.json         # Umbrales NDMI normalizados + mÃ©tricas de entrenamiento
```

---

## 6. Backend: FastAPI

El backend expone los modelos como una API REST construida con **FastAPI**. EstÃ¡ diseÃ±ado para ser el puente entre los datos satelitales, el modelo ML y el LLM.

### 6.1 Endpoints

| MÃ©todo | Ruta               | DescripciÃ³n                                         |
|--------|--------------------|-----------------------------------------------------|
| GET    | `/health`          | VerificaciÃ³n de que el servidor estÃ¡ en lÃ­nea       |
| GET    | `/parcels`         | Lista las parcelas disponibles (mÃ¡x. 200)           |
| POST   | `/analyze`         | DiagnÃ³stico por coordenadas GPS + foto opcional     |
| POST   | `/analyze/parcel`  | DiagnÃ³stico por ID de parcela directamente          |
| GET    | `/ui`              | Dashboard web (servido como archivos estÃ¡ticos)     |

### 6.2 Flujo de una peticiÃ³n `/analyze`

```
Usuario envÃ­a: { lat, lon, photo_b64 (opcional), skip_llm }
       â”‚
       â–¼
1. LocalCatalog.find_nearest(lat, lon)
   â””â”€â”€ Haversine sobre las 100 parcelas del CSV
   â””â”€â”€ Carga el .npz de la parcela mÃ¡s cercana
       â”‚
       â–¼
2. patch_to_timeseries(npz_path)
   â””â”€â”€ (T=277, C=5, H=50, W=53) â†’ promedio espacial â†’ (T=277, C=5)
       â”‚
       â–¼
3. extract_last_window(ts, t_mod, t_sev)
   â””â”€â”€ Ãšltimas 24 fechas â†’ 35 features estadÃ­sticas
       â”‚
       â–¼
4. EnsemblePredictor.predict(features)
   â””â”€â”€ MinMaxScaler â†’ E3 Stacking â†’ probabilidades â†’ clase + confianza
       â”‚
       â–¼
5. extract_trend_windows(ts, ..., n_windows=4)
   â””â”€â”€ 4 Ãºltimas ventanas â†’ tendencia NDMI (ascendente / estable / descendente)
       â”‚
       â–¼
6. generate_report(client=Ollama/OpenLLaMA, prediction, indices, trend, photo_b64)
   â””â”€â”€ Construye prompt con datos del diagnÃ³stico + foto (si la hay)
   â””â”€â”€ Ollama/OpenLLaMA devuelve reporte agronÃ³mico en espaÃ±ol
       â”‚
       â–¼
Respuesta JSON: { location, stress, indices, trend, llm_report }
```

### 6.3 ConfiguraciÃ³n

Las credenciales y rutas se gestionan con **pydantic-settings** y un archivo `.env`:

```
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=openllama
CDSE_USER=email@ejemplo.com       # Usuario CDSE para descargar Sentinel-2
CDSE_PASSWORD=...                 # ContraseÃ±a CDSE
```

> âš ï¸ El archivo `.env` **nunca se sube al repositorio** (estÃ¡ en `.gitignore`). Cada integrante del equipo debe crear el suyo a partir de `.env.example`.

---

## 7. IntegraciÃ³n LLM: Ollama/OpenLLaMA

### 7.1 Â¿Por quÃ© un LLM?

El modelo ML produce un nÃºmero de clase y una probabilidad â€” Ãºtiles tÃ©cnicamente, pero difÃ­ciles de interpretar para un agricultor sin formaciÃ³n tÃ©cnica. Ollama/OpenLLaMA convierte esos nÃºmeros en un **reporte agronÃ³mico en espaÃ±ol**, accionable y claro.

### 7.2 Prompt enviado a Ollama/OpenLLaMA

El sistema construye automÃ¡ticamente un mensaje que incluye:

- Coordenadas de la parcela analizada
- DiagnÃ³stico del modelo (clase, confianza, probabilidades)
- Los 5 Ã­ndices espectrales en su valor actual
- La tendencia de las Ãºltimas 4 ventanas (3 meses aproximados)
- La foto del campo en base64 (si el usuario la adjuntÃ³)

### 7.3 Rol del sistema (system prompt)

```
Eres un asesor agronÃ³mico especialista en cultivos de aguacate (Hass) en Jalisco.
Interpretas datos satelitales Sentinel-2 para diagnosticar estrÃ©s hÃ­drico.

Cuando recibas un diagnÃ³stico debes:
1. Explicar en lenguaje claro quÃ© significa para el agricultor.
2. Dar entre 3 y 5 recomendaciones concretas y accionables segÃºn la severidad.
3. Indicar el nivel de urgencia: INMEDIATA / ESTA SEMANA / MONITOREO.
4. Si hay foto, comentar si es consistente con los datos satelitales.
5. Cerrar con un dato breve de contexto climÃ¡tico o agronÃ³mico relevante.
```

### 7.4 Soporte multimodal

Si el agricultor adjunta una **foto del campo**, Ollama/OpenLLaMA la analiza visualmente junto con los datos satelitales, buscando seÃ±ales de estrÃ©s visibles (hojas enrolladas, coloraciÃ³n, necrosis) y comparÃ¡ndolas con lo que muestran los Ã­ndices.

### 7.5 Fallback sin LLM

Si Ollama/OpenLLaMA no estÃ¡ disponible (red, cuota agotada), el sistema devuelve un **reporte de reglas** basado solo en la clase predicha, sin llamar a la API. Esto garantiza que el sistema funcione siempre.

---

## 8. Frontend: Dashboard de la Cooperativa

El dashboard es una aplicaciÃ³n web de una sola pÃ¡gina (SPA) construida con **Vanilla JS** (sin frameworks) + **Leaflet.js** para el mapa + **Chart.js** para las grÃ¡ficas. Se sirve directamente desde FastAPI en `http://localhost:8000/ui`.

### 8.1 Componentes del dashboard

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ ðŸ¥‘ AguaVerde  Coop. Aguacatera Â· Jalisco   [stats]  [âš¡ Scan]  â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                             â”‚ ðŸ“ DiagnÃ³stico â”‚ âž• Nueva ubic.  â”‚
â”‚      MAPA LEAFLET           â”‚                                   â”‚
â”‚                             â”‚ Parcela H1 Â· 0.12 km             â”‚
â”‚  âš«âš«ðŸŸ¢ðŸ”´ðŸŸ¡ðŸŸ¢âš«âš«         â”‚ ðŸŸ¡ EstrÃ©s Moderado  94%          â”‚
â”‚       Jalisco, MÃ©xico       â”‚ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€    â”‚
â”‚                             â”‚ NDMI: 0.1823  NDVI: 0.6102       â”‚
â”‚                             â”‚ NDWI: 0.1045  NDRE: 0.3421       â”‚
â”‚  ðŸŸ¢ Sin estrÃ©s              â”‚                                   â”‚
â”‚  ðŸŸ¡ Moderado                â”‚ ðŸ“ˆ Tendencia NDMI [chart]        â”‚
â”‚  ðŸ”´ Severo                  â”‚ â¬‡ Tendencia descendente          â”‚
â”‚  âš« Sin analizar            â”‚                                   â”‚
â”‚                             â”‚ ðŸ¤– Reporte Ollama/OpenLLaMA                â”‚
â”‚                             â”‚ "Su parcela muestra estrÃ©s       â”‚
â”‚                             â”‚ moderado. Recomendamos..."       â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### 8.2 Funciones principales

**âš¡ Escanear mapa**
Analiza las primeras 30 parcelas en secuencia usando solo el modelo ML (sin llamar a Ollama/OpenLLaMA), para colorear el mapa rÃ¡pidamente. Muestra una barra de progreso durante el escaneo.

**Clic en marcador**
Al hacer clic en cualquier marcela del mapa, se lanza el anÃ¡lisis completo: modelo ML + reporte Ollama/OpenLLaMA. El panel lateral muestra en tiempo real los resultados, la grÃ¡fica de tendencia NDMI y el reporte agronÃ³mico. Si la parcela ya fue escaneada (solo ML), solicita el reporte Ollama/OpenLLaMA en ese momento.

**Nueva ubicaciÃ³n (pestaÃ±a)**
Permite ingresar cualquier par de coordenadas GPS manualmente, subir una foto del campo (opcional), y obtener el diagnÃ³stico completo. El sistema localiza la parcela Sentinel-2 mÃ¡s cercana mediante Haversine y ejecuta el pipeline completo.

**Filtros por clase**
Botones en el header para mostrar solo parcelas sin estrÃ©s / moderado / severo.

---

## 9. Arquitectura General del Sistema

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                        USUARIO / AGRICULTOR                      â”‚
â”‚              (navegador web o app mÃ³vil futura)                  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                   â”‚ HTTP (mismo origen)
                   â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                    FastAPI  (api/main.py)                         â”‚
â”‚                                                                  â”‚
â”‚  GET /parcels â”€â”€â–º LocalCatalog â†’ parcelas.csv                    â”‚
â”‚                                                                  â”‚
â”‚  POST /analyze                                                   â”‚
â”‚    â”‚                                                             â”‚
â”‚    â”œâ”€ LocalCatalog.find_nearest()  â”€â”€â–º parcelas.csv              â”‚
â”‚    â”‚         â””â”€â”€ haversine nearest â†’ carga .npz                  â”‚
â”‚    â”‚                                                             â”‚
â”‚    â”œâ”€ extract_last_window()        â”€â”€â–º features (35,)            â”‚
â”‚    â”‚                                                             â”‚
â”‚    â”œâ”€ EnsemblePredictor.predict()  â”€â”€â–º clase + confianza         â”‚
â”‚    â”‚         â””â”€â”€ MinMaxScaler â†’ Stacking â†’ LogReg                â”‚
â”‚    â”‚                                                             â”‚
â”‚    â””â”€ generate_report()            â”€â”€â–º Ollama local API      â”‚
â”‚              â””â”€â”€ texto agronÃ³mico + anÃ¡lisis foto                â”‚
â”‚                                                                  â”‚
â”‚  GET /ui â”€â”€â–º frontend/ (HTML/CSS/JS + Leaflet + Chart.js)        â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
         â”‚                              â”‚
         â–¼                              â–¼
 data/raw/parcels/               localhost:11434
   parcelas.csv                  claude-opus-4-8
   patches/*.npz
         â”‚
         â–¼
 models/
   ensemble_stacking.joblib
   ensemble_scaler.joblib
   ensemble_meta.json
```

---

## 10. Flujo Completo de Extremo a Extremo

```
 [ImÃ¡genes Sentinel-2 crudas]
          â”‚
          â–¼  src/ingestion/sentinel2_downloader.py
 [GeoTIFFs por parcela y fecha]
          â”‚
          â–¼  src/processing/spectral_indices.py
 [Ãndices NDVI/NDWI/NDMI/NDRE/EVI por fecha]
          â”‚
          â–¼  src/processing/time_series_builder.py
 [Parches .npz: (T=277, C=5, H=50, W=53)]
          â”‚
          â–¼  notebooks/Avance5.equipo16.ipynb
 [Promedio espacial â†’ (T=277, C=5)]
          â”‚
          â–¼  Ventana deslizante W=24, step=4
 [~6,400 muestras Ã— 35 features + etiqueta NDMI]
          â”‚
          â–¼  E3 Stacking (RF + XGB + SVM â†’ LogReg)
 [Modelo entrenado: F1-macro = 0.8868]
          â”‚
          â–¼  joblib.dump()
 [models/ensemble_stacking.joblib]
          â”‚
          â–¼  uvicorn api.main:app
 [API REST en http://localhost:8000]
          â”‚
     â”Œâ”€â”€â”€â”€â”´â”€â”€â”€â”€â”
     â–¼         â–¼
 [/docs]     [/ui]
 Swagger    Dashboard
            Leaflet
```

---

## 11. CÃ³mo Ejecutar el Proyecto

### Requisitos previos
- Python 3.12 recomendado para compatibilidad con el modelo serializado
- Credenciales CDSE (gratuitas en [dataspace.copernicus.eu](https://dataspace.copernicus.eu/))
- Ollama instalado localmente, sin API key

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

> Para el demo de la cooperativa, los pasos 3 y 4 ya estÃ¡n completados â€” los `.npz` y el modelo estÃ¡n en el repositorio/servidor. Solo se necesita el paso 5.

---

## 12. Estructura de Archivos

```
integrative-project/
â”‚
â”œâ”€â”€ api/                          # Backend FastAPI
â”‚   â”œâ”€â”€ main.py                   # Endpoints + mount del frontend
â”‚   â”œâ”€â”€ config.py                 # ConfiguraciÃ³n vÃ­a .env
â”‚   â”œâ”€â”€ sentinel.py               # LocalCatalog â€” haversine + carga .npz
â”‚   â”œâ”€â”€ features.py               # ExtracciÃ³n de 35 features por ventana
â”‚   â”œâ”€â”€ predictor.py              # EnsemblePredictor (lru_cache)
â”‚   â””â”€â”€ llm.py                    # Reporte agronÃ³mico con Ollama/OpenLLaMA
â”‚
â”œâ”€â”€ frontend/                     # Dashboard (sirve en GET /ui)
â”‚   â”œâ”€â”€ index.html                # Estructura del dashboard
â”‚   â”œâ”€â”€ style.css                 # Estilos (diseÃ±o cooperativa)
â”‚   â””â”€â”€ app.js                    # LÃ³gica: mapa, anÃ¡lisis, grÃ¡ficas
â”‚
â”œâ”€â”€ models/                       # Artefactos del modelo entrenado
â”‚   â”œâ”€â”€ ensemble_stacking.joblib  # Modelo E3 Stacking
â”‚   â”œâ”€â”€ ensemble_scaler.joblib    # Scaler MinMax
â”‚   â””â”€â”€ ensemble_meta.json        # Umbrales y mÃ©tricas
â”‚
â”œâ”€â”€ data/
â”‚   â””â”€â”€ raw/parcels/
â”‚       â”œâ”€â”€ parcelas.csv          # 100 parcelas georeferenciadas
â”‚       â”œâ”€â”€ patches/              # .npz por parcela (T,C,H,W)
â”‚       â””â”€â”€ *.kml                 # Archivo original de parcelas
â”‚
â”œâ”€â”€ notebooks/
â”‚   â””â”€â”€ Avance5.equipo16.ipynb    # Pipeline ML completo: EDA â†’ features â†’ E3 Stacking
â”‚
â”œâ”€â”€ src/                          # Scripts de ingesta y procesamiento
â”‚   â”œâ”€â”€ ingestion/
â”‚   â”‚   â”œâ”€â”€ kml_to_csv.py
â”‚   â”‚   â””â”€â”€ sentinel2_downloader.py
â”‚   â””â”€â”€ processing/
â”‚       â”œâ”€â”€ spectral_indices.py
â”‚       â””â”€â”€ time_series_builder.py
â”‚
â”œâ”€â”€ .env.example                  # Plantilla de variables de entorno
â”œâ”€â”€ requirements.txt              # Dependencias ML / notebooks
â”œâ”€â”€ requirements-api.txt          # Dependencias del backend
â”œâ”€â”€ Makefile                      # Comandos del proyecto
â””â”€â”€ README.md                     # DocumentaciÃ³n principal
```

---

## 13. Decisiones de DiseÃ±o

### Â¿Por quÃ© NDMI como Ã­ndice principal?

El NDMI usa las bandas B08 (NIR) y B11 (SWIR), donde la segunda es especialmente sensible al contenido de agua en la hoja. Es mÃ¡s especÃ­fico que el NDVI (que mide biomasa en general) y mÃ¡s robusto al ruido atmosfÃ©rico que el NDWI.

### Â¿Por quÃ© Stacking y no un solo modelo?

Cada base learner tiene fortalezas distintas: Random Forest maneja bien la no-linealidad, XGBoost es muy preciso en datos tabulares y el SVM con kernel RBF genera buenas fronteras de decisiÃ³n en espacios de alta dimensiÃ³n. El meta-learner aprende cuÃ¡ndo confiar en cada uno segÃºn el caso, superando a cualquiera individualmente.

### Â¿Por quÃ© ventanas de 24 fechas?

Con un paso de revisita de ~5 dÃ­as, 24 imÃ¡genes equivalen a aproximadamente **4 meses** de observaciÃ³n. Esto captura la variaciÃ³n estacional del estrÃ©s mientras mantiene el vector de features manejable (35 nÃºmeros).

### Â¿Por quÃ© FastAPI + archivos estÃ¡ticos en lugar de React/Vue?

Para un demo de cooperativa, la simplicidad de una SPA con Vanilla JS sirve directamente por FastAPI reduce las dependencias a cero (sin `npm`, sin build step). Un solo comando (`make dev`) levanta todo el sistema.

### Â¿Por quÃ© Ollama/OpenLLaMA y no otro LLM?

Ollama permite ejecutar el LLM localmente, sin costos de API y sin depender de credenciales externas. OpenLLaMA se puede usar como modelo local para generar reportes en espaÃ±ol con el tono correcto para agricultores.

---

## 14. Posibles Mejoras Futuras

| Mejora | DescripciÃ³n |
|--------|-------------|
| Datos en tiempo real | Integrar CDSE STAC API para descargar el tile Sentinel-2 mÃ¡s reciente al momento de la consulta |
| Alertas automÃ¡ticas | Enviar WhatsApp/email al agricultor cuando se detecte estrÃ©s severo |
| App mÃ³vil | PWA que tome la foto con la cÃ¡mara del celular directamente |
| HistÃ³rico por parcela | Guardar en base de datos el historial de diagnÃ³sticos por parcela |
| ViT for SITS | Sustituir el ensamble por el Vision Transformer entrenado en SITS para mayor precisiÃ³n |
| CalibraciÃ³n con datos de campo | Validar los umbrales de NDMI con mediciones reales de humedad en suelo |

---

*Documento generado para uso interno del Equipo 16 â€” Proyecto Integrador 2025*

