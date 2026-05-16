# Avocado Water Stress Monitor

**Proyecto de Monitoreo del Estrés Hídrico en Huertos de Aguacate**  
Imágenes satelitales Sentinel-2 · LSTM Autoencoder · Claude multimodal  
Sur de Jalisco, México — 2020 a la fecha

---

## Arquitectura del pipeline

```
KML (100 parcelas)
    │
    ▼
Sentinel Hub Process API
(Catalog API → escenas disponibles)
(Process API → GeoTIFF recortado al bbox de cada parcela)
Evalscript V3: NDVI, NDWI, NDMI, EVI2, MSI + máscara SCL
    │
    ▼
data/raw/sentinel2/<parcela>/<YYYY-MM-DD>.tif
(FLOAT32 · 6 bandas · 512×512 px · 10 m/px)
    │
    ▼
Series temporales por parcela
(estadísticos por banda: mean, std, p10, p90)
data/processed/time_series/<parcela>.parquet
    │
    ▼
Regularización a 8 días + ventanas de 46 timesteps
(StandardScaler ajustado sobre el conjunto completo)
data/processed/windows/X.npy  (N, 46, 17 features)
    │
    ▼
LSTM Autoencoder — entrenamiento no supervisado
Encoder → z ∈ R³² → Decoder → reconstrucción
Score de anomalía = MAE(x, x̂)  |  Umbral = p95
    │
    ▼
Claude Opus (LLM multimodal)
Figura PNG + estadísticos → reporte en Markdown
reports/<parcela>.md
```

---

## Inicio rápido

### 1. Clonar y configurar

```bash
git clone <repo-url>
cd avocado-water-stress
make setup
```

### 2. Credenciales

Edita `configs/credentials.yaml`:

```yaml
sentinel_hub:
  client_id: "TU_CLIENT_ID"       # apps.sentinel-hub.com → OAuth clients
  client_secret: "TU_CLIENT_SECRET"

anthropic:
  api_key: "TU_ANTHROPIC_API_KEY"
```

### 3. Pipeline completo

```bash
make all
```

O paso a paso:

```bash
make download     # descarga imágenes (puede tardar varias horas)
make ts           # series temporales y ventanas
make train        # LSTM Autoencoder + scores de anomalía
make report       # reportes con Claude multimodal
```

### 4. Prueba rápida (una parcela)

```bash
make download-test PARCEL_ID=parcela_001
make ts
make train-fast
make report-parcel PARCEL_ID=parcela_001
```

---

## Sentinel Hub — cómo obtener credenciales

1. Crea una cuenta gratuita en [apps.sentinel-hub.com](https://apps.sentinel-hub.com)
2. Ve a **User Settings → OAuth clients → Create new**
3. Copia `client_id` y `client_secret` en `configs/credentials.yaml`

> El plan gratuito de Sentinel Hub incluye 30,000 unidades de procesamiento/mes.  
> Para 100 parcelas × 5 años ≈ 6,000 imágenes estimadas — suficiente.

---

## Estructura del proyecto

```
avocado-water-stress/
├── configs/
│   ├── base.yaml            ← parámetros del pipeline
│   └── credentials.yaml     ← claves API (NO subir a Git)
├── data/
│   ├── raw/
│   │   ├── parcelas.kml     ← KML de Alpha Earth
│   │   └── sentinel2/       ← GeoTIFFs descargados
│   ├── processed/
│   │   ├── time_series/     ← series temporales por parcela
│   │   └── windows/         ← ventanas para el LSTM
│   └── models/lstm_ae/      ← pesos y scores del modelo
├── src/
│   ├── ingestion/
│   │   └── downloader.py    ← Catalog API + Process API
│   ├── processing/
│   │   ├── time_series.py   ← GeoTIFF → parquet
│   │   └── windows.py       ← ventanas deslizantes
│   ├── models/
│   │   └── lstm_autoencoder.py  ← modelo + MLflow
│   ├── reporting/
│   │   └── llm_report.py    ← Claude multimodal
│   └── utils/
│       ├── sentinel_auth.py ← OAuth2 token manager
│       └── kml_reader.py    ← lectura del KML
├── reports/                 ← reportes Markdown generados
├── Makefile
└── requirements.txt
```

---

## Índices espectrales calculados

| Índice | Bandas | Qué mide |
|--------|--------|----------|
| NDVI   | B08, B04 | Salud y densidad de vegetación |
| NDWI   | B08, B11 | Humedad de la hoja |
| NDMI   | B08, B11 | Humedad foliar (alias NDWI) |
| EVI2   | B08, B04 | Vegetación en zonas de alto NDVI |
| MSI    | B11, B8A | Estrés hídrico directo (↑ = más estrés) |

Los índices se calculan **en el servidor de Sentinel Hub** vía Evalscript V3, evitando la descarga de productos `.SAFE` de 1 GB.

---

## MLflow

El experimento `avocado-water-stress` registra automáticamente:
- Hiperparámetros del LSTM
- Loss de entrenamiento y validación por epoch
- Umbral de anomalía (percentil 95 del score MAE)
- Artefactos: pesos del modelo, scores, configuración

```bash
# mlflow ui  # abre en http://localhost:5000 Original
mlflow ui --backend-store-uri file:./mlruns
```
