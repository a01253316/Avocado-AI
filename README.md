# 🥑 Avocado Stress MLOps
**ViTs for SITS** — Vision Transformers for Satellite Image Time Series  
Detección de estrés hídrico en parcelas de aguacate (Jalisco / Michoacán) usando imágenes Sentinel-2.

---

## 🎯 Objetivo
Construir un pipeline MLOps end-to-end que:
1. Ingiera imágenes multiespectrales de Sentinel-2 por parcela y rango de fechas
2. Calcule índices de vegetación (NDVI, NDWI, NDMI, NDRE, EVI)
3. Construya series de tiempo por parcela (SITS)
4. Entrene modelos CNN / ViT para detectar estrés hídrico
5. Sirva predicciones vía API

## 📦 Stack
- **Datos**: Sentinel-2 (ESA), Alpha Earth API
- **Tracking**: MLflow + DVC
- **Orquestación**: Makefile → (futuro: Prefect / Airflow)
- **Modelos**: PyTorch — CNN pixel-level → ViT para series de tiempo
- **Serving**: FastAPI

---

## 🗂️ Estructura del Proyecto

```
avocado-stress-mlops/
├── data/
│   ├── raw/
│   │   ├── parcels/          # KML original + CSV extraído
│   │   └── sentinel2/        # TIFFs descargados por parcela/fecha
│   ├── processed/
│   │   ├── patches/          # Recortes (chips) por parcela
│   │   └── indices/          # Rasters de NDVI, NDWI, NDMI, NDRE, EVI
│   └── datasets/             # Datasets listos para entrenamiento (.npy / .pt)
│
├── src/
│   ├── ingestion/
│   │   ├── kml_to_csv.py         # ★ Extracción KML → CSV
│   │   ├── sentinel2_downloader.py   # Descarga por parcela + fecha
│   │   └── alpha_earth_client.py     # Cliente Alpha Earth API
│   ├── processing/
│   │   ├── spectral_indices.py   # Cálculo NDVI, NDWI, NDMI, NDRE, EVI
│   │   ├── patch_extractor.py    # Recorte de chips por parcela
│   │   └── time_series_builder.py # Apila imágenes en series temporales
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── cnn/
│   │   │   ├── pixel_classifier.py
│   │   │   └── train_cnn.py
│   │   └── vit/
│   │       ├── sits_vit.py       # ViT for SITS (paper base)
│   │       └── train_vit.py
│   └── utils/
│       ├── geo_utils.py
│       └── io_utils.py
│
├── notebooks/
│   ├── 01_eda_parcelas.ipynb
│   ├── 02_sentinel2_exploracion.ipynb
│   └── 03_indices_analisis.ipynb
│
├── configs/
│   ├── parcels.yaml          # Config de parcelas y buffer
│   └── sentinel2.yaml        # Bandas, fechas, tile
│
├── pipelines/
│   ├── ingestion_pipeline.py
│   └── processing_pipeline.py
│
├── tests/
│   ├── test_kml_to_csv.py
│   └── test_spectral_indices.py
│
├── .dvc/
├── MLproject
├── params.yaml
├── Makefile
├── requirements.txt
└── README.md
```

---

## 🚀 Inicio Rápido

```bash
# 1. Setup del entorno
make setup

# 2. Extracción KML → CSV
make extract-parcels

# 3. Descarga de imágenes Sentinel-2
make download-sentinel2 START=2020-01-01 END=2026-05-30

# 4. Calcular índices espectrales
make compute-indices

# 5. Entrenar CNN baseline
make train-cnn
```

---

## 📡 Bandas Sentinel-2 Usadas

| Índice | Bandas          | Qué mide                         |
|--------|-----------------|----------------------------------|
| NDVI   | B08, B04        | Vigor vegetal general            |
| NDWI   | B03, B08        | Contenido de agua en vegetación  |
| NDMI   | B08, B11        | Humedad en hoja/dosel ★          |
| NDRE   | B08, B05        | Estrés temprano (clorofila) ★    |
| EVI    | B08, B04, B02   | Vegetación en zonas densas       |

★ Más directos para estrés hídrico.

---

## 📌 Referencias
- Garnot & Landrieu (2021) — *ViTs for SITS: Vision Transformers for Satellite Image Time Series*
- ESA Sentinel-2 User Guide
