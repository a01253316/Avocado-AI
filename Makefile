# ============================================================
# avocado-stress-mlops — Makefile
# ============================================================
PYTHON      := python
PIP         := pip
CONDA_ENV   := avocado-mlops
KML_INPUT   := data/raw/parcels/aguacates_jalisco_5_5_26.kml
PARCELS_CSV := data/raw/parcels/parcelas.csv
START_DATE  := 2020-01-01
END_DATE    := $(shell date +%Y-%m-%d)
BUFFER_M    := 250

.PHONY: help setup conda-setup extract-parcels download-sentinel2 \
        compute-indices build-dataset train-cnn train-vit \
        test lint clean

# ── Ayuda ──────────────────────────────────────────────────
help:
	@echo ""
	@echo "🥑  Avocado Stress MLOps — comandos disponibles"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make conda-setup        Crea entorno conda con GDAL (recomendado en macOS)"
	@echo "  make setup              Instala deps vía pip (requiere GDAL en el sistema)"
	@echo "  make extract-parcels    KML → CSV (100 parcelas)"
	@echo "  make download-sentinel2 Descarga imágenes Sentinel-2"
	@echo "  make compute-indices    Calcula NDVI/NDWI/NDMI/NDRE/EVI"
	@echo "  make build-dataset      Arma el dataset para entrenamiento"
	@echo "  make train-cnn          Entrena CNN baseline"
	@echo "  make train-vit          Entrena ViT for SITS"
	@echo "  make test               Corre tests unitarios"
	@echo "  make lint               Lint con ruff"
	@echo "  make clean              Limpia archivos temporales"
	@echo ""

# ── Setup ──────────────────────────────────────────────────
conda-setup:
	@echo "🔧 Creando entorno conda '$(CONDA_ENV)' con GDAL via conda-forge..."
	conda create -n $(CONDA_ENV) python=3.11 -y
	conda run -n $(CONDA_ENV) conda install -c conda-forge \
		gdal fiona rasterio geopandas shapely pyproj -y
	conda run -n $(CONDA_ENV) pip install -r requirements.txt \
		--ignore-requires-python
	@echo ""
	@echo "✅ Entorno listo. Actívalo con:"
	@echo "   conda activate $(CONDA_ENV)"

setup:
	$(PIP) install -r requirements.txt
	@echo "✅ Entorno listo"

# ── Ingesta ────────────────────────────────────────────────
extract-parcels:
	@echo "📍 Extrayendo parcelas de $(KML_INPUT)..."
	$(PYTHON) src/ingestion/kml_to_csv.py \
		--input $(KML_INPUT) \
		--output $(PARCELS_CSV) \
		--buffer $(BUFFER_M)
	@echo "✅ CSV generado: $(PARCELS_CSV)"

download-sentinel2: $(PARCELS_CSV)
	@echo "🛰️  Descargando imágenes Sentinel-2 ($(START_DATE) → $(END_DATE))..."
	$(PYTHON) src/ingestion/sentinel2_downloader.py \
		--parcels $(PARCELS_CSV) \
		--start $(START_DATE) \
		--end $(END_DATE) \
		--output data/raw/sentinel2/

# ── Procesamiento ──────────────────────────────────────────
compute-indices:
	@echo "📊 Calculando índices espectrales..."
	$(PYTHON) src/processing/spectral_indices.py \
		--input  data/raw/sentinel2/ \
		--output data/processed/indices/

build-dataset:
	@echo "🗃️  Construyendo dataset de series temporales..."
	$(PYTHON) src/processing/time_series_builder.py \
		--parcels  $(PARCELS_CSV) \
		--indices  data/processed/indices/ \
		--output   data/datasets/

# ── Entrenamiento ──────────────────────────────────────────
train-cnn:
	@echo "🧠 Entrenando CNN baseline..."
	$(PYTHON) src/models/cnn/train_cnn.py \
		--config configs/sentinel2.yaml

train-vit:
	@echo "🤖 Entrenando ViT for SITS..."
	$(PYTHON) src/models/vit/train_vit.py \
		--config configs/sentinel2.yaml

# ── Calidad ────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v

lint:
	ruff check src/ tests/

# ── Limpieza ───────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete; \
	echo "🧹 Limpieza completada"
