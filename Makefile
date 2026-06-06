# ============================================================
# avocado-stress-mlops — Makefile
# ============================================================
PYTHON      := python
PIP         := pip
VIT_PRESET  := small   # tiny | small | base
KML_INPUT   := data/raw/parcels/aguacates_jalisco_5_5_26.kml
PARCELS_CSV := data/raw/parcels/parcelas.csv
START_DATE  := 2020-01-01
END_DATE    := $(shell date +%Y-%m-%d)
BUFFER_M    := 250

.PHONY: help setup extract-parcels download-sentinel2 \
        compute-indices build-dataset train-cnn train-vit \
        test lint clean

# ── Ayuda ──────────────────────────────────────────────────
help:
	@echo ""
	@echo "🥑  Avocado Stress MLOps — comandos disponibles"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make setup              Instala dependencias"
	@echo "  make extract-parcels    KML → CSV (parcelas)"
	@echo "  make download-sentinel2 Descarga GeoTIFFs via Process API (CDSE)"
	@echo "  make compute-indices    Calcula NDVI/NDWI/NDMI/NDRE/EVI por fecha"
	@echo "  make build-dataset      Arma el dataset SITS (patches + signals + split)"
	@echo "  make train-cnn          Entrena CNN baseline"
	@echo "  make train-vit          Entrena ViT for SITS"
	@echo "  make test               Corre tests unitarios"
	@echo "  make lint               Lint con ruff"
	@echo "  make clean              Limpia archivos temporales"
	@echo ""

# ── Setup ──────────────────────────────────────────────────
setup:
	$(PIP) install -r requirements.txt
	@echo "✅ Entorno listo"

# ── Ingesta ────────────────────────────────────────────────
extract-parcels:
	@echo "📍 Extrayendo parcelas de $(KML_INPUT)..."
	$(PYTHON) src/ingestion/kml_to_csv.py \
		--input  $(KML_INPUT) \
		--output $(PARCELS_CSV) \
		--buffer $(BUFFER_M)
	@echo "✅ CSV generado: $(PARCELS_CSV)"

# Descarga multibanda FLOAT32 via Process API de CDSE (Sentinel Hub)
# Fuente: sentinel-2-l2a | Credenciales en .env (CDSE_USER / CDSE_PASSWORD)
download-sentinel2: $(PARCELS_CSV)
	@echo "🛰️  Descargando Sentinel-2 ($(START_DATE) → $(END_DATE)) via Process API..."
	$(PYTHON) src/ingestion/sentinel2_downloader.py \
		--config  configs/sentinel2.yaml \
		--parcels $(PARCELS_CSV) \
		--start   $(START_DATE) \
		--end     $(END_DATE)
	@echo "✅ TIFFs guardados en data/raw/sentinel2/"

# ── Procesamiento ──────────────────────────────────────────

# Calcula índices espectrales desde parcel_multiband.tif (FLOAT32)
# Detecta automáticamente si los datos son FLOAT32 (Process API) o DN (uint16 legado)
compute-indices:
	@echo "📊 Calculando índices espectrales (NDVI/NDWI/NDMI/NDRE/EVI)..."
	$(PYTHON) src/processing/spectral_indices.py \
		--input  data/raw/sentinel2/ \
		--output data/processed/indices/
	@echo "✅ Índices guardados en data/processed/indices/"

# Construye el dataset SITS con split 70/15/15
build-dataset:
	@echo "🗃️  Construyendo dataset de series temporales..."
	$(PYTHON) src/processing/time_series_builder.py \
		--indices-dir data/processed/indices/ \
		--parcels     $(PARCELS_CSV) \
		--output      data/datasets/ \
		--mode        both \
		--normalize   minmax \
		--split
	@echo "✅ Dataset en data/datasets/ (patches/, signals/, manifest.csv, split.json)"

# ── Entrenamiento ──────────────────────────────────────────
train-cnn:
	@echo "🧠 Entrenando CNN baseline..."
	$(PYTHON) src/models/cnn/train_cnn.py

train-vit:
	@echo "🤖 Entrenando ViT for SITS (preset=$(VIT_PRESET))..."
	$(PYTHON) src/models/vit/train_vit.py \
		--preset     $(VIT_PRESET) \
		--dataset-dir data/datasets/ \
		--output-dir  models/ \
		--experiment  avocado-stress-vit

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