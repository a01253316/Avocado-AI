# ============================================================
# AguaVerde - Makefile
# Deteccion de Estres Hidrico en Aguacate - Jalisco, Mexico
# ============================================================

PYTHON      := python
PIP         := pip

# Data parameters
KML_INPUT   := data/raw/parcels/aguacates_jalisco_5_5_26.kml
PARCELS_CSV := data/raw/parcels/parcelas.csv
START_DATE  := 2020-01-01
END_DATE    := $(shell date +%Y-%m-%d)
BUFFER_M    := 250
VIT_PRESET  := small   # tiny | small | base
SAM2_CHECKPOINT ?= models/checkpoints/sam2.1_hiera_tiny.pt
SAM2_MODEL_CFG  ?= configs/sam2.1/sam2.1_hiera_t.yaml
SAM2_DEVICE     ?= auto
SAM2_CHECKPOINT_URL ?= https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

# Server parameters
HOST        := 127.0.0.1
PORT        := 8000

# Phony targets
.PHONY: help \
        setup setup-api setup-all \
        extract-parcels download-sentinel2 \
        compute-indices build-dataset \
        train-ensemble train-cnn train-vit prepare-sam2 download-sam2-checkpoint train-sam2 \
        dev serve \
        notebook \
        test lint \
        clean clean-cache

# ------------------------------------------------------------
# Help
# ------------------------------------------------------------
help:
	@echo ""
	@echo "AguaVerde - comandos disponibles"
	@echo "------------------------------------------------------------"
	@echo ""
	@echo "  SETUP"
	@echo "  -----"
	@echo "  make setup          Instala dependencias ML (requirements.txt)"
	@echo "  make setup-api      Instala dependencias API (requirements-api.txt)"
	@echo "  make setup-all      Instala ambos conjuntos de dependencias"
	@echo ""
	@echo "  PIPELINE DE DATOS"
	@echo "  -----------------"
	@echo "  make extract-parcels      KML -> CSV de parcelas"
	@echo "  make download-sentinel2   Descarga TIFFs Sentinel-2 via CDSE"
	@echo "  make compute-indices      Calcula NDVI/NDWI/NDMI/NDRE/EVI"
	@echo "  make build-dataset        Construye dataset SITS (patches + signals)"
	@echo ""
	@echo "  MODELOS"
	@echo "  -------"
	@echo "  make train-ensemble Entrena E3 Stacking (modelo principal)"
	@echo "  make train-cnn      Entrena CNN baseline"
	@echo "  make train-vit      Entrena ViT for SITS (preset=$(VIT_PRESET))"
	@echo "  make prepare-sam2   Exporta dataset SAM2 desde patches"
	@echo "  make download-sam2-checkpoint Descarga checkpoint SAM2.1 tiny"
	@echo "  make train-sam2     Fine-tuning SAM2 con checkpoint externo"
	@echo ""
	@echo "  SERVIDOR / DASHBOARD"
	@echo "  --------------------"
	@echo "  make dev            API + dashboard en modo desarrollo (--reload)"
	@echo "  make serve          API + dashboard en modo produccion"
	@echo "                      -> http://$(HOST):$(PORT)/ui"
	@echo ""
	@echo "  DESARROLLO"
	@echo "  ----------"
	@echo "  make notebook       Abre Jupyter Lab"
	@echo "  make test           Corre tests unitarios (pytest)"
	@echo "  make lint           Lint con ruff"
	@echo "  make clean          Limpia __pycache__ y .pyc"
	@echo "  make clean-cache    Limpia caches pesados (mlruns, .pytest_cache)"
	@echo ""

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
setup:
	$(PIP) install -r requirements.txt
	@echo "OK: Dependencias ML instaladas"

setup-api:
	$(PIP) install -r requirements-api.txt
	@echo "OK: Dependencias API instaladas"

setup-all: setup setup-api
	@echo "OK: Todas las dependencias instaladas"

# ------------------------------------------------------------
# Data pipeline
# ------------------------------------------------------------
extract-parcels:
	@echo "Extrayendo parcelas de $(KML_INPUT)..."
	$(PYTHON) src/ingestion/kml_to_csv.py \
		--input  $(KML_INPUT) \
		--output $(PARCELS_CSV) \
		--buffer $(BUFFER_M)
	@echo "OK: CSV generado: $(PARCELS_CSV)"

# Credenciales en .env: CDSE_USER / CDSE_PASSWORD
download-sentinel2: $(PARCELS_CSV)
	@echo "Descargando Sentinel-2 ($(START_DATE) -> $(END_DATE))..."
	$(PYTHON) src/ingestion/sentinel2_downloader.py \
		--config  configs/sentinel2.yaml \
		--parcels $(PARCELS_CSV) \
		--start   $(START_DATE) \
		--end     $(END_DATE)
	@echo "OK: TIFFs guardados en data/raw/sentinel2/"

compute-indices:
	@echo "Calculando indices espectrales..."
	$(PYTHON) src/processing/spectral_indices.py \
		--input  data/raw/sentinel2/ \
		--output data/processed/indices/
	@echo "OK: Indices guardados en data/processed/indices/"

build-dataset:
	@echo "Construyendo dataset SITS..."
	$(PYTHON) src/processing/time_series_builder.py \
		--indices-dir data/processed/indices/ \
		--parcels     $(PARCELS_CSV) \
		--output      data/datasets/ \
		--mode        both \
		--normalize   minmax \
		--split
	@echo "OK: Dataset en data/datasets/"

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

# Modelo principal - E3 Stacking (RF + XGBoost + SVM -> LogReg)
# Genera: models/ensemble_stacking.joblib, ensemble_scaler.joblib, ensemble_meta.json
train-ensemble:
	@echo "Entrenando E3 Stacking (modelo principal)..."
	$(PYTHON) -m jupyter nbconvert --to notebook --execute \
		--ExecutePreprocessor.timeout=600 \
		--output notebooks/Avance5.equipo16.executed.ipynb \
		notebooks/Avance5.equipo16.ipynb
	@echo "OK: Modelo guardado en models/"
	@echo "    ensemble_stacking.joblib / ensemble_scaler.joblib / ensemble_meta.json"

train-cnn:
	@echo "Entrenando CNN baseline..."
	$(PYTHON) src/models/cnn/train_cnn.py
	@echo "OK: Modelo guardado en models/best_pixel_cnn.pt"

train-vit:
	@echo "Entrenando ViT for SITS (preset=$(VIT_PRESET))..."
	$(PYTHON) src/models/vit/train_vit.py \
		--preset      $(VIT_PRESET) \
		--dataset-dir data/datasets/ \
		--output-dir  models/ \
		--experiment  avocado-stress-vit
	@echo "OK: Modelo guardado en models/best_sits_vit.pt"

prepare-sam2:
	$(PYTHON) src/models/sam2/prepare_sam2_dataset.py \
		--patches-dir data/datasets/patches \
		--norm-path data/datasets/normalizer_stats.json \
		--output-dir data/sam2
	@echo "OK: SAM2 dataset exported to data/sam2"

download-sam2-checkpoint:
	$(PYTHON) -c "from pathlib import Path; import urllib.request; p=Path('$(SAM2_CHECKPOINT)'); p.parent.mkdir(parents=True, exist_ok=True); print('Downloading $(SAM2_CHECKPOINT_URL) ->', p); urllib.request.urlretrieve('$(SAM2_CHECKPOINT_URL)', p)"

train-sam2:
	$(PYTHON) src/models/sam2/train_sam2_finetune.py \
		--manifest data/sam2/manifest.csv \
		--checkpoint $(SAM2_CHECKPOINT) \
		--model-cfg $(SAM2_MODEL_CFG) \
		--device $(SAM2_DEVICE) \
		--output models/sam2_avocado_finetuned.pt
	@echo "OK: SAM2 fine-tuned weights saved in models/sam2_avocado_finetuned.pt"

# ------------------------------------------------------------
# Server / dashboard
# ------------------------------------------------------------

# Desarrollo - recarga automatica al cambiar codigo
dev:
	@echo "Iniciando API en modo desarrollo..."
	@echo "   Dashboard -> http://$(HOST):$(PORT)/ui"
	@echo "   Swagger   -> http://$(HOST):$(PORT)/docs"
	@echo ""
	uvicorn api.main:app --host $(HOST) --port $(PORT) --reload

# Produccion - sin reload, varios workers
serve:
	@echo "Iniciando API en modo produccion..."
	@echo "   Dashboard -> http://$(HOST):$(PORT)/ui"
	uvicorn api.main:app --host $(HOST) --port $(PORT) --workers 2

# ------------------------------------------------------------
# Development
# ------------------------------------------------------------
notebook:
	@echo "Abriendo Jupyter Lab..."
	jupyter lab notebooks/

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	ruff check src/ api/ tests/

# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "OK: __pycache__ y .pyc eliminados"

clean-cache: clean
	rm -rf .pytest_cache .ruff_cache
	@echo "OK: Caches de herramientas eliminados"
