.PHONY: setup download indices ts train evaluate report all clean

# ── Configuración ─────────────────────────────────────────────────────────────
CFG       = configs/base.yaml
CRED      = configs/credentials.yaml
PARCEL_ID ?= parcela_001

# ── Instalación ───────────────────────────────────────────────────────────────
setup:
	pip install -r requirements.txt

# ── Pipeline completo ─────────────────────────────────────────────────────────
all: download indices ts train evaluate report

# ── Descarga de imágenes Sentinel Hub ─────────────────────────────────────────
download:
	python -m src.ingestion.downloader

download-test:
	python -c "from src.ingestion.downloader import run_download; run_download(parcel_ids=['$(PARCEL_ID)'])"

# ── Índices espectrales (ya calculados en el evalscript) ──────────────────────
# Los índices se calculan en el Process API (evalscript) durante la descarga.
# Este paso convierte los GeoTIFF en series temporales CSV/parquet.
indices: ts

# ── Series temporales ─────────────────────────────────────────────────────────
ts:
	python -m src.processing.time_series
	python -m src.processing.windows

# ── Entrenamiento LSTM Autoencoder ────────────────────────────────────────────
train:
	python -m src.models.lstm_autoencoder

train-fast:
	python -c "\
	import yaml; from pathlib import Path; \
	cfg = yaml.safe_load(Path('$(CFG)').read_text()); \
	cfg['model']['epochs'] = 10; \
	import tempfile, os; \
	tf = tempfile.NamedTemporaryFile(suffix='.yaml', delete=False, mode='w'); \
	yaml.dump(cfg, tf); tf.close(); \
	from src.models.lstm_autoencoder import train; train(tf.name)"

# ── Evaluación: genera scores de anomalía ─────────────────────────────────────
evaluate:
	python -c "from src.models.lstm_autoencoder import train; print('Scores generados en data/models/lstm_ae/scores.parquet')"

# ── Reportes con Claude multimodal ────────────────────────────────────────────
report:
	python -m src.reporting.llm_report

report-parcel:
	python -c "from src.reporting.llm_report import run_reports; run_reports(parcel_ids=['$(PARCEL_ID)'])"

# ── Limpieza ──────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

clean-data:
	rm -rf data/raw/sentinel2/* data/processed/time_series/* \
	       data/processed/windows/* data/models/ reports/
	@echo "Datos limpios."
