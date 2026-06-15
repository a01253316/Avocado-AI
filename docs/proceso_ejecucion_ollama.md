# Proceso de ejecucion con Ollama/OpenLLaMA

Esta guia describe el flujo normal para generar datos, entrenar el modelo y ejecutar el dashboard con un LLM local.

## 1. Entorno recomendado

El modelo usa dependencias recientes de scikit-learn. Usa Python 3.12:

```powershell
conda create -n avocado-ai python=3.12 -y
conda activate avocado-ai
python -m pip install -r requirements-api.txt
```

Verifica:

```powershell
python --version
python -c "import sklearn; print(sklearn.__version__)"
python -c "import xgboost; print(xgboost.__version__)"
```

## 2. Configurar modelo local

Ollama debe estar instalado y corriendo. Descarga el modelo local:

```powershell
ollama pull openllama
```

Si tu PC necesita una opcion mas ligera:

```powershell
ollama pull llama3.2:3b
```

En `.env`, deja solo un modelo activo:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=openllama
# OLLAMA_MODEL=llama3.2:3b
```

## 3. Generar datos

El dataset se genera desde el pipeline del proyecto:

```powershell
make extract-parcels
make download-sentinel2
make compute-indices
make build-dataset
```

Estos pasos crean los parches `.npz` y estadisticas necesarias para la inferencia:

```text
data/datasets/
  patches/*.npz
  normalizer_stats.json
```

## 4. Entrenar modelo

Entrena el modelo principal E3 Stacking:

```powershell
make train-ensemble
```

El entrenamiento genera:

```text
models/
  ensemble_stacking.joblib
  ensemble_scaler.joblib
  ensemble_meta.json
```

## 5. Ejecutar dashboard

Desde el entorno `avocado-ai`:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Abre:

```text
http://127.0.0.1:8000/ui
```

## 6. Cuando usar Makefile para dev

Tambien puedes usar:

```powershell
make dev
```

En Windows, si `make` apunta a otro Python, usa el comando explicito:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
