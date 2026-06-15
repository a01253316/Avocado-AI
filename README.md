# AguaVerde - Deteccion de Estres Hidrico en Aguacate

Sistema de monitoreo satelital para parcelas de aguacate en Jalisco. Combina datos Sentinel-2, modelos de ensemble y un reporte agronomico generado con un LLM local mediante Ollama.

## Resumen

El sistema permite:

1. Cargar parcelas georreferenciadas en un mapa.
2. Leer parches satelitales `.npz` por parcela.
3. Extraer indices espectrales y features de la ventana mas reciente.
4. Predecir estres hidrico con el modelo E3 Stacking.
5. Generar un reporte agronomico usando Ollama/OpenLLaMA u otro modelo local.
6. Visualizar diagnosticos, tendencias y segmentacion pixelada en un dashboard servido por FastAPI.

## Stack

| Capa | Tecnologia |
| --- | --- |
| Backend | FastAPI, Pydantic Settings, Uvicorn |
| Modelo ML | scikit-learn, XGBoost, joblib |
| LLM local | Ollama (`OLLAMA_MODEL`) |
| Modelo LLM recomendado | `openllama`; `llama3.2:3b` como alternativa ligera para PC sencilla |
| Frontend | HTML, CSS, Vanilla JS, Leaflet, Chart.js |
| Datos | Sentinel-2 / parches `.npz` por parcela |

## Datos y modelos

El flujo normal del proyecto genera los datos preprocesados y los modelos desde el codigo del repositorio. Primero se extraen las parcelas, despues se descargan y procesan las imagenes Sentinel-2, luego se construye el dataset SITS y finalmente se entrena el modelo principal.

```powershell
make extract-parcels
make download-sentinel2
make compute-indices
make build-dataset
make train-ensemble
```

Ese proceso crea los artefactos que usa el dashboard:

```text
data/datasets/
  patches/*.npz
  normalizer_stats.json

models/
  ensemble_stacking.joblib
  ensemble_scaler.joblib
  ensemble_meta.json
```

## Preparar entorno

Conda recomendado:

```powershell
conda create -n avocado-ai python=3.12 -y
conda activate avocado-ai
```

Instala dependencias del backend:

```powershell
python -m pip install -r requirements-api.txt
```

## Configurar Ollama / OpenLLaMA

Este proyecto usa Ollama como servidor local de LLM. No necesita API key para el reporte agronomico.

Instala Ollama y descarga un modelo:

```powershell
ollama pull openllama
```

Tambien puedes usar otra opcion ligera si tu PC lo necesita:

```powershell
ollama pull llama3.2:3b
```

Crea `.env` a partir de `.env.example`:

```powershell
copy .env.example .env
```

Configuracion minima:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=openllama
```

Si quieres usar el modelo ligero, cambia solo el modelo:

```env
OLLAMA_MODEL=llama3.2:3b
```

En `.env.example` hay varias opciones comentadas. Deja solo una linea `OLLAMA_MODEL=` sin comentar.

## Ejecutar dashboard

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

O, si tu `make` usa el mismo entorno `avocado-ai`:

```powershell
make dev
```

Abre:

```text
http://127.0.0.1:8000/ui
```

Swagger/API docs:

```text
http://127.0.0.1:8000/docs
```

## Funcionamiento del dashboard

El dashboard tiene tres vistas principales:

- `Diagnostico`: muestra el resultado de una parcela, indices espectrales, tendencia NDMI, probabilidades y reporte agronomico.
- `SAM2`: muestra una capa raster pixel por pixel sobre el mapa. Esta version usa el NDMI de los patches Sentinel-2 como preview de segmentacion; no es fine-tuning SAM2 real todavia.
- `Nueva ubicacion`: permite analizar coordenadas y, opcionalmente, una imagen de campo.

El boton `Escanear mapa` analiza las parcelas pendientes que esten visibles en la pantalla actual. El alcance depende del zoom y del area visible del mapa; acercate para analizar menos parcelas o alejate para cubrir mas.

En la pestana `SAM2`:

- `Analizar todo` genera mascaras pixeladas para las parcelas cargadas.
- Los filtros `Sin estres`, `Moderado`, `Severo` y `Sin mascara` solo cambian la visibilidad en el mapa.
- La lista `Parcelas activas` permite centrar el mapa en una parcela sin salir de la pestana.
- Las mascaras se calculan con `data/datasets/patches/<parcel_id>.npz` y `data/datasets/normalizer_stats.json`.

## Pipeline completo

Para generar el dataset desde cero:

```powershell
make extract-parcels
make download-sentinel2
make compute-indices
make build-dataset
```

Despues entrena el modelo principal:

```powershell
make train-ensemble
```

Flujo recomendado completo:

```powershell
conda activate avocado-ai
python -m pip install -r requirements-api.txt
ollama pull openllama
make extract-parcels
make download-sentinel2
make compute-indices
make build-dataset
make train-ensemble
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

## API

| Metodo | Ruta | Descripcion |
| --- | --- | --- |
| GET | `/health` | Estado del backend |
| GET | `/parcels` | Lista parcelas |
| POST | `/analyze` | Analiza coordenadas GPS |
| POST | `/analyze/parcel` | Analiza una parcela por ID |
| GET | `/sam2/mask/{parcel_id}` | Devuelve mascara pixelada por parcela |
| GET | `/ui` | Dashboard |
| GET | `/docs` | Swagger UI |

Ejemplo `POST /analyze/parcel`:

```json
{
  "parcel_id": "H40",
  "skip_llm": false
}
```

Respuesta resumida:

```json
{
  "location": { "parcel_id": "H40", "state": "Jalisco" },
  "stress": { "class": 1, "label": "Moderado", "confidence": 0.94 },
  "indices": { "NDMI": 0.1823, "NDVI": 0.6102 },
  "llm_report": { "model_used": "openllama", "fallback": false }
}
```

## Problemas comunes

### `No se encontro el parche .npz`

Falta el dataset preprocesado:

```text
data/datasets/patches/<parcel_id>.npz
```

Genera el dataset con el pipeline:

```powershell
make extract-parcels
make download-sentinel2
make compute-indices
make build-dataset
```

### `No se pudo conectar a Ollama`

Confirma que Ollama esta instalado y que el modelo existe:

```powershell
ollama list
ollama pull openllama
```

### El dashboard carga mapa pero no analiza parcelas

El CSV de parcelas esta disponible, pero faltan los `.npz` en `data/datasets/patches/`. Genera el dataset con `make build-dataset` despues de preparar los indices.

## Referencias

- ESA Sentinel-2 User Guide
- Ollama: https://ollama.com
- scikit-learn model persistence: https://scikit-learn.org/stable/model_persistence.html
