# Proceso de ejecucion con Ollama/OpenLLaMA

Esta guia describe el flujo normal para generar datos, entrenar el modelo y ejecutar el dashboard con un LLM local.

## 1. Entorno recomendado

El modelo usa dependencias recientes de scikit-learn. Usa Python 3.12:

```powershell
conda create -n avocado-ai python=3.12 -y
conda activate avocado-ai
python -m pip install -r requirements-api.txt
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

## 6. Usar el mapa y SAM2

En el dashboard:

- `Escanear mapa` analiza las parcelas pendientes dentro de la pantalla actual. Usa el zoom para controlar cuantas parcelas entran en el escaneo.
- `Diagnostico` muestra el resultado agronomico de una parcela.
- `SAM2` muestra una vista pixel por pixel basada en los patches Sentinel-2.
- `Analizar todo` dentro de `SAM2` genera las mascaras raster para las parcelas cargadas.
- `Ajustar overlap` compacta mas los bounds y promedia las clases calibradas cuando varias mascaras caen sobre el mismo pixel.
- Los filtros de `Pixeles visibles` ocultan o muestran grupos de pixeles (`Sin estres`, `Moderado`, `Severo`) sin recalcular el modelo.
- Los filtros de `Diagnostico Sentinel` controlan que parcelas aportan mascaras al raster segun su clase de analisis.
- Los filtros de `Coordenadas visibles` controlan que puntos del mapa se muestran por clase Sentinel, sin afectar el raster.
- `Sin mascara` muestra u oculta las parcelas pendientes sin raster generado.
- En la pestana `SAM2`, dar clic en un marcador oculta o muestra la mascara de esa parcela.
- La lista `Parcelas activas` tambien oculta o muestra la mascara de una parcela sin cambiar a la pestana de diagnostico.
- Las mascaras se calibran con el diagnostico de cada parcela para que el mapa completo sea proporcional al analisis por coordenada; por ejemplo, una parcela `Sin estres` no pinta rojo dominante.
- La mascara se muestra con un footprint compacto alrededor de la coordenada porque los bounds de Sentinel-2 corresponden al chip de datos, no al poligono real del huerto.
- Con `Ajustar overlap` activo, los pixeles solapados se resuelven por mayoria de votos entre mascaras calibradas, no por prioridad de severidad.

La version actual del dashboard usa una preview basada en NDMI por pixel. El fine-tuning de SAM2 se prepara con pseudo-mascaras generadas desde esos patches.

## 7. Fine-tuning SAM2

Para preparar el dataset de SAM2 desde los patches:

```powershell
make prepare-sam2
```

Instala las dependencias opcionales de SAM2:

```powershell
python -m pip install -r requirements-sam2.txt
```

Descarga el checkpoint base SAM2.1 tiny fuera de git:

```powershell
make download-sam2-checkpoint
```

Entrena el decoder de mascaras:

```powershell
make train-sam2
```

Por default `SAM2_DEVICE=auto`: usa GPU si tu PyTorch tiene CUDA y si no cae a CPU. Para forzar CPU:

```powershell
make train-sam2 SAM2_DEVICE=cpu
```

El resultado se guarda como:

```text
models/sam2_avocado_finetuned.pt
```

## 8. Cuando usar Makefile para dev

Tambien puedes usar:

```powershell
make dev
```

En Windows, si `make` apunta a otro Python, usa el comando explicito:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
