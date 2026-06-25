# Resumen de Experimentos D y E — AguaVerde
## Gaussian Process + Red Neuronal Probabilística para estrés hídrico

**Equipo 16 · Proyecto Integrador · Maestría en IA · Tec de Monterrey · Junio 2026**

---

## Tabla de contenido

1. [¿Por qué añadimos estos experimentos?](#1-por-qué-añadimos-estos-experimentos)
2. [Arquitectura general del sistema (3 capas)](#2-arquitectura-general-del-sistema-3-capas)
3. [Experimento D — Gaussian Process](#3-experimento-d--gaussian-process)
4. [Experimento E — Red Neuronal Probabilística](#4-experimento-e--red-neuronal-probabilística)
5. [Cómo leer el dashboard](#5-cómo-leer-el-dashboard)
6. [Cómo interpretar los resultados](#6-cómo-interpretar-los-resultados)
7. [Cómo reproducir / entrenar los modelos](#7-cómo-reproducir--entrenar-los-modelos)
8. [Limitaciones que hay que declarar](#8-limitaciones-que-hay-que-declarar)

---

## 1. ¿Por qué añadimos estos experimentos?

El modelo productivo del dashboard (**E3 Stacking**, F1-macro = 0.8868) clasifica cada
parcela en Sin estrés / Estrés moderado / Severo. Funciona bien.

El problema que señaló el **profesor asesor en la reunión del 24 de junio de 2026** es otro:
los umbrales que usamos para *etiquetar* los datos de entrenamiento son arbitrarios.

```
Etiquetado actual (api/features.py):
  NDMI normalizado > 0.249  →  Sin estrés
  NDMI normalizado < 0.057  →  Estrés severo

Problema: ¿por qué exactamente esos números?
  - Son percentiles sobre el dataset global.
  - El mismo NDMI de 0.10 puede ser "normal" para una parcela seca
    y "señal de alarma" para una parcela siempre húmeda.
  - No hay una distribución ajustada; son cortes fijos sin fundamento
    estadístico por parcela.
```

La propuesta del asesor: ajustar una **distribución** por parcela (o grupo de parcelas)
mediante *maximum likelihood* y clasificar por cuántas desviaciones estándar cae la
última observación respecto a lo esperado. Eso es exactamente lo que hacen los Experimentos D y E.

Ni el E ni el D **reemplazan** el E3 Stacking — son capas adicionales de evidencia que
justifican y complementan el diagnóstico oficial.

---

## 2. Arquitectura general del sistema (3 capas)

```
┌─────────────────────────────────────────────────────────────────────┐
│  CAPA 1 — DIAGNÓSTICO OFICIAL (pestaña "📍 Diagnóstico")            │
│                                                                     │
│  E3 Stacking  (RF + XGBoost + SVM → Logistic Regression)           │
│  Entrada: 35 features estadísticas · Ventana: 24 fechas            │
│  Salida:  clase (0/1/2) + probabilidades + reporte Claude          │
│  Métrica: F1-macro = 0.8868 · AUC-macro = 1.000                   │
│  ✅ Modelo en producción — NO se modificó                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  CAPA 2 — ALERTA POR DESVIACIÓN (pestaña "📈 Tendencias")           │
│                                                                     │
│  Experimento D — Gaussian Process                                   │
│  Ajusta una curva esperada para CADA parcela (o grupo de parcelas  │
│  de terreno similar) usando su historial de 277 fechas.            │
│  Clasifica por z-score: cuántas σ cae la última observación.       │
│  Útil cuando el E3 dice "sin estrés" pero la parcela se está       │
│  desviando de sus vecinas → alerta temprana.                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  CAPA 3 — DISTRIBUCIÓN APRENDIDA (pestaña "📈 Tendencias" / abajo)  │
│                                                                     │
│  Experimento E — StressLikelihoodNet (Transformer probabilístico)  │
│  Aprende μ y σ para los 5 índices condicionado a:                  │
│    · historial reciente (últimas 24 fechas)                        │
│    · huella espectral de la parcela (tipo de terreno proxy)        │
│    · día del año (estacionalidad)                                  │
│  Clasifica por CDF multivariada — sin umbral fijo.                │
└─────────────────────────────────────────────────────────────────────┘
```

**Flujo para la presentación:**
> "El E3 nos da el diagnóstico oficial validado (F1 = 0.8868). El GP nos muestra si la
> parcela se desvía de su propio historial o del de sus vecinas de terreno similar —
> alerta temprana. La red neuronal va un paso más allá: aprende qué distribución de
> índices es 'normal' para cada tipo de parcela en cada época del año, sin umbrales
> fijos. Las tres capas son complementarias, no competidoras."

---

## 3. Experimento D — Gaussian Process

### 3.1 Qué hace

Para cada parcela (o grupo de parcelas de terreno similar), el GP ajusta una función
de regresión sobre su serie histórica del índice elegido (NDMI por defecto). La función
ajustada da, para cualquier fecha, una **media esperada μ(t)** y una **incertidumbre
σ(t)**. Con eso podemos preguntar: ¿la última observación está dentro del rango normal
o está inusualmente baja?

```
Serie histórica NDMI (parcela H1, 277 fechas):
      ▲
  0.4 │          ···•···
  0.3 │       ··         ···
  0.2 │     ·                ···
  0.1 │   ··                    ·····
  0.0 │                               ●  ← observación reciente
 -0.1 │
      └──────────────────────────────────► tiempo
              ↑ Banda GP (±2σ en gris)

  Si ● está dentro de la banda → normal
  Si ● está debajo de −1σ → moderado
  Si ● está debajo de −2σ → severo
```

### 3.2 Dos modos: Individual y Grupo

**Individual**: el GP se entrena solo con el historial de esa parcela.
Útil para detectar anomalías respecto a su propio pasado.

**Grupo (terreno similar)**: agrupa las 100 parcelas en 4 grupos por similitud
de huella espectral (K-Means sobre la media histórica de 5 índices).
Un solo GP se ajusta sobre el promedio del grupo. Esto da más datos por modelo
y detecta cuando *varias parcelas* del mismo tipo caen juntas — algo que el GP
individual no puede ver.

| Grupo | Parcelas | NDMI medio | Interpretación |
|-------|----------|------------|----------------|
| 0     | 45       | +0.005     | Vegetación media |
| 1     | 23       | +0.087     | Parcelas más húmedas |
| 2     | 6        | +0.202     | Parcelas muy húmedas (riego frecuente) |
| 3     | 26       | −0.073     | Parcelas más secas |

**Resultado clave**: parcelas H1, H16, H50 — el GP individual las clasifica como "sin
estrés", pero el GP de grupo las marca como "moderado" (z > 1.5). Es decir: esas
parcelas están cayendo junto con todas sus vecinas de terreno similar, una señal de
estrés colectivo que la vista individual no detecta.

### 3.3 Cómo leer el z-score

```
z-score = (observado − μ_esperado) / σ_esperado

z > 0  → sobre lo esperado (más húmedo que lo normal)    ✅
z < 0  → bajo lo esperado  (más seco que lo normal)      ⚠️

Umbrales de clasificación:
  |z| < 1.0   →  Sin estrés   (dentro de ±1σ, 68% del tiempo normal)
  |z| ≥ 1.0   →  Moderado     (entre 1σ y 2σ por debajo)
  |z| ≥ 2.0   →  Severo       (más de 2σ por debajo)
```

A diferencia de un percentil global, estos cortes **tienen interpretación probabilística**:
caer 2σ por debajo de lo esperado para esa fecha y esa parcela es un evento que,
bajo distribución normal, ocurre el 2.3% de las veces — genuinamente anómalo.

### 3.4 Archivos relevantes

| Archivo | Propósito |
|---------|-----------|
| `src/models/gp/parcel_gp.py` | GP individual por parcela |
| `src/models/gp/terrain_groups.py` | Agrupamiento K-Means por huella espectral |
| `src/models/gp/group_gp.py` | GP de grupo + comparación individual/grupo |
| `api/trend.py` | Endpoint `GET /parcels/{id}/trend` |

---

## 4. Experimento E — Red Neuronal Probabilística

### 4.1 La idea central

El GP modela un solo índice a la vez. La red del Experimento E modela los **5 índices
simultáneamente**, condicionando la distribución esperada a tres fuentes de información:

1. **Historial reciente**: las últimas 24 fechas de los 5 índices (ventana deslizante)
2. **Huella estática de la parcela**: la media histórica completa de sus 5 índices
   (proxy del tipo de terreno — qué tan húmeda/seca es normalmente esta parcela)
3. **Día del año (DOY)**: para que el modelo sepa si estamos en temporada seca o lluvias

Con esas entradas, la red produce para cada uno de los 5 índices:
- **μ** (mu): el valor que se esperaría en condiciones normales para esta parcela, en esta época, dado su historial reciente
- **σ** (sigma): la incertidumbre de esa estimación

La función de pérdida que entrena la red es la **Negative Log-Likelihood Gaussiana**,
que es exactamente lo que el profesor pedía como "maximum likelihood":

```
L(μ, σ, y) = log(σ) + 0.5 × ((y − μ) / σ)²

Minimizar L equivale a encontrar los parámetros (μ, σ) que maximizan
la probabilidad de haber observado el valor y bajo una distribución N(μ, σ²).
```

### 4.2 Arquitectura del modelo

```
ENTRADA
  x_hist   (24, 5)  — ventana de historial (24 fechas × 5 índices)
  x_static (5,)     — huella estática de la parcela
  doy      (1,)     — día del año (1–365)

PROCESAMIENTO
  ┌─────────────────────────────────────────────────────────┐
  │  DOY Encoder                                            │
  │  [sin(2π·doy/365), cos(2π·doy/365),                   │
  │   sin(2π·doy/30.4), cos(2π·doy/30.4)]  →  Linear(4→32)│
  └────────────────────┬────────────────────────────────────┘
                       │ d_date = 32
  ┌────────────────────▼────────────────────────────────────┐
  │  Transformer Encoder                                    │
  │  Input projection: Linear(5 → 64)                       │
  │  Positional encoding aprendido (T=24)                   │
  │  2 capas × 4 cabezas de atención                        │
  │  → token [CLS] extraído → d_model = 64                 │
  └────────────────────┬────────────────────────────────────┘
                       │
  ┌────────────────────▼────────────────────────────────────┐
  │  Static Projection                                      │
  │  Linear(5 → 32)  →  d_static = 32                      │
  └────────────────────┬────────────────────────────────────┘
                       │
  ┌────────────────────▼────────────────────────────────────┐
  │  Fusion                                                 │
  │  Concat [CLS, static, date] → (64 + 32 + 32) = 128    │
  │  LayerNorm → Linear(128 → 64) → ReLU                   │
  └────────────────────┬────────────────────────────────────┘
                       │
  ┌────────────────────▼────────────────────────────────────┐
  │  Cabezas de salida                                      │
  │  mu_head:    Linear(64 → 5) → μ por índice             │
  │  sigma_head: Linear(64 → 5) → softplus + 1e-4 → σ > 0  │
  └─────────────────────────────────────────────────────────┘

SALIDA
  mu    (5,)  — valor esperado por índice
  sigma (5,)  — incertidumbre por índice (siempre positiva)
```

### 4.3 Clasificación por CDF multivariada

Una vez que tenemos (μ, σ) por índice, calculamos el z-score de la observación:

```
z_i = (y_observado_i − μ_i) / σ_i    para i ∈ {NDVI, NDWI, NDMI, NDRE, EVI}
```

Un z negativo significa que la observación está por debajo de lo esperado → señal de
estrés. Para combinar los 5 índices en una sola señal se usan pesos agronómicos:

```
Pesos de los índices:
  NDMI  40%  — índice dominante (humedad en dosel)
  NDWI  20%  — agua en vegetación
  NDVI  15%  — vigor general
  NDRE  15%  — estrés temprano (clorofila)
  EVI   10%  — vigor en zonas densas

señal_estrés = −Σ(peso_i × z_i)   ← negativo porque z bajo = estrés

percentil = Φ(señal_estrés) × 100   ← Φ es la CDF de N(0,1)
```

Los cortes de clasificación:

| Percentil CDF | Clase | Significado |
|---|---|---|
| < 25% | Sin estrés | Los índices están dentro o sobre lo esperado |
| 25% – 60% | Estrés moderado | Caída estadísticamente notable — atención |
| ≥ 60% | Estrés severo | Caída fuertemente anómala — acción inmediata |

**Por qué percentil y no z-score directo**: el percentil normaliza la señal entre 0 y
100%, lo que es más intuitivo para el dashboard y permite comparar parcelas en cualquier
época del año.

### 4.4 Resultado del entrenamiento

```
Dataset: 100 parcelas × 277 fechas × ventana T=24 → ~25,300 pares (x, y)
Split: 70% train / 15% validación / 15% test  (por parcela — sin fuga)

Entrenamiento: 60 épocas · AdamW · LR=3e-4 · cosine schedule · clip_norm=1.0

Mejor val NLL:  −2.7703
(NLL negativo = modelo bien calibrado: σ no es ni demasiado grande ni demasiado
pequeño. Un modelo trivial que pone σ=∞ da NLL ≈ 0. Un modelo perfecto da NLL → −∞)
```

### 4.5 Archivos relevantes

| Archivo | Propósito |
|---------|-----------|
| `src/models/likelihood_nn/stress_likelihood_net.py` | Arquitectura `StressLikelihoodNet` |
| `src/models/likelihood_nn/likelihood_dataset.py` | Dataset de ventanas deslizantes |
| `src/models/likelihood_nn/train_likelihood.py` | Loop de entrenamiento |
| `scripts/train_save_likelihood_nn.py` | Entry point CLI |
| `api/likelihood_predictor.py` | Inferencia + clasificación CDF |
| `api/main.py` | Endpoint `GET /parcels/{id}/likelihood` |
| `models/stress_likelihood_net.pt` | Pesos entrenados (generados por `make train-likelihood`) |
| `notebooks/03_experimento_e_likelihood_nn.ipynb` | Notebook documentado del experimento |

---

## 5. Cómo leer el dashboard

### Pestaña "📍 Diagnóstico"

- **Badge de clase** (verde/amarillo/rojo): resultado del E3 Stacking
- **Barra de confianza**: probabilidad que el meta-learner asignó a la clase ganadora
- **Índices espectrales**: valores de las últimas observaciones Sentinel-2
- **Tendencia NDMI**: mini-chart de los últimos ~50 valores de NDMI
- **Distribución de probabilidades**: probabilidad para cada una de las 3 clases
- **Reporte agronómico**: texto generado por Claude explicando el diagnóstico

### Pestaña "📈 Tendencias"

Esta pestaña tiene **dos secciones independientes**:

#### Sección superior — Experimento D (Gaussian Process)

1. Elige una parcela en el selector "Parcela"
2. Elige el índice espectral (NDMI por defecto)
3. Pulsa **🔮 Calcular tendencia**

Obtienes:
- **Gráfica GP**: la serie histórica (puntos azules), la media estimada (línea continua),
  la banda de confianza ±2σ (área sombreada). Las fechas recientes aparecen a la derecha.
- **Toggle Individual / Grupo**: cambia entre el GP ajustado solo sobre esta parcela
  vs. el GP del grupo de parcelas con terreno similar.
- **Cuadro "Última observación"**: fecha, valor observado, z-score, y clase resultante.
- **Pronóstico**: valor estimado para las próximas fechas.
- **Recomendación IA**: puedes subir una foto de campo y obtener un reporte de Claude
  que combina la señal del GP con la imagen.

#### Sección inferior — Experimento E (Red Neuronal)

Aparece tan pronto como eliges una parcela en el selector (no necesitas calcular el GP primero).

1. Pulsa **⚡ Calcular verosimilitud**

Obtienes:
- **Badge de clase** (verde/amarillo/rojo): resultado del Experimento E
- **Barra de percentil CDF**: qué percentil de la distribución multivariada ocupa la
  observación actual (eje 0–100%)
- **Gráfica de z-scores**: barras verticales, una por índice; negativo = por debajo de
  lo esperado. Líneas punteadas en ±1σ y ±2σ para referencia visual.
- **Tabla μ ± σ vs. observado**: para cada uno de los 5 índices, qué esperaba la red
  vs. qué se observó, con el z-score resultante. La fila de NDMI aparece en verde
  porque es el índice de mayor peso.

---

## 6. Cómo interpretar los resultados

### 6.1 El E3 dice "sin estrés" pero el GP dice "moderado" — ¿cuál creer?

No son contradictorios. El E3 clasifica con base en los valores *actuales* de los
índices comparados con los de entrenamiento (etiquetados por umbral global de NDMI).
El GP responde a la pregunta: *¿la última observación es anómala respecto a lo que era
esperable para esta parcela, en esta fecha, dado su historial?*

**Caso concreto:** H1, H16, H50 en el dataset actual.
El E3 dice "sin estrés" (los valores de NDMI están sobre 0.249). El GP de grupo dice
"moderado" (z > 1.5 respecto al grupo). Esto significa: todas las parcelas del mismo
tipo de terreno están cayendo juntas — patrón de estrés colectivo que el E3 todavía no
captura porque los valores absolutos aún no cruzaron el umbral de etiquetado.

**Recomendación de uso:**
- Si E3 = Sin estrés y GP = Sin estrés → parcela saludable
- Si E3 = Sin estrés y GP = Moderado → **prestar atención** (alerta temprana)
- Si E3 = Moderado/Severo → acción independientemente del GP

### 6.2 El Experimento E dice percentil 72% — ¿qué significa?

Un percentil de 72% significa que la señal de estrés combinada de los 5 índices
(ponderada por los pesos agronómicos) supera el 72% de la distribución esperada para
esa parcela en esa fecha. Dicho de otra forma: la observación actual es estadísticamente
más estresante que el 72% de los escenarios posibles que el modelo considera "normales"
para esta parcela en este día del año.

```
Percentil  Clase        Interpretación práctica
──────────────────────────────────────────────────
 0%– 24%   Sin estrés   Los índices están donde deberían estar
25%– 59%   Moderado     Hay algo inusual — revisar en campo próximamente
60%–100%   Severo       La combinación de índices es fuertemente anómala
```

### 6.3 ¿Qué me dice cada fila de la tabla del Experimento E?

La tabla tiene 5 filas (una por índice). Para cada una:

| Columna | Qué significa |
|---------|---------------|
| **Observado** | El valor del índice en la última imagen Sentinel-2 disponible (normalizado 0–1) |
| **μ esperado** | Lo que la red neuronal predice que debería valer ese índice, dado el historial + terreno + fecha actual |
| **±σ** | La incertidumbre de esa predicción. Un σ grande = el modelo no está seguro (poca señal en el historial para esta época) |
| **z-score** | (observado − μ) / σ. Negativo = más bajo de lo esperado. Cercano a 0 = normal |

**Colores del z-score:**
- Verde: |z| < 1.0 — dentro del rango normal
- Azul: |z| > 1.0 positivo — por encima de lo normal (más húmedo de lo esperado)
- Naranja: z < −1.0 — señal de estrés moderada
- Rojo: z < −2.0 — señal de estrés severa

**La fila NDMI aparece con fondo verde** porque es el índice de mayor peso (40%) en
la señal de estrés combinada.

### 6.4 ¿El NLL de −2.77 es bueno?

Sí. La función de pérdida NLL (Negative Log-Likelihood) mide qué tan bien calibradas
están las estimaciones μ y σ:

```
Un modelo que predice σ = ∞ (no sé nada):    NLL ≈ 0
Un modelo trivial (μ = media, σ = std):       NLL ≈ −1 a −2
Nuestro modelo entrenado (60 épocas):         NLL = −2.77 ✓

NLL más negativa = predicciones más confiadas y acertadas a la vez.
```

La validación cruzada por parcelas asegura que el NLL no es por memorizar los datos
de entrenamiento — el modelo vio parcelas distintas en train y test.

### 6.5 ¿Qué hacer si el botón dice "Modelo no entrenado"?

El archivo `models/stress_likelihood_net.pt` no existe todavía. Solo hay que ejecutar:

```bash
make train-likelihood
# equivalente a:
# python scripts/train_save_likelihood_nn.py \
#     --signals-dir data/datasets/signals \
#     --split-json  data/datasets/split.json \
#     --output-dir  models/ \
#     --epochs 60
```

El entrenamiento tarda ~3–5 minutos en CPU. Una vez generado el archivo `.pt`, el
endpoint `/parcels/{id}/likelihood` queda disponible automáticamente (carga el modelo
con `@lru_cache` la primera vez que se llama).

---

## 7. Cómo reproducir / entrenar los modelos

### Experimento D (Gaussian Process)

```bash
# Verificar agrupamiento K-Means (justifica k=4)
python -m src.models.gp.terrain_groups --scan-k

# Generar los grupos (guarda terrain_groups.json)
python -m src.models.gp.terrain_groups --n-groups 4

# Explorar GP individual de una parcela (con holdout de 6 fechas)
python -m src.models.gp.parcel_gp --parcel-ids H16 --holdout 6 --plot

# Comparar GP individual vs. grupo para una parcela
python -m src.models.gp.group_gp --parcel-id H16 --plot
```

El GP **no necesita entrenamiento offline**: se ajusta on-demand en el endpoint
`/parcels/{id}/trend`. Tarda ~2–4 segundos por parcela (primero carga el historial,
luego corre scipy.optimize para maximizar el log-likelihood marginal).

### Experimento E (Red Neuronal Probabilística)

```bash
# Entrenar y guardar (genera models/stress_likelihood_net.pt)
make train-likelihood

# Con parámetros personalizados
python scripts/train_save_likelihood_nn.py \
    --signals-dir data/datasets/signals \
    --split-json  data/datasets/split.json \
    --output-dir  models/ \
    --epochs 100 \
    --d-model 128 \
    --n-heads 4 \
    --n-layers 3

# Ver el notebook documentado
jupyter notebook notebooks/03_experimento_e_likelihood_nn.ipynb
```

### Iniciar el servidor completo

```bash
make dev
# o: uvicorn api.main:app --reload --port 8000
# Dashboard: http://localhost:8000/ui
```

---

## 8. Limitaciones que hay que declarar

Estas limitaciones deben mencionarse en la presentación — es mejor declararlas que
esperar que el profesor las señale:

### Experimento D

1. **Sin dato real de suelo o altitud**: el agrupamiento por "terreno similar" usa la
   huella espectral histórica como proxy, no una clasificación edafológica real. Las
   coordenadas `altitude_m` en `parcelas.csv` están en 0.0 para las 100 parcelas.

2. **Los umbrales 1σ/2σ siguen siendo una elección**: son más principistas que un
   percentil global (salen de una distribución ajustada), pero no son "los únicos
   correctos" — son la convención estadística estándar.

3. **El GP de grupo promedia sin ponderar**: si una parcela tiene más varianza que
   las demás del grupo, contribuye igual. Un modelo jerárquico bayesiano sería más
   correcto para trabajo futuro.

### Experimento E

4. **Sin los 10 canales raw de Sentinel-2**: solo tenemos los 5 índices calculados en
   los archivos `.npz`. El asesor propuso usar los canales raw (B02, B03, B04, B05,
   B08, B11, etc.) — con eso la red podría aprender correlaciones espectrales más
   ricas. La implementación actual usa la media histórica de 5 índices como proxy de
   terreno. Para producción, los canales raw son el camino.

5. **100 parcelas = dataset pequeño para un Transformer**: la ventana de T=24 con
   stride=1 genera ~25,300 muestras, pero todas vienen de las mismas 100 parcelas y
   las mismas 277 fechas. El modelo puede estar sobreajustando los patrones estacionales
   específicos de este tile Sentinel-2 (14QMF, Jalisco). Con 500+ parcelas sería
   mucho más robusto.

### Consideración general

Ninguno de los dos experimentos **reemplaza el E3 Stacking** como clasificador oficial.
La F1-macro del E3 (0.8868) fue validada con GroupShuffleSplit por parcela (sin fuga).
Los Experimentos D y E responden a una pregunta diferente: ¿hay fundamento estadístico
para los umbrales? — y la respuesta es sí, pero calculado de forma distribucional, no
por percentil global fijo.

---

## Apéndice — Comparación de los tres enfoques

| Aspecto | E3 Stacking | Exp. D — GP | Exp. E — NN Probabilística |
|---------|------------|-------------|---------------------------|
| **Modelo** | RF + XGB + SVM + LR | Gaussian Process sklearn | Transformer + NLL loss |
| **Entrada** | 35 features estadísticas | Serie temporal NDMI (o índice elegido) | Ventana 24×5 + huella estática + DOY |
| **Salida** | Clase + probabilidades | z-score + clase + pronóstico | μ, σ por índice + percentil CDF |
| **Umbrales** | Percentil NDMI global fijo | 1σ / 2σ de distribución ajustada | CDF multivariada: 25% / 60% |
| **Fundamento estadístico** | Etiquetado por regla heurística | Maximum likelihood (scikit-learn GP) | Maximum likelihood (Gaussian NLL) |
| **Requiere entrenamiento offline** | Sí (`make train`) | No (on-demand en el endpoint) | Sí (`make train-likelihood`) |
| **Tiempo de inferencia** | < 10 ms | 2–4 s (GP fit) | < 5 ms (carga modelo) |
| **Multi-índice** | Sí (35 features de 5 índices) | No (un índice a la vez) | Sí (5 índices simultáneos) |
| **Valor para la presentación** | Modelo productivo validado | Justificación estadística de umbrales | Fundamento matemático (MLE) propuesto por el asesor |

---

*Documento preparado para el equipo del Proyecto Integrador AguaVerde*
*Maestría en Inteligencia Artificial · Tecnológico de Monterrey · 24 de junio de 2026*
