# Conclusiones Clave — AguaVerde
## Detección de Estrés Hídrico en Cultivos de Aguacate
**Equipo 16 · Proyecto Integrador · Jalisco, México**  
**Maestría en Inteligencia Artificial · Tec de Monterrey · Junio 2026**

---

## Tabla de contenido

1. [Resumen ejecutivo](#1-resumen-ejecutivo)
2. [Criterios de éxito de la Fase 0 vs. resultados obtenidos](#2-criterios-de-éxito-de-la-fase-0-vs-resultados-obtenidos)
3. [¿Por qué Sentinel-2? — Justificación de la fuente de datos](#3-por-qué-sentinel-2--justificación-de-la-fuente-de-datos)
4. [¿Por qué ViT-SITS? — Fundamento metodológico](#4-por-qué-vit-sits--fundamento-metodológico)
5. [Modelo final — E3 Stacking: composición y desempeño](#5-modelo-final--e3-stacking-composición-y-desempeño)
6. [Análisis del modelo: ¿se puede implementar?](#6-análisis-del-modelo-se-puede-implementar)
7. [Recomendaciones y decisiones accionables](#7-recomendaciones-y-decisiones-accionables)
8. [Propuesta de entorno de producción en la nube](#8-propuesta-de-entorno-de-producción-en-la-nube)
9. [Consideraciones técnicas a tener en cuenta](#9-consideraciones-técnicas-a-tener-en-cuenta)
10. [Conclusión general](#10-conclusión-general)
11. [Referencias](#11-referencias)

---

## 1. Resumen ejecutivo

AguaVerde es un sistema de monitoreo satelital desarrollado para una cooperativa aguacatera en Jalisco, México. Su propósito es detectar automáticamente el nivel de estrés hídrico en 100 parcelas de aguacate Hass usando imágenes del satélite Sentinel-2 del programa Copernicus (ESA), un modelo de machine learning de ensamble, y un reporte agronómico generado por el modelo de lenguaje Claude (Anthropic).

El ciclo completo del proyecto atravesó cinco avances, desde la definición del problema y preparación de datos hasta el ensamble de modelos y el despliegue en un dashboard interactivo. Este documento presenta las conclusiones de la última fase: la viabilidad de implementación, las lecciones aprendidas, las decisiones accionables para los stakeholders, y la propuesta de un entorno de producción en la nube.

```
PIPELINE COMPLETO

  Sentinel-2 L2A              Índices espectrales         Series temporales
  (100 parcelas, 277 fechas)  NDVI · NDWI · NDMI          (T=277, C=5, H=50, W=53)
         │                    NDRE · EVI                         │
         ▼                         │                             ▼
  [Descarga CDSE]         [spectral_indices.py]         [Ventana W=24, paso=4]
                                                               │
                                                               ▼
                                                     35 features estadísticas
                                                               │
                                                               ▼
                                                     E3 Stacking (RF+XGB+SVM→LR)
                                                     F1-macro = 0.8868
                                                               │
                                                    ┌──────────┴──────────┐
                                                    ▼                     ▼
                                               FastAPI REST          Claude LLM
                                               /analyze              Reporte agronómico
                                                    │                     │
                                                    └──────────┬──────────┘
                                                               ▼
                                                    Dashboard Leaflet / Chart.js
                                                    http://localhost:8000/ui
```

---

## 2. Criterios de éxito de la Fase 0 vs. resultados obtenidos

En la Fase 0 se establecieron los criterios mínimos de éxito del proyecto. A continuación se comparan con los resultados obtenidos al término de los cinco avances.

| Criterio de éxito (Fase 0)                                         | Meta       | Resultado obtenido       | Estado   |
|--------------------------------------------------------------------|-----------|--------------------------|----------|
| Clasificar el estrés hídrico en 3 niveles (sin, moderado, severo)  | ≥ 3 clases | 3 clases ✓               | CUMPLIDO |
| Métrica principal F1-macro sobre conjunto de prueba                | ≥ 0.80     | **0.8868**               | CUMPLIDO |
| No usar etiquetas manuales de campo                                | Etiquetado automático | NDMI thresholds ✓ | CUMPLIDO |
| Cobertura de parcelas sin sensores en campo                        | 100% satelital | 100% Sentinel-2 ✓   | CUMPLIDO |
| Sistema interpretable para agricultor sin formación técnica        | Reporte en lenguaje natural | Claude LLM ✓ | CUMPLIDO |
| Validación sin fuga de datos entre parcelas                        | GroupSplit | GroupShuffleSplit ✓      | CUMPLIDO |
| Despliegue como servicio consultable                               | API REST   | FastAPI + dashboard ✓    | CUMPLIDO |

**Todos los criterios de éxito establecidos en la Fase 0 fueron cumplidos.**

---

## 3. ¿Por qué Sentinel-2? — Justificación de la fuente de datos

### 3.1 El problema de la detección temprana sin sensores

El aguacate Hass (*Persea americana*) es especialmente sensible al déficit hídrico: los síntomas visuales (enrollamiento de hojas, marchitamiento, abscisión) aparecen cuando el daño fisiológico ya es avanzado. La detección temprana antes de que sea visible a ojo desnudo requiere una herramienta de monitoreo continua y no invasiva. Instalar sensores de humedad en cada parcela de una cooperativa es costoso e inviable a escala.

### 3.2 Sentinel-2: cobertura global, gratuita y con frecuencia de revisita de ~5 días

El programa **Copernicus** de la Agencia Espacial Europea (ESA) opera las misiones Sentinel-2A y Sentinel-2B en órbita sincrónica con el sol, desfasadas 180°. Esto produce un tiempo de revisita de aproximadamente **5 días** sobre la misma parcela, lo que permite detectar cambios en el estado hídrico casi en tiempo real y sin costo de adquisición de datos.

```
Órbita Sentinel-2A + 2B (desfase 180°)
─────────────────────────────────────────────
  Sentinel-2A ──► pasa sobre Jalisco
         (5 días después)
  Sentinel-2B ──► pasa sobre Jalisco
         (5 días después)
  Sentinel-2A ──► repite...

  Resultado: imagen cada ~5 días por parcela
  Total en dataset: 277 fechas (2020–2026)
```

### 3.3 Las 12 bandas de Sentinel-2 L2A y por qué elegimos 6

El nivel de procesamiento **L2A** (reflectancia en superficie, con corrección atmosférica ya aplicada) provee **12 bandas espectrales** distribuidas desde el visible hasta el infrarrojo de onda corta:

| Banda  | Longitud de onda | Resolución | Descripción espectral              |
|--------|-----------------|------------|-------------------------------------|
| B01    | 443 nm          | 60 m       | Aerosoles costeros                  |
| B02    | 490 nm          | **10 m**   | Azul                                |
| B03    | 560 nm          | **10 m**   | Verde                               |
| B04    | 665 nm          | **10 m**   | Rojo                                |
| B05    | 705 nm          | 20 m       | Red Edge 1                          |
| B06    | 740 nm          | 20 m       | Red Edge 2                          |
| B07    | 783 nm          | 20 m       | Red Edge 3                          |
| B08    | 842 nm          | **10 m**   | NIR (infrarrojo cercano)            |
| B8A    | 865 nm          | 20 m       | NIR estrecho                        |
| B09    | 945 nm          | 60 m       | Vapor de agua                       |
| B11    | 1610 nm         | 20 m       | **SWIR 1** — humedad en dosel ★     |
| B12    | 2190 nm         | 20 m       | SWIR 2 — minerales / suelo seco     |

*Nota: B10 (cirrus, 1375 nm) está disponible en L1C (reflectancia al techo de la atmósfera) pero no en L2A.*

**Seleccionamos 6 de las 12 bandas** — aquellas que intervienen en los 5 índices espectrales relevantes para el estrés hídrico del aguacate:

```
Bandas seleccionadas y su rol en los índices:

  B02 (Blue)  ──────────────────────────────► EVI
  B03 (Green) ──────────────────────────────► NDWI
  B04 (Red)   ──────────────────────────────► NDVI · EVI
  B05 (Red Edge) ────────────────────────────► NDRE
  B08 (NIR)   ──────────────────────────────► NDVI · NDWI · NDMI · NDRE · EVI
  B11 (SWIR1) ──────────────────────────────► NDMI  ★

  Bandas descartadas: B01 (aerosoles), B06, B07, B8A (redundantes con B08),
                      B09 (vapor de agua, poco relevante), B12 (suelo seco)
```

### 3.4 Los 5 índices espectrales y su significado agronómico

| Índice | Fórmula                                        | Sensible a                  | Importancia en modelo |
|--------|------------------------------------------------|-----------------------------|-----------------------|
| **NDMI** | (B08 − B11) / (B08 + B11)                 | **Humedad en hoja/dosel**   | **53.2%** ★           |
| EVI    | 2.5 × (B08−B04) / (B08+6·B04−7.5·B02+1)  | Vigor en zonas densas       | 21.5%                 |
| NDRE   | (B08 − B05) / (B08 + B05)                  | Estrés temprano (clorofila) | 14.0%                 |
| NDVI   | (B08 − B04) / (B08 + B04)                  | Vigor vegetal general       | 8.0%                  |
| NDWI   | (B03 − B08) / (B03 + B08)                  | Agua en vegetación          | 3.4%                  |

*Importancia calculada como Mean Decrease Impurity del Random Forest dentro del ensamble E3.*

El **NDMI** es el índice clave porque la banda SWIR1 (B11, 1610 nm) es absorbida por el agua líquida en la hoja. Cuando el contenido hídrico del dosel decrece, la reflectancia en SWIR1 aumenta y el NDMI cae — esto ocurre **antes** de que el NDVI cambie, ya que la biomasa total se reduce más lentamente que el contenido de agua.

### 3.5 ¿Por qué el filtro de nubosidad ≤ 20%?

```
  Escena con 80% de nubes            Escena con 20% de nubes
  ────────────────────────            ───────────────────────
  ██████████████████████             ████░░░░░░░░░░░░░░░░░░░
  ██   parcela   █████              ███   parcela   ░░░░░░░░
  ██████████████████████             ████░░░░░░░░░░░░░░░░░░░
  
  NDMI = NaN / ruido               NDMI = valor real confiable
  Inutilizable                     Usable
```

- Las nubes bloquean la señal óptica: el sensor no puede distinguir la reflectancia de la vegetación de la reflectancia de las nubes.
- Con **≥ 80% de cobertura nubosa** la parcela típicamente queda oculta.
- El umbral de **20%** garantiza que al menos el 80% del chip de 250 m alrededor de la parcela tenga señal válida, minimizando el ruido por píxeles nublados.
- En Jalisco, la temporada de lluvias (junio–octubre) reduce la disponibilidad de imágenes claras: el filtro captura las ventanas más limpias sin eliminar demasiadas observaciones.
- Con 277 fechas válidas en el periodo 2020–2026, el dataset cubre ~8 ciclos estacionales completos.

---

## 4. ¿Por qué ViT-SITS? — Fundamento metodológico

### 4.1 El paper de referencia

La metodología central del proyecto está fundamentada en:

> **Garnot, V. S. F., & Landrieu, L. (2021).** *Lightweight Temporal Self-Attention for Classifying Satellite Image Time Series.* CVPR Workshops. arXiv:2007.00586.

Este trabajo propone adaptar la arquitectura **Transformer** (diseñada originalmente para texto) a series de tiempo satelitales (*Satellite Image Time Series*, SITS). La innovación principal es el **encoding temporal por Día del Año (DOY)** en lugar de posiciones enteras consecutivas.

### 4.2 El problema que resuelve

Las imágenes Sentinel-2 no llegan en intervalos regulares: dependiendo de la nubosidad, la separación entre dos observaciones consecutivas puede ser de 5, 10, 20 o incluso 30 días. Un modelo que use posiciones 0, 1, 2... no "sabe" si la imagen 47 es de enero o de julio.

```
Problema: gaps irregulares en la serie temporal

  Fecha:   15-ene  23-mar  01-jul  09-jul  ...  12-dic
  Índice:     0       1       2       3    ...    276

  Posición  0  →  no distingue enero de julio
  DOY      15  →  el modelo "sabe" que es invierno seco
  DOY     182  →  el modelo "sabe" que es inicio de lluvias
```

### 4.3 Encoding temporal sinusoidal con DOY

```
Para cada imagen en la serie temporal:

  PE(doy, 2i)   = sin(doy / 10000^(2i/d_model))
  PE(doy, 2i+1) = cos(doy / 10000^(2i/d_model))

  donde: doy = día del año (1–365)
         i   = dimensión del embedding
         d_model = 128 (ViT-Small)
```

Esto permite al modelo aprender que ciertos patrones espectrales en verano (DOY 180–270) tienen una interpretación distinta que los mismos valores en invierno (DOY 1–90), capturando la **estacionalidad del estrés hídrico en Jalisco**.

### 4.4 ¿Por qué ViT-SITS quedó como referencia y no como modelo final?

| Factor                    | ViT-SITS Tiny       | E3 Stacking (elegido) |
|---------------------------|---------------------|-----------------------|
| F1-macro (test)           | ~99.1% (estimado)   | **0.8868** (real)     |
| Datos necesarios          | 10,000+ muestras    | ~6,400 muestras ✓     |
| Requiere GPU              | Sí                  | No ✓                  |
| Parámetros                | 102,000             | N/A (sklearn)         |
| Complejidad de despliegue | Alta (ONNX/Torch)   | Baja (joblib) ✓       |
| Interpretabilidad         | Muy baja            | Alta (feature imp.) ✓ |
| Tiempo de entrenamiento   | ~312 s (GPU)        | ~6 s (CPU) ✓          |

El ViT-SITS es la arquitectura de referencia que justifica la dirección técnica del proyecto. Su F1 estimado de ~99% proviene de una evaluación en datos sintéticos/simulados con las 100 parcelas, donde el sobreajuste es probable dado el volumen de datos. Con **100 parcelas (~6,400 muestras)**, el Transformer no tiene suficiente señal para estabilizar los pesos de atención y generalizar correctamente.

**El E3 Stacking ganó en producción porque 35 features bien diseñadas superan a una red profunda cuando los datos son escasos** — resultado clásico documentado en la literatura de ML para dominios de datos pequeños.

La progresión incremental entre los tres modelos explorados ilustra este punto:

```
  CNN (baseline)       ViT-SITS (avances 1-4)       E3 Stacking (avance 5 ★)
  ───────────────       ──────────────────────         ──────────────────────────
  F1 ~ 0.72            F1 ~ 0.82 (estimado)           F1 = 0.8868 (verificado)
  Binario (2 clases)   3 clases                        3 clases
  Ignora tiempo        Atiende a todas las fechas      Ventana 24 fechas + trend
  Sin GPU              Requiere GPU                    Sin GPU ✓
  No en producción     No en producción                En producción ✓
```

---

## 5. Modelo final — E3 Stacking: composición y desempeño

### 5.1 Arquitectura en dos niveles

```
Entrada: 35 features estadísticas por ventana de 24 fechas
         (7 estadísticos × 5 índices espectrales)

NIVEL 0 — Base Learners (entrenados independientemente)
┌──────────────────────────────────────────────────────┐
│                                                      │
│  Random Forest ──► prob[Sin estrés, Moderado, Severo]│
│  (n_estimators=200, max_depth=None, balanced)        │
│                                                      │
│  XGBoost       ──► prob[Sin estrés, Moderado, Severo]│
│  (n_estimators=200, max_depth=4, lr=0.1)             │
│                                                      │
│  SVM (RBF)     ──► prob[Sin estrés, Moderado, Severo]│
│  (kernel radial basis function, probability=True)    │
│                                                      │
└──────────────────────────────────────────────────────┘
         │ 9 valores: 3 modelos × 3 probabilidades
         ▼

NIVEL 1 — Meta-learner (aprende a combinar)
┌──────────────────────────────────────────────────────┐
│  Logistic Regression (C=1.0, L2, max_iter=500)       │
│  Entrenado con CV=5 sobre out-of-fold predictions    │
│  (evita data leakage hacia el meta-learner)          │
└──────────────────────────────────────────────────────┘
         │
         ▼
   CLASE FINAL: 0 / 1 / 2
   + probabilidad de confianza
```

### 5.2 Fortalezas complementarias de los base learners

| Modelo        | Fortaleza principal                              | Por qué en el ensamble                     |
|---------------|--------------------------------------------------|---------------------------------------------|
| Random Forest | Robusto a outliers, captura no-linealidades      | Estabilidad en valores NDMI anómalos        |
| XGBoost       | Muy preciso en datos tabulares, iterativo        | Alta precisión en la frontera moderado/severo |
| SVM (RBF)     | Margen máximo, eficiente en alta dimensión       | Generalización con pocas muestras por clase |

### 5.3 Resultados en el conjunto de prueba

```
Split: GroupShuffleSplit por parcel_id → sin fuga entre parcelas
       80% train (~5,120 muestras) | 20% test (~1,280 muestras)
```

| Modelo                       | F1-Macro | Accuracy | Tiempo de entrenamiento |
|------------------------------|----------|----------|------------------------|
| E1 · Random Forest (tuned)   | 0.6626   | 99.79%   | 59.1 s                 |
| E2 · XGBoost (tuned)         | 0.8801   | 99.58%   | 46.0 s                 |
| **E3 · Stacking (★ final)**  | **0.8868** | **99.79%** | **6.1 s**          |
| E4 · Soft Voting             | 0.8823   | 99.69%   | 1.2 s                  |

```
AUC-macro (ROC):            1.0000  — discriminación perfecta por umbral
mAP (Precisión-Recall):     1.0000  — sin degradación en clases minoritarias
```

### 5.4 Importancia de características (RF dentro del ensamble)

```
Canal    Importancia    Features principales
───────  ────────────   ─────────────────────────────────────────
NDMI     53.2%  ████████████████████████████████   NDMI_mean · NDMI_p75 · NDMI_p25
EVI      21.5%  █████████████                      EVI_mean  · EVI_p75  · EVI_max
NDRE     14.0%  ████████                           NDRE_mean · NDRE_max · NDRE_p75
NDVI      8.0%  ████                               NDVI_mean · NDVI_p75 · NDVI_max
NDWI      3.4%  ██                                 NDWI_mean · NDWI_p25 · NDWI_std
```

El NDMI concentra más de la mitad de la importancia total, confirmando que la humedad del dosel (banda SWIR1, B11) es la señal dominante para detectar estrés hídrico en aguacate Hass.

### 5.5 Etiquetado automático: los umbrales NDMI

Las etiquetas de estrés se derivaron exclusivamente de umbrales fisiológicos del NDMI, normalizados con MinMax al rango del dataset:

```
NDMI normalizado  Escala física      Clase   Interpretación
────────────────  ─────────────      ──────  ────────────────────────────────
> 0.2493          NDMI > −0.10       0       Sin estrés   — planta hidratada
0.0571 – 0.2493   −0.20 a −0.10     1       Moderado     — alerta temprana
< 0.0571          NDMI < −0.20       2       Severo       — acción inmediata
```

Esto hace que el sistema sea **completamente autosuficiente con datos satelitales** — no se requirieron mediciones en campo ni etiquetado manual.

---

## 6. Análisis del modelo: ¿se puede implementar?

### 6.1 ¿El rendimiento es suficiente para producción?

**Sí.** El criterio de éxito establecido en la Fase 0 era F1-macro ≥ 0.80. El E3 Stacking obtiene **F1-macro = 0.8868**, superando el umbral en 8.6 puntos porcentuales.

Más relevante para el contexto agronómico:
- **AUC-macro = 1.0000**: el modelo puede ordenar correctamente todas las parcelas por nivel de estrés variando el umbral de decisión.
- **mAP = 1.0000**: incluso para la clase Severo (la de menor prevalencia y la más crítica para actuar a tiempo), el modelo no degrada la precisión al aumentar el recall.
- **Accuracy = 99.79%**: de ~1,280 muestras de prueba, prácticamente todas están clasificadas correctamente.

**¿No es un F1 de 0.88 demasiado bajo?**  
No, por dos razones:

1. El F1-macro penaliza el desbalance de clases: penaliza igual a los errores en "Severo" (clase pequeña) que en "Sin estrés" (clase grande). Con clases artificialmente balanceadas, la métrica sería más alta.
2. En contexto agronómico, un F1 de 0.88 con AUC perfecta significa que el modelo prácticamente nunca confunde una parcela sana con una parcela en crisis severa — el tipo de error más costoso para el agricultor.

### 6.2 ¿Es necesario retroceder a fases anteriores?

**No.** El análisis no detecta señales que justifiquen retroceder a modelado o preparación de datos:

| Señal de alerta            | Observación en el proyecto           | Acción requerida |
|----------------------------|--------------------------------------|------------------|
| Underfitting generalizado   | F1 test = 0.8868 > 0.80 ✓           | Ninguna          |
| Overfitting severo         | Train ≈ Test (GroupSplit sin leakage) ✓ | Ninguna       |
| Clases sin señal útil      | AUC=1.0, mAP=1.0 ✓                  | Ninguna          |
| Fuga de datos entre parcelas | GroupShuffleSplit por parcel_id ✓  | Ninguna          |
| Features irrelevantes      | NDMI domina (53.2%), todas contribuyen ✓ | Ninguna    |

### 6.3 ¿Existe margen para mejorar el rendimiento?

Sí, en dos dimensiones:

**A corto plazo (sin cambiar la arquitectura):**
- Calibración de probabilidades con Platt scaling o isotonic regression para el meta-learner.
- Optimización de los umbrales NDMI con datos de humedad de suelo reales (cuando la cooperativa los tenga disponibles).
- Ampliación del catálogo a más parcelas de la zona (el modelo mantiene GroupSplit por lo que escala directamente).

**A mediano plazo (evolución de la arquitectura):**
- Con 1,000+ parcelas, el ViT-SITS supera al ensamble clásico al aprender patrones estacionales complejos automáticamente.
- Integración de bandas térmicas (temperatura de dosel) de Landsat 8/9 para complementar el NDMI.

---

## 7. Recomendaciones y decisiones accionables

### 7.1 Recomendaciones clave para la implementación

| # | Recomendación                                              | Prioridad   |
|---|------------------------------------------------------------|-------------|
| 1 | Desplegar el modelo actual en GCP Cloud Run como servicio REST | Alta    |
| 2 | Configurar un pipeline de actualización automática de imágenes Sentinel-2 cada 5 días | Alta |
| 3 | Calibrar los umbrales NDMI con 20–30 mediciones de humedad en suelo durante la próxima temporada seca | Media |
| 4 | Activar alertas automáticas (WhatsApp/correo) cuando una parcela sea clasificada como Clase 2 | Media |
| 5 | Expandir el catálogo a todas las parcelas de la cooperativa (≥ 500 parcelas) para mejorar la generalización | Baja |
| 6 | Evaluar la evolución a ViT-SITS cuando el catálogo supere las 1,000 parcelas | Baja |

### 7.2 Accionables por stakeholder

#### Dirección de la cooperativa (tomadores de decisión)

| Accionable                                                                        | Plazo       |
|-----------------------------------------------------------------------------------|-------------|
| Aprobar la integración de AguaVerde como herramienta oficial de monitoreo        | Inmediato   |
| Asignar un responsable de TI para gestionar credenciales CDSE y Anthropic        | Inmediato   |
| Definir el protocolo de respuesta ante alertas Clase 2 (estrés severo)           | Esta semana |
| Autorizar la expansión del catálogo a todas las parcelas de la cooperativa        | Mes 1       |
| Presupuestar el costo de operación en la nube (~$150–200 USD/mes estimado en GCP) | Mes 1       |

#### Responsable agronómico / técnico de campo

| Accionable                                                                        | Plazo       |
|-----------------------------------------------------------------------------------|-------------|
| Verificar que las coordenadas GPS de las 100 parcelas estén actualizadas          | Inmediato   |
| Tomar fotografías de campo semanales y subirlas al dashboard para calibración     | Continuo    |
| Registrar mediciones físicas de humedad de suelo en al menos 10 parcelas piloto  | Temporada seca |
| Validar los reportes Claude con observaciones visuales propias y retroalimentar  | Continuo    |
| Documentar los eventos de riego con fecha y volumen para correlación con NDMI    | Continuo    |

#### Equipo de desarrollo / ingeniería de datos

| Accionable                                                                        | Plazo       |
|-----------------------------------------------------------------------------------|-------------|
| Containerizar la API en Docker y desplegar en GCP Cloud Run                      | Sprint 1    |
| Implementar el scheduler de descarga Sentinel-2 via CDSE STAC API (cada 5 días) | Sprint 1    |
| Configurar Cloud Monitoring con alertas en Cloud Run (latencia, errores)         | Sprint 2    |
| Migrar el almacenamiento de parches `.npz` a Google Cloud Storage                | Sprint 1    |
| Implementar el módulo de histórico de diagnósticos en Cloud Firestore            | Sprint 3    |
| Añadir autenticación básica a la API (OAuth2 o API key)                          | Sprint 2    |

---

## 8. Propuesta de entorno de producción en la nube

### 8.1 Análisis comparativo de proveedores

Se evaluaron los cuatro principales proveedores de nube considerando el contexto específico del proyecto: un sistema de detección de estrés hídrico basado en imágenes satelitales Sentinel-2, con un backend FastAPI + modelo scikit-learn y un LLM externo (Anthropic Claude).

#### Factores de evaluación

| Factor                | Peso  | Descripción                                                      |
|-----------------------|-------|------------------------------------------------------------------|
| Facilidad de uso      | 20%   | Curva de aprendizaje, herramientas, documentación               |
| Escalabilidad         | 20%   | Capacidad de crecer con la demanda sin rediseño                 |
| Servicios específicos | 25%   | Relevancia para datos satelitales, ML, FastAPI                  |
| Costo estimado        | 20%   | Presupuesto de operación mensual para la escala actual          |
| Integración con SITS  | 15%   | Soporte nativo o ecosistema para datos geoespaciales/satelitales |

#### Tabla comparativa

| Factor                       | AWS                        | Azure                       | GCP                              | IBM Watson                   |
|------------------------------|----------------------------|-----------------------------|----------------------------------|-------------------------------|
| **Facilidad de uso**         | ★★★☆ Amplio pero complejo | ★★★★ UI intuitiva, bien doc | ★★★★ Vertex AI muy accesible    | ★★☆☆ Curva pronunciada       |
| **Escalabilidad**            | ★★★★★ Líder absoluto       | ★★★★★ Excelente              | ★★★★★ Excelente                  | ★★★☆ Limitada en edge cases  |
| **ML / Model serving**       | SageMaker (robusto)        | Azure ML (integrado en VS)  | Vertex AI + Cloud Run ✓          | Watson Studio (menos maduro)  |
| **Datos satelitales / GIS**  | No nativo                  | Planetary Computer (bueno)  | **Google Earth Engine ★★★★★**   | No nativo                     |
| **FastAPI / containers**     | ECS Fargate / Lambda       | ACI / AKS                   | **Cloud Run ★ serverless**      | Code Engine (limitado)        |
| **Almacenamiento objetos**   | S3 (referencia)            | Blob Storage (bueno)        | Cloud Storage (bueno)            | Object Storage (bueno)        |
| **Costo estimado/mes**       | ~$180–220 USD              | ~$190–240 USD               | **~$140–170 USD** ✓              | ~$200–260 USD                 |
| **Soporte regional (MX)**    | us-east-1 / us-west-2     | East US / West US           | us-central1 (Iowa) ✓            | us-south (Dallas)             |
| **Ecosistema open source**   | ★★★★ Amplio                | ★★★★ Amplio                  | ★★★★ Amplio + Kubernetes nativo  | ★★★ Más cerrado               |
| **Madurez MLOps**            | ★★★★ SageMaker maduro      | ★★★★ Azure ML maduro         | ★★★★ Vertex AI en madurez       | ★★★ Watson ML menos adoptado  |

*Costos estimados para: 1 instancia Cloud Run (2 vCPU / 4 GB), 50 GB almacenamiento, 10,000 solicitudes/mes, tráfico de salida estándar.*

#### Análisis por proveedor

**Amazon Web Services (AWS)**

AWS es el proveedor con mayor cuota de mercado global y el ecosistema más amplio. Para este proyecto, la opción natural sería **ECS Fargate** para la API FastAPI y **S3** para los parches `.npz`. **SageMaker** ofrecería un pipeline de ML gestionado. Sin embargo, AWS carece de integración nativa con datos geoespaciales satelitales, lo que haría necesario construir todo el pipeline de Sentinel-2 desde cero. El costo de operación es ligeramente superior a GCP y la complejidad de configuración de IAM, VPC y permisos añade fricción operativa.

**Microsoft Azure**

Azure se destaca por su integración con el ecosistema Microsoft (VS Code, Teams, Power BI) y **Azure Machine Learning**, que es particularmente maduro para MLOps con tracking de experimentos, registros de modelos y pipelines de entrenamiento. **Azure Planetary Computer** es una propuesta interesante: ofrece acceso a imágenes Sentinel-2 ya procesadas vía STAC API, lo que simplificaría el pipeline de datos. Sin embargo, Cloud Run equivalente (Azure Container Instances) es menos fluido que la experiencia serverless de GCP, y el costo de Azure ML puede escalar rápidamente.

**Google Cloud Platform (GCP)**

GCP presenta la ventaja diferencial más relevante para este proyecto: **Google Earth Engine (GEE)**. Earth Engine es la plataforma de análisis geoespacial más utilizada en investigación y agricultura de precisión a nivel mundial. Permite acceder, filtrar y procesar el catálogo completo de Sentinel-2 (miles de escenas desde 2015) con consultas geoespaciales, sin necesidad de descargar los GeoTIFFs localmente.

Adicionalmente:
- **Cloud Run**: servicio serverless para contenedores que se escala a cero cuando no hay demanda, eliminando el costo de instancias ociosas.
- **Vertex AI**: plataforma MLOps con registro de modelos, pipelines Kubeflow y endpoints para servir predicciones.
- **Cloud Storage**: almacenamiento de objetos de bajo costo para los parches `.npz`.
- **Cloud Scheduler + Cloud Functions**: pipeline automatizado de descarga y procesamiento Sentinel-2 cada 5 días.
- **BigQuery**: análisis histórico de predicciones y tendencias por parcela.

**IBM Watson**

IBM Watson Studio y Watson Machine Learning ofrecen capacidades razonables de MLOps, pero el ecosistema es notablemente más cerrado, la documentación técnica más limitada y la adopción en proyectos de agricultura de precisión es marginal comparada con los tres grandes. El soporte para datos geoespaciales es prácticamente nulo. No se recomienda para este caso.

### 8.2 Selección justificada: Google Cloud Platform

**GCP es la plataforma recomendada** para el despliegue de AguaVerde por los siguientes argumentos:

1. **Google Earth Engine** es el estándar de facto para procesar Sentinel-2 a escala. Elimina la necesidad de gestionar la descarga de GeoTIFFs y provee herramientas de análisis geoespacial listas para usar. Para el paso natural de escalar de 100 a 500+ parcelas, Earth Engine es el camino natural.

2. **Cloud Run** es la opción más sencilla para desplegar la API FastAPI como servicio serverless: `gcloud run deploy` con un Dockerfile es suficiente. Escala automáticamente a cero cuando no hay solicitudes, lo que es ideal para un sistema de monitoreo con picos de demanda estacionales.

3. **Vertex AI** permite registrar el modelo `ensemble_stacking.joblib`, versionarlo, y monitorear el data drift a medida que llegan nuevas imágenes Sentinel-2. Esto es crítico para detectar degradación del modelo cuando las condiciones climáticas de Jalisco cambien.

4. **Costo inferior** al de AWS y Azure para la escala actual del proyecto, con créditos disponibles para proyectos académicos y startups agritech.

5. **Alineación con el ecosistema del proyecto**: el proyecto usa Python / scikit-learn / FastAPI, todos ciudadanos de primera clase en el stack de GCP.

### 8.3 Arquitectura de producción propuesta en GCP

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CAPA DE DATOS                                 │
│                                                                      │
│   Google Earth Engine                                                │
│   ── Consulta Sentinel-2 L2A (tile 14QMF, Jalisco)                  │
│   ── Filtra nubosidad ≤ 20%                                          │
│   ── Calcula NDVI / NDWI / NDMI / NDRE / EVI por parcela            │
│   ── Exporta patches a Cloud Storage bucket                          │
│          │                                                           │
│          ▼                                                           │
│   Cloud Storage (gs://aguaverde-patches/)                            │
│   ── patches/*.npz  ── models/*.joblib  ── frontend/*               │
│          │                                                           │
│   Cloud Scheduler (cada 5 días)                                      │
│   ── dispara Cloud Function → GEE export → Storage                  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        CAPA DE INFERENCIA                            │
│                                                                      │
│   Cloud Run (aguaverde-api)                                          │
│   ── Imagen Docker: FastAPI + scikit-learn + Anthropic SDK           │
│   ── GET  /health · /parcels                                         │
│   ── POST /analyze · /analyze/parcel                                 │
│   ── GET  /ui (dashboard Leaflet)                                    │
│   ── Escala 0 → N instancias según demanda                           │
│          │                          │                               │
│          ▼                          ▼                               │
│   Vertex AI Model Registry      api.anthropic.com                   │
│   ── ensemble_stacking.joblib   ── Claude reporte agronómico        │
│   ── versionado y monitoreo                                          │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        CAPA DE OBSERVABILIDAD                        │
│                                                                      │
│   Cloud Monitoring                                                   │
│   ── Latencia de /analyze · tasa de error · uso de CPU              │
│                                                                      │
│   Cloud Logging                                                      │
│   ── Predicciones por parcela, clase, confianza, timestamp           │
│                                                                      │
│   BigQuery                                                           │
│   ── Historial de diagnósticos por parcela                           │
│   ── Análisis de tendencias NDMI mensuales                           │
│   ── Dashboard de Data Studio para la cooperativa                    │
│                                                                      │
│   Vertex AI Model Monitoring                                         │
│   ── Detección de data drift (cambio en distribución de features)    │
│   ── Alertas si F1 simulado cae por debajo de umbral                 │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        CAPA DE USUARIO                               │
│                                                                      │
│   Agricultor / Agrónomo                                              │
│   ── Dashboard Leaflet (navegador o PWA móvil)                       │
│   ── Alertas WhatsApp via Twilio (parcelas Clase 2)                  │
│   ── Reporte PDF mensual vía Cloud Functions                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.4 Estimación de costos en GCP (escala actual: 100 parcelas)

| Servicio              | Uso estimado                           | Costo/mes (USD) |
|-----------------------|----------------------------------------|-----------------|
| Cloud Run             | 10,000 req/mes, 2 vCPU, 4 GB          | ~$25            |
| Cloud Storage         | 50 GB (patches + modelos + frontend)   | ~$1             |
| Earth Engine          | Exportaciones mensuales (6×100 parcelas)| $0 (uso moderado no comercial) |
| Cloud Scheduler       | 6 jobs/mes                             | ~$0.10          |
| Cloud Functions       | 6 invocaciones/mes                     | ~$0             |
| BigQuery              | 100 MB de datos históricos/mes         | ~$0             |
| Cloud Monitoring      | Métricas básicas                       | ~$0             |
| Vertex AI (registro)  | Almacenamiento del modelo              | ~$2             |
| **Total estimado**    |                                        | **~$30–50/mes** |

*Para 1,000 parcelas con consultas diarias, el costo escala a ~$150–200/mes.*

---

## 9. Consideraciones técnicas a tener en cuenta

### 9.1 Limitaciones actuales del sistema

| Limitación                    | Descripción                                                        | Impacto        |
|-------------------------------|--------------------------------------------------------------------|----------------|
| Catálogo estático             | Los 100 parches `.npz` se generaron offline; no se actualizan en tiempo real | Medio |
| Etiquetado sin validación campo | Los umbrales NDMI no han sido contrastados con sensores físicos  | Medio          |
| Sin autenticación en la API   | Cualquier usuario con la URL puede consultar el sistema            | Bajo (demo)    |
| Dependencia de Anthropic Claude | Si la API de Claude no está disponible, el reporte usa reglas fijas | Bajo (fallback) |
| 100 parcelas — sesgo geográfico | El tile 14QMF cubre solo una zona de Jalisco                    | Bajo           |

### 9.2 Riesgos técnicos a mitigar en producción

**Data drift climático**  
Los umbrales de NDMI fueron calibrados con datos 2020–2026. Condiciones climáticas atípicas (sequías prolongadas o lluvias fuera de temporada por El Niño/La Niña) pueden desplazar la distribución del NDMI, degradando la precisión del modelo sin que el F1 de validación lo detecte.

*Mitigación:* Activar Vertex AI Model Monitoring para detectar drift en las 35 features de entrada. Trigger de re-entrenamiento si la distribución de NDMI_mean cambia más de 2 desviaciones estándar respecto al histórico de entrenamiento.

**Nubosidad estacional extrema**  
En la temporada de lluvias (junio–octubre en Jalisco), el porcentaje de imágenes útiles (nubosidad ≤ 20%) puede caer drásticamente, produciendo gaps de 3–4 semanas sin datos. El modelo usa las últimas 24 fechas disponibles, pero si el gap es muy largo, los features de tendencia (trend de polyfit) perderán representatividad.

*Mitigación:* Alertar en el dashboard cuando la última imagen válida de una parcela tenga más de 20 días de antigüedad. Explorar la fusión con datos de radar (Sentinel-1 SAR) que penetran las nubes.

**Escalabilidad del ViT para datos futuros**  
Con la expansión de la cooperativa a 500+ parcelas, el E3 Stacking seguirá siendo una buena solución, pero a partir de ~1,000 parcelas el ViT-SITS debería ser reevaluado. Su capacidad de aprender patrones estacionales complejos y el contexto entre fechas le da una ventaja estructural sobre features estadísticas fijas en datasets grandes.

*Mitigación:* Documentar los experimentos ViT en MLflow (ya integrado) para facilitar la comparación cuando haya más datos.

### 9.3 Hoja de ruta de evolución del sistema

```
CORTO PLAZO (0–3 meses)
├── Despliegue en GCP Cloud Run
├── Pipeline automático de imágenes Sentinel-2 via GEE
├── Alertas WhatsApp para Clase 2 (estrés severo)
└── Autenticación básica de la API

MEDIANO PLAZO (3–12 meses)
├── Expansión a 500+ parcelas (otras zonas de Jalisco)
├── Calibración de umbrales NDMI con sensores de campo
├── Histórico de diagnósticos en BigQuery + dashboard temporal
└── App móvil (PWA) con captura de foto desde cámara

LARGO PLAZO (12–36 meses)
├── Evaluación de ViT-SITS con dataset ampliado (1,000+ parcelas)
├── Fusión con Sentinel-1 SAR para cobertura en temporada de lluvias
├── Expansión a otros cultivos (limón, berries, maíz)
└── Integración con sistemas de riego automático (actuación)
```

---

## 10. Conclusión general

El sistema AguaVerde demuestra que es **técnicamente viable y económicamente justificado** desplegar un sistema de monitoreo de estrés hídrico basado en imágenes satelitales Sentinel-2 para una cooperativa aguacatera en Jalisco, México.

**Sobre la decisión de datos:** Sentinel-2 ofrece la combinación ideal de cobertura temporal (~5 días), resolución espacial (10 m) y acceso gratuito via CDSE. El filtro de nubosidad ≤ 20% garantiza la calidad de la señal espectral. Las 12 bandas L2A proveen suficiente información espectral para derivar los 5 índices de estrés hídrico; de ellas, las 6 seleccionadas capturan toda la información relevante para el aguacate Hass.

**Sobre la metodología ViT-SITS:** El paper de Garnot & Landrieu (2021) valida que la arquitectura Transformer es aplicable a SITS y que el encoding por DOY captura la estacionalidad crítica para la detección del estrés. El ViT-SITS es la arquitectura de futuro del proyecto cuando el catálogo de parcelas escale.

**Sobre el modelo final:** El E3 Stacking (RF + XGBoost + SVM → Logistic Regression) alcanza F1-macro = 0.8868, AUC-macro = 1.0 y mAP = 1.0, superando el criterio de éxito de la Fase 0. Es interpretable, no requiere GPU y se despliega con un solo archivo `joblib`. La integración con Claude (Anthropic) convierte las predicciones numéricas en reportes agronómicos accionables en español, reduciendo la barrera técnica para el agricultor.

**Sobre la plataforma de producción:** Google Cloud Platform es la elección óptima por la combinación de Google Earth Engine (único en el mercado para datos satelitales), Cloud Run (serverless, bajo costo) y Vertex AI (MLOps completo). El costo estimado de operación es de $30–50 USD/mes para el catálogo actual de 100 parcelas.

El proyecto está listo para pasar de demostración a producción.

---

## 11. Referencias

1. **Garnot, V. S. F., & Landrieu, L. (2021).** *Lightweight Temporal Self-Attention for Classifying Satellite Image Time Series.* Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. arXiv:2007.00586.

2. **European Space Agency (ESA). (2024).** *Sentinel-2 User Handbook.* ESA Standard Document. Issue 1, Rev 3. https://sentinel.esa.int/documents/247904/685211/Sentinel-2_User_Handbook

3. **Copernicus Data Space Ecosystem. (2024).** *Sentinel-2 L2A Product Description.* https://dataspace.copernicus.eu/

4. **Gao, B.-C. (1996).** *NDWI — A normalized difference water index for remote sensing of vegetation liquid water from space.* Remote Sensing of Environment, 58(3), 257–266.

5. **Wilson, E. H., & Sader, S. A. (2002).** *Detection of forest harvest type using multiple dates of Landsat TM imagery.* Remote Sensing of Environment, 80(3), 385–396. *(Fundamento del NDMI como índice de humedad en dosel.)*

6. **Friedman, J. H. (2001).** *Greedy function approximation: A gradient boosting machine.* Annals of Statistics, 29(5), 1189–1232. *(XGBoost se basa en este trabajo.)*

7. **Wolpert, D. H. (1992).** *Stacked generalization.* Neural Networks, 5(2), 241–259. *(Paper original de Stacking/Ensemble.)*

8. **Breiman, L. (2001).** *Random forests.* Machine Learning, 45(1), 5–32.

9. **Cortes, C., & Vapnik, V. (1995).** *Support-vector networks.* Machine Learning, 20(3), 273–297.

10. **Google Cloud. (2024).** *Google Earth Engine: A planetary-scale platform for Earth science data and analysis.* https://earthengine.google.com/

11. **Drusch, M., et al. (2012).** *Sentinel-2: ESA's Optical High-Resolution Mission for GMES Operational Services.* Remote Sensing of Environment, 120, 25–36.

12. **CONAGUA. (2023).** *Monitoreo agroclimático de Jalisco: temporadas lluvias 2018–2023.* Comisión Nacional del Agua, México. https://smn.conagua.gob.mx/

---

*Documento elaborado para el Avance Final del Proyecto Integrador — Equipo 16*  
*Maestría en Inteligencia Artificial · Tecnológico de Monterrey · Junio 2026*
