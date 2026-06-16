# Fundamentos de Modelos — AguaVerde
## Progresión y justificación de las arquitecturas utilizadas
**Equipo 16 · Proyecto Integrador**

---

## ¿Por qué tres modelos?

En un proyecto de machine learning bien estructurado no se salta directamente al modelo más complejo. Se construye una **progresión incremental** donde cada modelo responde una pregunta específica y establece un punto de comparación claro para el siguiente.

```
Pregunta 1                Pregunta 2                 Pregunta 3
¿Es el problema           ¿Puede una red             ¿Podemos superar
resoluble con un          aprender el tiempo         la red con features
modelo simple?            correctamente?             diseñadas a mano?
      │                         │                          │
      ▼                         ▼                          ▼
  CNN Baseline            ViT for SITS               E3 Stacking
  (Avances 1-4)           (Avances 1-4)              (Avance 5 ★)
  2 clases                3 clases                   3 clases
  binario                 tri-clase                  tri-clase
  F1 ~0.72               F1 ~0.82                   F1 = 0.8868
```

---

## Modelo 1 — CNN (Baseline)

### ¿Qué es?

Una red neuronal convolucional que actúa como el punto de partida más simple posible. Está implementada en `src/models/cnn/pixel_classifier.py` con dos variantes:

- **PixelCNN**: trata cada píxel de la imagen de forma completamente independiente. Aplana toda la serie temporal en un vector y lo pasa por capas lineales.
- **PatchCNN**: trata el chip completo como una imagen multi-canal (arquitectura tipo U-Net). Preserva el contexto espacial entre píxeles vecinos.

### ¿Cómo funciona?

```
Entrada por píxel: (T=277 fechas × C=5 índices) → aplanado → vector de 1385 valores
                                  │
                    Linear(1385 → 256) → BN → ReLU → Dropout
                    Linear(256  → 128) → BN → ReLU → Dropout
                    Linear(128  → 64)  → BN → ReLU
                    Linear(64   → 1)   → logit binario
                                  │
                              Salida: 1 número (sano / estresado)
```

### Limitación clave: es binario

La salida del CNN es `nn.Linear(64, 1)` — **un solo logit**. Esto significa que solo puede clasificar entre **dos estados**: sano (0) o estresado (1). **No distingue estrés moderado de estrés severo**. Para la cooperativa esto no es suficiente: saber que hay estrés no dice cuánto riego aplicar ni con qué urgencia actuar.

### Otra limitación: ignora el tiempo

Al aplanar toda la serie temporal en un vector, el CNN **pierde completamente el orden cronológico**. No le importa si el NDMI bajó en enero o en julio, ni si la tendencia es creciente o decreciente. Trata todas las fechas como si fueran rasgos estáticos.

### ¿Para qué sirve entonces?

Sirve exactamente para lo que está diseñado: **establecer el piso de rendimiento**. Si un modelo posterior no supera al CNN, hay un problema en el diseño. También es muy rápido de entrenar y útil para hacer depuración del pipeline de datos.

---

## Modelo 2 — ViT for SITS (Baseline sofisticado)

### ¿Qué es?

Un **Vision Transformer adaptado para series de tiempo satelitales**. Implementado en `src/models/vit/sits_vit.py` siguiendo el paper:

> Garnot & Landrieu (2021) — *"Lightweight Temporal Self-Attention for Classifying Satellite Image Time Series"* — arXiv:2007.00586

La idea central del paper es que los Transformers, diseñados originalmente para texto (donde cada palabra presta atención a las demás), son perfectamente aplicables a series de tiempo satelitales, donde cada fecha "presta atención" a todas las demás fechas de la serie.

### ¿Cómo funciona?

```
Entrada: (B, T=277, C=5)  — serie temporal de 5 índices espectrales
         + doy: (B, T)    — día del año de cada imagen (1-365)
                │
    ┌───────────▼──────────────────────────────────────┐
    │  InputProjection:  Linear(C=5 → d_model=128)     │
    │  + LayerNorm                                     │
    └───────────┬──────────────────────────────────────┘
                │
    ┌───────────▼──────────────────────────────────────┐
    │  TemporalPositionalEncoding (DOY)                │
    │                                                  │
    │  PE(doy, 2i)   = sin(doy / 10000^(2i/d))        │
    │  PE(doy, 2i+1) = cos(doy / 10000^(2i/d))        │
    │                                                  │
    │  El modelo sabe si la imagen es de enero         │
    │  o de julio, aunque lleguen irregularmente.      │
    └───────────┬──────────────────────────────────────┘
                │
    ┌───────────▼──────────────────────────────────────┐
    │  TransformerEncoder × 4 capas                    │
    │                                                  │
    │  Para cada fecha:                                │
    │    "¿A qué otras fechas debo prestar atención    │
    │     para entender mi contexto?"                  │
    │                                                  │
    │  Enero ──►  ve que diciembre también fue seco    │
    │  Julio ──►  ve que la primavera fue húmeda       │
    │                                                  │
    │  MultiHeadAttention (8 cabezas)                  │
    │  + LayerNorm + Dropout + FFN                     │
    └───────────┬──────────────────────────────────────┘
                │
    ┌───────────▼──────────────────────────────────────┐
    │  MaskedTemporalPooling                           │
    │  Media de todos los embeddings (ignorando        │
    │  posiciones de padding si la serie es corta)     │
    └───────────┬──────────────────────────────────────┘
                │
    ┌───────────▼──────────────────────────────────────┐
    │  ClassificationHead                              │
    │  Linear(128 → 64) → GELU → Dropout              │
    │  Linear(64  → 3)  → 3 logits                    │
    └──────────────────────────────────────────────────┘
                │
            Salida: probas para [Sin estrés, Moderado, Severo]
```

### La innovación: encoding temporal con DOY

El Transformer estándar (BERT, GPT) usa posiciones 0, 1, 2, 3... para saber el orden de las palabras. Pero las imágenes Sentinel-2 **no llegan cada día** — pueden llegar cada 5, 8 o 12 días dependiendo de la nubosidad. Usar índices 0, 1, 2... no captura si una imagen es de invierno o de verano.

La solución del paper es usar el **Día del Año (DOY)** como posición:

```
Imagen del  15-ene → DOY = 15   → embedding sinusoidal de DOY=15
Imagen del  23-mar → DOY = 82   → embedding sinusoidal de DOY=82
Imagen del  01-jul → DOY = 182  → embedding sinusoidal de DOY=182
```

Así el modelo "sabe" que julio es temporada de lluvias en Jalisco y enero es más seco, aunque reciba las imágenes en cualquier orden y con gaps irregulares.

### Ya clasifica las 3 clases

A diferencia del CNN, el ViT produce **3 logits** (`nn.Linear(64, 3)`):

| Clase | Etiqueta     | Significado                        |
|-------|--------------|------------------------------------|
| 0     | Sin estrés   | NDMI alto, planta hidratada        |
| 1     | Moderado     | NDMI en zona de alerta             |
| 2     | Severo       | NDMI bajo, acción inmediata        |

### Tamaños disponibles

```python
Tiny  : d_model=64,  num_heads=4, num_layers=2  →  ~102K parámetros
Small : d_model=128, num_heads=8, num_layers=4  →  ~803K parámetros  ← default
Base  : d_model=256, num_heads=8, num_layers=6  →  ~4.8M parámetros
```

### ¿Por qué no quedó como modelo final?

Los Transformers son poderosos, pero **hambrientos de datos**. La atención necesita ver muchos ejemplos para aprender qué fechas son más informativas que otras. Con **100 parcelas** (~6,400 muestras), el ViT no tiene suficiente señal para estabilizar los pesos de atención. Con 10,000+ parcelas probablemente superaría al E3 Stacking.

---

## Modelo 3 — E3 Stacking (Modelo final en producción)

### ¿Qué es?

Un **ensamble de tres modelos clásicos de ML** cuyas predicciones son combinadas por un meta-modelo. No usa redes neuronales. En lugar de aprender features automáticamente, **nosotros diseñamos 35 features estadísticas** que capturan lo relevante de la serie temporal.

Implementado en `notebooks/Avance5.equipo16.ipynb`, serializado en `models/ensemble_stacking.joblib`.

### El insight clave: features diseñadas a mano vs. aprendidas

```
ViT (aprende solo):          E3 Stacking (features diseñadas):
─────────────────────         ─────────────────────────────────
Red ve los 277 valores        Humano extrae: ¿cuál fue la media
crudos y aprende              del NDMI en las últimas 24 fechas?
qué patrones importan.        ¿Bajó o subió? ¿Cuánto varió?
Necesita muchos datos.        35 números concretos y robustos.
```

### Extracción de features: ventana deslizante

```
Serie temporal (277 fechas × 5 índices)
│
│  Ventana W=24 fechas, paso=4
│
├── [fechas  0–23 ] → 35 features → etiqueta NDMI → muestra de entrenamiento
├── [fechas  4–27 ] → 35 features → etiqueta NDMI → muestra de entrenamiento
├── [fechas  8–31 ] → 35 features → etiqueta NDMI → muestra de entrenamiento
│   ...
└── [fechas 253–276] → 35 features → etiqueta NDMI → muestra de entrenamiento

100 parcelas × ~64 ventanas = ~6,400 muestras de entrenamiento
```

Para cada ventana de 24 fechas × 5 canales, se calculan **7 estadísticos por canal**:

| Estadístico | Qué captura                                         |
|-------------|-----------------------------------------------------|
| mean        | Nivel promedio de humedad en el período             |
| std         | Cuánto varía — alta std = irregularidad en el riego |
| min         | Momento de mayor estrés dentro de la ventana        |
| max         | Momento de mayor hidratación                        |
| p25         | Cuartil inferior — estado del 25% más estresado     |
| p75         | Cuartil superior — estado del 75% más hidratado     |
| trend       | Pendiente lineal — ¿está mejorando o empeorando?    |

**7 estadísticos × 5 canales = 35 features por ventana.**

### Arquitectura del ensamble

```
                ┌─────────────────────────────────────┐
 35 features ──►│  Random Forest                      │──► prob. [s0, s1, s2]
                ├─────────────────────────────────────┤
 35 features ──►│  XGBoost                            │──► prob. [s0, s1, s2]  ──► Logistic  ──► clase
                ├─────────────────────────────────────┤              Regression     final
 35 features ──►│  SVM (kernel RBF)                   │──► prob. [s0, s1, s2]
                └─────────────────────────────────────┘
                  BASE LEARNERS (nivel 1)                 META-LEARNER (nivel 2)
```

### ¿Por qué tres modelos distintos?

Cada base learner tiene fortalezas complementarias:

| Modelo          | Fortaleza                                              | Debilidad                          |
|-----------------|--------------------------------------------------------|------------------------------------|
| Random Forest   | Robusto a outliers, maneja bien la no-linealidad       | Puede ser conservador en fronteras |
| XGBoost         | Muy preciso en datos tabulares, captura interacciones  | Sensible a hiperparámetros         |
| SVM (RBF)       | Excelente con datos de alta dimensión, margen máximo   | Lento en datasets grandes          |

El meta-learner (Regresión Logística) aprende **cuándo confiar en cada uno**. Si para una muestra el RF y el SVM dicen "moderado" pero el XGBoost dice "severo", el meta-learner pondera quién tuvo más razón históricamente.

### Validación sin fuga de datos

Se usó `GroupShuffleSplit` con `groups=parcel_id` para garantizar que **ninguna ventana de una parcela del conjunto de entrenamiento aparezca en el conjunto de prueba**.

```
Sin GroupShuffleSplit (MAL):          Con GroupShuffleSplit (BIEN):
─────────────────────────────          ──────────────────────────────
Train: ventanas 0-23 de H1  ←         Train: todas las ventanas de H1
Test:  ventanas 4-27 de H1  ← FUGA    Test:  todas las ventanas de H51
       (misma parcela!)                       (parcela diferente)
```

Si las ventanas de la misma parcela estuvieran en train y test al mismo tiempo, el modelo simplemente "memoriza" esa parcela y el F1 reportado sería artificialmente alto.

### Resultado

| Métrica       | Valor      |
|---------------|------------|
| F1-macro      | **0.8868** |
| Split         | 80/20 a nivel de parcela |
| Muestras      | ~6,400 ventanas de 100 parcelas |

---

## Comparación final entre los tres modelos

| Dimensión              | CNN (baseline)          | ViT for SITS           | E3 Stacking (★ final)    |
|------------------------|-------------------------|------------------------|--------------------------|
| **Clases**             | 2 (binario)             | 3 ✓                    | 3 ✓                      |
| **Features**           | Aprendidas (red)        | Aprendidas (atención)  | Diseñadas (35 estadísticos) |
| **Contexto temporal**  | Ninguno (aplana todo)   | Atención × todas fechas| Ventana de 24 fechas      |
| **Contexto espacial**  | Píxel por píxel         | No (promedia)          | No (promedia)            |
| **DOY / estacionalidad**| No                    | Sí (encoding sinusoidal)| Indirectamente (trend)  |
| **Requiere GPU**       | No                      | Sí                     | No                       |
| **Datos necesarios**   | Poco                    | Mucho (Transformer)    | Poco                     |
| **Interpretabilidad**  | Media                   | Muy baja               | Alta (feature importance)|
| **F1-macro**           | ~0.72 (est.)            | ~0.82 (est.)           | **0.8868**               |
| **En producción**      | No                      | No                     | Sí ✓                     |

---

## ¿Cuándo escalaría cada modelo?

```
Parcelas disponibles:

  100 parcelas (actual)
  ─────────────────────
  E3 Stacking ★ gana: las 35 features estadísticas
  capturan bien la señal con datos escasos.
  El ViT no tiene suficientes datos para estabilizar
  los pesos de atención.

  1,000 – 5,000 parcelas
  ──────────────────────
  ViT empieza a competir: la atención aprende
  qué épocas del año son más informativas y
  qué patrones temporales preceden al estrés.

  10,000+ parcelas
  ─────────────────
  ViT probablemente gana: puede generalizar
  a condiciones climáticas regionales distintas
  y captura interacciones temporales complejas
  que los 35 features estadísticos no expresan.
```

---

## Conclusión

El CNN estableció que el problema es **resoluble con imágenes Sentinel-2**. El ViT demostró que el tiempo y la estacionalidad son **dimensiones importantes** del problema, y que la arquitectura Transformer es aplicable a SITS. El E3 Stacking ganó en producción porque **35 features bien diseñadas superan a una red profunda cuando los datos son escasos** — un resultado clásico en ML que vale la pena conocer.

Para una cooperativa con cientos de parcelas en el futuro, el ViT (o una variante más moderna como un modelo de Mamba o un Transformer eficiente) sería el camino natural de evolución.

---

*Fundamentos técnicos para uso interno del Equipo 16 — Proyecto Integrador 2025*
