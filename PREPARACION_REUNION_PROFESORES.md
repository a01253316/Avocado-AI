# Preparación para Reunión con Profesores — AguaVerde
## Defensa de la propuesta final de dashboard
**Equipo 16 · Proyecto Integrador · Jalisco, México · Maestría en IA, Tec de Monterrey**

---

## Cómo usar este documento

Este documento NO es un reporte técnico nuevo — los reportes técnicos completos ya existen (`CONCLUSIONES.md`, `REPORTE_EQUIPO.md`, `FUNDAMENTOS_MODELOS.md`). Esto es una **guía de defensa**: un resumen ejecutable para la reunión, con anticipación de preguntas y respuestas sugeridas, organizado para que cualquier integrante del equipo pueda responder con seguridad.

**Antes de la reunión, marca en la casilla lo que ya verificaste:**

- [ ] Dashboard corre localmente sin errores (`make dev` → `http://127.0.0.1:8000/ui`)
- [ ] Tienes a la mano una parcela de ejemplo con foto real para hacer la demo en vivo
- [ ] Revisaste la sección 6 (preguntas difíciles) y acordaron como equipo quién responde qué
- [ ] Confirmaron los puntos marcados **[POR CONFIRMAR]** en la sección del Experimento C (ver abajo)

---

## 1. Resumen de los tres experimentos

| | **Experimento A** | **Experimento B** | **Experimento C (propuesta final)** |
|---|---|---|---|
| **Fuente de datos** | Sentinel-2 L2A (CDSE), tabular | Alpha Earth / Google Satellite Embedding | Sentinel-2 L2A, 5 índices espectrales |
| **Resolución temporal** | Continua (~5 días, 277 fechas, 2020–2026) | Anual (2018–2025) | Trimestral / estacional (promedio por periodo) |
| **Dimensionalidad** | 5 índices (NDVI, NDWI, NDMI, NDRE, EVI) → 35 features por ventana | 64 canales embedding → 512 features estadísticos | 5 índices (NDVI, NDWI, NDMI, NDRE, EVI), promedio trimestral |
| **Unidad de observación** | Ventana de 24 fechas por parcela (~6,400 muestras) | Parcela-año (~800 observaciones) | Parcela-trimestre **[confirmar n° de muestras]** |
| **Modelo** | E3 Stacking (RF + XGBoost + SVM → Regresión Logística) | 7 modelos individuales comparados → mejor: SVM RBF balanceado (ajustado) | **[confirmar: mismo E3 Stacking o reentrenado sobre datos trimestrales]** |
| **F1-macro (test)** | **0.8868** | 0.8139 | **[confirmar valor exacto]** |
| **Interpretabilidad** | Alta (feature importance, NDMI domina 53.2%) | Media (embeddings no son interpretables agronómicamente) | Alta (mismos 5 índices, promedio estacional fácil de explicar) |
| **Reporte para el agricultor** | No incluido en el experimento base | No incluido | **Sí — Claude (Anthropic) multimodal, texto + foto de campo** |
| **Estado** | Completo, documentado a fondo | Completo, documentado a fondo | **Propuesta final a presentar** |

**Mensaje central para la reunión:** los tres experimentos no compiten entre sí — son una progresión de aprendizaje. A valida que Sentinel-2 + ML clásico resuelve el problema con alta precisión. B explora si un dataset alternativo (embeddings preentrenados, resolución anual) ofrece ventajas y concluye que es **viable pero inferior en F1 y en interpretabilidad agronómica**. C toma lo mejor de A (índices interpretables, validados agronómicamente) y lo combina con una resolución temporal pensada explícitamente para capturar estacionalidad (lluvias vs. secas) y un componente de explicabilidad para el usuario final (LLM multimodal).

---

## 2. Experimento A — Sentinel-2 tabular, serie continua (★ documentado a fondo)

Ya está cubierto en detalle en `CONCLUSIONES.md` y `REPORTE_EQUIPO.md`. Puntos clave que debes poder repetir de memoria:

- **100 parcelas** en Jalisco (tile Sentinel-2 `14QMF`), 277 fechas válidas (nubosidad ≤ 20%) entre 2020–2026.
- Pipeline: GeoTIFF → 5 índices espectrales → ventana deslizante (W=24 fechas, paso=4) → **35 features estadísticas** (7 estadísticos × 5 índices) → E3 Stacking.
- Etiquetado **100% automático** vía umbrales NDMI (sin mediciones de campo).
- Split `GroupShuffleSplit` por `parcel_id` (80/20) — sin fuga de datos entre parcelas.
- **F1-macro = 0.8868**, AUC-macro = 1.0, Accuracy = 99.79%.
- NDMI concentra 53.2% de la importancia del modelo — es la señal dominante (banda SWIR1, sensible a agua en hoja, cambia antes que el NDVI).

---

## 3. Experimento B — Alpha Earth / Google Satellite Embedding

Documentado en `notebooks/04_Modelos_Alternativos_AlphaEarth.ipynb`. Resumen para la reunión:

- Dataset: `alphaearth_dataset_2018_2025_with_annual_labels.csv` — **100 parcelas, 2018–2025, ~800 observaciones parcela-año**.
- Predictores: **512 variables** derivadas de los **64 canales de Alpha Earth** (embeddings preentrenados de Google, no índices espectrales interpretables).
- Etiqueta anual derivada de cuantiles de NDMI (igual filosofía de etiquetado automático que en A), **NDVI/NDMI se excluyeron como predictores** para evitar fuga conceptual (la etiqueta se construyó a partir de ellos).
- Se compararon **7 modelos individuales** (no ensambles, por requisito de la rúbrica): regresión logística, SVM lineal, SVM RBF, KNN, árbol de decisión, Naive Bayes, MLP — más un `DummyClassifier` de referencia.
- Validación: `GroupShuffleSplit` 75/25 por parcela + `GroupKFold` por parcela.
- Se afinaron los 2 mejores con `GridSearchCV`. Modelo final: **SVM RBF balanceado ajustado** → **F1-macro = 0.8139**, balanced accuracy = 0.8112, recall clase severa = 0.7692.
- Se probó también una **CNN experimental** directamente sobre los parches `H×W×64` (sin reducir a estadísticos) → F1-macro = 0.7336, con señales de sobreajuste moderado (curva train/val se separa) — esperable con ~800 observaciones.

**¿Por qué B no se eligió como propuesta final?**
1. **F1 inferior** al de A (0.8139 vs. 0.8868), incluso usando 64 canales en vez de 5.
2. **Resolución anual** es demasiado gruesa para detectar estrés hídrico de forma accionable — un agricultor necesita saber esta semana, no este año.
3. **Los embeddings no son interpretables agronómicamente**: no se puede decirle al agricultor "el canal A37 bajó", mientras que con NDMI sí se puede explicar "tu dosel perdió humedad".
4. Las etiquetas siguen siendo proxy de NDMI — B no resuelve el problema de validación de campo que tiene A, y además pierde interpretabilidad a cambio de nada.

---

## 4. Experimento C — Propuesta final (5 índices, agregación trimestral + LLM multimodal)

### 4.1 Lo que sabemos con certeza (ya implementado y documentado)

- Mismos **5 índices espectrales** que el Experimento A: NDVI, NDWI, NDMI, NDRE, EVI — máxima interpretabilidad agronómica.
- Integración con **Claude (Anthropic)** como LLM: recibe el diagnóstico del modelo + los índices + la tendencia + (opcional) una **foto real de campo**, y genera un reporte agronómico en español con recomendaciones accionables y nivel de urgencia.
- El sistema tiene **fallback sin LLM** (reporte basado en reglas) si la API no está disponible — el sistema nunca se cae por dependencia externa.
- Dashboard interactivo (Leaflet + Chart.js) ya funcional, con mapa de parcelas, escaneo batch, vista de tendencia NDMI y reporte LLM.

### 4.2 Lo que debes confirmar como equipo antes de la reunión **[POR CONFIRMAR]**

No encontré en el repositorio un dataset o notebook separado que documente específicamente la **agregación trimestral con promedio por índice** que describes (a diferencia de A, que usa fechas individuales en ventanas de 24, y de B, que usa agregación anual). Antes de presentar el Experimento C como propuesta final, el equipo debe tener lista la respuesta a:

1. **¿Cuántos trimestres y años cubre el dataset trimestral?** (¿2020–2026 igual que A? ¿menos?)
2. **¿Cuántas muestras resultan?** (100 parcelas × N trimestres = ?)
3. **¿Es un dataset nuevo desde GeoTIFFs Sentinel-2, o es una re-agregación de los datos de A?**
4. **¿El modelo final usa el mismo E3 Stacking reentrenado sobre datos trimestrales, o es un modelo nuevo?** Si es el mismo, ¿qué F1-macro se obtuvo con la agregación trimestral vs. el F1=0.8868 de A?
5. **¿Por qué trimestral y no mensual?** Ten lista la justificación agronómica: el aguacate Hass en Jalisco tiene ciclos de floración/fructificación ligados a temporada de lluvias (junio–octubre) vs. seca (noviembre–mayo); el trimestre captura esa transición sin generar ruido de corto plazo por nubosidad puntual.
6. **¿El LLM (Claude) usa el `ANTHROPIC_API_KEY` por defecto, o el sistema en producción usa Ollama local con Claude como opción?** — Importante: el `README.md` actual indica que el proveedor **por defecto es Ollama local**, con Claude como alternativa configurable. Si van a presentar "Claude" como la pieza central de la propuesta, aclaren si:
   - (a) van a fijar Claude como proveedor de producción para la cooperativa, o
   - (b) van a mantener Ollama como default y Claude como opción premium/respaldo.
   Cualquiera de las dos es defendible, pero deben decir lo mismo todos los integrantes del equipo.

> Si alguno de estos puntos no se resuelve antes de la reunión, es mejor decir explícitamente "esto está en proceso de definirse" que dar una cifra inventada — los profesores notan inconsistencias entre lo dicho y lo que puedan ver en el repo/demo.

### 4.3 Argumento de venta del Experimento C como propuesta final

- Combina la **interpretabilidad y el desempeño validado de A** (mismos 5 índices, mismo tipo de modelo) con una resolución temporal pensada para **estacionalidad agronómica real** (lluvias vs. secas), no solo para maximizar una métrica.
- Agrega la pieza que A y B no tienen: un **reporte en lenguaje natural y multimodal** que cierra la brecha entre "salida de un modelo de ML" y "algo que un agricultor sin formación técnica puede usar para actuar hoy mismo".
- Es el único de los tres experimentos diseñado pensando en el **usuario final y el flujo operativo completo** (parcela → predicción → explicación → acción), no solo en la métrica de clasificación.

---

## 5. Tabla comparativa para mostrar en la reunión

| Criterio | A (Sentinel-2 continuo) | B (Alpha Earth anual) | C (Sentinel-2 trimestral + LLM) |
|---|---|---|---|
| F1-macro | 0.8868 | 0.8139 | [confirmar] |
| Interpretabilidad técnica | Alta | Baja-media | Alta |
| Interpretabilidad para el agricultor | Media (requiere traducir clase + % a lenguaje natural) | Baja | **Alta (reporte LLM en español)** |
| Frecuencia de actualización | Cada ~5 días | Anual | Trimestral |
| Requiere foto de campo | No | No | **Sí (opcional, mejora el diagnóstico)** |
| Costo de cómputo / cloud | Bajo (sklearn, sin GPU) | Bajo (sklearn) + CNN opcional con GPU | Bajo + costo de API LLM |
| Validado contra campo real | No (proxy NDMI) | No (proxy NDMI) | No (proxy NDMI) — mismo punto ciego que A y B |
| Listo para producción | Sí | No (queda como referencia/exploración) | **Propuesta — pendiente confirmar puntos de 4.2** |

---

## 6. Preguntas anticipadas de los profesores y respuestas sugeridas

### Sobre datos y ventanas temporales

**P: ¿Cada cuánto se tomaron las muestras de cada parcela?**
R: En A, cada imagen Sentinel-2 disponible con nubosidad ≤20% — en promedio cada ~5 días gracias al par de satélites Sentinel-2A/2B, dando 277 fechas por parcela entre 2020–2026. En C, esas mismas observaciones se agregan a nivel trimestral (promedio del periodo) para suavizar ruido de corto plazo y resaltar el patrón estacional.

**P: ¿Por qué no usar resolución diaria o semanal si Sentinel-2 lo permite?**
R: El paso de revisita real (~5 días) ya impone un límite; bajar a diario no aporta información nueva. Subir a trimestral, en cambio, es una decisión deliberada para alinear la unidad de análisis con los ciclos fisiológicos del aguacate (floración, cuajado de fruto) y con los periodos de lluvia/seca, que es donde el agricultor toma decisiones de riego.

**P: ¿Cómo manejan la nubosidad y los huecos en la serie?**
R: Filtro de nubosidad ≤20% en la fuente; si hay un hueco prolongado, el sistema alerta cuando la última imagen válida de una parcela supera 20 días de antigüedad (ver sección de riesgos en `CONCLUSIONES.md`).

**P: ¿Las 100 parcelas son representativas de toda la cooperativa?**
R: No — cubren solo el tile Sentinel-2 `14QMF` en Jalisco. Es una limitación reconocida; el roadmap de expansión a 500+ parcelas está en `CONCLUSIONES.md` sección 7.

### Sobre el proceso y la metodología

**P: ¿Cómo evitan que el modelo "memorice" una parcela y luego haga trampa en la validación?**
R: `GroupShuffleSplit` agrupando por `parcel_id` en los tres experimentos — ninguna ventana/observación de una parcela del set de entrenamiento aparece en el de prueba.

**P: ¿Cómo generaron las etiquetas de estrés sin mediciones de campo?**
R: Umbrales fisiológicos sobre NDMI normalizado (validados en literatura: Gao 1996, Wilson & Sader 2002). Es una limitación reconocida explícitamente — las etiquetas son un **proxy**, no verdad de campo. Mencionarlo proactivamente antes de que lo pregunten reduce el riesgo de que parezca que lo están ocultando.

**P: ¿Por qué tres experimentos distintos y no uno solo desde el principio?**
R: Es una progresión de validación: A confirma que el enfoque base funciona con alto F1; B prueba una fuente de datos alternativa (embeddings) para ver si mejora algo, y concluye que no — pierde interpretabilidad y F1 sin ganar nada a cambio; C toma la fuente de datos validada en A y la rediseña para el caso de uso real (estacionalidad + explicabilidad para el usuario final).

### Sobre efectividad del modelo

**P: ¿Por qué el F1 de B (con 512 features de embeddings) es más bajo que el de A (con solo 35 features hechas a mano)?**
R: Buena pregunta que conviene anticipar con datos: B usa una **unidad de observación más gruesa** (parcela-año, ~800 muestras vs. ~6,400 ventanas en A) y los embeddings de Alpha Earth no fueron entrenados específicamente para detectar humedad foliar — son representaciones genéricas de cobertura terrestre. Los 35 features de A están diseñados explícitamente alrededor del NDMI, que es la señal fisiológicamente más relevante. Es un caso clásico de "features bien diseñadas > muchas features genéricas" cuando los datos son escasos (ver `FUNDAMENTOS_MODELOS.md`).

**P: ¿El modelo distingue bien estrés moderado de severo, o solo "sano vs. no sano"?**
R: En A, `recall_severo` y AUC=1.0/mAP=1.0 muestran que el modelo no degrada en la clase minoritaria (severo), que es la más costosa de fallar. En B, el recall de la clase severa del modelo final fue 0.7692 — más bajo, otro punto a favor de A/C sobre B.

**P: ¿Cómo saben que el modelo no está sobreajustado?**
R: Train≈Test con GroupSplit sin fuga, F1 de test por encima del umbral de éxito (0.80) sin señales de underfitting ni overfitting (tabla de diagnóstico en `CONCLUSIONES.md` sección 6.2). Para la CNN experimental de B sí se observó sobreajuste moderado — y por eso no se usó como modelo final, ni en B ni en C.

### Sobre la integración del LLM (Claude)

**P: ¿Por qué confiar en un LLM para dar recomendaciones agronómicas? ¿No puede "alucinar"?**
R: El LLM no reemplaza al modelo de ML — solo **traduce** la salida numérica (clase, confianza, índices, tendencia) a lenguaje natural accionable. El prompt del sistema restringe explícitamente la estructura de la respuesta (explicación, 3-5 recomendaciones, nivel de urgencia, comentario sobre la foto si la hay). Es una capa de comunicación, no la fuente de la decisión.

**P: ¿Qué pasa si la foto no es consistente con los datos satelitales?**
R: El system prompt le pide a Claude comentar explícitamente la consistencia entre foto e índices — si hay discrepancia, el reporte lo señala en vez de ocultarlo. Bueno tener un ejemplo concreto de demo donde esto se vea.

**P: ¿Qué pasa si Claude/el LLM no está disponible?**
R: Fallback a reporte basado en reglas fijas según la clase predicha — el sistema sigue funcionando, solo pierde la riqueza de explicación.

**P: ¿Usan Claude o un modelo local?**
R: **[Responder según lo que decidan en el punto 4.2.6]** — ser consistentes como equipo.

### Sobre áreas de oportunidad y limitaciones

**P: ¿Cuál es la limitación más importante del sistema completo?**
R: Ninguno de los tres experimentos tiene **validación contra mediciones de campo reales** (sensores de humedad de suelo). Las etiquetas son proxy de NDMI en los tres casos. Es la limitación transversal más honesta que se puede presentar, y ya está priorizada en el roadmap (calibración con 20–30 mediciones de humedad en la próxima temporada seca).

**P: ¿Qué harían distinto si tuvieran más tiempo/datos?**
R: Expandir a 500+ parcelas, calibrar umbrales NDMI con sensores físicos, evaluar ViT-SITS cuando el catálogo crezca (tiene ventaja teórica con datos abundantes), y para C específicamente: completar la comparación cuantitativa entre agregación trimestral vs. continua para justificar con números la elección temporal.

**P: ¿Cómo escalaría esto a otras cooperativas o cultivos?**
R: La arquitectura es agnóstica al cultivo — cambiarían los umbrales NDMI (calibrados para aguacate Hass) y posiblemente los índices relevantes. Sentinel-2 tiene cobertura global gratuita, así que la escalabilidad geográfica no es una limitación técnica fuerte.

---

## 7. Checklist final antes de entrar a la reunión

- [ ] Acordar como equipo las respuestas a los 6 puntos de la sección 4.2 (Experimento C)
- [ ] Tener el dashboard corriendo y una parcela de ejemplo lista para demo en vivo (con foto real)
- [ ] Decidir quién presenta cada experimento (A, B, C) y quién responde preguntas técnicas vs. de negocio
- [ ] Tener a la mano las cifras clave de memoria: F1=0.8868 (A), F1=0.8139 (B), F1=? (C)
- [ ] Preparar un cierre claro: "C es la propuesta final porque combina interpretabilidad, estacionalidad agronómica real y explicabilidad para el usuario — ¿qué necesitamos para su aprobación?"
