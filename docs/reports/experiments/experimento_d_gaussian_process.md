# Experimento D — Gaussian Process por parcela y por grupo de terreno

> Resumen corto de números y decisiones para usar en el resumen ejecutivo.
> Origen: sugerencia de los profesores asesores en la reunión del 2026-06-18
> — no reemplaza el E3 Stacking ya productivizado, es un experimento
> adicional sobre la justificación de umbrales NDMI.

---

## 1. Problema que resuelve

En la reunión, el profesor señaló que el umbral NDMI usado para el
pseudo-etiquetado (decil/percentil fijo) es **arbitrario**: con 5 índices
no hay una sola "línea de corte" coherente, y la misma medida de NDMI
puede significar estrés o no según el tipo de terreno de la parcela.

Propuesta del profesor: en vez de un umbral global, ajustar una
distribución (gaussiana / Gaussian Process) por parcela o por grupo de
parcelas similares, vía *maximum likelihood*, y clasificar por
desviaciones estándar respecto a lo esperado — no por percentil fijo.

---

## 2. Qué se construyó

### 2.1 GP individual (por parcela)

- Un `GaussianProcessRegressor` (scikit-learn) por parcela, sobre su
  serie histórica de NDMI (277 fechas, 2020-01-03 a 2026-05-28).
- Kernel: `RBF` (tendencia suave, length_scale 10–365 días) +
  `ExpSineSquared` (estacionalidad anual, periodo fijo = 365.25 días) +
  `WhiteKernel` (ruido). Los hiperparámetros se ajustan maximizando el
  likelihood marginal — es el "maximum likelihood" que pidió el profesor,
  resuelto por sklearn, sin implementación manual.
- Clasificación de la última observación por z-score respecto a la media
  que el GP esperaba para esa fecha:
  `z < 1` → sin estrés · `1 ≤ z < 2` → moderado · `z ≥ 2` → severo.
- Código: `src/models/gp/parcel_gp.py`.

### 2.2 GP de grupo (por tipo de terreno)

- El catálogo **no tiene dato real de suelo/elevación** —
  `altitude_m` viene en `0.0` para las 100 parcelas (placeholder sin
  datos en `data/raw/parcels/parcelas.csv`). Se usó como proxy de
  "tipo de terreno" la **huella espectral histórica** de cada parcela
  (promedio 2020–2026 de sus 5 índices, ya en `data/datasets/manifest.csv`).
- Agrupación con K-Means (4 grupos) sobre esa huella, estandarizada.
  El número de grupos no se eligió a ojo: se comparó silhouette score
  para k=2..9 — k=2 da el mayor silhouette (0.57) pero solo separa
  "baseline alto/bajo" (30/70 parcelas); **k=4** se eligió por mantener
  buena separación (0.51) con grupos de tamaño razonable.

  | Grupo | Parcelas | NDMI medio | NDVI medio | EVI medio |
  |---|---|---|---|---|
  | 0 | 45 | +0.0047 | 0.456 | 0.288 |
  | 1 | 23 | +0.0869 | 0.554 | 0.349 |
  | 2 | 6  | +0.2023 | 0.681 | 0.426 |
  | 3 | 26 | −0.0732 | 0.385 | 0.231 |

- Por cada parcela se calcula su **baseline** (promedio histórico propio)
  y se centra su serie. Las desviaciones de todas las parcelas del mismo
  grupo se promedian **fecha por fecha** (las 100 parcelas comparten
  exactamente las mismas 277 fechas — mismo tile Sentinel-2, mismo filtro
  de nubes) y se ajusta un solo GP sobre esa serie promediada.
  Promediar entre parcelas (no solo concatenar) es lo que reduce el ruido
  idiosincrático de una parcela con pocos eventos de estrés severo en su
  propio historial — el mecanismo que pidió el profesor.
- Pronóstico para una parcela: `baseline_parcela + GP_grupo.media(t)`.
- Código: `src/models/gp/terrain_groups.py` (clustering) y
  `src/models/gp/group_gp.py` (GP de grupo + comparación individual/grupo).

### 2.3 Integración

- Endpoint `GET /parcels/{parcel_id}/trend?index=NDMI&horizon=5`
  (`api/trend.py`, `api/main.py`) devuelve ambos bloques (individual y
  grupo) en una sola respuesta.
- Pestaña **"Tendencias"** en el dashboard, con toggle
  Individual / Grupo (terreno similar). Verificado visualmente.
- `data/` está en `.gitignore` — `data/datasets/terrain_groups.json` no
  se versiona. Cualquiera que clone esta branch necesita correr
  `python -m src.models.gp.terrain_groups --n-groups 4` una vez antes de
  que el bloque "group" del endpoint deje de salir en `null` (la vista
  individual funciona igual sin ese paso).

---

## 3. Resultado clave (para el resumen ejecutivo)

Validado con holdout (entrenar sin las últimas 6-8 fechas y comparar
contra lo observado): el GP individual y el de grupo predicen bien — los
z-scores en el holdout caen mayormente dentro de ±0.3-0.8σ.

Comparando individual vs. grupo en parcelas reales del catálogo (índice
NDMI, fecha 2026-05-28):

| Parcela | Individual (z, clase) | Grupo (z, clase) |
|---|---|---|
| H1  | −0.00, sin estrés | **+1.60, moderado** |
| H16 | +0.50, sin estrés | **+1.66, moderado** |
| H50 | −0.13, sin estrés | **+1.41, moderado** |
| H63 | +0.15, sin estrés | +0.54, sin estrés |
| H21 | +0.05, sin estrés | −3.46, sin estrés (atípico hacia arriba) |

**Mensaje para el resumen ejecutivo:** varias parcelas que su propio
historial individual no marca como anómalas sí lo son **respecto a otras
parcelas de terreno similar** en la misma fecha — la vista de grupo
detecta un patrón (caída conjunta de NDMI) que la vista aislada no ve.
Esto es justo el valor que el profesor anticipaba al pedir agrupar por
terreno: más muestras por distribución, especialmente útil para parcelas
con pocos eventos de estrés severo en su propia historia.

---

## 4. ¿Cómo se aplica esto a la clasificación final?

**Todavía no se fusiona con el E3 Stacking — es una decisión deliberada,
no un pendiente olvidado.**

Hoy el dashboard muestra dos señales en paralelo, en pestañas distintas:

- **Diagnóstico** → la clase oficial (sin estrés/moderado/severo) del
  E3 Stacking, el modelo ya validado (F1-macro=0.8868). **No se modificó.**
- **Tendencias** → el z-score del GP (individual y de grupo), una señal
  complementaria de "qué tan atípica es la última observación".

La integración completa significaría usar el z-score del GP para
**redefinir el criterio de etiquetado** que hoy usa un umbral NDMI fijo
(`api/features.py: label_window()`, -0.10/-0.20 crudo — exactamente el
punto que cuestionó el profesor) y **reentrenar** el E3 Stacking con esas
nuevas etiquetas. Eso es un cambio de fondo al modelo productivo a días
de la presentación final, así que **no se hizo** para esta entrega.

**Recomendación para esta entrega:** presentar el GP como una capa de
**alerta temprana / atención**, no como un clasificador competidor — es
literalmente lo que el profesor describió como "semáforos de alerta" en
la reunión. Cuando el GP de grupo dice "moderado" y el diagnóstico oficial
dice "sin estrés" (el caso de H1/H16/H50), no es una contradicción: es una
señal de que esa parcela se está desviando de sus pares de terreno similar
*antes* de que el clasificador oficial lo capture. Es valor añadido, no
una segunda verdad en competencia con el modelo productivo.

Si el equipo quiere ir más allá (re-etiquetar y reentrenar con el criterio
del GP), eso sería un **Experimento E** declarado como trabajo futuro —
no algo a resolver esta semana.

---

## 5. Limitaciones a declarar (honestidad > impresionar)

1. **No hay dato real de suelo/altitud** — el agrupamiento por terreno es
   un proxy (huella espectral histórica), no una clasificación edafológica
   real. Mencionarlo como limitación, no presentarlo como "tipo de suelo
   medido".
2. **Recorte de percentil 1/99 en la normalización existente**
   (`src/processing/time_series_builder.py`, `TSNormalizer.fit`) afecta a
   las parcelas más extremas del catálogo (ej. H16, la más estresada):
   sus valores más bajos quedan "aplanados" en el percentil 1 global, lo
   que sesga levemente su baseline hacia menos extremo. No es un bug del
   experimento D, es preexistente al pipeline de datos.
3. **El promedio entre parcelas del grupo no pondera por cuántas parcelas
   aportan dato en cada fecha** — simplificación válida para esta primera
   iteración, pero no es un modelo jerárquico completo.
4. **Los cortes de 1σ/2σ son una elección razonable, no la única posible**
   — más principista que un percentil arbitrario (porque salen de una
   distribución ajustada por likelihood), pero siguen siendo una decisión
   de diseño que vale la pena declarar explícitamente.
5. Este experimento **no reemplaza el E3 Stacking** (F1-macro 0.8868) que
   sigue siendo el modelo productivo del dashboard — es una pieza adicional
   para justificar/refinar los umbrales, no un modelo de clasificación
   final.

---

## 6. Cómo reproducir

```bash
# 1. Agrupar parcelas por terreno (huella espectral) - data/ esta gitignored,
#    hay que correr esto despues de cada clone fresco
python -m src.models.gp.terrain_groups --scan-k          # justifica k
python -m src.models.gp.terrain_groups --n-groups 4       # genera terrain_groups.json

# 2. GP individual de una parcela
python -m src.models.gp.parcel_gp --parcel-ids H16 --holdout 6 --plot

# 3. GP de grupo vs. individual, lado a lado
python -m src.models.gp.group_gp --parcel-id H16 --plot
```

Plots de ejemplo en `outputs/gp_prototype/` (no versionados).
Notebook con la narrativa visual completa:
`notebooks/02_experimento_d_gaussian_process.ipynb`.
