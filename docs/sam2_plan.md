# Plan de soporte SAM2

Este branch inicia la integracion de segmentacion como una capa incremental sobre el dashboard.

## Estado actual

- La pestana `SAM2` del dashboard muestra una capa preview sobre el mapa.
- La capa crea un mosaico con todas las parcelas cargadas.
- Las parcelas pendientes se muestran en gris.
- Al generar mascaras, cada parcela se pinta pixel por pixel usando el NDMI de la ventana reciente:
  - verde: sin estres
  - amarillo: estres moderado
  - rojo: estres severo
- El pipeline de fine-tuning ya esta preparado, pero requiere instalar SAM2 y descargar un checkpoint fuera de git.
- El endpoint `/sam2/mask/{parcel_id}` devuelve una mascara raster ligera por parcela.
- El boton `Analizar todo` genera las mascaras para alimentar la capa completa.
- El boton `Ajustar overlap` compacta mas los bounds y resuelve pixeles solapados por mayoria de votos entre clases calibradas.
- Los controles de visibilidad permiten ocultar grupos de pixeles por clase sin recalcular las mascaras.
- Los filtros de diagnostico Sentinel permiten elegir que parcelas aportan mascaras al raster segun su clase de analisis.
- Los filtros de coordenadas visibles permiten ocultar puntos por clase Sentinel sin modificar el raster.
- En modo SAM2, los marcadores y la lista de parcelas activas funcionan como toggles por mascara.
- El boton general `Escanear mapa` analiza las parcelas visibles en el viewport actual del mapa.
- La lista `Parcelas activas` muestra todas las parcelas con mascara visible y solo centra el mapa al seleccionarlas.
- Los datasets nuevos guardan `bounds_wgs84` en cada patch `.npz`; SAM2 usa esos bounds reales para colocar el raster en Leaflet.
- El frontend compacta visualmente esos bounds alrededor de la coordenada de la parcela porque representan el chip Sentinel-2 completo, no el poligono real del huerto.
- Si dos mascaras se sobreponen, el frontend compone una sola capa por viewport y cada pixel queda con una sola clase final; los filtros solo muestran u ocultan esa clase, no recalculan el ganador del pixel.
- Antes de pintar, cada mascara se calibra con el diagnostico de la parcela: `Sin estres` baja rojo a moderado y moderado a verde; `Moderado` baja rojo a moderado; `Severo` conserva las clases originales.

## Por que empezar asi

El modelo actual clasifica el estres por parcela usando datos Sentinel-2 procesados. SAM2, en cambio, necesita una imagen y prompts o mascaras para producir segmentacion espacial. Antes de hacer fine-tuning conviene validar el flujo visual y definir que se va a usar como verdad de entrenamiento.

## Siguiente etapa tecnica

1. Exportar imagenes y mascaras para SAM2:

```powershell
make build-dataset
make prepare-sam2
```

`make build-dataset` es importante si tus `.npz` fueron generados antes de guardar `bounds_wgs84`. Sin esos bounds, el dashboard usa una caja aproximada por compatibilidad.

2. Descargar un checkpoint SAM2.1 de Meta fuera de git:

```powershell
make download-sam2-checkpoint
```

3. Instalar dependencias opcionales:

```powershell
python -m pip install -r requirements-sam2.txt
```

4. Ejecutar fine-tuning:

```powershell
make train-sam2
```

5. Reemplazar gradualmente las pseudo-mascaras NDMI por mascaras revisadas:
   - mascaras manuales por zonas del huerto,
   - mascaras derivadas de indices espectrales,
   - o mascaras aproximadas revisadas por agronomos.

6. Conectar el checkpoint fine-tuned a la inferencia del dashboard.

## Datos minimos para SAM2 real

- Imagenes por parcela o recortes georreferenciados.
- Relacion entre `parcel_id`, imagen y fecha.
- Mascara o etiqueta espacial por clase de estres.
- Metricas para comparar la segmentacion con el diagnostico del modelo.

La vista actual permite avanzar con la UX y el contrato de datos sin bloquearse por el entrenamiento de SAM2.
