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
- Todavia no ejecuta SAM2 real ni fine-tuning.
- El endpoint `/sam2/mask/{parcel_id}` devuelve una mascara raster ligera por parcela.
- El boton `Analizar todo` genera las mascaras para alimentar la capa completa.
- Los controles de visibilidad permiten ocultar grupos de pixeles por clase sin recalcular las mascaras.
- El boton general `Escanear mapa` analiza las parcelas visibles en el viewport actual del mapa.
- La lista `Parcelas activas` muestra todas las parcelas con mascara visible y solo centra el mapa al seleccionarlas.

## Por que empezar asi

El modelo actual clasifica el estres por parcela usando datos Sentinel-2 procesados. SAM2, en cambio, necesita una imagen y prompts o mascaras para producir segmentacion espacial. Antes de hacer fine-tuning conviene validar el flujo visual y definir que se va a usar como verdad de entrenamiento.

## Siguiente etapa tecnica

1. Generar o reunir imagenes raster por parcela con la misma fecha/ventana usada por el modelo.
2. Definir mascaras de entrenamiento:
   - mascaras manuales por zonas del huerto,
   - mascaras derivadas de indices espectrales,
   - o mascaras aproximadas revisadas por agronomos.
3. Crear un endpoint de backend para devolver geometria o mascaras por `parcel_id`.
4. Reemplazar la capa preview por poligonos/masks reales.
5. Evaluar fine-tuning de SAM2 cuando existan suficientes pares imagen-mascara.

## Datos minimos para SAM2 real

- Imagenes por parcela o recortes georreferenciados.
- Relacion entre `parcel_id`, imagen y fecha.
- Mascara o etiqueta espacial por clase de estres.
- Metricas para comparar la segmentacion con el diagnostico del modelo.

La vista actual permite avanzar con la UX y el contrato de datos sin bloquearse por el entrenamiento de SAM2.
