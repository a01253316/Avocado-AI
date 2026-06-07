"""
kml_to_csv.py
=============
Extrae parcelas de aguacate desde un archivo KML y genera un CSV
con coordenadas, metadata geográfica y parámetros para descarga
de imágenes Sentinel-2 / Alpha Earth.

Uso:
    python kml_to_csv.py --input data/raw/parcels/aguacates.kml
    python kml_to_csv.py --input data/raw/parcels/aguacates.kml --buffer 500
"""

import argparse
import csv
import logging
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KML namespace helpers
# ---------------------------------------------------------------------------
KML_NS = {
    "kml": "http://www.opengis.net/kml/2.2",
    "gx": "http://www.google.com/kml/ext/2.2",
}


def _strip_ns(tag: str) -> str:
    """Elimina el namespace de un tag XML."""
    return re.sub(r"\{[^}]+\}", "", tag)


# ---------------------------------------------------------------------------
# Parseo del KML
# ---------------------------------------------------------------------------
def parse_kml(kml_path: Path) -> list[dict]:
    """
    Lee un KML con Placemarks de tipo Point y devuelve una lista de dicts
    con la información de cada parcela.

    Campos extraídos:
        - parcel_id   : nombre del Placemark (ej. "H1")
        - longitude   : coordenada X (lon) del punto
        - latitude    : coordenada Y (lat) del punto
        - altitude_m  : altitud en metros (0 si no está disponible)
    """
    logger.info(f"Leyendo KML: {kml_path}")
    tree = ET.parse(kml_path)
    root = tree.getroot()

    parcels: list[dict] = []

    # Buscar todos los Placemarks sin importar la profundidad del árbol
    for pm in root.iter():
        if _strip_ns(pm.tag) != "Placemark":
            continue

        # --- Nombre de la parcela ---
        name_el = pm.find(".//{http://www.opengis.net/kml/2.2}name")
        parcel_id = name_el.text.strip() if name_el is not None else "UNKNOWN"

        # --- Coordenadas del Point ---
        coords_el = pm.find(".//{http://www.opengis.net/kml/2.2}coordinates")
        if coords_el is None or coords_el.text is None:
            logger.warning(f"Parcela {parcel_id} sin coordenadas, se omite.")
            continue

        raw = coords_el.text.strip()
        parts = raw.split(",")
        if len(parts) < 2:
            logger.warning(f"Formato de coordenadas inválido en {parcel_id}: {raw}")
            continue

        lon = float(parts[0])
        lat = float(parts[1])
        alt = float(parts[2]) if len(parts) > 2 else 0.0

        parcels.append(
            {
                "parcel_id": parcel_id,
                "latitude": lat,
                "longitude": lon,
                "altitude_m": alt,
            }
        )

    logger.info(f"Parcelas extraídas: {len(parcels)}")
    return parcels


# ---------------------------------------------------------------------------
# Enriquecimiento del CSV
# ---------------------------------------------------------------------------
def enrich(df: pd.DataFrame, buffer_m: int = 250) -> pd.DataFrame:
    """
    Añade columnas útiles para el pipeline de descarga y procesamiento:

        - state         : estado inferido desde coordenadas (heurística bbox)
        - buffer_m      : radio en metros para recorte de imagen por parcela
        - bbox_west/east/south/north : bounding box aproximado para la API
        - sentinel2_tile : tile UTM de Sentinel-2 que cubre la zona (aprox.)
        - download_ready : flag para controlar qué parcelas ya tienen imagen
    """
    # Heurística simple: sur de Jalisco vs Michoacán por longitud/latitud
    # Jalisco: bbox aprox [-105.7, -101.5] lon, [18.9, 22.7] lat
    # Michoacán: bbox aprox [-103.7, -100.0] lon, [17.9, 20.4] lat
    def infer_state(row):
        lon, lat = row["longitude"], row["latitude"]
        # Si está dentro del rango más occidental, Jalisco; si se traslapa, Jalisco primero
        if lon <= -101.5:
            return "Jalisco"
        return "Michoacan"

    df["state"] = df.apply(infer_state, axis=1)

    # Buffer para bounding box
    # Latitud: 1° ≈ 111 320 m (constante)
    # Longitud: 1° ≈ 111 320 × cos(lat) m  → varía con la latitud
    # En Jalisco (~19-20°) cos(lat) ≈ 0.942, así que Δlon > Δlat para el mismo buffer
    import math
    lat_rad = df["latitude"].apply(math.radians)
    delta_lat = buffer_m / 111_320
    delta_lon = buffer_m / (111_320 * lat_rad.apply(math.cos))

    df["buffer_m"] = buffer_m
    df["bbox_west"]  = (df["longitude"] - delta_lon).round(8)
    df["bbox_east"]  = (df["longitude"] + delta_lon).round(8)
    df["bbox_south"] = (df["latitude"]  - delta_lat).round(8)
    df["bbox_north"] = (df["latitude"]  + delta_lat).round(8)

    # Tile Sentinel-2 más común para el sur de Jalisco: 14QMF / 14QMG
    # Se puede refinar con la API de Sentinel pero sirve como placeholder
    df["sentinel2_tile"] = "14QMF"

    # Flags de estado para el pipeline
    df["download_ready"] = False
    df["processed"] = False
    df["indices_computed"] = False

    return df


# ---------------------------------------------------------------------------
# Ordenamiento natural (H1, H2 ... H10, H11 vs H1, H10, H11, H2...)
# ---------------------------------------------------------------------------
def _natural_sort_key(parcel_id: str):
    parts = re.split(r"(\d+)", parcel_id)
    return [int(p) if p.isdigit() else p for p in parts]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extrae parcelas KML → CSV")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/parcels/aguacates_jalisco_5_5_26.kml"),
        help="Ruta al archivo KML de entrada",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta del CSV de salida (default: mismo dir que input)",
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=250,
        help="Buffer en metros alrededor de cada punto para bbox de descarga (default: 250)",
    )
    args = parser.parse_args()

    # Validaciones
    if not args.input.exists():
        logger.error(f"Archivo KML no encontrado: {args.input}")
        sys.exit(1)

    # Output por defecto: mismo directorio que el KML
    output_path = args.output or args.input.with_suffix(".csv")

    # Parseo
    parcels = parse_kml(args.input)
    if not parcels:
        logger.error("No se extrajeron parcelas. Revisa el KML.")
        sys.exit(1)

    # DataFrame
    df = pd.DataFrame(parcels)

    # Ordenar de forma natural (H1, H2, ..., H10, H11, ...)
    df = df.sort_values(
        "parcel_id", key=lambda col: col.map(_natural_sort_key)
    ).reset_index(drop=True)

    # Enriquecer
    df = enrich(df, buffer_m=args.buffer)

    # Guardar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"CSV guardado en: {output_path}")
    logger.info(f"Total parcelas: {len(df)}")
    logger.info(f"Preview:\n{df[['parcel_id','latitude','longitude','state','buffer_m']].head(10).to_string()}")


if __name__ == "__main__":
    main()
