"""
utils/kml_reader.py
Lee el KML de Alpha Earth y devuelve un GeoDataFrame con las 100 parcelas.
"""
from pathlib import Path
import geopandas as gpd
import fiona


def load_parcels(kml_path: str = "data/raw/parcelas.kml") -> gpd.GeoDataFrame:
    """
    Lee el KML y devuelve un GeoDataFrame con columnas:
        name       – nombre/ID de la parcela
        geometry   – geometría original (Polygon o Point)
        centroid   – centroide como shapely Point
        lon, lat   – coordenadas del centroide
    """
    fiona.drvsupport.supported_drivers["LIBKML"] = "rw"
    fiona.drvsupport.supported_drivers["KML"] = "rw"

    gdf = gpd.read_file(kml_path, driver="KML")
    gdf = gdf.to_crs(epsg=4326)

    # Normaliza nombre
    if "Name" in gdf.columns:
        gdf = gdf.rename(columns={"Name": "name"})
    elif "name" not in gdf.columns:
        gdf["name"] = [f"parcela_{i:03d}" for i in range(len(gdf))]

    # Centroide
    gdf["centroid"] = gdf.geometry.centroid
    gdf["lon"] = gdf["centroid"].x
    gdf["lat"] = gdf["centroid"].y

    return gdf[["name", "geometry", "centroid", "lon", "lat"]].reset_index(drop=True)


def parcel_bbox(row, buffer_m: float = 50.0):
    """
    Devuelve el bounding box de una parcela con buffer, en grados decimales.
    Para buffer_m ≈ 50 m en latitudes mexicanas: 1° lat ≈ 111_000 m.
    """
    from shapely.geometry import box

    delta = buffer_m / 111_000
    geom = row.geometry
    minx, miny, maxx, maxy = geom.bounds
    return box(minx - delta, miny - delta, maxx + delta, maxy + delta)
