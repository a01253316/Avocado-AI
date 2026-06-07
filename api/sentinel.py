"""
Capa de acceso a datos satelitales.

Modo actual (local): busca la parcela más cercana en parcelas.csv y carga su .npz.
Modo producción (CDSE): descargaría el tile Sentinel-2 en tiempo real para las
coordenadas exactas del usuario.
"""
from __future__ import annotations

import pathlib
from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd

from .features import patch_to_timeseries


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia en km entre dos puntos (grados decimales)."""
    R = 6371.0
    la1, lo1, la2, lo2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
    return 2 * R * asin(sqrt(a))


class LocalCatalog:
    """
    Catálogo local: carga parcelas.csv y resuelve coordenadas GPS a .npz.

    En producción esto se reemplazaría por una consulta STAC a CDSE para
    descargar el tile Sentinel-2 más reciente que cubra el punto del usuario.
    """

    def __init__(self, parcelas_csv: pathlib.Path, patches_dir: pathlib.Path):
        self._df       = pd.read_csv(parcelas_csv)
        self._patches  = patches_dir
        self._validate()

    def _validate(self):
        required = {"parcel_id", "latitude", "longitude"}
        missing  = required - set(self._df.columns)
        if missing:
            raise ValueError(f"parcelas.csv falta columnas: {missing}")

    def find_nearest(self, lat: float, lon: float) -> dict:
        """
        Devuelve la parcela más cercana al punto dado y carga su serie temporal.
        """
        df = self._df.copy()
        df["dist_km"] = df.apply(
            lambda r: haversine_km(lat, lon, r["latitude"], r["longitude"]), axis=1
        )
        nearest = df.loc[df["dist_km"].idxmin()]

        parcel_id = nearest["parcel_id"]
        npz_path  = self._patches / f"{parcel_id}.npz"

        if not npz_path.exists():
            raise FileNotFoundError(
                f"No se encontró el parche .npz para {parcel_id} en {self._patches}. "
                "Ejecuta: make build-dataset"
            )

        ts = patch_to_timeseries(npz_path)

        return {
            "parcel_id":  parcel_id,
            "lat":        float(nearest["latitude"]),
            "lon":        float(nearest["longitude"]),
            "dist_km":    round(float(nearest["dist_km"]), 3),
            "state":      str(nearest.get("state", "Jalisco")),
            "ts":         ts,           # (T, C) array — no serializable, uso interno
            "npz_path":   str(npz_path),
            "data_source": "local_catalog",
            "note": (
                "Datos del parche más cercano en el catálogo local. "
                "En producción se descargaría el tile Sentinel-2 exacto via CDSE."
                if nearest["dist_km"] > 1.0 else
                "Datos del parche local correspondiente a la ubicación."
            ),
        }
