"""
ingestion/downloader.py
Descarga imágenes Sentinel-2 L2A por parcela usando la Process API de Sentinel Hub.

Flujo por parcela:
  1. Catalog API  → busca escenas disponibles en el rango de fechas con baja nubosidad
  2. Process API  → descarga un GeoTIFF recortado al bbox de la parcela con el evalscript
  3. Guarda en    data/raw/sentinel2/<nombre_parcela>/<YYYY-MM-DD>.tif
"""
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

import requests
import rasterio
import numpy as np
import yaml

from src.utils.sentinel_auth import SentinelHubAuth
from src.utils.kml_reader import load_parcels, parcel_bbox

logger = logging.getLogger(__name__)

# ── Evalscript V3 ─────────────────────────────────────────────────────────────
# Devuelve 6 bandas FLOAT32: NDVI, NDWI, NDMI, EVI2, máscara válida, SCL
# La normalización DN→reflectancia (÷10000) se hace aquí en el servidor.

EVALSCRIPT = """
//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["B04", "B08", "B8A", "B11", "B12", "SCL", "dataMask"],
            units: "DN"
        }],
        output: {
            bands: 6,
            sampleType: "FLOAT32"
        }
    };
}

function evaluatePixel(s) {
    // SCL válidos: 4=vegetación, 5=suelo desnudo, 6=agua
    // Excluimos: 0=sin datos, 1-3=saturado/sombra oscura, 7-11=nubes y nieve
    let valid = s.dataMask && (s.SCL == 4 || s.SCL == 5 || s.SCL == 6);
    if (!valid) return [-9999, -9999, -9999, -9999, 0, s.SCL];

    const NO_DATA = -9999;
    const EPS     = 1e-10;

    let RED  = s.B04  / 10000.0;
    let NIR  = s.B08  / 10000.0;
    let NIR2 = s.B8A  / 10000.0;   // 20 m, menos ruido para MSI
    let SWIR1= s.B11  / 10000.0;
    let SWIR2= s.B12  / 10000.0;
    let GREEN= 0;                   // B03 no se pide → usamos proxy EVI2

    // --- Índices ---
    let NDVI = (NIR  - RED)   / (NIR  + RED   + EPS);
    let NDWI = (NIR  - SWIR1) / (NIR  + SWIR1 + EPS);   // humedad hoja
    let NDMI = (NIR  - SWIR1) / (NIR  + SWIR1 + EPS);   // alias NDWI
    let EVI2 = 2.5  * (NIR - RED) / (NIR + 2.4 * RED + 1.0);
    let MSI  = SWIR1 / (NIR2 + EPS);                     // estrés hídrico directo

    return [NDVI, NDWI, NDMI, EVI2, MSI, 1.0];
}
"""

BAND_NAMES = ["NDVI", "NDWI", "NDMI", "EVI2", "MSI", "valid_mask"]


# ── Catalog API ───────────────────────────────────────────────────────────────

def search_catalog(auth: SentinelHubAuth, bbox: list, date_start: str,
                   date_end: str, max_cloud: int = 20) -> list[dict]:
    """
    Consulta el Catalog STAC de Sentinel Hub para encontrar escenas disponibles.
    Devuelve lista de dicts con {date, cloud_cover, id}.
    """
    url = "https://services.sentinel-hub.com/api/v1/catalog/1.0.0/search"
    payload = {
        "bbox": bbox,
        "datetime": f"{date_start}T00:00:00Z/{date_end}T23:59:59Z",
        "collections": ["sentinel-2-l2a"],
        "limit": 100,
        "filter": f"eo:cloud_cover <= {max_cloud}",
        "filter-lang": "cql2-text",
        "fields": {
            "include": ["id", "properties.datetime", "properties.eo:cloud_cover"],
            "exclude": []
        }
    }
    scenes = []
    while True:
        resp = requests.post(url, json=payload, headers=auth.headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            scenes.append({
                "id":          feat["id"],
                "date":        props["datetime"][:10],
                "cloud_cover": props.get("eo:cloud_cover", 0),
            })
        # Paginación
        next_token = data.get("context", {}).get("next")
        if not next_token:
            break
        payload["next"] = next_token

    # Dedup por fecha (queda la de menor nubosidad)
    best: dict[str, dict] = {}
    for s in scenes:
        d = s["date"]
        if d not in best or s["cloud_cover"] < best[d]["cloud_cover"]:
            best[d] = s
    return sorted(best.values(), key=lambda x: x["date"])


# ── Process API ───────────────────────────────────────────────────────────────

def download_scene(auth: SentinelHubAuth, bbox: list, date: str,
                   resolution: int = 10) -> np.ndarray | None:
    """
    Descarga una escena recortada al bbox para una fecha específica.
    Devuelve ndarray (6, H, W) en FLOAT32, o None si falla.
    """
    url = "https://services.sentinel-hub.com/api/v1/process"
    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{date}T00:00:00Z",
                        "to":   f"{date}T23:59:59Z",
                    },
                    "maxCloudCoverage": 20,
                    "mosaickingOrder": "leastCC",  # pide la de menor nubosidad
                },
                "processing": {
                    "harmonizeValues": True,       # corrección entre sensores S2A/S2B
                }
            }]
        },
        "evalscript": EVALSCRIPT,
        "output": {
            "width":  512,
            "height": 512,
            "responses": [{
                "identifier": "default",
                "format": {"type": "image/tiff", "parameters": {"compression": "LZW"}}
            }]
        }
    }

    headers = auth.headers()
    headers["Content-Type"] = "application/json"
    headers["Accept"] = "image/tiff"

    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    if resp.status_code != 200:
        logger.warning(f"  Process API error {resp.status_code}: {resp.text[:200]}")
        return None

    # Escribe a bytes temporales y lee con rasterio
    import io
    with rasterio.open(io.BytesIO(resp.content)) as src:
        arr = src.read().astype(np.float32)  # shape (6, H, W)
    return arr


# ── Pipeline de descarga ───────────────────────────────────────────────────────

def download_parcel(parcel_row, cfg: dict, auth: SentinelHubAuth,
                    out_dir: Path, retry: int = 3) -> int:
    """
    Descarga todas las escenas disponibles para una parcela.
    Guarda cada escena como <out_dir>/<name>/<YYYY-MM-DD>.tif
    Retorna el número de imágenes descargadas.
    """
    name = parcel_row["name"]
    bbox_geom = parcel_bbox(parcel_row, buffer_m=cfg["data"]["buffer_meters"])
    minx, miny, maxx, maxy = bbox_geom.bounds
    bbox = [minx, miny, maxx, maxy]

    parcel_dir = out_dir / name
    parcel_dir.mkdir(parents=True, exist_ok=True)

    # Buscar escenas disponibles
    scenes = search_catalog(
        auth, bbox,
        date_start=cfg["data"]["date_start"],
        date_end=cfg["data"]["date_end"],
        max_cloud=cfg["data"]["max_cloud_cover"],
    )
    logger.info(f"  {name}: {len(scenes)} escenas encontradas")

    downloaded = 0
    for scene in scenes:
        out_file = parcel_dir / f"{scene['date']}.tif"
        if out_file.exists():
            logger.debug(f"    ✓ {scene['date']} ya existe, omitiendo")
            downloaded += 1
            continue

        for attempt in range(retry):
            arr = download_scene(auth, bbox, scene["date"],
                                 resolution=cfg["data"]["resolution"])
            if arr is not None:
                # Guardar como GeoTIFF con metadatos mínimos
                profile = {
                    "driver": "GTiff",
                    "dtype": "float32",
                    "count": arr.shape[0],
                    "height": arr.shape[1],
                    "width": arr.shape[2],
                    "crs": "EPSG:4326",
                    "transform": rasterio.transform.from_bounds(*bbox, arr.shape[2], arr.shape[1]),
                    "compress": "lzw",
                    "nodata": -9999.0,
                }
                with rasterio.open(out_file, "w", **profile) as dst:
                    dst.write(arr)
                    dst.update_tags(
                        parcel=name,
                        date=scene["date"],
                        cloud_cover=scene["cloud_cover"],
                        bands=",".join(BAND_NAMES),
                    )
                downloaded += 1
                logger.info(f"    ✓ {scene['date']} (CC={scene['cloud_cover']:.1f}%)")
                break
            else:
                if attempt < retry - 1:
                    time.sleep(2 ** attempt)
        else:
            logger.warning(f"    ✗ {scene['date']} falló tras {retry} intentos")

    return downloaded


def run_download(cfg_path: str = "configs/base.yaml",
                 cred_path: str = "configs/credentials.yaml",
                 parcel_ids: list[str] | None = None):
    """
    Punto de entrada principal.
    parcel_ids: lista de nombres para descargar solo algunas parcelas (útil para pruebas).
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    cfg  = yaml.safe_load(Path(cfg_path).read_text())
    auth = SentinelHubAuth(cred_path)
    gdf  = load_parcels(cfg["paths"]["kml"])

    if parcel_ids:
        gdf = gdf[gdf["name"].isin(parcel_ids)]

    out_dir = Path(cfg["paths"]["raw_sentinel"])
    total   = 0

    logger.info(f"Iniciando descarga para {len(gdf)} parcelas")
    for _, row in gdf.iterrows():
        try:
            n = download_parcel(row, cfg, auth, out_dir)
            total += n
            logger.info(f"✓ {row['name']}: {n} imágenes descargadas")
        except Exception as e:
            logger.error(f"✗ {row['name']}: {e}")

    logger.info(f"Descarga completa: {total} imágenes totales")


if __name__ == "__main__":
    run_download()
