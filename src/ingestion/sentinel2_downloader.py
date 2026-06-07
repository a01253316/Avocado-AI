"""
sentinel2_downloader.py
=======================
Descarga imágenes Sentinel-2 por parcela y rango de fechas usando la API
de Copernicus Data Space Ecosystem (CDSE) — gratuita y oficial.

Arquitectura actualizada (Process API / Sentinel Hub - FLOAT32):
    CDSEAuth                → obtiene/renueva el token OAuth2
    STACSearcher            → busca productos via STAC API por bbox + fecha + nubosidad
    ProcessAPIDownloader    → solicita y descarga un GeoTIFF multibanda recortado en FLOAT32
    AlphaEarthClient        → stub para futura integración con Alpha Earth
    SentinelDownloadManager → orquestador: lee el CSV, coordina todo, actualiza estados

Uso:
    # Descarga todas las parcelas del CSV
    python sentinel2_downloader.py --config configs/sentinel2.yaml \
                                   --parcels data/raw/parcels/parcelas.csv

    # Solo las primeras 5 parcelas (para pruebas)
    python sentinel2_downloader.py --parcels data/raw/parcels/parcelas.csv \
                                   --parcel-ids H1 H2 H3 H4 H5

Credenciales necesarias en .env:
    CDSE_USER=tu_email@ejemplo.com
    CDSE_PASSWORD=tu_contraseña
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("s2_downloader")


# ---------------------------------------------------------------------------
# Constantes CDSE
# ---------------------------------------------------------------------------
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)
CDSE_STAC_URL  = "https://stac.dataspace.copernicus.eu/v1"

# Tipo de colección
S2_COLLECTION  = "sentinel-2-l2a"

# Reintentos y tiempos de espera
MAX_RETRIES = 4
RETRY_BACKOFF = 2       # segundos base (se duplica en cada reintento)
DOWNLOAD_PAUSE = 1.0    # segundos entre descargas para no saturar la API


# ---------------------------------------------------------------------------
# Dataclasses de configuración
# ---------------------------------------------------------------------------
@dataclass
class DownloadConfig:
    """Parámetros de descarga leídos del YAML de configuración."""
    start_date: str
    end_date: str
    max_cloud_cover: int
    bands: list[str]
    output_dir: Path
    resolution_m: int = 10


@dataclass
class Parcel:
    """Datos de una parcela listos para búsqueda y descarga."""
    parcel_id: str
    latitude: float
    longitude: float
    bbox_west: float
    bbox_east: float
    bbox_south: float
    bbox_north: float
    output_dir: Path = field(init=False)

    def __post_init__(self):
        self.output_dir = Path("data/raw/sentinel2") / self.parcel_id

    @property
    def bbox(self) -> list[float]:
        """Formato [west, south, east, north] que usa STAC."""
        return [self.bbox_west, self.bbox_south, self.bbox_east, self.bbox_north]


# ---------------------------------------------------------------------------
# 1. Autenticación CDSE (OAuth2)
# ---------------------------------------------------------------------------
class CDSEAuth:
    """
    Gestiona el token de acceso OAuth2 para CDSE.
    Renueva el token automáticamente antes de que expire.
    """
    def __init__(self, username: str, password: str):
        self._user     = username
        self._password = password
        self._token: str | None = None
        self._expires_at: float = 0.0

    def token(self) -> str:
        """Retorna un token válido, renovándolo si es necesario."""
        if self._token is None or time.time() >= self._expires_at - 60:
            self._refresh()
        return self._token  # type: ignore[return-value]

    def _refresh(self) -> None:
        logger.debug("Renovando token CDSE...")
        resp = requests.post(
            CDSE_TOKEN_URL,
            data={
                "grant_type": "password",
                "client_id":  "cdse-public",
                "username":   self._user,
                "password":   self._password,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Error de autenticación CDSE ({resp.status_code}): {resp.text}"
            )
        body = resp.json()
        self._token      = body["access_token"]
        self._expires_at = time.time() + body.get("expires_in", 600)
        logger.info("Token CDSE obtenido (válido ~%d min)", body.get("expires_in", 600) // 60)

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token()}"}


# ---------------------------------------------------------------------------
# 2. Búsqueda STAC
# ---------------------------------------------------------------------------
class STACSearcher:
    """Busca productos Sentinel-2 disponibles en CDSE via STAC API."""
    def __init__(self, auth: CDSEAuth):
        self._auth = auth

    def search(
        self,
        parcel: Parcel,
        config: DownloadConfig,
        page_size: int = 100,
    ) -> Generator[dict, None, None]:
        
        endpoint = f"{CDSE_STAC_URL}/search"
        payload = {
            "collections": [S2_COLLECTION],
            "bbox":        parcel.bbox,
            "datetime":    f"{config.start_date}T00:00:00Z/{config.end_date}T23:59:59Z",
            "limit":       page_size,
            "filter-lang": "cql2-json",
            "filter": {
                "op": "<=",
                "args": [{"property": "eo:cloud_cover"}, config.max_cloud_cover]
            },
        }

        # Estado de paginación — se actualiza en cada vuelta según la spec STAC API
        # https://api.stacspec.org/v1.0.0/item-search/#tag/Item-Search/operation/getItemSearch
        next_url:    str | None = None
        next_method: str        = "POST"   # método HTTP del link "next"
        next_body:   dict       = {}       # cuerpo si method == POST
        next_merge:  bool       = False    # si True, fusionar con el payload original
        page_num = 1

        while True:
            # Primera página: POST con el payload completo.
            # Páginas siguientes: respetar el método que indica el link "next".
            #   CDSE devuelve next con method=GET pero sin el parámetro collections,
            #   lo que causa HTTP 400 si se sigue ciegamente como GET.
            #   El fix: cuando el GET-link no contenga collections, re-enviar como
            #   POST con el payload original + el token de paginación del link.
            if next_url:
                if next_method == "POST":
                    body = {**payload, **next_body} if next_merge else (next_body or payload)
                    data = self._post_with_retry(next_url, body)
                else:
                    # GET: solo usamos si la URL ya lleva collections embebido
                    if "collections" in next_url:
                        data = self._get_with_retry(next_url)
                    else:
                        # CDSE bug: GET link sin collections → reconvertir a POST
                        logger.debug(
                            "Paginación: next link GET sin 'collections' → re-enviando como POST"
                        )
                        data = self._post_with_retry(endpoint, {**payload, **next_body})
            else:
                data = self._post_with_retry(endpoint, payload)

            if not data:
                break

            features = data.get("features", [])
            if not features:
                break

            logger.debug(
                "Parcela %s: página %d → %d productos",
                parcel.parcel_id, page_num, len(features),
            )
            for feature in features:
                yield feature

            links    = data.get("links", [])
            next_obj = next((l for l in links if l.get("rel") == "next"), None)
            if next_obj:
                next_url    = next_obj.get("href")
                next_method = next_obj.get("method", "GET").upper()
                next_body   = next_obj.get("body",  {}) or {}
                next_merge  = bool(next_obj.get("merge", False))
                page_num   += 1
            else:
                break

    def _post_with_retry(self, url: str, payload: dict) -> dict:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    url,
                    json    = payload,
                    headers = {**self._auth.headers(), "Content-Type": "application/json"},
                    timeout = 60,
                )
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 429:
                    wait = RETRY_BACKOFF ** attempt
                    logger.warning("Rate limit CDSE, esperando %ds...", wait)
                    time.sleep(wait)
                    continue
                logger.error("STAC POST HTTP %d: %s", resp.status_code, resp.text[:400])
                return {}
            except requests.RequestException as e:
                if attempt == MAX_RETRIES:
                    logger.error("STAC POST falló definitivamente: %s", e)
                    return {}
                wait = RETRY_BACKOFF ** attempt
                time.sleep(wait)
        return {}

    def _get_with_retry(self, url: str) -> dict:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(
                    url,
                    headers = self._auth.headers(),
                    timeout = 60,
                )
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 429:
                    time.sleep(RETRY_BACKOFF ** attempt)
                    continue
                logger.error("STAC GET HTTP %d: %s", resp.status_code, resp.text[:300])
                return {}
            except requests.RequestException:
                if attempt == MAX_RETRIES:
                    return {}
                time.sleep(RETRY_BACKOFF ** attempt)
        return {}


# ---------------------------------------------------------------------------
# 3. Descarga de bandas (Process API / Sentinel Hub - Arreglado FLOAT32)
# ---------------------------------------------------------------------------
class ProcessAPIDownloader:
    """
    Descarga un GeoTIFF recortado y multibanda usando la Process API de CDSE.
    Optimizado en precisión FLOAT32 para evitar truncamiento a cero (TIFs negros).
    """
    def __init__(self, auth: CDSEAuth, dry_run: bool = False):
        self._auth = auth
        self._dry_run = dry_run
        self._endpoint = "https://sh.dataspace.copernicus.eu/api/v1/process"

    def download_product(
        self,
        product: dict,
        parcel: Parcel,
        bands: list[str],
    ) -> Path | None:
        
        date_str = self._parse_date(product)
        out_dir  = parcel.output_dir / date_str
        out_file = out_dir / "parcel_multiband.tif"

        if out_file.exists():
            logger.debug("  ↳ %s/%s ya descargado (TIFF), skip", parcel.parcel_id, date_str)
            return None

        if self._dry_run:
            logger.info("  [DRY-RUN] Solicitaría TIFF recortado para %s/%s", parcel.parcel_id, date_str)
            return None

        out_dir.mkdir(parents=True, exist_ok=True)

        # Guardar metadata del STAC
        (out_dir / "metadata.json").write_text(
            json.dumps(product.get("properties", {}), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Usamos FLOAT32 para que los decimales de reflectancia (0.0 a 1.0) no se trunquen a 0
        bands_json = json.dumps(bands)
        evalscript = f"""//VERSION=3
        function setup() {{
            return {{
                input: {bands_json},
                output: {{ bands: {len(bands)}, sampleType: "FLOAT32" }}
            }};
        }}
        function evaluatePixel(sample) {{
            return [{", ".join([f"sample.{b}" for b in bands])}];
        }}
        """

        payload = {
            "input": {
                "bounds": {
                    "bbox": parcel.bbox,
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {
                                "from": f"{date_str}T00:00:00Z",
                                "to": f"{date_str}T23:59:59Z"
                            }
                        }
                    }
                ]
            },
            "output": {
                "resx": 0.00009,
                "resy": 0.00009,
                "responses": [
                    {
                        "identifier": "default",
                        "format": {"type": "image/tiff"}
                    }
                ]
            },
            "evalscript": evalscript
        }

        success = self._download_tiff(payload, out_file)
        
        if success:
            logger.info("  ✓ %s/%s (TIFF Multibanda guardado)", parcel.parcel_id, date_str)
            return out_dir
        else:
            logger.warning("  ↳ %s/%s: falló la generación del TIFF", parcel.parcel_id, date_str)
            return None

    def _parse_date(self, product: dict) -> str:
        raw = product.get("properties", {}).get("datetime", "")
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        except ValueError:
            return raw[:10] if len(raw) >= 10 else "unknown-date"

    def _download_tiff(self, payload: dict, dest: Path) -> bool:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    self._endpoint,
                    headers={
                        "Authorization": f"Bearer {self._auth.token()}",
                        "Content-Type": "application/json",
                        "Accept": "image/tiff"
                    },
                    json=payload,
                    timeout=120
                )
                
                if resp.status_code in [429, 500, 502, 503, 504]:
                    wait = RETRY_BACKOFF ** attempt
                    logger.warning("Process API ocupada (HTTP %d). Reintento en %ds", resp.status_code, wait)
                    time.sleep(wait)
                    continue
                    
                resp.raise_for_status()
                
                # Validación crítica: Verificar que no sea un JSON de error disfrazado de TIFF
                content_type = resp.headers.get("Content-Type", "")
                if "image/tiff" not in content_type:
                    logger.error("    Error: El servidor no devolvió un TIFF válido. Devolvió: %s. Body: %s", content_type, resp.text[:200])
                    return False

                with dest.open("wb") as f:
                    f.write(resp.content)
                return True
                
            except requests.RequestException as e:
                if attempt == MAX_RETRIES:
                    logger.error("    Fallo definitivo en Process API: %s", e)
                    if dest.exists():
                        dest.unlink()
                    return False
                wait = RETRY_BACKOFF ** attempt
                time.sleep(wait)
                
        return False


# ---------------------------------------------------------------------------
# 4. Alpha Earth (stub — listo para integrar)
# ---------------------------------------------------------------------------
class AlphaEarthClient:
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._base_url = os.getenv("ALPHA_EARTH_BASE_URL", "https://api.alphaearth.ai/v1")

    def search(self, parcel: Parcel, config: DownloadConfig) -> list[dict]:
        logger.warning("AlphaEarthClient.search() aún no implementado.")
        return []

    def download(self, product: dict, parcel: Parcel, bands: list[str]) -> Path | None:
        logger.warning("AlphaEarthClient.download() aún no implementado.")
        return None

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}", "Accept": "application/json"}


# ---------------------------------------------------------------------------
# 5. Orquestador principal
# ---------------------------------------------------------------------------
class SentinelDownloadManager:
    def __init__(
        self,
        config: DownloadConfig,
        auth: CDSEAuth,
        dry_run: bool = False,
    ):
        self._config     = config
        self._searcher   = STACSearcher(auth)
        self._downloader = ProcessAPIDownloader(auth, dry_run=dry_run)
        self._dry_run    = dry_run

    def run(
        self,
        parcels_csv: Path,
        parcel_ids: list[str] | None = None,
    ) -> None:
        df = pd.read_csv(parcels_csv)

        if parcel_ids:
            df = df[df["parcel_id"].isin(parcel_ids)].copy()
            logger.info("Filtrando a %d parcelas: %s", len(df), parcel_ids)

        total    = len(df)
        ok_count = 0
        audit: list[dict] = []

        logger.info("=" * 60)
        logger.info("Iniciando descarga — %d parcelas | %s → %s",
                    total, self._config.start_date, self._config.end_date)
        logger.info("Bandas: %s | Nubes ≤ %d%%",
                    self._config.bands, self._config.max_cloud_cover)
        if self._dry_run:
            logger.info("[DRY-RUN] No se escribirá nada en disco")
        logger.info("=" * 60)

        for idx, row in df.iterrows():
            parcel = Parcel(
                parcel_id  = row["parcel_id"],
                latitude   = row["latitude"],
                longitude  = row["longitude"],
                bbox_west  = row["bbox_west"],
                bbox_east  = row["bbox_east"],
                bbox_south = row["bbox_south"],
                bbox_north = row["bbox_north"],
            )

            logger.info("[%d/%d] Parcela %s", idx + 1, total, parcel.parcel_id)

            products_found = 0
            products_ok    = 0

            for product in self._searcher.search(parcel, self._config):
                products_found += 1
                result = self._downloader.download_product(
                    product, parcel, self._config.bands
                )
                if result is not None:
                    products_ok += 1
                time.sleep(DOWNLOAD_PAUSE)

            logger.info(
                "  → %d productos encontrados, %d descargados",
                products_found, products_ok
            )

            if products_ok > 0 and not self._dry_run:
                df.at[idx, "download_ready"] = True
                ok_count += 1

            audit.append({
                "parcel_id":        parcel.parcel_id,
                "products_found":   products_found,
                "products_downloaded": products_ok,
                "timestamp":        datetime.now(timezone.utc).isoformat(),
            })

        if not self._dry_run:
            df.to_csv(parcels_csv, index=False)
            logger.info("CSV actualizado: %s", parcels_csv)

        audit_path = self._config.output_dir / "download_audit.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._dry_run:
            audit_path.write_text(
                json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        logger.info("=" * 60)
        logger.info("Descarga completa: %d/%d parcelas con datos", ok_count, total)
        logger.info("Auditoría guardada en: %s", audit_path)
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Helpers de configuración
# ---------------------------------------------------------------------------
def load_config(config_path: Path) -> DownloadConfig:
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return DownloadConfig(
        start_date      = cfg["time_range"]["start"],
        end_date        = cfg["time_range"]["end"],
        max_cloud_cover = cfg["image"]["max_cloud_cover_pct"],
        bands           = [b["name"] for b in cfg["bands"]],
        output_dir      = Path(cfg["paths"]["raw_sentinel2"]),
        resolution_m    = cfg["image"]["resolution_m"],
    )


def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        logger.error(
            "Variable de entorno '%s' no definida. "
            "Agrega tus credenciales en el archivo .env",
            name,
        )
        sys.exit(1)
    return val


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga imágenes Sentinel-2 para parcelas (Process API - FLOAT32)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sentinel2.yaml"),
        help="Ruta al YAML de configuración (default: configs/sentinel2.yaml)",
    )
    parser.add_argument(
        "--parcels",
        type=Path,
        default=Path("data/raw/parcels/parcelas.csv"),
        help="CSV de parcelas generado por kml_to_csv.py",
    )
    parser.add_argument(
        "--parcel-ids",
        nargs="+",
        metavar="ID",
        help="IDs específicos a descargar (ej: H1 H2 H5). Si se omite, descarga todo.",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Fecha de inicio YYYY-MM-DD (sobreescribe el YAML)",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="Fecha de fin YYYY-MM-DD (sobreescribe el YAML)",
    )
    parser.add_argument(
        "--max-clouds",
        type=int,
        help="% máximo de nubosidad (sobreescribe el YAML)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra qué descargaría sin escribir nada en disco",
    )
    parser.add_argument(
        "--source",
        choices=["cdse", "alpha-earth"],
        default="cdse",
        help="Fuente de datos (default: cdse)",
    )
    args = parser.parse_args()

    if not args.parcels.exists():
        logger.error("CSV de parcelas no encontrado: %s", args.parcels)
        sys.exit(1)
    if not args.config.exists():
        logger.error("Config no encontrada: %s", args.config)
        sys.exit(1)

    config = load_config(args.config)

    if args.start:
        config.start_date = args.start
    if args.end:
        config.end_date = args.end
    if args.max_clouds is not None:
        config.max_cloud_cover = args.max_clouds

    if args.source == "cdse":
        auth    = CDSEAuth(require_env("CDSE_USER"), require_env("CDSE_PASSWORD"))
        manager = SentinelDownloadManager(config, auth, dry_run=args.dry_run)
    else:
        api_key = require_env("ALPHA_EARTH_API_KEY")
        client  = AlphaEarthClient(api_key)
        logger.error("Alpha Earth downloader pendiente de implementación completa.")
        sys.exit(1)

    manager.run(args.parcels, parcel_ids=args.parcel_ids)


if __name__ == "__main__":
    main()