"""
sentinel2_downloader.py
=======================
Descarga imágenes Sentinel-2 por parcela y rango de fechas usando la API
de Copernicus Data Space Ecosystem (CDSE) de ESA — gratuita y oficial.

Arquitectura:
    CDSEAuth            → obtiene/renueva el token OAuth2
    STACSearcher        → busca productos via STAC API por bbox + fecha + nubosidad
    BandDownloader      → descarga las bandas .tif específicas de cada producto
    AlphaEarthClient    → stub para futura integración con Alpha Earth
    SentinelDownloadManager → orquestador: lee el CSV, coordina todo, actualiza estados

Uso:
    # Descarga todas las parcelas del CSV
    python sentinel2_downloader.py --config configs/sentinel2.yaml \\
                                   --parcels data/raw/parcels/parcelas.csv

    # Solo las primeras 5 parcelas (para pruebas)
    python sentinel2_downloader.py --parcels data/raw/parcels/parcelas.csv \\
                                   --parcel-ids H1 H2 H3 H4 H5

    # Dry-run: muestra qué descargaría sin hacerlo
    python sentinel2_downloader.py --parcels data/raw/parcels/parcelas.csv \\
                                   --dry-run

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
from datetime import datetime, timedelta
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
CDSE_STAC_URL  = "https://catalogue.dataspace.copernicus.eu/stac"
CDSE_ODATA_URL = "https://zipper.dataspace.copernicus.eu/odata/v1"

# Tipo de producto: S2MSI2A = Level-2A (reflectancia de superficie, ya corregido)
# Es lo que queremos para índices de vegetación
S2_COLLECTION  = "SENTINEL-2"
S2_PRODUCT_TYPE = "S2MSI2A"

# Bandas que descargamos (solo las que necesitamos para los índices)
TARGET_BANDS = ["B02", "B03", "B04", "B05", "B08", "B11"]

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

    El token dura 10 minutos. Esta clase lo renueva automáticamente
    antes de que expire, de forma transparente para el resto del código.
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
    """
    Busca productos Sentinel-2 disponibles en CDSE via STAC API.

    Filtra por:
        - bbox de la parcela
        - rango de fechas
        - porcentaje máximo de nubosidad
        - tipo de producto (S2MSI2A = Level-2A)
    """

    def __init__(self, auth: CDSEAuth):
        self._auth = auth

    def search(
        self,
        parcel: Parcel,
        config: DownloadConfig,
        page_size: int = 100,
    ) -> Generator[dict, None, None]:
        """
        Genera los productos disponibles para una parcela (paginado).

        Cada producto es un dict GeoJSON Feature con los campos:
            id, geometry, properties (datetime, cloudCover, etc.), assets
        """
        endpoint = f"{CDSE_STAC_URL}/search"
        payload  = {
            "collections": [S2_COLLECTION],
            "bbox":        parcel.bbox,
            "datetime":    f"{config.start_date}T00:00:00Z/{config.end_date}T23:59:59Z",
            "query": {
                "eo:cloud_cover": {"lte": config.max_cloud_cover},
                "s2:product_type": {"eq": S2_PRODUCT_TYPE},
            },
            "limit": page_size,
            "sortby": [{"field": "datetime", "direction": "asc"}],
        }

        page = 1
        while True:
            payload["page"] = page
            resp = self._request_with_retry(endpoint, payload)
            features = resp.get("features", [])
            if not features:
                break
            logger.debug(
                "Parcela %s: página %d → %d productos",
                parcel.parcel_id, page, len(features)
            )
            for feature in features:
                yield feature
            # Si la página está completa seguimos paginando, si no, terminamos
            if len(features) < page_size:
                break
            page += 1

    def _request_with_retry(self, url: str, payload: dict) -> dict:
        """POST con reintentos exponenciales ante errores de red/servidor."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    headers=self._auth.headers(),
                    timeout=60,
                )
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 429:
                    wait = RETRY_BACKOFF ** attempt
                    logger.warning("Rate limit CDSE, esperando %ds...", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
            except requests.RequestException as e:
                if attempt == MAX_RETRIES:
                    raise
                wait = RETRY_BACKOFF ** attempt
                logger.warning("Error en búsqueda (intento %d/%d): %s. Reintentando en %ds",
                               attempt, MAX_RETRIES, e, wait)
                time.sleep(wait)
        return {}


# ---------------------------------------------------------------------------
# 3. Descarga de bandas
# ---------------------------------------------------------------------------
class BandDownloader:
    """
    Descarga bandas .tif individuales de un producto Sentinel-2.

    Estructura de salida:
        data/raw/sentinel2/
        └── {parcel_id}/
            └── {YYYY-MM-DD}/
                ├── B02.tif
                ├── B03.tif
                ├── B04.tif
                ├── B05.tif
                ├── B08.tif
                ├── B11.tif
                └── metadata.json
    """

    def __init__(self, auth: CDSEAuth, dry_run: bool = False):
        self._auth    = auth
        self._dry_run = dry_run

    def download_product(
        self,
        product: dict,
        parcel: Parcel,
        bands: list[str],
    ) -> Path | None:
        """
        Descarga las bandas de un producto para una parcela.

        Retorna la ruta del directorio descargado, o None si ya existía
        (skip inteligente: no re-descarga lo que ya está en disco).
        """
        # Extraer fecha del producto
        date_str = self._parse_date(product)
        out_dir  = parcel.output_dir / date_str

        if self._already_downloaded(out_dir, bands):
            logger.debug("  ↳ %s/%s ya descargado, skip", parcel.parcel_id, date_str)
            return None

        if self._dry_run:
            logger.info("  [DRY-RUN] Descargaría %s/%s", parcel.parcel_id, date_str)
            return None

        out_dir.mkdir(parents=True, exist_ok=True)

        # Guardar metadata del producto
        (out_dir / "metadata.json").write_text(
            json.dumps(product.get("properties", {}), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Descargar cada banda
        assets      = product.get("assets", {})
        downloaded  = 0
        failed: list[str] = []

        for band in bands:
            asset = self._find_band_asset(assets, band)
            if asset is None:
                logger.warning("  ↳ Banda %s no encontrada en producto %s", band, product.get("id"))
                failed.append(band)
                continue
            out_file = out_dir / f"{band}.tif"
            success  = self._download_file(asset["href"], out_file)
            if success:
                downloaded += 1
            else:
                failed.append(band)

        if failed:
            logger.warning("  ↳ %s/%s: fallaron bandas %s", parcel.parcel_id, date_str, failed)
        if downloaded > 0:
            logger.info("  ✓ %s/%s (%d/%d bandas)", parcel.parcel_id, date_str, downloaded, len(bands))

        return out_dir if downloaded > 0 else None

    # ── helpers ──────────────────────────────────────────────

    def _parse_date(self, product: dict) -> str:
        """Extrae YYYY-MM-DD del campo datetime del producto."""
        raw = product.get("properties", {}).get("datetime", "")
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        except ValueError:
            return raw[:10] if len(raw) >= 10 else "unknown-date"

    def _find_band_asset(self, assets: dict, band: str) -> dict | None:
        """
        Busca el asset correspondiente a una banda en el dict de assets del producto.
        Los nombres de asset en CDSE siguen el patrón: B02, B02_10m, etc.
        """
        # Búsqueda exacta primero, luego por prefijo
        for key, val in assets.items():
            if key.upper() == band.upper():
                return val
        for key, val in assets.items():
            if key.upper().startswith(band.upper()):
                return val
        return None

    def _already_downloaded(self, out_dir: Path, bands: list[str]) -> bool:
        """Retorna True si todas las bandas ya existen en disco."""
        if not out_dir.exists():
            return False
        return all((out_dir / f"{b}.tif").exists() for b in bands)

    def _download_file(self, url: str, dest: Path) -> bool:
        """Descarga un archivo con streaming y reintentos."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(
                    url,
                    headers=self._auth.headers(),
                    stream=True,
                    timeout=120,
                )
                resp.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        f.write(chunk)
                return True
            except requests.RequestException as e:
                if attempt == MAX_RETRIES:
                    logger.error("    Fallo definitivo descargando %s: %s", dest.name, e)
                    if dest.exists():
                        dest.unlink()  # Limpia archivo parcial
                    return False
                wait = RETRY_BACKOFF ** attempt
                logger.debug("    Reintento %d/%d para %s en %ds", attempt, MAX_RETRIES, dest.name, wait)
                time.sleep(wait)
        return False


# ---------------------------------------------------------------------------
# 4. Alpha Earth (stub — listo para integrar)
# ---------------------------------------------------------------------------
class AlphaEarthClient:
    """
    Cliente para la API de Alpha Earth.

    Por ahora es un stub. Cuando tengas las credenciales y la documentación
    de la API, implementa `search()` y `download()` siguiendo el mismo
    patrón que CDSEDownloader.

    Alpha Earth puede usarse como fuente alternativa o complementaria
    para imágenes de alta resolución o datos de cobertura de suelo.
    """

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._base_url = os.getenv("ALPHA_EARTH_BASE_URL", "https://api.alphaearth.ai/v1")

    def search(self, parcel: Parcel, config: DownloadConfig) -> list[dict]:
        """TODO: implementar búsqueda de imágenes en Alpha Earth."""
        logger.warning("AlphaEarthClient.search() aún no implementado.")
        return []

    def download(self, product: dict, parcel: Parcel, bands: list[str]) -> Path | None:
        """TODO: implementar descarga de bandas desde Alpha Earth."""
        logger.warning("AlphaEarthClient.download() aún no implementado.")
        return None

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}", "Accept": "application/json"}


# ---------------------------------------------------------------------------
# 5. Orquestador principal
# ---------------------------------------------------------------------------
class SentinelDownloadManager:
    """
    Orquesta la descarga de imágenes Sentinel-2 para todas las parcelas.

    Flujo por parcela:
        1. Leer bbox y estado del CSV
        2. Buscar productos disponibles via STAC
        3. Descargar las bandas de cada producto
        4. Actualizar la columna `download_ready` en el CSV
        5. Registrar un resumen de descargas en un JSON de auditoría
    """

    def __init__(
        self,
        config: DownloadConfig,
        auth: CDSEAuth,
        dry_run: bool = False,
    ):
        self._config     = config
        self._searcher   = STACSearcher(auth)
        self._downloader = BandDownloader(auth, dry_run=dry_run)
        self._dry_run    = dry_run

    def run(
        self,
        parcels_csv: Path,
        parcel_ids: list[str] | None = None,
    ) -> None:
        """
        Ejecuta el pipeline de descarga para las parcelas del CSV.

        Args:
            parcels_csv: Ruta al CSV generado por kml_to_csv.py
            parcel_ids:  Lista de IDs a procesar. Si es None, procesa todas.
        """
        df = pd.read_csv(parcels_csv)

        # Filtrar parcelas si se especificaron IDs concretos
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

            # Actualizar flag en el DataFrame
            if products_ok > 0 and not self._dry_run:
                df.at[idx, "download_ready"] = True
                ok_count += 1

            audit.append({
                "parcel_id":        parcel.parcel_id,
                "products_found":   products_found,
                "products_downloaded": products_ok,
                "timestamp":        datetime.utcnow().isoformat(),
            })

        # Persistir el CSV actualizado
        if not self._dry_run:
            df.to_csv(parcels_csv, index=False)
            logger.info("CSV actualizado: %s", parcels_csv)

        # Guardar auditoría
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
    """Lee el YAML de configuración y retorna un DownloadConfig."""
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
    """Lee una variable de entorno, lanza error claro si no existe."""
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
        description="Descarga imágenes Sentinel-2 para parcelas de aguacate",
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

    # Validaciones básicas
    if not args.parcels.exists():
        logger.error("CSV de parcelas no encontrado: %s", args.parcels)
        sys.exit(1)
    if not args.config.exists():
        logger.error("Config no encontrada: %s", args.config)
        sys.exit(1)

    # Cargar configuración
    config = load_config(args.config)

    # Sobreescrituras desde CLI
    if args.start:
        config.start_date = args.start
    if args.end:
        config.end_date = args.end
    if args.max_clouds is not None:
        config.max_cloud_cover = args.max_clouds

    # Inicializar cliente según fuente
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
