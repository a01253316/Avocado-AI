"""
utils/sentinel_auth.py
Autenticación OAuth2 con Sentinel Hub.
Gestiona el token y lo renueva automáticamente antes de que expire.
"""
import time
import requests
import yaml
from pathlib import Path


class SentinelHubAuth:
    """
    Obtiene y renueva el token de acceso de Sentinel Hub (OAuth2 client_credentials).
    Uso:
        auth = SentinelHubAuth("configs/credentials.yaml")
        headers = auth.headers()
    """

    TOKEN_URL = "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token"

    def __init__(self, credentials_path: str = "configs/credentials.yaml"):
        creds = yaml.safe_load(Path(credentials_path).read_text())["sentinel_hub"]
        self._client_id = creds["client_id"]
        self._client_secret = creds["client_secret"]
        self._token: str | None = None
        self._expires_at: float = 0.0

    # ── API pública ──────────────────────────────────────────────────────────

    def token(self) -> str:
        """Devuelve un token válido, renovándolo si está a menos de 60 s de expirar."""
        if time.time() >= self._expires_at - 60:
            self._refresh()
        return self._token

    def headers(self) -> dict:
        """Devuelve las cabeceras HTTP listas para usar en requests."""
        return {
            "Authorization": f"Bearer {self.token()}",
            "Content-Type": "application/json",
        }

    # ── Internos ─────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        resp = requests.post(
            self.TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        self._expires_at = time.time() + data.get("expires_in", 3600)
