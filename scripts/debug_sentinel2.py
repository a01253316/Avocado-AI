"""
debug_sentinel2.py v2
=====================
Prueba TRES APIs de CDSE para encontrar cuál devuelve datos reales:
    A) STAC API       — stac.dataspace.copernicus.eu/v1
    B) OData API      — catalogue.dataspace.copernicus.eu/odata/v1
    C) OpenSearch API — catalogue.dataspace.copernicus.eu/resto
"""
import json, os, sys, requests
from dotenv import load_dotenv

load_dotenv()
SEP  = "─" * 60
def hdr(t): print(f"\n{SEP}\n  {t}\n{SEP}")
def ok(m):  print(f"  ✅  {m}")
def fail(m):print(f"  ❌  {m}")
def info(m):print(f"       {m}")

# ── Coordenadas H1 ──────────────────────────────────────────
BBOX = [-103.48985528, 19.66091863, -103.48508559, 19.66541018]
POLY = "POLYGON((-103.4899 19.6609,-103.4899 19.6654,-103.4851 19.6654,-103.4851 19.6609,-103.4899 19.6609))"
START, END = "2023-01-01", "2023-03-31"

CDSE_TOKEN_URL = ("https://identity.dataspace.copernicus.eu"
                  "/auth/realms/CDSE/protocol/openid-connect/token")

# ── Auth ────────────────────────────────────────────────────
user = os.getenv("CDSE_USER"); pwd = os.getenv("CDSE_PASSWORD")
resp = requests.post(CDSE_TOKEN_URL,
    data={"grant_type":"password","client_id":"cdse-public",
          "username":user,"password":pwd}, timeout=30)
if resp.status_code != 200:
    print("❌ Auth fallida"); sys.exit(1)
token = resp.json()["access_token"]
ok(f"Token OK ({user})")
HDR = {"Authorization": f"Bearer {token}"}

# ============================================================
# A — STAC sin filtro de colección (busca lo que sea)
# ============================================================
hdr("A — STAC sin filtro de colección (cualquier dato en el bbox)")

for stac_url in [
    "https://stac.dataspace.copernicus.eu/v1/search",
    "https://catalogue.dataspace.copernicus.eu/stac/search",
]:
    info(f"Probando: {stac_url}")
    r = requests.post(stac_url, json={
        "bbox": BBOX,
        "datetime": f"{START}T00:00:00Z/{END}T23:59:59Z",
        "limit": 3,
    }, headers=HDR, timeout=30)
    info(f"  HTTP {r.status_code}")
    if r.status_code == 200:
        feats = r.json().get("features", [])
        if feats:
            ok(f"  {len(feats)} productos — colección: "
               f"{feats[0].get('collection', feats[0].get('properties',{}).get('collection','?'))}")
            break
        else:
            info(f"  0 resultados")

# ============================================================
# B — STAC probando nombres de colección conocidos
# ============================================================
hdr("B — STAC probando distintos nombres de colección S2")

S2_NAMES = [
    "sentinel-2-l2a", "SENTINEL-2", "SENTINEL-2-L2A",
    "sentinel-2-l1c", "Sentinel2",  "sentinel2",
]
STAC_SEARCH = "https://stac.dataspace.copernicus.eu/v1/search"

for col in S2_NAMES:
    r = requests.post(STAC_SEARCH, json={
        "collections": [col],
        "bbox": BBOX,
        "datetime": f"{START}T00:00:00Z/{END}T23:59:59Z",
        "limit": 1,
    }, headers=HDR, timeout=30)
    n = len(r.json().get("features", [])) if r.status_code == 200 else -1
    status = "✅  ENCONTRADO" if n > 0 else ("HTTP " + str(r.status_code) if r.status_code != 200 else "0 resultados")
    print(f"  '{col}' → {status}")
    if n > 0:
        props = r.json()["features"][0].get("properties", {})
        info(f"  cloud={props.get('eo:cloud_cover','?')} date={props.get('datetime','?')[:10]}")
        break

# ============================================================
# C — STAC lista de colecciones del nuevo endpoint
# ============================================================
hdr("C — Colecciones en stac.dataspace.copernicus.eu/v1")

r = requests.get("https://stac.dataspace.copernicus.eu/v1/collections",
                 headers=HDR, timeout=30)
info(f"HTTP {r.status_code}")
if r.status_code == 200:
    cols = r.json()
    # Puede ser lista directa o dict con 'collections'
    if isinstance(cols, list):
        items = cols
    else:
        items = cols.get("collections", cols.get("items", []))
    ids = [c.get("id","?") for c in items] if items else []
    ok(f"{len(ids)} colecciones")
    s2 = [c for c in ids if "sent" in c.lower() or "s2" in c.lower()]
    if s2:
        ok(f"Colecciones Sentinel-2: {s2}")
    else:
        info(f"Sin Sentinel-2. Primeras 15 colecciones:")
        for c in ids[:15]: info(f"  - {c}")
    # Mostrar JSON crudo si está vacío
    if not ids:
        info(f"Respuesta cruda: {json.dumps(cols)[:400]}")

# ============================================================
# D — OData API (catálogo clásico de Copernicus)
# ============================================================
hdr("D — OData API (catalogue.dataspace.copernicus.eu/odata/v1)")

odata_url = (
    "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    f"?$filter=Collection/Name eq 'SENTINEL-2'"
    f" and OData.CSC.Intersects(area=geography'SRID=4326;{POLY}')"
    f" and ContentDate/Start gt {START}T00:00:00.000Z"
    f" and ContentDate/Start lt {END}T23:59:59.000Z"
    f" and Attributes/OData.CSC.StringAttribute/any("
    f"att:att/Name eq 'productType' and att/OData.CSC.StringAttributeValue/Value eq 'S2MSI2A')"
    f"&$top=3&$orderby=ContentDate/Start desc"
)
info(f"GET OData...")
r = requests.get(odata_url, headers=HDR, timeout=60)
info(f"HTTP {r.status_code}")
if r.status_code == 200:
    items = r.json().get("value", [])
    if items:
        ok(f"OData funciona — {len(items)} productos encontrados")
        p = items[0]
        ok(f"Nombre: {p.get('Name','?')}")
        info(f"  Fecha : {p.get('ContentDate',{}).get('Start','?')[:10]}")
        info(f"  ID    : {p.get('Id','?')}")
        info(f"  Size  : {p.get('ContentLength',0)/1e6:.1f} MB")
    else:
        fail("OData respondió pero 0 productos")
        info(f"Resp: {json.dumps(r.json())[:300]}")
else:
    fail(f"OData HTTP {r.status_code}: {r.text[:300]}")

# ============================================================
# E — OpenSearch API
# ============================================================
hdr("E — OpenSearch API (REST clásico)")

opensearch_url = (
    "https://catalogue.dataspace.copernicus.eu/resto/api/collections"
    f"/Sentinel2/search.json"
    f"?box={BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]}"
    f"&startDate={START}T00:00:00Z"
    f"&completionDate={END}T23:59:59Z"
    f"&cloudCover=[0,20]"
    f"&productType=S2MSI2A"
    f"&maxRecords=3"
)
info(f"GET OpenSearch...")
r = requests.get(opensearch_url, headers=HDR, timeout=60)
info(f"HTTP {r.status_code}")
if r.status_code == 200:
    data  = r.json()
    feats = data.get("features", [])
    if feats:
        ok(f"OpenSearch funciona — {len(feats)} productos")
        p = feats[0].get("properties", {})
        ok(f"Título: {p.get('title','?')}")
        info(f"  Fecha : {p.get('startDate','?')[:10]}")
        info(f"  Nubes : {p.get('cloudCover','?')}%")
        info(f"  ID    : {feats[0].get('id','?')}")
    else:
        fail(f"0 resultados. Total: {data.get('totalResults',0)}")
else:
    fail(f"HTTP {r.status_code}: {r.text[:200]}")

# ============================================================
# RESUMEN
# ============================================================
hdr("RESUMEN")
print("""
  Comparte este output completo para ver qué API funciona
  y actualizar el downloader al método correcto.
""")

# ============================================================
# F — STAC con sentinel-2-l2a + CQL2-JSON (confirmar filtros)
# ============================================================
hdr("F — STAC sentinel-2-l2a + filtro CQL2 de nubosidad")

r = requests.post("https://stac.dataspace.copernicus.eu/v1/search",
    json={
        "collections": ["sentinel-2-l2a"],
        "bbox": BBOX,
        "datetime": f"{START}T00:00:00Z/{END}T23:59:59Z",
        "limit": 3,
        "filter-lang": "cql2-json",
        "filter": {
            "op": "<=",
            "args": [{"property": "eo:cloud_cover"}, 20]
        },
    }, headers=HDR, timeout=30)
info(f"HTTP {r.status_code}")
if r.status_code == 200:
    feats = r.json().get("features", [])
    ok(f"Con CQL2 nubosidad≤20%: {len(feats)} productos")
    for f in feats:
        p = f.get("properties", {})
        info(f"  {p.get('datetime','?')[:10]} | nubes={p.get('eo:cloud_cover','?')}%")
else:
    fail(f"CQL2 no funciona: {r.text[:200]}")
    info("→ Se usará solo bbox+fecha sin filtro de nubosidad")

# ============================================================
# G — Assets del primer producto STAC (¿hay TIFFs por banda?)
# ============================================================
hdr("G — Assets del primer producto (bandas disponibles)")

r = requests.post("https://stac.dataspace.copernicus.eu/v1/search",
    json={
        "collections": ["sentinel-2-l2a"],
        "bbox": BBOX,
        "datetime": f"{START}T00:00:00Z/{END}T23:59:59Z",
        "limit": 1,
    }, headers=HDR, timeout=30)

if r.status_code == 200 and r.json().get("features"):
    feat   = r.json()["features"][0]
    assets = feat.get("assets", {})
    ok(f"Producto: {feat.get('id','?')[:60]}")
    ok(f"Total assets: {len(assets)}")

    # Buscar bandas específicas
    target = ["B02","B03","B04","B05","B08","B11","B8A"]
    print()
    info("── Assets de bandas relevantes ──")
    found_bands = []
    for key, val in assets.items():
        ku = key.upper()
        if any(b in ku for b in target) or any(ku.startswith(b) for b in target):
            href = val.get("href","")
            info(f"  [{key}] → {href[-60:] if len(href)>60 else href}")
            found_bands.append(key)

    if not found_bands:
        info("No se encontraron assets de bandas por nombre.")
        info("Todos los assets disponibles:")
        for key, val in list(assets.items())[:20]:
            info(f"  [{key}] type={val.get('type','?')} href_end=...{val.get('href','')[-50:]}")
else:
    fail("No se pudo obtener assets")
