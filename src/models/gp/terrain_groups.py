"""
terrain_groups.py
==================
Agrupa parcelas por "tipo de terreno" (Experimento D.2 - sugerencia de los
profesores asesores: ajustar el GP sobre un conjunto de parcelas con
"mismo tipo de terreno, mismo tipo de arbol, mismo tipo de subsuelo" en
vez de una parcela aislada, para tener mas muestras por distribucion).

El catalogo no tiene un dato de suelo/elevacion real: `altitude_m` viene
en 0.0 para las 100 parcelas (placeholder sin datos, ver
data/raw/parcels/parcelas.csv). El proxy disponible es la **huella
espectral historica** de cada parcela - el promedio 2020-2026 de sus 5
indices (`mean_ndvi/ndwi/ndmi/ndre/evi` en data/datasets/manifest.csv).
Dos parcelas con la misma huella espectral comparten tipo de dosel y
retencion de agua a lo largo de varios anos - es una variable de nivel
("baseline"), no del evento de estres puntual que luego se mide con el GP
estacional, asi que agrupar por ella no es circular con la clasificacion.

El numero de grupos no se fija a ojo: se elige revisando silhouette
score para k=2..9 (ver --scan-k). k=2 da el silhouette mas alto (0.57)
pero solo separa "baseline alto" vs. "bajo" (30/70); k=4 mantiene buena
separacion (0.51) con grupos de tamano razonable (sin grupos de 1
parcela) - es el default, pero es un argumento, no un valor fijo.

Uso:
    python -m src.models.gp.terrain_groups --scan-k
    python -m src.models.gp.terrain_groups --n-groups 4
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("terrain_groups")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

FEATURES        = ["mean_ndvi", "mean_ndwi", "mean_ndmi", "mean_ndre", "mean_evi"]
DEFAULT_N_GROUPS = 4


def load_terrain_features(manifest_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    return df[["parcel_id", *FEATURES]]


def scan_k(df: pd.DataFrame, k_min: int = 2, k_max: int = 9, random_state: int = 42) -> None:
    X = StandardScaler().fit_transform(df[FEATURES].values)
    logger.info("-- Silhouette score por numero de grupos --")
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X)
        sil = silhouette_score(X, km.labels_)
        sizes = np.bincount(km.labels_).tolist()
        logger.info("  k=%d  silhouette=%.3f  tamanos=%s", k, sil, sizes)


def cluster_parcels(df: pd.DataFrame, n_groups: int, random_state: int = 42) -> dict:
    X = StandardScaler().fit_transform(df[FEATURES].values)
    km = KMeans(n_clusters=n_groups, random_state=random_state, n_init=10).fit(X)

    df = df.copy()
    df["group"] = km.labels_.astype(str)

    groups: dict[str, list[str]] = {
        g: df.loc[df["group"] == g, "parcel_id"].tolist()
        for g in sorted(df["group"].unique())
    }
    parcel_to_group = dict(zip(df["parcel_id"], df["group"]))

    return {
        "n_groups":        n_groups,
        "features":        FEATURES,
        "groups":          groups,
        "parcel_to_group": parcel_to_group,
    }


def save_groups(result: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Grupos guardados en %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agrupa parcelas por tipo de terreno (huella espectral historica)",
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/datasets/manifest.csv"))
    parser.add_argument("--n-groups", type=int, default=DEFAULT_N_GROUPS)
    parser.add_argument("--output", type=Path, default=Path("data/datasets/terrain_groups.json"))
    parser.add_argument("--scan-k", action="store_true", help="Solo imprime silhouette score por k, no guarda nada")
    args = parser.parse_args()

    df = load_terrain_features(args.manifest)
    logger.info("Parcelas con huella espectral valida: %d", len(df))

    if args.scan_k:
        scan_k(df)
        return

    result = cluster_parcels(df, args.n_groups)

    for g, ids in result["groups"].items():
        sub = df[df["parcel_id"].isin(ids)]
        logger.info(
            "Grupo %s | %2d parcelas | NDMI medio=%+.4f (sigma=%.4f) | NDVI medio=%.4f | EVI medio=%.4f",
            g, len(ids), sub["mean_ndmi"].mean(), sub["mean_ndmi"].std(),
            sub["mean_ndvi"].mean(), sub["mean_evi"].mean(),
        )

    save_groups(result, args.output)


if __name__ == "__main__":
    main()
