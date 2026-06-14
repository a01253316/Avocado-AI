"""
train_save_model.py
===================
Entrena el ensamble E3 (Stacking: RF + XGB + SVM -> LR) sobre los parches reales
y guarda el modelo + scaler con joblib para ser cargados por la API.

Uso:
    python scripts/train_save_model.py
    python scripts/train_save_model.py --patches data/datasets/patches --out models/
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False

# ── Constantes (deben coincidir con api/features.py) ────────────────────────
CHANNEL_NAMES  = ["NDVI", "NDWI", "NDMI", "NDRE", "EVI"]
STAT_NAMES     = ["mean", "std", "min", "max", "p25", "p75", "trend"]
NDMI_CH        = 2
WINDOW_SIZE    = 24
WINDOW_STEP    = 4
CLASS_NAMES    = ["Sin estrés", "Moderado", "Severo"]


def load_normalizer(norm_path: pathlib.Path) -> tuple[float, float]:
    with open(norm_path) as f:
        ns = json.load(f)
    stats = ns["stats"]["NDMI"]
    denom = stats["max"] - stats["min"]
    t_mod = (-0.10 - stats["min"]) / denom
    t_sev = (-0.20 - stats["min"]) / denom
    return t_mod, t_sev


def extract_window_features(window: np.ndarray) -> np.ndarray:
    feats = []
    t = np.arange(window.shape[0], dtype=np.float32)
    for c in range(window.shape[1]):
        col = window[:, c]
        slope_raw = np.polyfit(t, col, 1)[0]
        slope = float(slope_raw) if np.isfinite(slope_raw) else 0.0
        feats += [
            float(np.mean(col)),
            float(np.std(col)),
            float(np.min(col)),
            float(np.max(col)),
            float(np.percentile(col, 25)),
            float(np.percentile(col, 75)),
            slope,
        ]
    return np.array(feats, dtype=np.float32)


def load_patch_features(
    patches_dir: pathlib.Path,
    t_mod: float,
    t_sev: float,
    window_size: int = WINDOW_SIZE,
    step: int = WINDOW_STEP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    npz_files = sorted(patches_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No hay archivos .npz en {patches_dir}")

    all_X, all_y, all_g = [], [], []
    for npz_path in npz_files:
        p    = np.load(npz_path)
        data = np.nan_to_num(p["data"].astype(np.float32), nan=0.0)
        ts   = data.mean(axis=(2, 3))
        T    = len(ts)
        for start in range(0, T - window_size + 1, step):
            win = ts[start:start + window_size]
            m   = float(np.mean(win[:, NDMI_CH]))
            lbl = 2 if m < t_sev else (1 if m < t_mod else 0)
            all_X.append(extract_window_features(win))
            all_y.append(lbl)
            all_g.append(npz_path.stem)

    X      = np.vstack(all_X)
    y      = np.array(all_y)
    gmap   = {g: i for i, g in enumerate(sorted(set(all_g)))}
    groups = np.array([gmap[g] for g in all_g])
    return X, y, groups


def build_stacking() -> StackingClassifier:
    rf = RandomForestClassifier(
        n_estimators=200, max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    boosting = (
        xgb.XGBClassifier(
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", random_state=42, n_jobs=-1,
        )
        if HAS_XGB
        else GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42,
        )
    )
    svm = CalibratedClassifierCV(
        SVC(C=10, kernel="rbf", gamma="scale", random_state=42), cv=3
    )
    return StackingClassifier(
        estimators=[("rf", rf), ("xgb", boosting), ("svm", svm)],
        final_estimator=LogisticRegression(max_iter=500, C=1.0, random_state=42),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches", default="data/datasets/patches")
    parser.add_argument("--norm",    default="data/datasets/normalizer_stats.json")
    parser.add_argument("--out",     default="models")
    args = parser.parse_args()

    patches_dir = pathlib.Path(args.patches)
    norm_path   = pathlib.Path(args.norm)
    out_dir     = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando umbrales NDMI...")
    t_mod, t_sev = load_normalizer(norm_path)
    print(f"  t_mod={t_mod:.4f}  t_sev={t_sev:.4f}")

    print("Extrayendo características de parches...")
    X, y, groups = load_patch_features(patches_dir, t_mod, t_sev)
    counts = np.bincount(y, minlength=3)
    print(f"  Dataset: {X.shape}  clases={dict(zip(CLASS_NAMES, counts))}")

    # Split train/test con GroupShuffleSplit (sin data-leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    scaler   = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("Entrenando E3 Stacking...")
    t0    = time.perf_counter()
    model = build_stacking()
    model.fit(X_train_s, y_train)
    elapsed = time.perf_counter() - t0
    print(f"  Entrenamiento: {elapsed:.1f}s")

    y_pred = model.predict(X_test_s)
    f1m    = f1_score(y_test, y_pred, average="macro")
    print(f"\nF1-macro (test): {f1m:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # Guardar modelo, scaler y umbrales
    model_path  = out_dir / "ensemble_stacking.joblib"
    scaler_path = out_dir / "ensemble_scaler.joblib"
    meta_path   = out_dir / "ensemble_meta.json"

    joblib.dump(model,  model_path)
    joblib.dump(scaler, scaler_path)
    meta = {
        "t_mod": t_mod,
        "t_sev": t_sev,
        "f1_macro_test": round(f1m, 4),
        "n_features": X.shape[1],
        "class_names": CLASS_NAMES,
        "feature_names": [f"{c}_{s}" for c in CHANNEL_NAMES for s in STAT_NAMES],
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nModelo guardado en:  {model_path}")
    print(f"Scaler guardado en:  {scaler_path}")
    print(f"Metadata guardada en: {meta_path}")


if __name__ == "__main__":
    main()
