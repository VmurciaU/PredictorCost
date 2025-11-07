# app/models/loader.py
from __future__ import annotations
import logging, os, pickle, time
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# --------------------------------------------
# Dependencias opcionales
# --------------------------------------------
try:
    from joblib import load as joblib_load  # type: ignore
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

try:
    import pandas as pd  # type: ignore
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

from app.core.config import settings

__all__ = [
    "get_model",
    "reload_model",
    "get_expected_features",
    "predict_one",
    "predict_many",
    "predict_proba_one",
    "model_info",
    "model_schema",
]

# --------------------------------------------
# Logging
# --------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    level = getattr(logging, getattr(settings, "LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.setLevel(level)

# --------------------------------------------
# Estado global
# --------------------------------------------
_model: Optional[Any] = None
_model_lock = Lock()
_expected_features: Optional[List[str]] = None
_model_mtime: Optional[float] = None
_feature_source: Optional[str] = None

# --------------------------------------------
# Config (con defaults sensatos)
# --------------------------------------------
MODEL_PATH = getattr(settings, "MODEL_PATH", "./model.pkl")
MODEL_NAME = getattr(settings, "MODEL_NAME", "unknown")
MODEL_VERSION = getattr(settings, "MODEL_VERSION", "0")
ENVIRONMENT = getattr(settings, "ENVIRONMENT", "local")
APP_NAME = getattr(settings, "APP_NAME", "app")
AUTO_RELOAD = bool(getattr(settings, "MODEL_AUTO_RELOAD", True))

# --------------------------------------------
# Helpers base
# --------------------------------------------
def _normalize_key(k: str) -> str:
    return k.strip().lower()

def _load_from_disk(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modelo no encontrado en: {path}")
    logger.info(f"Cargando modelo desde: {path}")
    if _HAS_JOBLIB:
        try:
            return joblib_load(path)
        except Exception as e:
            logger.warning(f"Fallo joblib.load, reintentando con pickle: {e!r}")
    with open(path, "rb") as f:
        return pickle.load(f)

def _maybe_autoreload(path: str) -> None:
    global _model, _model_mtime, _expected_features, _feature_source
    if not AUTO_RELOAD:
        return
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return
    if _model is not None and _model_mtime is not None and mtime > _model_mtime:
        logger.info("Detectado cambio en el archivo de modelo; recargando…")
        _model = None
        _expected_features = None
        _feature_source = None
        _model_mtime = None
        _ = get_model()

# --------------------------------------------
# Modelo
# --------------------------------------------
def get_model() -> Any:
    global _model, _expected_features, _model_mtime, _feature_source
    _maybe_autoreload(MODEL_PATH)
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            mdl = _load_from_disk(MODEL_PATH)
            names = getattr(mdl, "feature_names_in_", None)
            if names is not None:
                _expected_features = list(map(str, names))
                _feature_source = "model.feature_names_in_"
            else:
                _expected_features = None
                _feature_source = None
            _model = mdl
            try:
                _model_mtime = os.path.getmtime(MODEL_PATH)
            except OSError:
                _model_mtime = None
            logger.info(f"Modelo cargado OK ({MODEL_NAME} / {MODEL_VERSION}).")
    return _model

def reload_model() -> Tuple[Any, Optional[List[str]]]:
    global _model, _expected_features, _model_mtime, _feature_source
    with _model_lock:
        _model = None
        _expected_features = None
        _feature_source = None
        _model_mtime = None
        mdl = get_model()
    return mdl, _expected_features

def get_expected_features() -> Optional[List[str]]:
    global _expected_features
    if _expected_features is not None:
        return _expected_features
    _ = get_model()
    return _expected_features

# --------------------------------------------
# Limpieza dict → fila DataFrame (mixto)
# --------------------------------------------
def _coerce_mixed_row_for_df(payload: Dict[str, Any], feats: List[str]) -> Dict[str, Any]:
    """
    Convierte a float lo que sea numérico y preserva strings para categóricas.
    Respeta el orden/nombres exactos esperados por el modelo.
    """
    row: Dict[str, Any] = {}
    norm = {_normalize_key(str(k)): v for k, v in payload.items()}
    for f in feats:
        fn = _normalize_key(f)
        if fn not in norm:
            raise ValueError(f"Falta la feature requerida: '{f}'")
        v = norm[fn]
        if isinstance(v, str):
            vv = v.strip()
            try:
                row[f] = float(vv)
            except ValueError:
                row[f] = vv  # categórica textual (OneHotEncoder)
        else:
            row[f] = v
    return row

# --------------------------------------------
# Predicciones
# --------------------------------------------
def _predict_with_dataframe_dict(mdl: Any, payload: Dict[str, Any]) -> float:
    if not _HAS_PANDAS:
        raise RuntimeError("Este modelo requiere pandas (dict→DataFrame por nombre de columnas).")
    feats = get_expected_features()
    if not feats:
        raise RuntimeError("El modelo no expone feature_names_in_.")
    row = _coerce_mixed_row_for_df(payload, feats)
    df = pd.DataFrame([row], columns=list(feats))
    yhat = mdl.predict(df)
    return float(yhat[0])

def predict_one(features: Union[Dict[str, Any], "pd.DataFrame"]) -> float:
    mdl = get_model()
    # Si el modelo usa nombres de columna y llega dict → SIEMPRE DataFrame
    if isinstance(features, dict) and get_expected_features():
        return _predict_with_dataframe_dict(mdl, features)
    if _HAS_PANDAS and isinstance(features, pd.DataFrame):
        yhat = mdl.predict(features)
        return float(yhat[0])
    raise TypeError("Entrada no soportada: usa dict o pandas.DataFrame")

def predict_many(features: "pd.DataFrame") -> np.ndarray:
    mdl = get_model()
    if not _HAS_PANDAS:
        raise RuntimeError("pandas es requerido para predicción por lotes.")
    return np.asarray(mdl.predict(features), dtype=float)

def predict_proba_one(features: Dict[str, Any]) -> Optional[np.ndarray]:
    mdl = get_model()
    if not hasattr(mdl, "predict_proba"):
        return None
    feats = get_expected_features() or []
    if not _HAS_PANDAS:
        raise RuntimeError("pandas es requerido para predict_proba con columnas por nombre.")
    row = _coerce_mixed_row_for_df(features, feats)
    df = pd.DataFrame([row], columns=list(feats))
    return np.asarray(mdl.predict_proba(df))

# --------------------------------------------
# Info / schema
# --------------------------------------------
def model_info() -> Dict[str, Any]:
    m = get_model()
    feats = get_expected_features()
    try:
        path_exists = os.path.exists(MODEL_PATH)
        size = os.path.getsize(MODEL_PATH) if path_exists else None
        mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(MODEL_PATH))) if path_exists else None
    except Exception:
        path_exists = None
        size = None
        mtime = None
    return {
        "app_name": APP_NAME,
        "environment": ENVIRONMENT,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_path": MODEL_PATH,
        "model_exists": path_exists,
        "model_size_bytes": size,
        "model_mtime": mtime,
        "model_class": m.__class__.__name__,
        "expected_features": feats,
        "n_expected_features": len(feats) if feats else None,
        "feature_source": _feature_source,
        "auto_reload": AUTO_RELOAD,
    }

def model_schema() -> Dict[str, Any]:
    """
    Inspecta ColumnTransformer/OneHotEncoder para mostrar columnas asignadas y categorías.
    """
    out: Dict[str, Any] = {"has_column_transformer": False, "transformers": []}
    try:
        from sklearn.compose import ColumnTransformer  # type: ignore
        from sklearn.preprocessing import OneHotEncoder  # type: ignore
        m = get_model()
        ct = None
        if hasattr(m, "named_steps"):
            for _, step in m.named_steps.items():
                if isinstance(step, ColumnTransformer):
                    ct = step
                    break
        if isinstance(m, ColumnTransformer):
            ct = m
        if ct is None:
            return out
        out["has_column_transformer"] = True
        for name, trans, cols in ct.transformers_:
            entry: Dict[str, Any] = {
                "name": name,
                "cols": list(cols) if hasattr(cols, "__iter__") else cols,
                "transformer": str(type(trans)),
            }
            if "OneHotEncoder" in str(type(trans)):
                try:
                    entry["onehot_categories"] = [
                        [str(x) for x in cat_list] for cat_list in getattr(trans, "categories_", [])
                    ]
                except Exception:
                    entry["onehot_categories"] = None
            out["transformers"].append(entry)
    except Exception as e:
        out["error"] = repr(e)
    return out

# --------------------------------------------
# CLI
# --------------------------------------------
if __name__ == "__main__":
    import argparse, json, sys

    parser = argparse.ArgumentParser(description="CLI de pruebas para el modelo")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="Imprime metadatos del modelo")
    sub.add_parser("schema", help="Muestra esquema/transformers del modelo")
    sub.add_parser("template", help="Muestra plantilla de features esperadas")

    p_one = sub.add_parser("predict-one", help="Predice una muestra en JSON (string)")
    p_one.add_argument("json_features", help="Dict JSON con features")

    p_file = sub.add_parser("predict-file", help="Predice leyendo un archivo JSON")
    p_file.add_argument("json_path", help="Ruta al archivo .json")

    p_stdin = sub.add_parser("predict-stdin", help="Lee JSON desde STDIN")

    p_csv = sub.add_parser("predict-csv", help="Predice desde CSV")
    p_csv.add_argument("csv_path", help="Ruta al archivo CSV (con encabezados exactos)")

    args = parser.parse_args()

    if args.cmd == "info":
        print(json.dumps(model_info(), indent=2, ensure_ascii=False)); sys.exit(0)

    if args.cmd == "schema":
        print(json.dumps(model_schema(), indent=2, ensure_ascii=False)); sys.exit(0)

    if args.cmd == "template":
        feats = get_expected_features()
        tpl = {f: 0.0 for f in feats} if feats else {}
        print(json.dumps(tpl, indent=2, ensure_ascii=False)); sys.exit(0)

    if args.cmd == "predict-one":
        payload = json.loads(args.json_features)
        y = _predict_with_dataframe_dict(get_model(), payload)
        print(json.dumps({"prediction": y}, indent=2, ensure_ascii=False)); sys.exit(0)

    if args.cmd == "predict-file":
        with open(args.json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        y = _predict_with_dataframe_dict(get_model(), payload)
        print(json.dumps({"prediction": y}, indent=2, ensure_ascii=False)); sys.exit(0)

    if args.cmd == "predict-stdin":
        payload = json.loads(sys.stdin.read())
        y = _predict_with_dataframe_dict(get_model(), payload)
        print(json.dumps({"prediction": y}, indent=2, ensure_ascii=False)); sys.exit(0)

    if args.cmd == "predict-csv":
        if not _HAS_PANDAS:
            print("pandas no disponible", file=sys.stderr); sys.exit(2)
        df = pd.read_csv(args.csv_path)
        feats = get_expected_features()
        if feats:
            missing = [c for c in feats if c not in df.columns]
            if missing:
                print(f"Faltan columnas en el CSV: {missing}", file=sys.stderr); sys.exit(2)
            df = df[list(feats)]  # reordenar
        y = np.asarray(get_model().predict(df), dtype=float)
        df_out = df.copy(); df_out["prediction"] = y
        print(df_out.to_csv(index=False)); sys.exit(0)
