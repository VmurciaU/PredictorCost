import time
from typing import Any, Dict, Tuple

from app.models.loader import (
    predict_one as _loader_predict_one,
    get_expected_features,
)

def _validate_keys(features: Dict[str, Any]) -> None:
    """
    Valida que vengan todas las columnas que el pipeline espera.
    Si el modelo no expone expected features, no valida (compatibilidad).
    """
    expected = get_expected_features()
    if expected is None:
        return
    missing = [c for c in expected if c not in features]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

def predict_one(features: Dict[str, Any]) -> Tuple[float, float]:
    """
    Ejecuta la predicción delegando toda la lógica al loader y mide latencia.
    Retorna: (predicción, latencia_ms)
    """
    _validate_keys(features)

    t0 = time.perf_counter()
    y = float(_loader_predict_one(features))  # el loader ya hace coerción/orden
    ms = (time.perf_counter() - t0) * 1000.0
    return y, ms
