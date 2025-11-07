# app/routes/predict.py
from __future__ import annotations

import os
from typing import Dict, Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import app.models.loader as loader
from app.core.config import settings

router = APIRouter(prefix="/api", tags=["predict"])

# ====== Esquemas ======
class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Mapa de {feature: valor}. Deben ser numéricos.",
        examples=[{"age": 42, "bmi": 27.3, "children": 2, "smoker": 1, "region": 3}],
    )

class PredictResponse(BaseModel):
    prediction: float
    model: str
    version: str
    used_order: Optional[List[str]] = None


# ====== Endpoints ======
@router.get("/health")
def health():
    """
    Health no debe cargar el modelo (para no tirar 500).
    Solo verifica que el server está vivo y si el path del modelo existe.
    """
    try:
        exists = os.path.exists(settings.MODEL_PATH)
        return {
            "status": "ok",
            "app": settings.APP_NAME,
            "env": settings.ENVIRONMENT,
            "model_path_exists": exists,
            "model_path": settings.MODEL_PATH,
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "app": settings.APP_NAME,
            "env": settings.ENVIRONMENT,
        }


@router.get("/info")
def info():
    """Resumen del modelo; si cargar falla, responde en modo degradado."""
    try:
        return loader.model_info()
    except Exception as e:
        return {
            "app_name": settings.APP_NAME,
            "environment": settings.ENVIRONMENT,
            "model_name": settings.MODEL_NAME,
            "model_version": settings.MODEL_VERSION,
            "model_path": settings.MODEL_PATH,
            "error_loading_model": str(e),
        }


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """Recibe un dict de features y devuelve la predicción."""
    try:
        yhat = loader.predict_one(payload.features)
        order = loader.get_expected_features() or list(payload.features.keys())
        return PredictResponse(
            prediction=float(yhat),
            model=settings.MODEL_NAME,
            version=settings.MODEL_VERSION,
            used_order=order,
        )
    except ValueError as ve:
        # errores de validación (features faltantes/no numéricas)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno en predicción: {e!r}")


@router.post("/reload")
def reload():
    """Recarga el modelo desde disco (útil tras desplegar un modelo nuevo)."""
    mdl, feats = loader.reload_model()
    return {
        "reloaded": True,
        "model_class": mdl.__class__.__name__,
        "has_expected_features": feats is not None,
        "n_expected_features": len(feats) if feats else None,
        "expected_features": feats,
    }
