from fastapi import APIRouter, HTTPException
from app.schemas.request import PredictPayload
from app.schemas.response import PredictResponse, HealthResponse
from app.services.predictor import predict_one
from app.core.config import settings

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", version=settings.MODEL_VERSION)

@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictPayload):
    try:
        # ⬇️ convertir a dict antes de llamar al servicio
        y, latency_ms = predict_one(payload.features.model_dump())
        return PredictResponse(
            prediction=y,
            model=settings.MODEL_VERSION,
            inference_ms=round(latency_ms, 3),
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=500, detail="Inference error")

