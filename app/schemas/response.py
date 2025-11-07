from pydantic import BaseModel, Field

class PredictResponse(BaseModel):
    """
    Respuesta del endpoint /predict
    """
    prediction: float = Field(..., description="Valor predicho por el modelo.")
    model: str = Field(..., description="Versión/identificador del modelo usado.")
    inference_ms: float = Field(..., description="Tiempo de inferencia en milisegundos.")

    # Ejemplo visible en Swagger
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prediction": 15243.75,
                    "model": "nb06-gridsearch",
                    "inference_ms": 3.72
                }
            ]
        }
    }

class HealthResponse(BaseModel):
    """
    Respuesta del endpoint /health
    """
    status: str = Field(..., description="Estado del servicio ('ok' si está funcionando).")
    version: str = Field(..., description="Versión del modelo actualmente cargado.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"status": "ok", "version": "nb06-gridsearch"}
            ]
        }
    }
