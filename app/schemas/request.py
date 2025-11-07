from typing import Literal
from pydantic import BaseModel, Field, conlist

# --- Tipos literales permitidos ---
SexoT = Literal["male", "female"]
FumadorT = Literal["yes", "no"]
RegionT = Literal["northeast", "northwest", "southeast", "southwest"]

# --- Estructura principal del registro ---
class Features(BaseModel):
    Edad: float = Field(..., description="Edad en años")
    BMI: float = Field(..., description="Índice de masa corporal (usar punto decimal)")
    Hijos: float = Field(..., description="Número de hijos")
    Sexo: SexoT = Field(..., description='Valores válidos: "male" o "female"')
    Fumador: FumadorT = Field(..., description='Valores válidos: "yes" o "no"')
    Region: RegionT = Field(
        ..., description='Valores válidos: "northeast", "northwest", "southeast" o "southwest"'
    )

# --- Cuerpo principal del POST /predict ---
class PredictPayload(BaseModel):
    """
    Cuerpo del POST /predict.
    Recibe {"features": {...}} con las columnas esperadas por el modelo.
    """
    features: Features

    # Ejemplo visible en Swagger (/docs)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": {
                        "Edad": 46,
                        "BMI": 31.1,
                        "Hijos": 2,
                        "Sexo": "male",
                        "Fumador": "yes",
                        "Region": "southeast"
                    }
                }
            ]
        }
    }

# --- Opcional: para procesamiento en lote (no usado aún) ---
class BatchPredictPayload(BaseModel):
    items: conlist(PredictPayload, min_length=1)
