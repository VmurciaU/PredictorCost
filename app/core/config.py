from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    # --- Información general ---
    APP_NAME: str = os.getenv("APP_NAME", "ModelPredictorCost")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "local")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    # --- Configuración del modelo ---
    MODEL_NAME: str = os.getenv("MODEL_NAME", "RandomForestRegressor")
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "nb06-gridsearch")
    MODEL_PATH: str = os.path.abspath(os.getenv("MODEL_PATH", "model/model.pkl"))

    # --- Servidor ---
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8080"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    RELOAD: bool = os.getenv("RELOAD", "False").lower() == "true"

    class Config:
        validate_assignment = True

    # --- Métodos auxiliares ---
    def info(self):
        return {
            "app_name": self.APP_NAME,
            "environment": self.ENVIRONMENT,
            "debug": self.DEBUG,
            "model_name": self.MODEL_NAME,
            "model_version": self.MODEL_VERSION,
            "model_path": self.MODEL_PATH,
            "host": self.HOST,
            "port": self.PORT,
            "log_level": self.LOG_LEVEL,
        }

settings = Settings()

if settings.DEBUG:
    print("🔧 Configuración cargada:")
    for k, v in settings.info().items():
        print(f"  {k}: {v}")
