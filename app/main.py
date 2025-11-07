# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
# Asegúrate de importar el router correcto:
from app.api.endpoints import router as predict_router

app = FastAPI(title=settings.APP_NAME, version=settings.MODEL_VERSION)

# --- CORS (por si usas frontend o pruebas remotas) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # cambia por dominios específicos en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Rutas principales ---
app.include_router(predict_router, prefix="/api")

@app.get("/")
def root():
    return {
        "app": settings.APP_NAME,
        "env": settings.ENVIRONMENT,
        "status": "running",
        "model_version": settings.MODEL_VERSION
    }
