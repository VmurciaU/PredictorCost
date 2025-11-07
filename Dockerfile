# ===== Imagen base ligera de Python =====
FROM python:3.12-slim

# Mejoras de rendimiento y logs en stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Carpeta de trabajo dentro del contenedor
WORKDIR /app

# (Opcional) Paquetes del sistema si algún día se requieren; lo dejamos mínimo
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copiar dependencias e instalarlas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Exponer el puerto (Render usa env PORT; local usamos 8080)
EXPOSE 8080

# Comando de arranque: usa PORT si Render la inyecta; 8080 por defecto
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
