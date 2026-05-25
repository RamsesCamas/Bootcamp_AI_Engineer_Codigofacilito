# DocOps Agent — Clase 16
# Optimizado para Hugging Face Spaces (Docker SDK).
# HF Spaces espera puerto 7860.

FROM python:3.11-slim

# ── Dependencias de sistema mínimas ──
# build-essential: requerido por sentence-transformers / rank_bm25 wheels en algunos arches.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Layer de dependencias (cacheable) ──
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Código ──
COPY . .

# ── Directorio de datos persistente para ChromaDB ──
# En HF Spaces, /data es el volumen persistente estándar.
RUN mkdir -p /data/chroma /app/outputs \
 && chmod -R 777 /data /app/outputs

# Variables de entorno (valores por defecto; secrets se inyectan en runtime).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRACING_ENABLED=false \
    PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006 \
    CHROMA_PATH=/data/chroma \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port", "7860", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true", \
     "--server.enableCORS", "false"]
