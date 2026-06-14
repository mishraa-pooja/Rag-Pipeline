# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OS deps required by torch / sentence-transformers
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first to leverage Docker layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Copy app
COPY . .

# Create non-root user and give it ownership of the app + data dirs.
RUN useradd --create-home --shell /bin/bash appuser \
 && mkdir -p /app/data/uploads \
 && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Security: drop default capabilities, run as non-root, do not run privileged.
# Health check hits FastAPI's /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
