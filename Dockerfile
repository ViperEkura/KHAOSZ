# AstrAI Dockerfile - Minimal

# Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY astrai/ ./astrai/
COPY pyproject.toml .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# Production stage
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS production

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libpython3.12 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY astrai/ ./astrai/
COPY scripts/ ./scripts/
COPY assets/ ./assets/
COPY pyproject.toml .
COPY README.md .

RUN useradd -m -u 1000 astrai && chown -R astrai:astrai /app
USER astrai

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1