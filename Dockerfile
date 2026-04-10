# AstrAI Dockerfile - Multi-stage Build (Optimized)

# Build stage - use base image with minimal build tools
FROM nvidia/cuda:12.6.0-base-ubuntu24.04 AS builder

WORKDIR /app

# Install Python 3.12 and minimal build dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtual environment
RUN python3.12 -m venv --copies /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code and install dependencies
COPY astrai/ ./astrai/
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir . \
    --extra-index-url https://download.pytorch.org/whl/cu126

# Production stage
FROM nvidia/cuda:12.6.0-base-ubuntu24.04 AS production

WORKDIR /app

# Install Python 3.12 runtime
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.12 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY astrai/ ./astrai/
COPY scripts/ ./scripts/
COPY assets/ ./assets/
COPY pyproject.toml .
COPY README.md .

# Create non-root user
RUN useradd -m astrai && chown -R astrai:astrai /app
USER astrai

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1