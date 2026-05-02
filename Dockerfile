# ==============================================================================
# SAGE 5.0 — Multi-stage Docker build
# ==============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY sage/__init__.py sage/__init__.py

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir tiktoken>=0.5.0 tabulate>=0.9.0

COPY . .
RUN pip install --no-cache-dir -e .

# ==============================================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="Gaurav Batule" \
      org.opencontainers.image.title="SAGE" \
      org.opencontainers.image.description="Hybrid Graph-Cortex Language Model — Wave Propagation + Resonance Memory" \
      org.opencontainers.image.url="https://github.com/gauravbatule/SAGE" \
      org.opencontainers.image.version="5.0.0" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

EXPOSE 8888

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8888/api/health')" || exit 1

ENTRYPOINT ["python"]
CMD ["serve.py"]
