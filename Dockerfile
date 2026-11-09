
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HYBRID_RAG_QDRANT_PATH=/app/data/qdrant_storage \
    HYBRID_RAG_SQLITE_PATH=/app/data/documents.db \
    HF_HOME=/app/data/huggingface

WORKDIR /app

# Runtime library needed by some ML dependencies.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Install the application and runtime dependencies.
RUN python -m pip install --upgrade pip \
    && python -m pip install "torch==2.5.1" \
    && python -m pip install "transformers>=4.41,<5" \
    && python -m pip install .

# Prepare persistent storage and a non-root user.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app/data

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "hybrid_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
