# RESEARCHOS API (CPU). Build:  docker build -t researchos-api .
# The image contains no model weights, datasets or databases: mount operator checkpoints read-only at /models and keep state in /data.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# Dependencies first so source edits do not invalidate this layer. CPU PyTorch keeps the image small.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml ./
RUN pip install ".[api,corpus,postgres]"

COPY model ./model
COPY agent_engine ./agent_engine
COPY research ./research
COPY api ./api
COPY scripts ./scripts
COPY configs ./configs
COPY alembic.ini ./
COPY artifacts/tokenizer ./artifacts/tokenizer

# Unprivileged user; all mutable state lives under /data.
RUN useradd --system --uid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin app \
    && mkdir -p /data /models \
    && chown app:app /data
USER 10001:10001

ENV RESEARCHOS_DATABASE_URL=sqlite:////data/researchos.db \
    RESEARCHOS_ENGINE_DB=/data/engine.db \
    RESEARCHOS_DATA_DIR=/data/files \
    RESEARCHOS_MODELS_DIR=/models
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["python", "scripts/serve_api.py", "--host", "0.0.0.0", "--port", "8000"]
