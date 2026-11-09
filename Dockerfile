# syntax=docker/dockerfile:1
# RESEARCHOS production image (API, worker and migration entry points share it).   docker build -t researchos:local .
#
# Three stages: the React app is built with Node, the Python dependencies with pip into a virtualenv, and the runtime image receives only
# the finished artifacts (no Node, no compilers, no pip cache, no test code). It contains no secrets, model weights, datasets or databases:
# configuration comes from RESEARCHOS_* environment variables at run time, operator checkpoints are mounted read-only at /models and all
# state lives under /data. Nothing here has been built or run on the development machine (Docker is not installed there); the CI "docker"
# job and docs/DOCKER.md describe how to verify it.

# ---------------------------------------------------------------- 1. frontend (Vite + React + TypeScript)
FROM node:20-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build          # tsc --noEmit && vite build  ->  /build/dist

# ---------------------------------------------------------------- 2. python dependencies
FROM python:3.11-slim-bookworm AS pybuild
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# CPU PyTorch first (the default PyPI wheel pulls CUDA libraries and is several GB). Versions of everything else are pinned by
# requirements/constraints.txt so that two builds of the same commit install the same packages.
RUN pip install --upgrade pip && pip install "torch>=2.4" --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml requirements/constraints.txt ./
RUN pip install -c constraints.txt ".[api,corpus,postgres]"

# ---------------------------------------------------------------- 3. runtime
FROM python:3.11-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:$PATH"
# tini runs as PID 1: it forwards SIGTERM to the application and reaps the helper processes the API starts (document extraction,
# training and experiment children), which a bare Python PID 1 would leave as zombies.
RUN apt-get update && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*
COPY --from=pybuild /opt/venv /opt/venv
WORKDIR /app
COPY model ./model
COPY agent_engine ./agent_engine
COPY research ./research
COPY api ./api
COPY scripts ./scripts
COPY configs ./configs
COPY alembic.ini ./
COPY artifacts/tokenizer ./artifacts/tokenizer
COPY --from=frontend /build/dist ./frontend/dist

# Unprivileged user; all mutable state lives under /data (a volume), everything else is read-only-capable.
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

STOPSIGNAL SIGTERM
ENTRYPOINT ["tini", "--"]
# uvicorn (no reload, no dev server); forwarded headers are honoured only from RESEARCHOS_TRUSTED_PROXIES.
CMD ["python", "scripts/serve_api.py", "--host", "0.0.0.0", "--port", "8000"]
