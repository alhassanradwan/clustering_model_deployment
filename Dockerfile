# ---------- Stage 1: builder ----------
# Has uv and does the installing. None of this stage ends up in the final
# image except the finished virtual environment copied out below.
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /usr/local/bin/uv

WORKDIR /app

# Compile .pyc files now so the API starts faster; copy rather than link so
# the environment is self-contained when moved to the next stage.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install only the `serve` group from pyproject.toml, at the versions pinned
# in uv.lock. The training tooling (DVC, MLflow, Hydra) is left out.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --only-group serve --no-install-project


# ---------- Stage 2: runtime ----------
# A fresh copy of the same base image. No uv, no build leftovers.
FROM python:3.12-slim

# A normal, unprivileged account to run the API. Docker runs as root by
# default; the API only reads files, so it has no need for that.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# The only thing taken from the builder: the installed packages. The path
# must match the builder's, because the environment records where it lives.
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# The code and the model artifacts the API loads at startup.
COPY src/ src/
COPY models/ models/
COPY reports/cluster_profile.csv reports/

EXPOSE 8000

# Everything after this line, including the CMD, runs as appuser.
USER appuser

# Docker calls /health every 30 seconds and marks the container unhealthy after
# three failures in a row. The slim image has no curl, so Python makes the
# request; urlopen raises on an error status, which exits non-zero.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 0.0.0.0, not 127.0.0.1: inside a container, 127.0.0.1 means "only the
# container itself", so nothing outside could reach the API.
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
