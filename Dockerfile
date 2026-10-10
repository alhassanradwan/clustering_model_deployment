# ---------- Stage 1: builder ----------
# Has uv and does the installing. None of this stage ends up in the final
# image except the finished virtual environment copied out below.
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /usr/local/bin/uv 
#This puts uv inside the image.
#You use uv to install your Python packages.

WORKDIR /app
#That creates the folder inside the image, on the container's own Linux filesystem. 
#Nothing is created on Windows.
#Then every COPY lands inside it, because paths are relative to the WORKDIR:

#Python doesn't run .py files directly — it first converts them to a faster internal form called bytecode (.pyc files).
#Normally that happens the first time something is imported, at runtime.
#For a container, that means the very first request pays the cost of compiling pandas, scikit-learn and everything else.
#This setting says: do it now, during the build. The .pyc files are baked into the image, so the container starts ready to go.
#Do some Python startup work during the build so the container starts faster.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

#Your Dockerfile uses copy because you're moving the .venv from the builder image to the final image.    



#"These are the Python packages my project needs."
#It does not copy your code yet.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --only-group serve --no-install-project
#--frozen  → use exactly the versions in uv.lock

#--only-group serve → install only API/serving packages (FastAPI, Uvicorn, pandas, sklearn, etc.)

# ---------- Stage 2: runtime ----------
# A fresh copy of the same base image. No uv, no build leftovers.
FROM python:3.12-slim

# A normal, unprivileged account to run the API. Docker runs as root by
# default; the API only reads files, so it has no need for that.
RUN useradd --create-home --uid 1000 appuser
#creates the employee, and tells Docker: "run the application as this employee."

WORKDIR /app
#"From this point onward, use /app as the working folder inside the container."

# The only thing taken from the builder: the installed packages. The path
# must match the builder's, because the environment records where it lives.
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

#The $ sign in this case means:
#"Use the value stored inside this variable.


#Copy /app/.venv from the builder stage into /app/.venv in the new runtime stage.

#This creates a build-time variable called APP_VERSION.
ARG APP_VERSION=dev
ENV APP_VERSION=$APP_VERSION
#This takes the value from the build-time ARG and puts it into an environment variable that exists when the container runs.


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
#
# Run through a shell so ${PORT} is expanded. Hosting platforms choose the
# port and pass it in this variable; locally there is none, so it falls back
# to 8000 and nothing about Compose, Swarm or Kubernetes changes.
#
# `exec` replaces the shell with uvicorn, so uvicorn stays process 1 and
# receives the stop signal directly. Without it the shell would hold PID 1,
# swallow the signal, and every shutdown would wait for a 10-second timeout.
CMD ["sh", "-c", "exec uvicorn api:app --app-dir src --host 0.0.0.0 --port ${PORT:-8000}"]
#Uvicorn takes that FastAPI application and starts a web server for it.
#sh = a small Linux program that runs commands.
#-c = tells sh: "execute the command that comes next."