# JANNUS container image.
#
# Rewritten in v1.50. The v1.40 image copied compiled site-packages from a
# Debian bookworm builder into an Ubuntu jammy runtime and then installed a
# different Python than the one the packages were built for. It also discarded
# /install/bin (so no console scripts survived), shipped no pip (so nothing
# could be repaired in place), and ran everything as root. CI built the image
# but never ran it, so none of that was ever noticed.
#
# This version uses one base image throughout, installs the package properly,
# and runs as an unprivileged user.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Lets jannus.core.paths find configs/ and model/ regardless of cwd.
    JANNUS_HOME=/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip ca-certificates && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer first so source edits do not invalidate the wheel cache.
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN python -m pip install --upgrade pip && \
    python -m pip install ".[segmentation,nnunet,api]"

COPY configs/ configs/
COPY weights.lock.json ./

# Checkpoints are mounted at runtime, never baked in: they are large, and they
# are distributed under different terms than the code.
RUN mkdir -p /app/model /app/outputs

# Run unprivileged. Any file-write defect is then contained to the app user
# rather than being root-level.
RUN useradd --create-home --uid 10001 jannus && \
    chown -R jannus:jannus /app
USER jannus

EXPOSE 8000

# Bind localhost inside the container by default; publish with `-p` and put a
# TLS-terminating proxy in front. See SECURITY.md.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)" || exit 1

# Default to the CLI, which is what an external validation site needs.
# Override for the API service:
#   docker run ... jannus:1.50 python run_server.py --host 0.0.0.0
ENTRYPOINT ["jannus"]
CMD ["--help"]
