# strong-crane: the pipeline with its exact Python environment. The inputs are not in the image: mount them at /data
# (layout in docker/paths.json and the README, "Run with Docker"); results are written to /app/results.
#   docker run --rm --gpus all -v <inputs>:/data -v <out>:/app/results ghcr.io/somnathab3/strong-crane make smoke
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 MPLBACKEND=Agg
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 make && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install -r requirements.txt nbformat==5.10.4 nbclient==0.11.0 ipykernel==7.3.0 pytest

COPY . .
RUN pip install --no-deps -e . \
 && cp docker/paths.json configs/paths.json \
 && ln -s /data/features data/features \
 && python -m ipykernel install --sys-prefix --name python3

# GPU by default (needs --gpus all and the NVIDIA Container Toolkit); V77_DEVICE=CPU runs without a GPU
CMD ["make", "help"]
