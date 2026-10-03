# CardioSolv imaging service: DICOM CT/MR -> cardiac segmentation -> heart USD
# Build:  docker build -f docker/Dockerfile -t cardiosolv-imaging .
# Run:    docker compose -f docker/docker-compose.yml up -d
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 \
    CARDIOSOLV_MODELS_DIR=/opt/cardiosolv \
    NV_SEGMENT_CTMR_ROOT=/opt/cardiosolv/NV-Segment-CTMR/NV-Segment-CTMR \
    TOTALSEG_HOME_DIR=/opt/cardiosolv/totalsegmentator \
    CARDIOSOLV_IMAGING_WORKDIR=/data
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY cardiosolv_imaging ./cardiosolv_imaging
COPY scripts ./scripts
RUN pip install ".[nvsegment,totalsegmentator,server,dicom-extra]"

# Model weights are fetched at build time unless SKIP_MODELS=1 (then mount /opt/cardiosolv as a volume
# and run scripts/setup_models.sh once). NV-Segment-CTMR weights: NVIDIA non-commercial licence.
ARG SKIP_MODELS=0
ARG TOTALSEG_LICENSE=""
RUN if [ "$SKIP_MODELS" = "0" ]; then TOTALSEG_LICENSE="$TOTALSEG_LICENSE" bash scripts/setup_models.sh /opt/cardiosolv; fi

EXPOSE 8040
VOLUME ["/data"]
HEALTHCHECK CMD curl -fs http://localhost:8040/health || exit 1
CMD ["uvicorn", "cardiosolv_imaging.server:app", "--host", "0.0.0.0", "--port", "8040"]
