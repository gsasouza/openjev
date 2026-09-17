FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ARG OPENJEV_MODEL=Qwen/Qwen3.5-4B
ARG OPENJEV_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OPENJEV_MODEL=${OPENJEV_MODEL} \
    OPENJEV_REVISION=${OPENJEV_REVISION}

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# The CUDA build has to come from the PyTorch index; the pyproject pin
# (torch==2.10.0) is satisfied by 2.10.0+cu128, so installing the package
# afterwards does not pull a second copy.
RUN pip install --upgrade pip \
    && pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install . runpod==1.12.0

# Bake the pinned weights into the image so a scale-to-zero cold start never
# downloads them. Keep this after the dependency layers: it is the large one.
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${OPENJEV_MODEL}', revision='${OPENJEV_REVISION}')"
ENV HF_HUB_OFFLINE=1

COPY handler.py test_input.json ./

CMD ["python", "-u", "handler.py"]
