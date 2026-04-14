# ACE-Step Music Generator — Jetson Orin Nano 8GB (aarch64, CUDA 12.6, JetPack R36)
# Optimised for 8GB unified memory: cpu_offload + overlapped_decode
#
# Build on Borg:  docker build -t hailstorm-ace-step .
# Model auto-downloads from HuggingFace on first run (~7 GB).
# Mount a persistent volume at /app/checkpoints to cache it.

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    # ACE-Step writes outputs here by default
    ACE_OUTPUT_DIR=/app/outputs

# ── System deps ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip python3.10-dev \
    build-essential git curl libsndfile1 ffmpeg sox libsox-dev \
    libopenblas0 \
    cuda-cupti-12-6 \
    libcudss0-cuda-12 \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# ── CUDA library paths (matches Jetson AI Lab convention) ──
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/libcudss/12:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# ── Python venv ─────────────────────────────────────────────
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── PyTorch for CUDA 12.6 (Jetson aarch64 wheels from NVIDIA AI Lab) ──
# Standard PyPI/PyTorch wheels are x86 only. Jetson needs NVIDIA's builds.
# Pinned to match Borg's JetPack R36 + CUDA 12.6 environment.
RUN curl -fSL -o /tmp/torch-2.9.1-cp310-cp310-linux_aarch64.whl \
      https://pypi.jetson-ai-lab.io/jp6/cu126/+f/02f/de421eabbf626/torch-2.9.1-cp310-cp310-linux_aarch64.whl && \
    curl -fSL -o /tmp/torchaudio-2.9.1-cp310-cp310-linux_aarch64.whl \
      https://pypi.jetson-ai-lab.io/jp6/cu126/+f/d12/bede7113e6b00/torchaudio-2.9.1-cp310-cp310-linux_aarch64.whl && \
    curl -fSL -o /tmp/torchvision-0.24.1-cp310-cp310-linux_aarch64.whl \
      https://pypi.jetson-ai-lab.io/jp6/cu126/+f/d5b/caaf709f11750/torchvision-0.24.1-cp310-cp310-linux_aarch64.whl && \
    pip install --no-cache-dir \
      /tmp/torch-2.9.1-cp310-cp310-linux_aarch64.whl \
      /tmp/torchaudio-2.9.1-cp310-cp310-linux_aarch64.whl \
      /tmp/torchvision-0.24.1-cp310-cp310-linux_aarch64.whl && \
    rm -f /tmp/*.whl

# ── ACE-Step from source ───────────────────────────────────
# Install with --no-deps to prevent PyPI from overwriting our Jetson PyTorch wheels.
# Then install the actual dependencies manually (minus torch/torchaudio/torchvision/gradio).
WORKDIR /app
RUN git clone --depth 1 https://github.com/ace-step/ACE-Step.git /app/ace-step
RUN pip install --no-cache-dir --no-deps -e /app/ace-step

# ── ACE-Step dependencies (excluding PyTorch stack + gradio) ──
# torch, torchaudio, torchvision: already installed from Jetson wheels above
# gradio: not needed — we use our own FastAPI server
RUN pip install --no-cache-dir \
    "datasets==3.4.1" \
    "diffusers>=0.33.0" \
    "librosa==0.11.0" \
    "loguru==0.7.3" \
    "matplotlib==3.10.1" \
    "numpy" \
    "pypinyin==0.53.0" \
    "pytorch_lightning==2.5.1" \
    "soundfile==0.13.1" \
    "tqdm" \
    "transformers==4.50.0" \
    "py3langid==0.3.0" \
    "hangul-romanize==0.1.0" \
    "num2words==0.5.14" \
    "spacy>=3.8,<4.0" \
    "accelerate==1.6.0" \
    "cutlet" \
    "fugashi[unidic-lite]" \
    "click" \
    "peft" \
    "safetensors" \
    "huggingface-hub" \
    "einops"

# ── FastAPI server deps ────────────────────────────────────
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ── Our custom server ──────────────────────────────────────
COPY server.py /app/server.py

# ── Volumes ────────────────────────────────────────────────
# checkpoints: model cache (~7 GB, auto-downloaded on first run)
# outputs: generated audio files
RUN mkdir -p /app/checkpoints /app/outputs
VOLUME ["/app/checkpoints", "/app/outputs"]

EXPOSE 8006

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:8005/health || exit 1

CMD ["python3", "/app/server.py"]
