# ACE-Step on Jetson Orin Nano 8GB

Running [ACE-Step](https://github.com/ace-step/ACE-Step) — a state-of-the-art text-to-music model — on a **Jetson Orin Nano 8GB** (aarch64, JetPack R36, CUDA 12.6). Sovereign music generation, no cloud required.

---

## The Achievement

ACE-Step's official Docker image targets x86_64 and a standard NVIDIA GPU setup. Getting it running on the Jetson required solving three independent problems that aren't documented anywhere together:

### 1. PyTorch doesn't ship aarch64 wheels on PyPI

Standard `pip install torch` on a Jetson pulls nothing useful. NVIDIA maintains their own wheel index at `pypi.jetson-ai-lab.io` — pinned to JetPack releases — but the URLs aren't obvious and the version matrix is a research project in itself.

This repo pins the exact wheels for JetPack R36 / CUDA 12.6:

```
torch 2.9.1   → pypi.jetson-ai-lab.io/jp6/cu126/...
torchaudio 2.9.1
torchvision 0.24.1
```

The Dockerfile fetches them directly via `curl` and installs with `--no-deps` first to prevent PyPI overwriting them with x86 garbage later.

### 2. 8GB unified memory requires careful orchestration

Jetson unified memory means CPU and GPU share the same physical pool. ACE-Step's model is ~7 GB. Loading it naively will OOM.

Two flags solve this:
- `cpu_offload=True` — keeps weights on the CPU side and streams to GPU for each op
- `overlapped_decode=True` — pipelines data transfer with compute to hide latency

These are memory-only flags — CUDA still handles all actual computation.

### 3. The cuBLAS contiguous memory trap

This one took the most debugging. On Jetson unified memory, loading model weights via `cpu_offload` fragments the physical address space. cuBLAS needs a *contiguous* workspace allocation — if you create the cuBLAS handle after loading the model, NvMap can't find a large enough contiguous block and inference fails with cryptic memory errors (largest free block drops to ~2MB).

**Fix:** pre-allocate the cuBLAS workspace with a dummy matmul *before* model loading, then keep the handle alive. PyTorch caches it per-device.

```python
# Do this BEFORE loading model weights
a = torch.randn(64, 64, device="cuda", dtype=torch.float32)
_ = torch.matmul(a, b)
torch.cuda.synchronize()
# Do NOT call empty_cache — we want the cuBLAS workspace to persist
```

This one line saves you from ~4 hours of memory fragmentation debugging.

---

## Performance

| Clip Length | Inference Steps | Generation Time |
|-------------|-----------------|-----------------|
| 60s         | 27 (fast)       | ~30-45s         |
| 60s         | 60 (quality)    | ~70-90s         |
| 120s        | 27              | ~60-90s         |

Not bad for 8GB unified memory running a 7GB model.

---

## Setup

### Prerequisites

- Jetson Orin Nano 8GB (or larger — this is the floor)
- JetPack R36 (Ubuntu 22.04 base)
- CUDA 12.6
- Docker with NVIDIA Container Runtime (`nvidia-container-toolkit`)

### Deploy

```bash
# Clone this repo onto your Jetson
git clone <this repo> /opt/hailstorm/ace-step
cd /opt/hailstorm/ace-step

# Create persistent dirs
mkdir -p /opt/hailstorm/ace-step/checkpoints
mkdir -p /opt/hailstorm/ace-step/outputs

# Build and start
docker compose up -d
```

First run downloads the ACE-Step model (~7 GB from HuggingFace). This takes a while depending on your connection. Watch the logs:

```bash
docker logs -f hailstorm-ace-step
```

When you see `Pipeline loaded and ready.`, you're good.

### Health check

```bash
curl http://localhost:8006/health
```

Returns:
```json
{
  "status": "healthy",
  "model_status": "ready",
  "device": "cuda (Orin)",
  "checkpoint_dir": "/app/checkpoints"
}
```

---

## API

### Generate music

```bash
curl -X POST http://localhost:8006/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "cinematic orchestral, tense strings, rising tension, film score",
    "lyrics": "",
    "audio_duration": 60.0,
    "infer_step": 27
  }'
```

Response:
```json
{
  "status": "complete",
  "filename": "ace_a1b2c3d4e5f6.wav",
  "audio_url": "/output/ace_a1b2c3d4e5f6.wav",
  "duration_requested": 60.0,
  "generation_time": 38.4,
  "seed_used": 1839201
}
```

Retrieve the file:
```bash
curl http://localhost:8006/output/ace_a1b2c3d4e5f6.wav -o output.wav
```

### With lyrics

```bash
curl -X POST http://localhost:8006/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "indie folk, acoustic guitar, warm vocal, intimate",
    "lyrics": "[verse]\nWords go here\nMore words here\n\n[chorus]\nThe chorus lives here",
    "audio_duration": 90.0,
    "infer_step": 60
  }'
```

---

## Architecture Notes

- **FastAPI server** instead of Gradio — headless, no web UI overhead, plays nicely with load balancers and GPU swap orchestrators
- **Single worker, pipeline lock** — one GPU, one inference at a time. Requests queue rather than racing for memory
- **Async health endpoint** — stays responsive during 30-90s generation runs; Docker health checks work correctly
- **Background thread loading** — model loads in a thread at startup; server accepts requests immediately, returns 503 until ready

---

## What Is This Part Of?

This is a component of [Hailstorm / Genesis Node](https://ukstudio.world) — a sovereign cloud platform built around the principle that your data, your compute, and your AI should belong to you.

Borg (the Jetson Orin Nano) handles local GPU inference. ACE-Step sits alongside Chatterbox TTS and a local LLM, all orchestrated by a GPU swap layer that ensures only one model uses the GPU at a time — keeping everything within the 8GB envelope.

---

## If This Saved You Time

Running a 7GB music model on 8GB unified memory with a bespoke cuBLAS warmup hack is not what the documentation suggests is possible. If this repo got you there faster:

[![Support on Liberapay](https://liberapay.com/assets/widgets/donate.svg)](https://liberapay.com/muttleydosomething/donate)

Hardware costs money. More VRAM means more models. The Jetson B70 (32GB) is next.

---

## Credits

- [ACE-Step](https://github.com/ace-step/ACE-Step) — the model
- [NVIDIA Jetson AI Lab](https://www.jetson-ai-lab.io) — aarch64 PyTorch wheels
- Holmes — for the cuBLAS fix and the stubbornness to keep debugging until it worked
