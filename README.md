# ACE-Step on Jetson Orin Nano 8GB

Running [ACE-Step](https://github.com/ace-step/ACE-Step) -- a state-of-the-art text-to-music model -- on a **Jetson Orin Nano 8GB** (aarch64, JetPack R36, CUDA 12.6). Sovereign music generation, no cloud required.

---

## The Achievement

ACE-Step is designed for desktop GPUs with 12-24 GB of dedicated VRAM. The Jetson Orin Nano has **8 GB of unified memory** -- shared between CPU and GPU, with the OS and system services taking their cut. That leaves roughly 6 GB of usable headroom for a model that weighs 7 GB.

**Nobody documents how to do this, because you're not supposed to be able to.**

Getting here required solving five independent problems across the CUDA, PyTorch, ARM64, and Python stack -- each one a dead end without the others. The final solution is a 600-line custom FastAPI server that monkey-patches ACE-Step's entire inference pipeline, replaces its memory management layer, and orchestrates model loading at a granularity the original code never anticipated.

### 1. PyTorch doesn't ship aarch64 wheels on PyPI

Standard `pip install torch` on a Jetson pulls nothing useful. NVIDIA maintains their own wheel index at `pypi.jetson-ai-lab.io` -- pinned to JetPack releases -- but the URLs aren't obvious and the version matrix is a research project in itself. One wrong version and CUDA operations silently produce garbage or segfault.

This repo pins the exact wheels for JetPack R36 / CUDA 12.6:

```
torch 2.9.1   -> pypi.jetson-ai-lab.io/jp6/cu126/...
torchaudio 2.9.1
torchvision 0.24.1
```

The Dockerfile fetches them directly via `curl` and installs with `--no-deps` first to prevent PyPI overwriting them with x86 garbage when installing ACE-Step's other dependencies.

### 2. 8 GB unified memory requires surgical orchestration

Jetson unified memory means CPU and GPU share the same physical pool. You can't "offload to CPU" in the traditional sense -- there's nowhere to offload *to*. The model must be loaded, used, and then **completely destroyed** before the next component can take its place.

ACE-Step's built-in `CpuOffloader` assumes traditional dedicated VRAM -- it moves tensors between CPU RAM and GPU VRAM. On Jetson, that just shuffles bytes within the same pool, wasting bandwidth and solving nothing.

**The fix:** A complete replacement memory manager (`JetsonOffloader`) that:
- Loads each model component from disk checkpoints on demand
- Tracks reference counts for re-entrant calls (e.g. `calc_v` called inside `diffusion_process`)
- Truly deletes and garbage-collects each component when done -- not just `.to("cpu")`, but `del model` + `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.synchronize()`
- Handles accelerate dispatch hooks (which break if you call `.to()` on a device-mapped model)

This alone required monkey-patching 6 methods on the ACE-Step pipeline class, intercepting the `__call__` method, and replacing the `CpuOffloader` class in ACE-Step's own module before any pipeline instance is created.

### 3. The accelerate device_map split with disk offload

Even with surgical loading, the 6.2 GB transformer doesn't fit in available memory as a single unit. The solution uses HuggingFace `accelerate` to split the transformer across CUDA (3 GB pinned), with remaining layers offloaded to disk and streamed in one block at a time during inference.

**Critical discovery:** You cannot include CPU in the device map. ARM64 CPU cores lack float16 conv2d kernels entirely -- any layer placed on CPU fails silently with "*GET was unable to find an engine to execute this computation*". This is not documented anywhere in NVIDIA's or HuggingFace's materials. The only valid configuration is CUDA + disk, with zero CPU allocation:

```python
device_map="auto",
max_memory={0: "3000MiB"},  # CUDA only -- NO CPU (ARM64 lacks fp16 kernels)
offload_folder="/tmp/ace_offload",
```

### 4. The bfloat16/float16 dtype mismatch

ACE-Step's text encoder can produce bfloat16 embeddings depending on the checkpoint. On x86, this is fine -- the CPU handles bfloat16 natively and CUDA converts as needed. On ARM64, bfloat16 operations have no CPU kernel at all. The fix intercepts the pipeline's `__call__` method to force float16 across all phases, regardless of what the upstream model config specifies.

### 5. The cuBLAS contiguous memory trap

This one took the most debugging. On Jetson unified memory, loading model weights via the offloader fragments the physical address space. cuBLAS needs a *contiguous* workspace allocation -- if you create the cuBLAS handle after loading the model, NvMap can't find a large enough contiguous block and inference fails with cryptic memory errors (largest free block drops to ~2 MB).

**Fix:** Pre-allocate the cuBLAS workspace with a dummy matmul *before* model loading, then keep the handle alive. PyTorch caches it per-device.

```python
# Do this BEFORE loading model weights
a = torch.randn(64, 64, device="cuda", dtype=torch.float32)
_ = torch.matmul(a, a)
torch.cuda.synchronize()
# Do NOT call empty_cache -- we want the cuBLAS workspace to persist
```

This is a memory fragmentation race condition. You will never find it in the documentation because the documentation assumes you have dedicated VRAM. On unified memory, allocation order determines whether inference is possible at all.

### What this adds up to

The `server.py` in this repo is **600 lines of custom code** that:
- Replaces ACE-Step's memory management with a Jetson-specific offloader
- Monkey-patches 8 methods on the pipeline class
- Implements reference-counted model lifecycle management
- Splits the transformer across CUDA and disk via accelerate
- Pre-allocates cuBLAS workspaces to prevent memory fragmentation
- Forces float16 across all pipeline phases for ARM64 compatibility
- Wraps everything in a production-ready FastAPI server with health checks, async endpoints, and background model loading

The Dockerfile is another 105 lines of careful dependency pinning -- fetching aarch64-specific PyTorch wheels, installing ACE-Step with `--no-deps` to prevent dependency pollution, and configuring CUDA library paths for the Jetson's non-standard layout.

**This is not a configuration change. It's a port.**

---

## Performance

| Clip Length | Inference Steps | Generation Time |
|-------------|-----------------|-----------------|
| 60s         | 27 (fast)       | ~30-45s         |
| 60s         | 60 (quality)    | ~70-90s         |
| 120s        | 27              | ~60-90s         |

Not bad for 8 GB unified memory running a 7 GB model with disk-offloaded layers.

---

## Setup

### Prerequisites

- Jetson Orin Nano 8GB (or larger -- this is the floor)
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

- **FastAPI server** instead of Gradio -- headless, no web UI overhead, plays nicely with load balancers and GPU swap orchestrators
- **Single worker, pipeline lock** -- one GPU, one inference at a time. Requests queue rather than racing for memory
- **Async health endpoint** -- stays responsive during 30-90s generation runs; Docker health checks work correctly
- **Background thread loading** -- model loads in a thread at startup; server accepts requests immediately, returns 503 until ready

---

## What Is This Part Of?

This is a component of [Hailstorm / Genesis Node](https://ukstudio.world) -- a sovereign cloud platform built around the principle that your data, your compute, and your AI should belong to you.

Borg (the Jetson Orin Nano) handles local GPU inference. ACE-Step sits alongside Chatterbox TTS and a local LLM, all orchestrated by a GPU swap layer that ensures only one model uses the GPU at a time -- keeping everything within the 8 GB envelope.

---

## If This Saved You Time

Running a 7 GB music model on 8 GB unified memory with a bespoke cuBLAS warmup hack, disk-offloaded transformer layers, and 8 monkey-patched pipeline methods is not what the documentation suggests is possible. If this repo got you there faster:

[![Support on Liberapay](https://liberapay.com/assets/widgets/donate.svg)](https://liberapay.com/muttleydosomething/donate)

Hardware costs money. More VRAM means more models. The Jetson B70 (32 GB) is next.

---

## Credits

- [ACE-Step](https://github.com/ace-step/ACE-Step) -- the model
- [NVIDIA Jetson AI Lab](https://www.jetson-ai-lab.io) -- aarch64 PyTorch wheels
- Holmes -- for the cuBLAS fix and the stubbornness to keep debugging until it worked
