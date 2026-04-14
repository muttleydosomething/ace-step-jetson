"""
ACE-Step Music Generator -- FastAPI Service for Borg (Jetson Orin Nano 8GB)

Loads the ACE-Step pipeline ONCE at startup with memory-optimised flags,
then serves generation requests over HTTP. Designed to run behind the
Hailstorm GPU Swap Orchestrator.

Endpoint summary:
    POST /generate   -- Generate music (synchronous, returns when complete)
    GET  /health     -- Health check + model status
    GET  /output/{f} -- Serve a generated audio file

Model loading happens in a background task so the health endpoint is
available immediately -- important for Docker health checks and the
orchestrator. First run downloads ~7 GB from HuggingFace.
"""

import os
import time
import uuid
import asyncio
import threading
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# -- Configuration -----------------------------------------------------------

CHECKPOINT_DIR = os.environ.get("ACE_CHECKPOINT_DIR", "/app/checkpoints")
OUTPUT_DIR = os.environ.get("ACE_OUTPUT_DIR", "/app/outputs")
HOST = os.environ.get("ACE_HOST", "0.0.0.0")
PORT = int(os.environ.get("ACE_PORT", "8006"))
DTYPE = os.environ.get("ACE_DTYPE", "bfloat16")

# Jetson 8GB flags -- non-negotiable for this hardware
CPU_OFFLOAD = True
OVERLAPPED_DECODE = True
TORCH_COMPILE = False  # aarch64 Triton is limited

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -- Global state ------------------------------------------------------------

pipeline = None
pipeline_lock = asyncio.Lock()
model_status = "initialising"  # initialising -> downloading -> loading -> ready | error
startup_error = None


def _warmup_cublas():
    """Pre-allocate cuBLAS workspace BEFORE model loading.

    On Jetson unified memory, model weights loaded via cpu_offload scatter
    pages across the entire physical address space. cuBLAS needs a contiguous
    workspace allocation — if we create the handle after loading, NvMap can't
    find a large enough contiguous block (lfb drops to ~2MB).

    Fix: create a cuBLAS handle (via a dummy matmul) while memory is still
    contiguous, then keep it alive. PyTorch caches the handle per-device.
    """
    if not torch.cuda.is_available():
        print("[ace-step] CUDA not available, skipping cuBLAS warmup")
        return
    print("[ace-step] Pre-allocating cuBLAS workspace (before model load)...")
    try:
        # Force cuBLAS handle creation with a small matmul
        a = torch.randn(64, 64, device="cuda", dtype=torch.float32)
        b = torch.randn(64, 64, device="cuda", dtype=torch.float32)
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        # Also warm up bfloat16 path since that's what the model uses
        a16 = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        b16 = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        _ = torch.matmul(a16, b16)
        torch.cuda.synchronize()
        del a, b, a16, b16
        # Do NOT call empty_cache — we want the cuBLAS workspace to persist
        print(f"[ace-step] cuBLAS workspace allocated. "
              f"CUDA reserved: {torch.cuda.memory_reserved(0) / 1024**2:.0f} MB")
    except Exception as e:
        print(f"[ace-step] WARNING: cuBLAS warmup failed: {e}")


def _load_pipeline_blocking():
    """Load the ACE-Step pipeline (blocking). Runs in a background thread."""
    global pipeline, model_status, startup_error
    try:
        # Warm up cuBLAS FIRST — must happen before model weights fragment memory
        _warmup_cublas()

        from acestep.pipeline_ace_step import ACEStepPipeline

        model_status = "downloading"
        print(f"[ace-step] Initialising pipeline (checkpoint_dir={CHECKPOINT_DIR}, "
              f"dtype={DTYPE}, cpu_offload={CPU_OFFLOAD}, "
              f"overlapped_decode={OVERLAPPED_DECODE})")

        pipeline = ACEStepPipeline(
            checkpoint_dir=CHECKPOINT_DIR,
            dtype=DTYPE,
            cpu_offload=CPU_OFFLOAD,
            overlapped_decode=OVERLAPPED_DECODE,
            torch_compile=TORCH_COMPILE,
        )

        model_status = "loading"
        print("[ace-step] Downloading/loading checkpoint...")
        pipeline.load_checkpoint(CHECKPOINT_DIR)

        model_status = "ready"
        print("[ace-step] Pipeline loaded and ready.")

    except Exception as e:
        model_status = "error"
        startup_error = str(e)
        print(f"[ace-step] ERROR loading pipeline: {e}")
        import traceback
        traceback.print_exc()


# -- FastAPI app -------------------------------------------------------------

app = FastAPI(
    title="ACE-Step Music Generator",
    description="Sovereign music generation on Borg (Jetson Orin Nano 8GB)",
    version="1.0.0",
)


@app.on_event("startup")
async def startup():
    """Kick off model loading in a background thread."""
    thread = threading.Thread(target=_load_pipeline_blocking, daemon=True)
    thread.start()
    print("[ace-step] Model loading started in background thread.")


@app.on_event("shutdown")
async def shutdown():
    """Clean up GPU memory."""
    global pipeline
    if pipeline is not None:
        del pipeline
        pipeline = None
        torch.cuda.empty_cache()
        print("[ace-step] Pipeline unloaded.")


# -- Request / Response models -----------------------------------------------

class GenerateRequest(BaseModel):
    """Music generation parameters.

    Only `prompt` is required. Everything else has sensible defaults
    tuned for the Jetson's capabilities.
    """
    prompt: str = Field(
        ...,
        description="Genre/style tags, e.g. 'pop, upbeat, female vocal, 120bpm'",
    )
    lyrics: str = Field(
        default="",
        description="Lyrics with structure tags: [verse], [chorus], [bridge]",
    )
    audio_duration: float = Field(
        default=60.0,
        ge=5.0,
        le=240.0,
        description="Duration in seconds (5-240). Longer = slower.",
    )
    infer_step: int = Field(
        default=27,
        ge=10,
        le=100,
        description="Inference steps. 27 = fast, 60 = quality.",
    )
    guidance_scale: float = Field(default=15.0, ge=1.0, le=30.0)
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility. None = random.",
    )
    format: str = Field(
        default="wav",
        description="Output format: wav or ogg",
    )

    # Advanced params -- sensible defaults, most users won't touch these
    scheduler_type: str = Field(default="euler")
    cfg_type: str = Field(default="apg")
    omega_scale: float = Field(default=10.0)
    guidance_interval: float = Field(default=0.5)
    guidance_interval_decay: float = Field(default=0.0)
    min_guidance_scale: float = Field(default=3.0)
    use_erg_tag: bool = Field(default=True)
    use_erg_lyric: bool = Field(default=True)
    use_erg_diffusion: bool = Field(default=True)
    oss_steps: Optional[str] = Field(default=None)
    guidance_scale_text: float = Field(default=0.0)
    guidance_scale_lyric: float = Field(default=0.0)


class GenerateResponse(BaseModel):
    status: str
    filename: str
    audio_url: str
    duration_requested: float
    generation_time: float
    seed_used: int


class HealthResponse(BaseModel):
    status: str
    model_status: str
    device: str
    checkpoint_dir: str
    error: Optional[str] = None


# -- Endpoints ---------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check. Returns model status and device info.

    Docker/orchestrator: returns 200 as soon as the server is up.
    Check `model_status` field to know if generation is available.
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = f"cuda ({torch.cuda.get_device_name(0)})"

    return HealthResponse(
        status="healthy" if model_status == "ready" else "starting",
        model_status=model_status,
        device=device,
        checkpoint_dir=CHECKPOINT_DIR,
        error=startup_error,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate music from text prompt and optional lyrics.

    This is synchronous -- the request blocks until generation completes.
    On Jetson Orin Nano 8GB, expect ~30-90s for a 60s clip at 27 steps.
    """
    if model_status != "ready":
        detail = f"Model not ready (status: {model_status})."
        if startup_error:
            detail += f" Error: {startup_error}"
        raise HTTPException(status_code=503, detail=detail)

    # Determine seed
    seed = req.seed if req.seed is not None else int(time.time()) % (2**31)

    # Generate a unique output filename
    job_id = uuid.uuid4().hex[:12]
    filename = f"ace_{job_id}.{req.format}"
    output_path = os.path.join(OUTPUT_DIR, filename)

    start = time.time()

    # Pipeline.__call__ is blocking (PyTorch inference).
    # Run in executor to keep the event loop (and /health) responsive.
    # Lock ensures only one generation at a time (single GPU).
    async with pipeline_lock:
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                _generate_sync,
                req,
                seed,
                output_path,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    elapsed = time.time() - start

    return GenerateResponse(
        status="complete",
        filename=filename,
        audio_url=f"/output/{filename}",
        duration_requested=req.audio_duration,
        generation_time=round(elapsed, 2),
        seed_used=seed,
    )


def _generate_sync(req: GenerateRequest, seed: int, output_path: str):
    """Run pipeline inference (blocking). Called via run_in_executor."""
    result = pipeline(
        format=req.format,
        audio_duration=req.audio_duration,
        prompt=req.prompt,
        lyrics=req.lyrics,
        infer_step=req.infer_step,
        guidance_scale=req.guidance_scale,
        scheduler_type=req.scheduler_type,
        cfg_type=req.cfg_type,
        omega_scale=req.omega_scale,
        manual_seeds=str(seed),
        guidance_interval=req.guidance_interval,
        guidance_interval_decay=req.guidance_interval_decay,
        min_guidance_scale=req.min_guidance_scale,
        use_erg_tag=req.use_erg_tag,
        use_erg_lyric=req.use_erg_lyric,
        use_erg_diffusion=req.use_erg_diffusion,
        oss_steps=req.oss_steps,
        guidance_scale_text=req.guidance_scale_text,
        guidance_scale_lyric=req.guidance_scale_lyric,
        save_path=output_path,
        batch_size=1,
    )
    return result


@app.get("/output/{filename}")
async def serve_output(filename: str):
    """Serve a generated audio file."""
    # Sanitise: prevent path traversal
    safe_name = Path(filename).name
    filepath = os.path.join(OUTPUT_DIR, safe_name)

    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Determine content type
    if safe_name.endswith(".ogg"):
        media_type = "audio/ogg"
    else:
        media_type = "audio/wav"

    return FileResponse(filepath, media_type=media_type, filename=safe_name)


# -- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=HOST,
        port=PORT,
        workers=1,  # Single worker -- one GPU, one pipeline instance
        log_level="info",
    )
