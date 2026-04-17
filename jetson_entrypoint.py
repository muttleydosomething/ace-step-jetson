#!/usr/bin/env python3
"""ACE-Step 1.5 — Jetson Orin Nano 8GB Entrypoint.

Memory strategy for 7.5 GB UNIFIED memory (CPU + GPU share same DRAM):
  1. DiT decoder: Load to CPU → quanto INT8 on CPU → per-param to CUDA
  2. ALL conditioning (encoder, tokenizer, detokenizer): stay on CPU in BFLOAT16
     (NOT float32 — float32 doubles memory usage on unified memory!)
  3. prepare_condition() wrapper: inputs CPU, autocast bfloat16, outputs → CUDA
  4. Text encoder: CPU-only hot-swap per request
  5. VAE: Load to CUDA after text encoder unload (0.34 GB)
  6. Batch size capped at 1 to halve intermediate allocations

On Jetson, CPU and GPU share the same physical DRAM. Moving modules "to CPU"
does NOT free GPU memory — it uses the same pool. Every byte counts twice.

Environment (REQUIRED):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
"""

from __future__ import annotations

import ctypes
import gc
import os
import threading
import time
from typing import Optional, Tuple

import os
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"
import torch
from loguru import logger

# ═══════════════════════════════════════════════════════════════
# GLOBAL: Disable cuDNN for ALL convolutions on Jetson
# ═══════════════════════════════════════════════════════════════
# cuDNN 9.5.1 on Jetson Orin SM 8.7 has ZERO working conv1d/conv2d
# kernels for bfloat16 or float16. Every attempt fails with:
#   "GET was unable to find an engine to execute this computation"
# This affects both the DiT decoder (2 conv layers) and the VAE
# (74 conv layers: 37 encoder + 37 decoder including ConvTranspose1d).
# PyTorch's native CUDA conv kernels handle all dtypes without
# workspace allocation and have negligible performance difference
# for the tensor sizes involved.
torch.backends.cudnn.enabled = False
logger.info("[JETSON] cuDNN GLOBALLY DISABLED — native CUDA kernels for all convolutions")

# =================================================================
# Patch 14: Fix torchaudio.save (torchcodec broken on Jetson)
# =================================================================
# torchaudio 2.10.0 delegates save() to torchcodec, which crashes on Jetson
# (libnvrtc.so.13 missing). Use soundfile (WAV) + ffmpeg (MP3) instead.
import subprocess
import tempfile
import torchaudio
import soundfile as _sf

def _jetson_torchaudio_save(filepath, src, sample_rate, channels_first=True, **kwargs):
    """Drop-in torchaudio.save using soundfile + ffmpeg."""
    filepath = str(filepath)
    ext = os.path.splitext(filepath)[1].lower()
    if src.is_cuda:
        src = src.cpu()
    src = src.float()
    audio_np = src.numpy()
    if channels_first:
        audio_np = audio_np.T
    if ext == ".wav":
        _sf.write(filepath, audio_np, sample_rate, subtype="PCM_16")
    elif ext in (".mp3", ".ogg", ".flac", ".aac"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            _sf.write(tmp_path, audio_np, sample_rate, subtype="PCM_16")
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_path]
            if ext == ".mp3":
                cmd += ["-codec:a", "libmp3lame", "-q:a", "2"]
            elif ext == ".ogg":
                cmd += ["-codec:a", "libvorbis", "-q:a", "4"]
            elif ext == ".flac":
                cmd += ["-codec:a", "flac"]
            cmd.append(filepath)
            subprocess.run(cmd, check=True, capture_output=True)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        _sf.write(filepath, audio_np, sample_rate)

torchaudio.save = _jetson_torchaudio_save
logger.info("[JETSON] Patch 14: torchaudio.save monkey-patched (soundfile + ffmpeg, bypasses broken torchcodec)")


# ═══════════════════════════════════════════════════════════════
# Fix: Symlink examples into site-packages
# ═══════════════════════════════════════════════════════════════
import glob as _glob
_site = os.path.dirname(os.path.dirname(__import__("acestep").__file__))
_examples_link = os.path.join(_site, "examples")
if not os.path.exists(_examples_link) and os.path.isdir("/app/ace-step/examples"):
    try:
        os.symlink("/app/ace-step/examples", _examples_link)
    except OSError:
        pass


CHECKPOINT_DIR = os.environ.get("ACESTEP_CHECKPOINTS_DIR", "/app/checkpoints")
TEXT_ENCODER_NAME = "Qwen3-Embedding-0.6B"

# Maximum batch size for Jetson — unified memory can't handle batch=2
JETSON_MAX_BATCH_SIZE = 1

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _aggressive_free():
    """Free as much memory as possible: GC + CUDA cache + glibc trim."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except RuntimeError as e:
            logger.warning(f'[JETSON] empty_cache() NVML failure (non-fatal): {e}')
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def _drop_page_cache():
    """Drop OS page cache to free physical DRAM for NvMap.

    On Jetson unified memory, NvMap needs contiguous physical pages. When RAM
    is tight and swap is under pressure, kernel-held page-cache can be the
    difference between NvMap satisfying a request and an NVML assertion firing.
    Requires CAP_SYS_ADMIN (container runs privileged). Non-fatal on failure.
    """
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except Exception as e:
        logger.warning(f"[JETSON] drop_caches failed (non-fatal): {e}")


def _safe_to_cuda(tensor, device="cuda", dtype=None, label="tensor", max_retries=2):
    """Move a tensor to CUDA with retry-on-NVML-assert pressure relief.

    Patch 15: the .to(cuda) path in CPU-wrapped encoders can hit the same
    NVML allocator assert as the decoder restore (CUDACachingAllocator.cpp:1165)
    when memory pressure has accumulated across requests. Mirror the Patch 14
    pattern: on NVML/CUDA/OOM error, drop page cache + aggressive_free + retry.

    Caller is responsible for the non-CUDA fast path. If all retries fail,
    raises the last error rather than silently falling back, so the caller
    can decide whether CPU-resident output is acceptable.
    """
    if tensor is None:
        return None
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            if dtype is not None:
                return tensor.to(device, dtype=dtype)
            return tensor.to(device)
        except RuntimeError as e:
            last_error = e
            msg = str(e)
            if "NVML" not in msg and "CUDA" not in msg and "out of memory" not in msg:
                raise
            logger.warning(
                f"[JETSON] Patch 15: {label}.to({device}) attempt "
                f"{attempt + 1}/{max_retries + 1} hit NVML/CUDA assert: "
                f"{msg.splitlines()[0]}. Relieving pressure and retrying..."
            )
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            _aggressive_free()
            _drop_page_cache()
            _aggressive_free()
    raise RuntimeError(
        f"[JETSON] Patch 15: {label}.to({device}) failed after "
        f"{max_retries + 1} attempts with pressure relief: {last_error}"
    )


def _move_model_to_cuda_per_param(model, dtype=torch.bfloat16, skip_prefixes=None, pace_every=50):
    """Move model parameters to CUDA one at a time, skipping specified prefixes.

    This avoids the bulk NvMap allocation that `model.to("cuda")` triggers,
    which fails on Jetson when CMA headroom is < 1 GB. Per-parameter
    transfer lets the caching allocator make small incremental NvMap requests
    that the kernel can satisfy even under memory pressure.

    Patch 14: paced migration — every `pace_every` transfers we call
    empty_cache() to release allocator reservations that accumulate during a
    long migration. Gives NvMap breathing room on restore paths.

    skip_prefixes: list of parameter name prefixes to leave on CPU.
    """
    skip_prefixes = skip_prefixes or []
    skipped = 0
    moved = 0
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in skip_prefixes):
            skipped += 1
            continue
        param.data = param.data.to("cuda", dtype=dtype)
        moved += 1
        if pace_every > 0 and moved % pace_every == 0:
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass
    for name, buf in model.named_buffers():
        if buf is not None:
            if any(name.startswith(p) for p in skip_prefixes):
                continue
            parts = name.split(".")
            mod = model
            for p in parts[:-1]:
                mod = getattr(mod, p)
            mod._buffers[parts[-1]] = buf.to("cuda")
    if skip_prefixes:
        logger.info(f"[JETSON] Per-param CUDA: {moved} moved, {skipped} kept on CPU")
    _aggressive_free()



def _move_quanto_weights_to_cuda(model, skip_prefixes=None, pace_every=50):
    """Move quanto QBytesTensor internals (_data, _scale) to CUDA.

    quanto quantized weights are NOT exposed via named_parameters().
    They are custom tensor subclasses stored as module attributes (QLinear.qweight).
    This walks the module tree and moves their internal storage to CUDA
    one tensor at a time (same incremental pattern as per-param transfer).

    Without this, the decoder's quantized linear layers have activations
    on CUDA but weight data on CPU -> matmul device mismatch crash.

    Patch 14: paced migration — every `pace_every` transfers we call
    empty_cache() to release allocator reservations. On restore paths (post-
    VAE), the allocator pool is fragmented and the stock unpaced walk can
    trigger an NVML assert on a single `qw._data.cuda()` call.
    """
    skip_set = [p.rstrip(".") for p in (skip_prefixes or [])]
    moved = 0
    for name, mod in model.named_modules():
        # Skip conditioning modules that must stay on CPU
        if any(name == sp or name.startswith(sp + ".") for sp in skip_set):
            continue
        # quanto QLinear stores quantized weights in .qweight (a QBytesTensor)
        if hasattr(mod, "qweight"):
            qw = mod.qweight
            # Move internal _data (int8 weights) to CUDA
            if hasattr(qw, "_data") and isinstance(qw._data, torch.Tensor) and not qw._data.is_cuda:
                qw._data = qw._data.cuda()
                moved += 1
            # Move internal _scale (per-channel scales) to CUDA
            if hasattr(qw, "_scale") and isinstance(qw._scale, torch.Tensor) and not qw._scale.is_cuda:
                qw._scale = qw._scale.cuda()
                moved += 1
            if pace_every > 0 and moved > 0 and moved % pace_every == 0:
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass
    if moved:
        logger.info(f"[JETSON] Moved {moved} quanto internal tensors (_data/_scale) to CUDA")
    _aggressive_free()




def _fix_conv_bfloat16(model, skip_prefixes=None):
    """Fix Conv1d/ConvTranspose1d: bypass cuDNN entirely on Jetson.

    cuDNN 9.5.1 on Jetson Orin SM 8.7 reports "GET was unable to find an
    engine" for Conv1d(192, 2048, k=2, s=2) in BOTH bfloat16 AND float16
    when CUDA memory is fragmented after many diffusion steps.

    Previous approach (v1): cast conv to float16 + dtype hooks.  Failed
    because cuDNN still couldn't find an engine under memory pressure.

    Current approach (v2): bypass cuDNN entirely.  PyTorch's native CUDA
    conv kernel handles all dtypes without workspace allocation.  For the
    2 tiny patch-embedding convs (proj_in + proj_out), this costs nothing.
    Weights stay in bfloat16 — no dtype conversion needed.
    """
    skip_set = [p.rstrip(".") for p in (skip_prefixes or [])]
    count = 0
    for name, mod in model.named_modules():
        if any(name == sp or name.startswith(sp + ".") for sp in skip_set):
            continue
        if isinstance(mod, (torch.nn.Conv1d, torch.nn.Conv2d,
                            torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d)):

            # Wrap forward to disable cuDNN for this specific module
            _orig_forward = mod.forward

            def _make_wrapper(orig_fn):
                def _no_cudnn_forward(*args, **kwargs):
                    with torch.backends.cudnn.flags(enabled=False):
                        return orig_fn(*args, **kwargs)
                return _no_cudnn_forward

            mod.forward = _make_wrapper(_orig_forward)
            count += 1

    if count:
        logger.info(f"[JETSON] {count} conv layers: cuDNN BYPASSED — native CUDA kernel (no workspace needed)")


def _fix_vae_conv_float32(vae_model):
    """Patch 12: Wrap ALL VAE conv layers with float32 compute + cuDNN bypass.

    Even with torch.backends.cudnn.enabled=False at module level, the VAE's
    OobleckDecoder conv1d layers still hit "GET was unable to find an engine"
    on SM 8.7 in bfloat16.  Root cause: something in the diffusers/accelerate
    call chain re-enables cuDNN for the VAE decode path.

    The DiT's 2 conv layers survive because they have per-layer cuDNN context
    manager wrappers (via _fix_conv_bfloat16).  The VAE's 74 conv layers had
    no per-layer protection.

    This fix is STRONGER than _fix_conv_bfloat16: it casts inputs AND weights
    to float32 for the conv operation (creating temporary copies), runs with
    cuDNN force-disabled, then casts the output back to the original dtype.
    float32 conv ALWAYS works on CUDA regardless of cuDNN state or SM version.

    Memory cost: negligible.  The largest conv is Conv1d(64, 2048, k=7) =
    64*2048*7*4 = 3.4 MB temporary.  Freed immediately after each conv call.
    """
    import torch.nn.functional as F

    count = 0
    for name, mod in vae_model.named_modules():
        if isinstance(mod, torch.nn.Conv1d):
            # Capture module reference in closure
            def _make_conv1d_wrapper(conv):
                def _float32_forward(input):
                    orig_dtype = input.dtype
                    with torch.backends.cudnn.flags(enabled=False):
                        out = F.conv1d(
                            input.float(), conv.weight.float(),
                            conv.bias.float() if conv.bias is not None else None,
                            conv.stride, conv.padding, conv.dilation, conv.groups,
                        )
                    return out.to(orig_dtype)
                return _float32_forward
            mod.forward = _make_conv1d_wrapper(mod)
            count += 1

        elif isinstance(mod, torch.nn.ConvTranspose1d):
            def _make_convt1d_wrapper(conv):
                def _float32_forward(input, output_size=None):
                    orig_dtype = input.dtype
                    with torch.backends.cudnn.flags(enabled=False):
                        out = F.conv_transpose1d(
                            input.float(), conv.weight.float(),
                            conv.bias.float() if conv.bias is not None else None,
                            conv.stride, conv.padding, conv.output_padding,
                            conv.groups, conv.dilation,
                        )
                    return out.to(orig_dtype)
                return _float32_forward
            mod.forward = _make_convt1d_wrapper(mod)
            count += 1

        elif isinstance(mod, torch.nn.Conv2d):
            def _make_conv2d_wrapper(conv):
                def _float32_forward(input):
                    orig_dtype = input.dtype
                    with torch.backends.cudnn.flags(enabled=False):
                        out = F.conv2d(
                            input.float(), conv.weight.float(),
                            conv.bias.float() if conv.bias is not None else None,
                            conv.stride, conv.padding, conv.dilation, conv.groups,
                        )
                    return out.to(orig_dtype)
                return _float32_forward
            mod.forward = _make_conv2d_wrapper(mod)
            count += 1

    if count:
        logger.info(
            f"[JETSON] Patch 12: {count} VAE conv layers wrapped with float32 compute "
            f"+ cuDNN bypass (bulletproof against SM 8.7 bfloat16 conv failures)"
        )


# ═══════════════════════════════════════════════════════════════
# Patch 10: Diffusion loop memory — KV cache reset + per-step cleanup
# ═══════════════════════════════════════════════════════════════

def _install_diffusion_memory_fixes(model):
    """Prevent unbounded KV cache growth across diffusion steps.

    Root cause of ~360-400s OOM: generate_audio() passes use_cache=True
    and accumulates past_key_values across ALL diffusion steps via
    DynamicCache.  Each step appends its full KV pairs to the cache:

      Per step: 24 layers × 2 (K+V) × 8 heads × seq_len × 128 head_dim × 2B
              ≈ 281 MB for a 60s track (seq_len ~3000 after patch_size=2)

    After ~14 steps the cache alone exceeds Jetson's free memory (~3.9 GB).
    The Conv1d "unable to find engine" error is cuDNN failing to allocate
    workspace after the cache has consumed everything.

    Fix: pre-hook on decoder resets the cache each call so no cross-step
    accumulation occurs, plus post-hook releases the CUDA cache to prevent
    fragmentation between steps.

    The Heun corrector already uses use_cache=False with a fresh cache —
    the developers knew correctors shouldn't accumulate.  We apply the
    same principle to the main predictor call.
    """
    from transformers.cache_utils import DynamicCache, EncoderDecoderCache

    def _reset_cache_pre_hook(module, args, kwargs):
        """Reset KV cache before each decoder forward.

        NOTE: gc.collect() was removed (v2) — full Python GC sweeps between
        diffusion steps cause the glibc allocator to release pages back to
        the OS, which fragments the unified memory pool.  CUDA empty_cache
        alone is sufficient and less disruptive.
        """
        kwargs['past_key_values'] = EncoderDecoderCache(DynamicCache(), DynamicCache())
        kwargs['use_cache'] = False
        torch.cuda.empty_cache()
        return args, kwargs

    def _cleanup_post_hook(module, args, output):
        """Release CUDA cache after each decoder forward to prevent fragmentation."""
        torch.cuda.empty_cache()
        return output

    model.decoder.register_forward_pre_hook(_reset_cache_pre_hook, with_kwargs=True)
    model.decoder.register_forward_hook(_cleanup_post_hook)
    logger.info(
        "[JETSON] Patch 10: KV cache reset per decoder call + inter-step CUDA cleanup "
        "(prevents ~281 MB/step accumulation)"
    )


# ═══════════════════════════════════════════════════════════════
# Patch 11: Decoder ↔ CPU swap for VAE CUDA decode
# ═══════════════════════════════════════════════════════════════

def _move_decoder_to_cpu(model):
    """Move decoder from CUDA to CPU, including quanto QBytesTensor internals.

    On Jetson unified memory, this frees the CUDA allocator's hold on ~1.58 GB
    of decoder weights. The physical DRAM is still occupied (CPU side), but the
    CUDA allocator can now satisfy new allocation requests (for VAE) from the
    freed NvMap/CMA pages.

    quanto's QBytesTensor stores _data (int8 weights) and _scale (per-channel
    scales) as internal tensors NOT exposed via named_parameters(), so they
    need explicit handling.
    """
    decoder = model.decoder
    # Regular parameters
    for param in decoder.parameters():
        if param.data.is_cuda:
            param.data = param.data.cpu()
    # Buffers
    for name, buf in decoder.named_buffers():
        if buf is not None and buf.is_cuda:
            parts = name.split(".")
            mod = decoder
            for p in parts[:-1]:
                mod = getattr(mod, p)
            mod._buffers[parts[-1]] = buf.cpu()
    # quanto QBytesTensor internals
    for _name, mod in decoder.named_modules():
        if hasattr(mod, "qweight"):
            qw = mod.qweight
            if hasattr(qw, "_data") and isinstance(qw._data, torch.Tensor) and qw._data.is_cuda:
                qw._data = qw._data.cpu()
            if hasattr(qw, "_scale") and isinstance(qw._scale, torch.Tensor) and qw._scale.is_cuda:
                qw._scale = qw._scale.cpu()
    _aggressive_free()


def _restore_decoder_to_cuda(model, dtype, max_retries=2):
    """Restore decoder to CUDA after VAE decode completes.

    Uses the same per-param transfer + quanto weight migration as initial load.
    Conv bypass and KV cache hooks survive the round-trip because they are
    registered on Module objects (not individual tensors).

    Patch 14: retry-with-pressure-relief. If an NVML/CUDA assert fires mid-
    migration (classic symptom on Jetson unified memory under pressure),
    drop OS page cache + aggressive_free and retry. Observed failures have
    been transient — the allocator recovers once physical pages are released.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            _move_model_to_cuda_per_param(
                model, dtype=dtype, skip_prefixes=list(_CPU_PREFIXES)
            )
            _move_quanto_weights_to_cuda(model, skip_prefixes=list(_CPU_PREFIXES))
            _aggressive_free()
            return
        except RuntimeError as e:
            last_error = e
            msg = str(e)
            if "NVML" not in msg and "CUDA" not in msg and "out of memory" not in msg:
                raise
            logger.warning(
                f"[JETSON] Patch 14: Restore attempt {attempt + 1}/{max_retries + 1} "
                f"hit NVML/CUDA assert: {msg.splitlines()[0]}. "
                "Relieving memory pressure and retrying..."
            )
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            _aggressive_free()
            _drop_page_cache()
            _aggressive_free()
    raise RuntimeError(
        f"[JETSON] Patch 14: Decoder restore failed after {max_retries + 1} "
        f"attempts with pressure relief: {last_error}"
    )


# ═══════════════════════════════════════════════════════════════
# CPU prepare_condition wrapper
# ═══════════════════════════════════════════════════════════════

def _install_cpu_prepare_condition(model, target_device, target_dtype):
    """Wrap prepare_condition: run ALL conditioning on CPU, outputs to CUDA.

    prepare_condition() runs ONCE per generation. It uses encoder, tokenizer,
    and detokenizer — all on CPU in bfloat16. This wrapper:
      1. Moves all input tensors from CUDA to CPU (preserving dtype)
      2. Calls the original prepare_condition under torch.autocast("cpu", bfloat16)
         so that mixed-precision ops (RMSNorm float32 intermediates hitting
         bfloat16 linear weights) are handled automatically
      3. Moves the 3 output tensors to CUDA for the diffusion loop

    CRITICAL: Conditioning modules stay in BFLOAT16, not float32.
    On Jetson unified memory, float32 doubles RAM usage (killing headroom).
    autocast handles the dtype mismatches that float32 was solving.
    """
    _orig = model.prepare_condition  # bound method — captures self=model

    def _to_cpu(x):
        """Move tensor to CPU, preserving dtype. No float32 promotion!"""
        if isinstance(x, torch.Tensor) and x.is_cuda:
            if x.is_floating_point():
                return x.to("cpu")  # keep original dtype (bfloat16)
            else:
                return x.to("cpu")  # preserve integer dtypes (Long, Int, Bool)
        return x

    def _cpu_prepare_condition(*args, **kwargs):
        cpu_args = tuple(_to_cpu(a) for a in args)
        cpu_kwargs = {k: _to_cpu(v) for k, v in kwargs.items()}
        _aggressive_free()  # clear CUDA cache before CPU-heavy conditioning
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            result = _orig(*cpu_args, **cpu_kwargs)
        # Move outputs to CUDA for the diffusion loop
        if isinstance(result, tuple):
            out = tuple(
                t.to(target_device, dtype=target_dtype)
                if isinstance(t, torch.Tensor) else t
                for t in result
            )
        elif isinstance(result, torch.Tensor):
            out = result.to(target_device, dtype=target_dtype)
        else:
            out = result
        _aggressive_free()  # clean up CPU allocations
        return out

    model.prepare_condition = _cpu_prepare_condition
    logger.info("[JETSON] prepare_condition wrapped: CPU bfloat16 autocast, CUDA output only")


# ═══════════════════════════════════════════════════════════════
# CPU Text Encoder Proxy (for hot-swap in preprocess_batch)
# ═══════════════════════════════════════════════════════════════

class _CPUTextEncoderProxy:
    """Proxy that keeps the text encoder on CPU but produces CUDA outputs.

    Used during preprocess_batch (before prepare_condition) for text/lyric
    embedding. The Qwen3-Embedding-0.6B forward pass takes ~2-3s on CPU
    vs ~0.3s on CUDA — trivial compared to diffusion.
    """
    def __init__(self, model, target_device="cuda", target_dtype=torch.bfloat16):
        object.__setattr__(self, '_model', model)
        object.__setattr__(self, '_device', target_device)
        object.__setattr__(self, '_dtype', target_dtype)

    def __call__(self, *args, **kwargs):
        cpu_args = tuple(
            a.to("cpu") if isinstance(a, torch.Tensor) and a.is_cuda else a
            for a in args
        )
        cpu_kwargs = {
            k: v.to("cpu") if isinstance(v, torch.Tensor) and v.is_cuda else v
            for k, v in kwargs.items()
        }
        with torch.no_grad():
            output = self._model(*cpu_args, **cpu_kwargs)
        if hasattr(output, 'last_hidden_state') and output.last_hidden_state is not None:
            output.last_hidden_state = _safe_to_cuda(
                output.last_hidden_state,
                device=self._device,
                dtype=self._dtype,
                label="text_encoder.last_hidden_state",
            )
        if hasattr(output, 'pooler_output') and output.pooler_output is not None:
            output.pooler_output = _safe_to_cuda(
                output.pooler_output,
                device=self._device,
                dtype=self._dtype,
                label="text_encoder.pooler_output",
            )
        return output

    def __getattr__(self, name):
        return getattr(self._model, name)

    def embed_tokens(self, input_ids):
        """Intercept embed_tokens: move input to CPU, run lookup, return on CUDA."""
        if isinstance(input_ids, torch.Tensor) and input_ids.is_cuda:
            input_ids = input_ids.to('cpu')
        with torch.no_grad():
            result = self._model.embed_tokens(input_ids)
        return _safe_to_cuda(
            result,
            device=self._device,
            dtype=self._dtype,
            label="text_encoder.embed_tokens",
        )

    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._model, name, value)

    def eval(self):
        self._model.eval()
        return self

    def to(self, *args, **kwargs):
        logger.debug("[JETSON] CPUTextEncoderProxy: intercepted .to() — staying on CPU")
        return self

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return self._dtype


# ═══════════════════════════════════════════════════════════════
# Sentinel for hot-swapped text encoder
# ═══════════════════════════════════════════════════════════════

class _HotSwapPlaceholder:
    """Non-None placeholder so readiness checks pass before hot-swap."""
    pass

_HOTSWAP_SENTINEL = _HotSwapPlaceholder()


# ═══════════════════════════════════════════════════════════════
# Patch 1: Quantization validation — accept quanto
# ═══════════════════════════════════════════════════════════════

def _patched_validate_quantization_setup(self, *, quantization, compile_model):
    """Allow quanto as an alternative to torchao for quantization."""
    if quantization is None:
        return
    try:
        import torchao  # noqa: F401
        return
    except Exception:
        pass
    try:
        from optimum.quanto import quantize  # noqa: F401
        logger.info("[JETSON] torchao unavailable; using optimum-quanto for quantization")
        return
    except Exception:
        pass
    raise ImportError("Neither torchao nor optimum-quanto available for quantization")


# ═══════════════════════════════════════════════════════════════
# Patch 2: DiT loading — CPU load → quanto on CPU → DECODER-ONLY to CUDA
# ═══════════════════════════════════════════════════════════════

# Conditioning components that stay on CPU — they run once per generation
# inside prepare_condition(). Only the decoder needs CUDA for the 100+
# iteration diffusion loop.
_CPU_PREFIXES = ("encoder.", "tokenizer.", "detokenizer.")

def _patched_load_main_model(
    self, *, model_checkpoint_path, device, use_flash_attention,
    compile_model, quantization
):
    """Load DiT with Jetson memory-safe pipeline.

    Key difference from stock: encoder, tokenizer, and detokenizer NEVER
    touch CUDA. They stay on CPU in BFLOAT16 from load through inference.
    Only the decoder (and its null_condition_emb) goes to CUDA.

    CRITICAL MEMORY NOTE: On Jetson unified memory, CPU and GPU share the
    same physical DRAM. Casting conditioning modules to float32 DOUBLES
    their memory usage (1.6 GB → 3.2 GB) in the SAME pool that CUDA
    allocations use. torch.autocast handles dtype mismatches instead.
    """
    from transformers import AutoModel

    if not os.path.exists(model_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_checkpoint_path}")

    if torch.cuda.is_available() and getattr(self, "model", None) is not None:
        del self.model
        self.model = None
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except RuntimeError:
            pass

    attn = "sdpa"
    logger.info("[JETSON] Loading DiT to CPU (low_cpu_mem_usage)...")
    self.model = AutoModel.from_pretrained(
        model_checkpoint_path,
        trust_remote_code=True,
        attn_implementation=attn,
        dtype=self.dtype,
        low_cpu_mem_usage=True,
    )
    self.model.config._attn_implementation = attn
    self.config = self.model.config
    self._sync_alignment_config()
    self._apply_cuda_bool_argsort_workaround()
    self.model.eval()

    # Quanto INT8 on CPU — halves decoder weight memory before touching CUDA
    if quantization is not None:
        try:
            from optimum.quanto import quantize, freeze, qint8
            logger.info("[JETSON] Applying quanto INT8 to decoder on CPU...")
            t0 = time.time()
            if hasattr(self.model, "decoder"):
                quantize(self.model.decoder, weights=qint8)
                freeze(self.model.decoder)
            else:
                quantize(self.model, weights=qint8)
                freeze(self.model)
            gc.collect()
            logger.info(f"[JETSON] INT8 done on CPU ({time.time() - t0:.1f}s)")
        except Exception as exc:
            logger.warning(f"[JETSON] quanto failed ({exc}), loading unquantized")

    # Per-parameter transfer to CUDA — SKIP conditioning components
    if device.startswith("cuda"):
        logger.info("[JETSON] Moving DECODER ONLY to CUDA (per-parameter, skipping conditioning)...")
        t0 = time.time()
        _move_model_to_cuda_per_param(
            self.model,
            dtype=self.dtype,
            skip_prefixes=list(_CPU_PREFIXES),
        )
        self.model.decoder = self.model.decoder.to(self.dtype)
        _aggressive_free()
        # Move quanto internal weight tensors (_data, _scale) to CUDA.
        _move_quanto_weights_to_cuda(self.model, skip_prefixes=list(_CPU_PREFIXES))

        # Fix: cuDNN on Jetson SM 8.7 has no bfloat16 conv1d/conv2d kernels.
        _fix_conv_bfloat16(self.model, skip_prefixes=list(_CPU_PREFIXES))

        # Fix: KV cache grows unboundedly across diffusion steps (~281 MB/step).
        _install_diffusion_memory_fixes(self.model)

        alloc = torch.cuda.memory_allocated() / 1e9
        logger.info(f"[JETSON] Decoder on CUDA ({time.time() - t0:.1f}s, {alloc:.2f} GB)")
        logger.info("[JETSON] Conditioning (encoder + tokenizer + detokenizer) stays on CPU in BFLOAT16")
        logger.info("[JETSON] NO float32 cast — unified memory can't afford the 2x overhead")

        # Wrap prepare_condition to run ALL conditioning on CPU with autocast
        _install_cpu_prepare_condition(self.model, "cuda", self.dtype)

    elif not self.offload_to_cpu:
        self.model = self.model.to(device).to(self.dtype)
    else:
        self.model = self.model.to("cpu").to(self.dtype)

    # Silence latent
    sl_path = os.path.join(model_checkpoint_path, "silence_latent.pt")
    if os.path.exists(sl_path):
        self.silence_latent = torch.load(sl_path, weights_only=True).transpose(1, 2)
        sl_device = "cpu" if self.offload_to_cpu and self.offload_dit_to_cpu else device
        self.silence_latent = self.silence_latent.to(sl_device).to(self.dtype)

    return attn


# ═══════════════════════════════════════════════════════════════
# Patch 3: Skip _apply_dit_quantization (already done in loader)
# ═══════════════════════════════════════════════════════════════

def _noop_dit_quantization(self, quantization):
    """No-op — quantization already applied during CPU loading phase."""
    pass


# ═══════════════════════════════════════════════════════════════
# Patch 4: Text encoder — hot-swap (no permanent CUDA resident)
# ═══════════════════════════════════════════════════════════════

def _patched_load_text_encoder_and_tokenizer(self, *, checkpoint_dir, device):
    """Load only the tokenizer. Text encoder is hot-swapped per request."""
    from transformers import AutoTokenizer

    text_encoder_path = os.path.join(checkpoint_dir, TEXT_ENCODER_NAME)
    if not os.path.exists(text_encoder_path):
        raise FileNotFoundError(f"Text encoder not found at {text_encoder_path}")

    self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
    self.text_encoder = _HOTSWAP_SENTINEL
    self._text_encoder_path = text_encoder_path
    self._text_encoder_lock = threading.Lock()

    logger.info("[JETSON] Text encoder: tokenizer loaded, model hot-swapped per request (CPU-only)")
    return text_encoder_path


# ═══════════════════════════════════════════════════════════════
# Patch 5: _load_model_context for text_encoder — CPU-only hot-swap
# ═══════════════════════════════════════════════════════════════

from contextlib import contextmanager

def _patched_load_model_context(self, model_name):
    """Override _load_model_context for text encoder hot-swap and VAE decoder swap."""
    # Patch 14: Lazy restore check. If the previous VAE cycle deferred the
    # decoder CPU->CUDA migration, do it now — before any other model context
    # loads. This is the calm moment in the generation cycle: VAE fully
    # released, diffusion hasn't started, allocator under minimum pressure.
    # Skip for VAE re-entry (decoder must stay on CPU during VAE anyway).
    if getattr(self, "_decoder_needs_cuda_restore", False) and model_name != "vae":
        logger.info(
            f"[JETSON] Patch 14: Lazy restore triggered by "
            f"load_model_context('{model_name}')..."
        )
        t0 = time.time()
        _aggressive_free()
        _drop_page_cache()
        _aggressive_free()
        try:
            _restore_decoder_to_cuda(self.model, self.dtype)
            alloc = torch.cuda.memory_allocated() / 1e9
            logger.info(
                f"[JETSON] Decoder -> CUDA ({time.time() - t0:.1f}s, lazy). "
                f"CUDA allocated: {alloc:.2f} GB -- ready for diffusion"
            )
        finally:
            self._decoder_needs_cuda_restore = False

    # Text encoder: hot-swap from disk (CPU-only)
    if model_name == "text_encoder" and (
        self.text_encoder is None
        or isinstance(self.text_encoder, _HotSwapPlaceholder)
    ):
        return _hotswap_text_encoder_context(self)

    # VAE: swap decoder off CUDA first to free VRAM for GPU decode
    if model_name == "vae":
        return _vae_with_decoder_swap(self)

    return self._original_load_model_context(model_name)


@contextmanager
def _hotswap_text_encoder_context(handler):
    """Context manager: load text encoder on CPU, wrap with CUDA output proxy."""
    from transformers import AutoModel

    enc_path = getattr(handler, "_text_encoder_path", None)
    if enc_path is None:
        raise RuntimeError("Text encoder path not set — init not called?")

    lock = getattr(handler, "_text_encoder_lock", threading.Lock())
    with lock:
        logger.info("[JETSON] Hot-swap: loading text encoder (CPU-only, CUDA output proxy)...")
        t0 = time.time()

        encoder = AutoModel.from_pretrained(enc_path, low_cpu_mem_usage=True)
        encoder.eval()

        proxy = _CPUTextEncoderProxy(
            encoder,
            target_device="cuda",
            target_dtype=handler.dtype,
        )
        handler.text_encoder = proxy

        logger.info(
            f"[JETSON] Text encoder on CPU with CUDA output proxy ({time.time() - t0:.1f}s)"
        )

        try:
            yield
        finally:
            handler.text_encoder = _HOTSWAP_SENTINEL
            del proxy
            del encoder
            _aggressive_free()
            logger.info(
                f"[JETSON] Text encoder unloaded (total: {time.time() - t0:.1f}s)"
            )


# Patch 16: VAE-on-CUDA with probe + sticky fallback.
#
# Patch 13 forced VAE to CPU permanently to dodge NVML allocator asserts.
# That works but costs ~13 minutes per 4-minute track (CPU conv1d is slow
# on ARM cores). Patches 14+15 closed the allocator-assert recovery gaps,
# so we can now try CUDA first, with safe fallback.
#
# Strategy:
#   1. Probe free CUDA memory after decoder swap; need >= min_free_gb.
#   2. If probe passes AND sticky flag is enabled, try to move VAE to CUDA
#      with _safe_to_cuda retry-with-pressure-relief.
#   3. On ANY failure (probe too low, CUDA move crashes, or NVML assert
#      during decode), flip sticky flag off for rest of process lifetime
#      and fall back to Patch 13 CPU path.
#   4. On success, keep latents on CUDA (no CPU hook) — decode runs fully
#      on GPU, which is ~50-100x faster than ARM CPU.
#
# min_free_gb = 1.5 covers the VAE weight payload (~340 MB) plus float32
# activation temporaries (Patch 12 keeps convs in float32). This is a
# conservative threshold — headroom prevents mid-decode NVML asserts.
_VAE_CUDA_STATE = {
    "enabled": True,
    "min_free_gb": 1.5,
    "disabled_reason": None,
    "successes": 0,
    "failures": 0,
}


def _disable_vae_cuda(reason):
    """Flip VAE-CUDA sticky flag off for rest of process lifetime."""
    if _VAE_CUDA_STATE["enabled"]:
        _VAE_CUDA_STATE["enabled"] = False
        _VAE_CUDA_STATE["disabled_reason"] = reason
        _VAE_CUDA_STATE["failures"] += 1
        logger.warning(
            f"[JETSON] Patch 16: VAE-CUDA DISABLED for session. Reason: {reason}. "
            f"Falling back to Patch 13 CPU path. "
            f"(successes={_VAE_CUDA_STATE['successes']}, "
            f"failures={_VAE_CUDA_STATE['failures']})"
        )


def _try_move_vae_to_cuda(handler):
    """Attempt VAE -> CUDA with retry. Returns True on success, False on failure.

    Checks free-memory probe first; if below threshold, bails without trying.
    On failure, flips the sticky flag via _disable_vae_cuda.
    """
    if not _VAE_CUDA_STATE["enabled"]:
        return False

    # Probe free VRAM (on Jetson unified memory this is effectively free RAM)
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / 1024**3
        total_gb = total_bytes / 1024**3
    except Exception as e:
        _disable_vae_cuda(f"mem_get_info failed: {e}")
        return False

    min_free = _VAE_CUDA_STATE["min_free_gb"]
    if free_gb < min_free:
        logger.info(
            f"[JETSON] Patch 16: VAE-CUDA skipped this request — "
            f"free={free_gb:.2f} GB < {min_free:.2f} GB threshold "
            f"(total={total_gb:.2f} GB). Falling back to CPU for this decode."
        )
        return False

    logger.info(
        f"[JETSON] Patch 16: VAE-CUDA probe OK (free={free_gb:.2f} GB / "
        f"total={total_gb:.2f} GB). Moving VAE to CUDA..."
    )

    t0 = time.time()
    try:
        # Move VAE to CUDA via per-param to avoid bulk NvMap allocation.
        # VAE is small (~340 MB) and unquantized so regular per-param works.
        _move_model_to_cuda_per_param(
            handler.vae, dtype=torch.float32, pace_every=20
        )
        _aggressive_free()
    except (RuntimeError, AssertionError) as e:
        msg = str(e)
        is_mem = ("NVML" in msg or "CUDA" in msg or
                  "out of memory" in msg.lower())
        if is_mem:
            logger.warning(
                f"[JETSON] Patch 16: VAE CUDA move failed "
                f"({msg.splitlines()[0]}). Falling back to CPU."
            )
            # Best-effort revert: move VAE back to CPU so decode hook works
            try:
                handler.vae = handler.vae.to("cpu")
            except Exception:
                pass
            _aggressive_free()
            _drop_page_cache()
            _aggressive_free()
            _disable_vae_cuda(f"CUDA move error: {msg.splitlines()[0]}")
            return False
        raise  # non-memory error — let it propagate

    logger.info(
        f"[JETSON] Patch 16: VAE on CUDA ({time.time() - t0:.1f}s). "
        f"GPU decode path engaged."
    )
    return True


@contextmanager
def _vae_with_decoder_swap(handler):
    """Swap decoder off CUDA, try VAE on CUDA, fall back to CPU on failure.

    Patch 11 (decoder swap) + Patch 13 (CPU fallback) + Patch 16 (CUDA preferred).

    Previously (Patch 13): VAE always forced to CPU to bypass NVML asserts.
    That's safe but costs ~13 minutes/track on ARM cores.

    Patch 16: Try CUDA first with memory probe + sticky fallback. If CUDA
    works (most of the time on a warm container), VAE decode runs in seconds
    instead of minutes. If anything goes wrong, we fall back to CPU for this
    request AND all subsequent requests in this process (sticky flag).

    On Jetson unified memory, CPU and CUDA share the same physical DRAM —
    the CPU path remains viable as a permanent fallback with no data copies.
    """
    decoder = handler.model.decoder if hasattr(handler.model, 'decoder') else None
    decoder_was_cuda = False

    if decoder is not None:
        try:
            p = next(decoder.parameters())
            decoder_was_cuda = p.device.type == 'cuda'
        except StopIteration:
            pass

    if decoder_was_cuda:
        logger.info("[JETSON] Patch 11: Swapping decoder off CUDA for VAE decode...")
        t0 = time.time()
        _move_decoder_to_cpu(handler.model)
        alloc = torch.cuda.memory_allocated() / 1e9
        logger.info(
            f"[JETSON] Decoder -> CPU ({time.time() - t0:.1f}s). "
            f"CUDA allocated: {alloc:.2f} GB"
        )

    # Patch 16: Try VAE on CUDA (fast path). Falls through to Patch 13 CPU
    # path on probe failure, CUDA move failure, or sticky flag disabled.
    vae_on_cuda = _try_move_vae_to_cuda(handler)

    if not vae_on_cuda:
        # Patch 13: CPU VAE decode (fallback). The VAE is already on CPU
        # (offload_to_cpu=True) or was reverted to CPU by _try_move_vae_to_cuda.
        handler.vae = handler.vae.to("cpu")
        _aggressive_free()
        logger.info(
            "[JETSON] Patch 13: VAE forced to CPU for decode "
            "(bypasses CUDA CachingAllocator NVML assertion failures)"
        )

    # Hook: ensure latents match the VAE device before OobleckDecoder forward.
    # The diffusion loop produces latents on CUDA.
    # - VAE on CPU: move latents to CPU (Patch 13 behaviour).
    # - VAE on CUDA: leave latents on CUDA (Patch 16 behaviour — noop move).
    target_device = "cuda" if vae_on_cuda else "cpu"

    def _latents_to_target(module, args):
        return tuple(
            a.to(target_device) if isinstance(a, torch.Tensor) and
            a.device.type != target_device else a
            for a in args
        )

    vae_decoder_hook = handler.vae.decoder.register_forward_pre_hook(_latents_to_target)

    try:
        yield  # Patch 16: decode runs on chosen device
    except (RuntimeError, AssertionError) as e:
        # If we picked CUDA and it NVML-asserted during decode, flip the
        # sticky flag off so subsequent requests use CPU. Then re-raise —
        # the caller (user) will retry and get the CPU path automatically.
        if vae_on_cuda:
            msg = str(e)
            if ("NVML" in msg or "CUDA" in msg or
                    "out of memory" in msg.lower()):
                _disable_vae_cuda(f"decode error: {msg.splitlines()[0]}")
        raise
    else:
        if vae_on_cuda:
            _VAE_CUDA_STATE["successes"] += 1
            logger.info(
                f"[JETSON] Patch 16: VAE-CUDA decode success "
                f"(session: {_VAE_CUDA_STATE['successes']} OK / "
                f"{_VAE_CUDA_STATE['failures']} failed)"
            )
    finally:
        vae_decoder_hook.remove()
        # If VAE was moved to CUDA, return it to CPU to free VRAM for the
        # decoder restore on the next request.
        if vae_on_cuda:
            try:
                handler.vae = handler.vae.to("cpu")
                _aggressive_free()
            except Exception as e:
                logger.warning(
                    f"[JETSON] Patch 16: VAE -> CPU cleanup failed "
                    f"(non-fatal): {e}"
                )
        if decoder_was_cuda:
            # Patch 14: Lazy restore. Do NOT migrate decoder back to CUDA here.
            # At VAE-exit the allocator is under maximum pressure (VAE
            # activations still dropping, latents being cleaned, swap churn).
            # This is where the NVML_SUCCESS == r INTERNAL ASSERT fires.
            # Flag the handler; the next call to _patched_load_model_context
            # will do the restore during the calm before the next generation,
            # with pressure relief and retry.
            handler._decoder_needs_cuda_restore = True
            logger.info(
                "[JETSON] Patch 14: Decoder restore deferred (lazy). "
                "Will migrate to CUDA at next load_model_context entry."
            )


# ═══════════════════════════════════════════════════════════════
# Patch 6: VAE loading — CPU load → per-param to CUDA
# ═══════════════════════════════════════════════════════════════

def _patched_load_vae_model(self, *, checkpoint_dir, device, compile_model):
    """Load VAE with per-parameter CUDA transfer (same pattern as DiT)."""
    from diffusers.models import AutoencoderOobleck

    vae_path = os.path.join(checkpoint_dir, "vae")
    if not os.path.exists(vae_path):
        raise FileNotFoundError(f"VAE not found at {vae_path}")

    vae_dtype = self._get_vae_dtype(device) if not self.offload_to_cpu else self._get_vae_dtype("cpu")

    self.vae = AutoencoderOobleck.from_pretrained(vae_path, torch_dtype=vae_dtype)

    if not self.offload_to_cpu and device.startswith("cuda"):
        logger.info("[JETSON] Moving VAE to CUDA (per-parameter)...")
        _move_model_to_cuda_per_param(self.vae, dtype=vae_dtype)
    elif not self.offload_to_cpu:
        self.vae = self.vae.to(device).to(vae_dtype)
    else:
        self.vae = self.vae.to("cpu").to(vae_dtype)

    # Patch 12: float32 conv wrappers — ALWAYS apply regardless of offload setting.
    # The wrappers modify .forward methods and are device-agnostic (they reference
    # conv.weight at call time). When offload_to_cpu=True, _load_model_context moves
    # the VAE to CUDA at generation time, and the wrappers must already be in place.
    _fix_vae_conv_float32(self.vae)

    self.vae.eval()

    if compile_model:
        self._ensure_len_for_compile(self.vae, "vae")
        self.vae = torch.compile(self.vae)

    return vae_path


# ═══════════════════════════════════════════════════════════════
# Patch 7: Inject quantization from environment variable
# ═══════════════════════════════════════════════════════════════

_original_initialize_service = None

def _patched_initialize_service(
    self,
    project_root,
    config_path,
    device="auto",
    use_flash_attention=False,
    compile_model=False,
    offload_to_cpu=False,
    offload_dit_to_cpu=False,
    quantization=None,
    prefer_source=None,
    use_mlx_dit=True,
):
    """Inject ACESTEP_QUANTIZATION env var if quantization not passed."""
    if quantization is None:
        env_quant = os.environ.get("ACESTEP_QUANTIZATION")
        if env_quant:
            quantization = env_quant
            logger.info(f"[JETSON] Injecting quantization from env: {quantization}")
    return _original_initialize_service(
        self,
        project_root=project_root,
        config_path=config_path,
        device=device,
        use_flash_attention=use_flash_attention,
        compile_model=compile_model,
        offload_to_cpu=offload_to_cpu,
        offload_dit_to_cpu=offload_dit_to_cpu,
        quantization=quantization,
        prefer_source=prefer_source,
        use_mlx_dit=use_mlx_dit,
    )


# ═══════════════════════════════════════════════════════════════
# Patch 8: Allow hot-swap text encoder in readiness check
# ═══════════════════════════════════════════════════════════════

def _patched_validate_readiness(self):
    """Skip text_encoder check — it's hot-swapped per request on Jetson."""
    if self.model is None or self.vae is None or self.text_tokenizer is None:
        return {
            "audios": [],
            "status_message": "Model not fully initialized.",
            "extra_outputs": {},
            "success": False,
            "error": "Model not fully initialized",
        }
    if self.text_encoder is None and not hasattr(self, "_text_encoder_path"):
        return {
            "audios": [],
            "status_message": "Text encoder not configured.",
            "extra_outputs": {},
            "success": False,
            "error": "Text encoder not configured",
        }
    return None


# ═══════════════════════════════════════════════════════════════
# Patch 9: Force batch_size=1 on Jetson unified memory
# ═══════════════════════════════════════════════════════════════

def _patch_gpu_config_batch_size():
    """Force max batch size to 1 on Jetson — unified memory can't handle batch=2.

    The GPU config sets batch_size based on VRAM tiers designed for discrete GPUs.
    On Jetson where CPU+GPU share 7.6 GB, batch=2 doubles intermediate allocations
    in the same memory pool and causes NvMap OOM during batch preparation.

    We patch at TWO levels for belt-and-braces:
    1. GPU_TIER_CONFIGS dict — catches any future get_gpu_config() calls
    2. check_batch_size_limit — final gate at generation time
    """
    try:
        from acestep import gpu_config

        # Level 1: Patch ALL tier configs (before config object is created)
        for tier_name, tier_cfg in gpu_config.GPU_TIER_CONFIGS.items():
            if tier_cfg.get('max_batch_size_with_lm', 0) > JETSON_MAX_BATCH_SIZE:
                tier_cfg['max_batch_size_with_lm'] = JETSON_MAX_BATCH_SIZE
            if tier_cfg.get('max_batch_size_without_lm', 0) > JETSON_MAX_BATCH_SIZE:
                tier_cfg['max_batch_size_without_lm'] = JETSON_MAX_BATCH_SIZE

        # Level 2: Patch the runtime check that validates batch_size at generation time
        _orig_check_batch = gpu_config.check_batch_size_limit

        def _capped_check_batch(*args, **kwargs):
            result = _orig_check_batch(*args, **kwargs)
            if isinstance(result, int) and result > JETSON_MAX_BATCH_SIZE:
                return JETSON_MAX_BATCH_SIZE
            return result

        gpu_config.check_batch_size_limit = _capped_check_batch

        logger.info(f"[JETSON] GPU config patched: all tiers capped to batch_size={JETSON_MAX_BATCH_SIZE}")
    except Exception as e:
        logger.warning(f"[JETSON] Could not patch GPU config batch_size: {e}")


# ═══════════════════════════════════════════════════════════════
# Apply all patches
# ═══════════════════════════════════════════════════════════════

def apply_jetson_patches():
    """Apply all Jetson Orin Nano memory-management patches."""
    global _original_initialize_service
    from acestep.core.generation.handler.init_service_setup import InitServiceSetupMixin
    from acestep.core.generation.handler.init_service_loader import InitServiceLoaderMixin
    from acestep.core.generation.handler.init_service_loader_components import InitServiceLoaderComponentsMixin
    from acestep.core.generation.handler.init_service_offload_context import InitServiceOffloadContextMixin
    from acestep.core.generation.handler.init_service_orchestrator import InitServiceOrchestratorMixin
    from acestep.core.generation.handler.generate_music_request import GenerateMusicRequestMixin

    # Save originals
    InitServiceOffloadContextMixin._original_load_model_context = (
        InitServiceOffloadContextMixin._load_model_context
    )
    _original_initialize_service = InitServiceOrchestratorMixin.initialize_service

    # Apply patches 1-8
    InitServiceSetupMixin._validate_quantization_setup = _patched_validate_quantization_setup
    InitServiceLoaderMixin._load_main_model_from_checkpoint = _patched_load_main_model
    InitServiceLoaderMixin._apply_dit_quantization = _noop_dit_quantization
    InitServiceLoaderComponentsMixin._load_text_encoder_and_tokenizer = _patched_load_text_encoder_and_tokenizer
    InitServiceLoaderComponentsMixin._load_vae_model = _patched_load_vae_model
    InitServiceOffloadContextMixin._load_model_context = _patched_load_model_context

    # Patch 7: inject quantization from env
    InitServiceOrchestratorMixin.initialize_service = _patched_initialize_service

    # Patch 8: allow hot-swap text encoder in readiness check
    GenerateMusicRequestMixin._validate_generate_music_readiness = _patched_validate_readiness

    # Patch 9: force batch_size=1
    _patch_gpu_config_batch_size()

    # Verify allocator config
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments:False" not in alloc_conf:
        logger.warning(
            "[JETSON] PYTORCH_CUDA_ALLOC_CONF missing expandable_segments:False! "
            "NvMap pages may not be released on model unload."
        )

    logger.info(
        "[JETSON] All 16 patches applied "
        "(CPU-quanto + decoder-only-CUDA + CPU-prepare-condition + "
        "CPU-text-encoder + hot-swap + env-quant + readiness + batch-cap + "
        "KV-cache-reset + decoder-VAE-swap + VAE-float32-conv + CPU-VAE-decode + "
        "torchaudio-save-fix + safe-to-cuda + VAE-CUDA-probe-fallback)"
    )


# Main
if __name__ == "__main__":
    apply_jetson_patches()

    import sys
    mode = os.environ.get("ACESTEP_MODE", "gradio")

    if mode == "api":
        from acestep.api_server import main
        main()
    else:
        sys.argv = [
            "acestep",
            "--server-name", os.environ.get("ACESTEP_API_HOST", "0.0.0.0"),
            "--port", os.environ.get("ACESTEP_API_PORT", "8006"),
            "--enable-api",
            "--init_service", "true",
            "--config_path", os.environ.get("ACESTEP_CONFIG_PATH", "acestep-v15-turbo"),
            "--device", "cuda",
            "--offload_to_cpu", "true",
            "--offload_dit_to_cpu", "false",
            "--quantization", os.environ.get("ACESTEP_QUANTIZATION", "int8_weight_only"),
            "--init_llm", os.environ.get("ACESTEP_INIT_LLM", "false"),
            "--lm_model_path", os.environ.get("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-0.6B"),
            "--backend", os.environ.get("ACESTEP_LM_BACKEND", "pt"),
            "--allowed-path", "/app/outputs",
        ]
        from acestep.acestep_v15_pipeline import main
        main()
