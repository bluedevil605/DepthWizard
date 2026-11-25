"""
Depth Module for DepthWizard
Performs monocular relative depth estimation using Depth Anything V2.

PRODUCTION SYSTEM IMPROVEMENT NOTES:
- Model Selection: In a production setting, allow runtime model selection:
  e.g., Depth-Anything-V2-Base or Depth-Anything-V2-Large for maximum geometric fidelity,
  or multi-view stereopsis / shadow-depth fusion networks trained specifically on overhead orthoimagery.
- Batching & Tiling: Aerial gigapixel orthomosaics require seamless overlapping sliding-window
  tiling with Gaussian blending across tile boundaries to avoid seam artifacts.
- GPU Acceleration: Deploy with TensorRT or ONNX Runtime inference server (Triton) with FP16/BF16.
"""

import logging
import os
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)

# Global depth model cache
_DEPTH_PIPELINE = None


def get_depth_pipeline():
    """
    Lazy loader for Depth Anything V2 pipeline.
    Uses 'depth-anything/Depth-Anything-V2-Small-hf' for speed in hackathon demo.
    Falls back gracefully to offline relief estimation if HF download is inaccessible.
    """
    global _DEPTH_PIPELINE
    if _DEPTH_PIPELINE is not None:
        return _DEPTH_PIPELINE

    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation, pipeline

        device = 0 if torch.cuda.is_available() else -1
        model_id = "depth-anything/Depth-Anything-V2-Small-hf"
        logger.info(f"Loading Depth-Anything-V2-Small-hf (device: {device})...")
        
        # Try local cache first for fast, deterministic offline startup
        try:
            img_proc = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
            model = AutoModelForDepthEstimation.from_pretrained(model_id, local_files_only=True)
            _DEPTH_PIPELINE = pipeline(
                task="depth-estimation",
                model=model,
                image_processor=img_proc,
                device=device,
            )
        except Exception:
            _DEPTH_PIPELINE = pipeline(
                task="depth-estimation",
                model=model_id,
                device=device,
            )

        logger.info(f"SUCCESS: Depth Anything V2 pipeline ('{model_id}') loaded successfully on device: {device}.")
        return _DEPTH_PIPELINE
    except Exception as e:
        logger.warning(
            f"Failed to load Depth Anything V2 pipeline ({e}). "
            "Offline/demo fallback relief estimator will be used."
        )
        return None


def estimate_relative_depth(image: Image.Image) -> np.ndarray:
    """
    Estimate monocular relative depth from an RGB PIL image.
    
    Returns:
        np.ndarray: 2D float32 array normalized to [0.0, 1.0], where 1.0 is highest elevation/relief.
    """
    pipe = get_depth_pipeline()

    # Ensure RGB format
    if image.mode != "RGB":
        image = image.convert("RGB")

    orig_w, orig_h = image.size
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("[DEPTH] Starting inference")
    logger.info("[DEPTH] Model: depth-anything/Depth-Anything-V2-Small-hf")
    logger.info(f"[DEPTH] Device: {device_name}")
    logger.info(f"[DEPTH] Input: {orig_w}x{orig_h}")

    # Pre-scale large imagery to max 1024px for depth inference.
    # Depth Anything V2 natively resamples internally to ~518px ViT patches anyway.
    # Passing 4K images to pipeline() causes severe memory bloat and slow 35s CPU processing;
    # pre-scaling preserves full topographic resolution while finishing in ~4s.
    max_dim = 1024
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / float(max(orig_w, orig_h))
        proc_w = max(16, int(round(orig_w * scale)))
        proc_h = max(16, int(round(orig_h * scale)))
        proc_img = image.resize((proc_w, proc_h), Image.Resampling.BILINEAR)
    else:
        proc_w, proc_h = orig_w, orig_h
        proc_img = image

    if pipe is not None:
        try:
            result = pipe(proc_img)
            # Result can be a dict with key 'depth' or a PIL image
            depth_img = result.get("depth", result) if isinstance(result, dict) else result
            depth_arr = np.array(depth_img, dtype=np.float32)

            # Ensure depth map matches processing dimensions
            if depth_arr.shape != (proc_h, proc_w):
                depth_pil = Image.fromarray(depth_arr)
                depth_arr = np.array(
                    depth_pil.resize((proc_w, proc_h), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                )

            # Convert to float32
            depth_arr = depth_arr.astype(np.float32)
            
            # Handle invalid values safely
            if not np.isfinite(depth_arr).all():
                depth_arr[~np.isfinite(depth_arr)] = np.nanmedian(depth_arr)

            # In Depth Anything, output is metric/inverse-depth where closer/higher relief is higher.
            # Compute raw diagnostics before correction
            raw_min = float(depth_arr.min())
            raw_max = float(depth_arr.max())
            raw_mean = float(depth_arr.mean())
            raw_std = float(depth_arr.std())

            # Robust percentile bounds to ignore extreme monolithic errors
            p_lo, p_hi = np.percentile(depth_arr, 2), np.percentile(depth_arr, 98)
            
            # Clip and normalize to [0, 1]
            clipped = np.clip(depth_arr, p_lo, p_hi)
            if p_hi > p_lo:
                normalized = (clipped - p_lo) / (p_hi - p_lo)
            else:
                normalized = np.zeros_like(clipped)

            # --- Outlier Rejection: Keep Raw Structural Step Discontinuities ---
            # Reject only extreme single-pixel spike artifacts (deviation > 0.25 relief in 3x3)
            # Avoids eroding sharp building corners or walls.
            from scipy.ndimage import median_filter, gaussian_filter
            local_med3 = median_filter(normalized, size=3)
            spike_mask = np.abs(normalized - local_med3) > 0.25
            normalized[spike_mask] = local_med3[spike_mask]

            # --- Multi-scale Unsharp Mask for Depth Sharpening ---
            # Depth Anything V2 (ViT-based) produces smooth depth maps due to patch-wise
            # attention. We apply a two-scale unsharp mask to recover sharp edges at
            # building walls, roof boundaries and terrain ridgelines without over-sharpening
            # flat areas (which have near-zero gradient and are barely affected).
            # Scale 1 (sigma=1.0): fine edge detail — sharpens pixel-level discontinuities
            # Scale 2 (sigma=3.0): coarse structure — lifts building masses vs ground plane
            _fine_blur = gaussian_filter(normalized, sigma=1.0)
            _coarse_blur = gaussian_filter(normalized, sigma=3.0)
            # alpha_fine=0.8 boosts hairline edges; alpha_coarse=0.4 lifts structure masses
            normalized = normalized + 0.8 * (normalized - _fine_blur) + 0.4 * (normalized - _coarse_blur)
            normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)

            raw_structural_depth = normalized

            logger.info(
                f"[DIAGNOSTICS] Depth processing: Raw min={raw_min:.3f}, max={raw_max:.3f}, mean={raw_mean:.3f}, std={raw_std:.3f}. "
                f"Percentile bounds: [{p_lo:.3f}, {p_hi:.3f}]."
            )

            # Resize depth map back to source dimensions if pre-scaled
            if (proc_w, proc_h) != (orig_w, orig_h):
                raw_pil = Image.fromarray(raw_structural_depth)
                final_depth = np.array(
                    raw_pil.resize((orig_w, orig_h), Image.Resampling.BILINEAR),
                    dtype=np.float32
                )
            else:
                final_depth = raw_structural_depth

            logger.info(f"[DEPTH] Output shape: {final_depth.shape}")
            logger.info(f"[DEPTH] Min: {float(np.nanmin(final_depth)):.4f}")
            logger.info(f"[DEPTH] Max: {float(np.nanmax(final_depth)):.4f}")
            logger.info(f"[DEPTH] Mean: {float(np.nanmean(final_depth)):.4f}")

            return final_depth
        except Exception as err:
            logger.error(f"Inference error in depth pipeline: {err}. Using fallback relief.")

    # Offline hackathon fallback: Frequency and luminance decomposition relief heuristic
    # Combines low-frequency lighting gradients and local contrast as an elevation proxy
    # ensuring offline demonstrations NEVER crash.
    logger.info("Computing offline relief fallback...")
    gray = np.array(image.convert("L"), dtype=np.float32) / 255.0
    from scipy.ndimage import gaussian_filter, laplace
    
    low_freq = gaussian_filter(gray, sigma=max(orig_w, orig_h) / 15.0)
    high_freq = np.abs(laplace(gray))
    combined = low_freq * 0.7 + high_freq * 0.3
    
    c_min = float(combined.min())
    c_max = float(combined.max())
    c_mean = float(combined.mean())
    logger.info(
        f"[DEBUG] Fallback relief raw output: min={c_min:.4f}, max={c_max:.4f}, mean={c_mean:.4f}"
    )
    if c_max > c_min:
        rel_depth = (combined - c_min) / (c_max - c_min)
    else:
        rel_depth = np.zeros_like(combined)

    logger.info(
        f"[DEBUG] Fallback relief normalized: min={float(rel_depth.min()):.4f}, "
        f"max={float(rel_depth.max()):.4f}, mean={float(rel_depth.mean()):.4f}"
    )
    return rel_depth.astype(np.float32)


def generate_depth_preview_png(rel_depth: np.ndarray, output_path: str):
    """
    Renders a normalized 2D relative depth map as a colorful high-contrast preview PNG.
    Uses an internal turbo-style elevation gradient.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Simple, fast turbo/terrain color ramp without heavy matplotlib dependency
    # Values 0.0 -> 1.0 map from deep navy blue -> cyan -> yellow-green -> orange -> white-peak
    x = np.clip(rel_depth, 0.0, 1.0)
    
    r = np.clip(1.5 * x - 0.2, 0.0, 1.0)
    g = np.clip(1.0 - np.abs(2.0 * x - 1.0), 0.0, 1.0)
    b = np.clip(1.2 * (1.0 - 1.5 * x), 0.0, 1.0)
    
    rgb = np.stack([r, g, b], axis=-1)
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    
    preview_img = Image.fromarray(rgb_uint8, mode="RGB")
    preview_img.save(output_path, "PNG")
    return output_path

# GAMUS is intentionally a second model: it estimates metric AGL rather than generic relative depth.
_GAMUS_PIPELINE = None
GAMUS_CHECKPOINT_VERSION = "v2"  # Change to "v1" to use the fallback/reference checkpoint.
_GAMUS_CHECKPOINTS = {
    "v1": ("depth_anything_gamus_finetuned", "gamus_finetuned_weights.zip"),
    "v2": ("depth_anything_gamus_v2_finetuned", "gamus_v2_finetuned_weights.zip"),
}
if GAMUS_CHECKPOINT_VERSION not in _GAMUS_CHECKPOINTS:
    raise ValueError(f"Unsupported GAMUS_CHECKPOINT_VERSION: {GAMUS_CHECKPOINT_VERSION}")
GAMUS_MODEL_NAME, GAMUS_ARCHIVE_NAME = _GAMUS_CHECKPOINTS[GAMUS_CHECKPOINT_VERSION]
GAMUS_ARCHIVE = os.path.join(os.path.dirname(os.path.dirname(__file__)), GAMUS_ARCHIVE_NAME)
GAMUS_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", GAMUS_MODEL_NAME)

def get_gamus_pipeline():
    """Lazy-load the local GAMUS fine-tune, which emits metric AGL in meters."""
    global _GAMUS_PIPELINE
    if _GAMUS_PIPELINE is not None:
        return _GAMUS_PIPELINE
    try:
        import zipfile
        config_path = os.path.join(GAMUS_MODEL_DIR, "config.json")
        if not os.path.exists(config_path):
            if not os.path.exists(GAMUS_ARCHIVE):
                raise FileNotFoundError(f"GAMUS checkpoint archive not found: {GAMUS_ARCHIVE}")
            os.makedirs(GAMUS_MODEL_DIR, exist_ok=True)
            with zipfile.ZipFile(GAMUS_ARCHIVE) as archive:
                archive.extractall(GAMUS_MODEL_DIR)
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation, pipeline
        device = 0 if torch.cuda.is_available() else -1
        processor = AutoImageProcessor.from_pretrained(GAMUS_MODEL_DIR, local_files_only=True)
        model = AutoModelForDepthEstimation.from_pretrained(GAMUS_MODEL_DIR, local_files_only=True)
        _GAMUS_PIPELINE = pipeline("depth-estimation", model=model, image_processor=processor, device=device)
        logger.info("Loaded GAMUS fine-tuned metric-AGL checkpoint (device: %s).", device)
        return _GAMUS_PIPELINE
    except Exception as exc:
        logger.warning("Could not load GAMUS metric-AGL checkpoint: %s", exc)
        return None

def estimate_gamus_agl(image: Image.Image) -> np.ndarray:
    """Return GAMUS per-pixel AGL predictions in metres (never normalized)."""
    pipe = get_gamus_pipeline()
    if pipe is None:
        raise RuntimeError("GAMUS checkpoint is unavailable; cannot run metric AGL calibration")
    image = image.convert("RGB")
    orig_w, orig_h = image.size
    scale = min(1.0, 1024.0 / max(orig_w, orig_h))
    proc_w, proc_h = max(16, round(orig_w * scale)), max(16, round(orig_h * scale))
    proc_img = image.resize((proc_w, proc_h), Image.Resampling.BILINEAR) if scale < 1 else image
    result = pipe(proc_img)
    agl = np.asarray(result.get("depth", result) if isinstance(result, dict) else result, dtype=np.float32).copy()
    if agl.shape != (proc_h, proc_w):
        agl = np.asarray(Image.fromarray(agl).resize((proc_w, proc_h), Image.Resampling.BILINEAR), dtype=np.float32).copy()
    if (proc_w, proc_h) != (orig_w, orig_h):
        agl = np.asarray(Image.fromarray(agl).resize((orig_w, orig_h), Image.Resampling.BILINEAR), dtype=np.float32).copy()
    agl[~np.isfinite(agl)] = np.nanmedian(agl)
    return np.clip(agl, 0.0, None).astype(np.float32)