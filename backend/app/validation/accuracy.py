"""
Topographic Accuracy Validation Module for DepthWizard
Computes statistical accuracy metrics (RMSE, MAE, Pearson correlation)
between generated DSM and high-precision reference rasters (e.g., aerial LiDAR).
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from scipy import stats

logger = logging.getLogger(__name__)


def validate_against_reference(
    generated_dsm: np.ndarray,
    reference_file_path: str,
    target_crs: Optional[str] = None,
    target_transform = None
) -> Dict[str, Any]:
    """
    Resamples the reference raster (e.g. mock LiDAR or survey DEM) to the generated DSM's grid
    and computes RMSE, MAE, Pearson correlation, and bias metrics.
    
    Args:
        generated_dsm: 2D float32 array of the generated DSM
        reference_file_path: Filepath to the reference raster
        target_crs: CRS of the generated DSM
        target_transform: Affine transform of the generated DSM
        
    Returns:
        Dict containing statistical error metrics
    """
    target_h, target_w = generated_dsm.shape

    try:
        with rasterio.open(reference_file_path) as ref_src:
            ref_data = ref_src.read(1).astype(np.float32)
            ref_nodata = ref_src.nodata

            # Resample reference raster to align precisely with generated DSM pixel grid
            if ref_src.crs and target_crs and target_transform:
                aligned_ref = np.zeros((target_h, target_w), dtype=np.float32)
                reproject(
                    source=ref_data,
                    destination=aligned_ref,
                    src_transform=ref_src.transform,
                    src_crs=ref_src.crs,
                    dst_transform=target_transform,
                    dst_crs=CRS.from_user_input(target_crs),
                    resampling=Resampling.bilinear
                )
            else:
                # If no CRS available, resize directly using PIL
                from PIL import Image
                ref_pil = Image.fromarray(ref_data)
                aligned_ref = np.array(
                    ref_pil.resize((target_w, target_h), Image.Resampling.BILINEAR),
                    dtype=np.float32
                )
    except Exception as err:
        logger.warning(f"Failed to read reference with rasterio ({err}), attempting fallback image reader...")
        from PIL import Image
        with Image.open(reference_file_path) as img:
            ref_pil = img.convert("F")
            aligned_ref = np.array(
                ref_pil.resize((target_w, target_h), Image.Resampling.BILINEAR),
                dtype=np.float32
            )
        ref_nodata = None

    # Filter out nodata and invalid non-finite values
    mask = np.isfinite(generated_dsm) & np.isfinite(aligned_ref)
    if ref_nodata is not None:
        mask = mask & (aligned_ref != ref_nodata)
    
    # Exclude standard nodata sentinel values (-9999, etc.)
    mask = mask & (aligned_ref > -1000.0) & (aligned_ref < 9000.0)

    dsm_vals = generated_dsm[mask].astype(np.float64)
    ref_vals = aligned_ref[mask].astype(np.float64)

    n_samples = len(dsm_vals)
    total_pixels = target_h * target_w
    valid_percentage = (n_samples / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    
    if n_samples < 10:
        logger.warning("Fewer than 10 overlapping valid pixels found for validation.")
        return {
            "rmse": None,
            "mae": None,
            "pearson_r": None,
            "r_squared": None,
            "mean_bias": None,
            "std_error": None,
            "sample_count": n_samples,
            "valid_percentage": round(valid_percentage, 2),
            "message": "Insufficient overlapping spatial coverage between DSM and reference raster."
        }

    # Statistical error calculations
    dsm_min = float(np.min(dsm_vals))
    dsm_max = float(np.max(dsm_vals))
    dsm_mean = float(np.mean(dsm_vals))
    ref_min = float(np.min(ref_vals))
    ref_max = float(np.max(ref_vals))
    ref_mean = float(np.mean(ref_vals))
    
    likely_mode = "relative (0-1 or small arbitrary range)" if dsm_max - dsm_min < 256 and dsm_mean < 1000 else "calibrated absolute meters"
    
    logger.info(
        f"[DIAGNOSTIC] Predicted DSM (generated_dsm) array stats: min={dsm_min:.4f}, max={dsm_max:.4f}, mean={dsm_mean:.4f}"
    )
    logger.info(
        f"[DIAGNOSTIC] Reference LiDAR raster (ref_vals) stats: min={ref_min:.4f}, max={ref_max:.4f}, mean={ref_mean:.4f}"
    )
    logger.info(
        f"[DIAGNOSTIC] Predicted array appears to be in mode: {likely_mode}"
    )

    diff = dsm_vals - ref_vals
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    std_error = float(np.std(diff))

    # R-squared (Coefficient of Determination)
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((ref_vals - np.mean(ref_vals)) ** 2)
    r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    # Pearson correlation coefficient
    if np.std(dsm_vals) > 1e-6 and np.std(ref_vals) > 1e-6:
        r_coeff, p_val = stats.pearsonr(dsm_vals, ref_vals)
        pearson_r = float(r_coeff)
    else:
        pearson_r = 1.0

    results = {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "pearson_r": round(pearson_r, 4),
        "r_squared": round(r_squared, 4),
        "mean_bias": round(bias, 2),
        "std_error": round(std_error, 2),
        "sample_count": int(n_samples),
        "valid_percentage": round(valid_percentage, 2),
        "reference_min": round(float(np.min(ref_vals)), 1),
        "reference_max": round(float(np.max(ref_vals)), 1)
    }

    logger.info(
        f"[VALIDATION] RMSE={results['rmse']}m, "
        f"MAE={results['mae']}m, "
        f"R²={results['r_squared']}, "
        f"N_samples={results['sample_count']} ({results['valid_percentage']}%)"
    )

    return results
