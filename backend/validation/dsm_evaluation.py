"""
Advanced DSM Evaluation Framework for DepthWizard

This module isolates evaluation experiments (multi-scale, cross-validation, 
terrain masking, and robust calibration analysis) without mutating the core 
production calibration logic.
"""

import logging
import numpy as np
from typing import Dict, Any, List
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from scipy import stats

from app.validation.accuracy import validate_against_reference

logger = logging.getLogger(__name__)

def _aggregate_raster(dsm: np.ndarray, scale_factor: int) -> np.ndarray:
    """Downsample a raster by local mean aggregation to simulate multi-scale resolution."""
    if scale_factor <= 1:
        return dsm
    
    h, w = dsm.shape
    new_h, new_w = h // scale_factor, w // scale_factor
    
    aggregated = np.zeros((new_h, new_w), dtype=np.float32)
    for i in range(new_h):
        for j in range(new_w):
            block = dsm[i*scale_factor:(i+1)*scale_factor, j*scale_factor:(j+1)*scale_factor]
            valid_mask = np.isfinite(block)
            if np.any(valid_mask):
                aggregated[i, j] = np.mean(block[valid_mask])
            else:
                aggregated[i, j] = np.nan
    return aggregated

def evaluate_multiscale(
    predicted_dsm: np.ndarray,
    reference_file_path: str,
    target_crs: str,
    target_transform: Any,
    scales: List[int] = [1, 5, 10, 30]
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates DSM accuracy at multiple spatial resolutions.
    """
    results = {}
    
    # Align the reference to the native grid first, then aggregate both natively
    try:
        with rasterio.open(reference_file_path) as ref_src:
            ref_data = ref_src.read(1).astype(np.float32)
            ref_nodata = ref_src.nodata
            
            target_h, target_w = predicted_dsm.shape
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
            
            if ref_nodata is not None:
                aligned_ref[aligned_ref == ref_nodata] = np.nan
                
    except Exception as e:
        logger.error(f"Failed to align reference raster: {e}")
        return {}

    for scale in scales:
        agg_pred = _aggregate_raster(predicted_dsm, scale)
        agg_ref = _aggregate_raster(aligned_ref, scale)
        
        mask = np.isfinite(agg_pred) & np.isfinite(agg_ref)
        mask = mask & (agg_ref > -1000.0) & (agg_ref < 9000.0)
        
        dsm_vals = agg_pred[mask]
        ref_vals = agg_ref[mask]
        
        n_samples = len(dsm_vals)
        total_pixels = agg_pred.shape[0] * agg_pred.shape[1]
        
        if n_samples < 10:
            results[f"scale_{scale}x"] = {"rmse": None, "mae": None, "r_squared": None}
            continue
            
        diff = dsm_vals - ref_vals
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))
        
        ss_res = np.sum(diff ** 2)
        ss_tot = np.sum((ref_vals - np.mean(ref_vals)) ** 2)
        r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        
        results[f"{scale}x"] = {
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "r_squared": round(r_squared, 4),
            "valid_percentage": round((n_samples / total_pixels) * 100, 2) if total_pixels > 0 else 0
        }
        
    return results

def evaluate_spatial_cross_validation(
    predicted_dsm: np.ndarray,
    reference_file_path: str,
    target_crs: str,
    target_transform: Any,
    grid_size: tuple = (4, 4)
) -> Dict[str, Any]:
    """
    Evaluates DSM accuracy by slicing the geographic extent into a grid of blocks.
    Reports the mean and variance of RMSE/R2 across blocks.
    """
    # Align reference
    target_h, target_w = predicted_dsm.shape
    aligned_ref = np.zeros((target_h, target_w), dtype=np.float32)
    try:
        with rasterio.open(reference_file_path) as ref_src:
            ref_data = ref_src.read(1).astype(np.float32)
            reproject(
                source=ref_data,
                destination=aligned_ref,
                src_transform=ref_src.transform,
                src_crs=ref_src.crs,
                dst_transform=target_transform,
                dst_crs=CRS.from_user_input(target_crs),
                resampling=Resampling.bilinear
            )
            if ref_src.nodata is not None:
                aligned_ref[aligned_ref == ref_src.nodata] = np.nan
    except Exception as e:
        logger.error(f"Cross-val alignment failed: {e}")
        return {}
        
    h_step = target_h // grid_size[0]
    w_step = target_w // grid_size[1]
    
    block_metrics = []
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            pred_block = predicted_dsm[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
            ref_block = aligned_ref[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
            
            mask = np.isfinite(pred_block) & np.isfinite(ref_block) & (ref_block > -1000) & (ref_block < 9000)
            
            p_vals = pred_block[mask]
            r_vals = ref_block[mask]
            
            if len(p_vals) < 100: # Need decent sample size per block
                continue
                
            diff = p_vals - r_vals
            rmse = np.sqrt(np.mean(diff**2))
            mae = np.mean(np.abs(diff))
            
            ss_res = np.sum(diff ** 2)
            ss_tot = np.sum((r_vals - np.mean(r_vals)) ** 2)
            r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
            
            block_metrics.append({
                "block": (i, j),
                "rmse": rmse,
                "mae": mae,
                "r_squared": r_squared
            })
            
    if not block_metrics:
        return {"status": "failed", "reason": "Not enough valid blocks."}
        
    rmse_list = [b["rmse"] for b in block_metrics]
    r2_list = [b["r_squared"] for b in block_metrics]
    
    return {
        "status": "success",
        "num_valid_blocks": len(block_metrics),
        "mean_rmse": round(float(np.mean(rmse_list)), 2),
        "std_rmse": round(float(np.std(rmse_list)), 2),
        "mean_r2": round(float(np.mean(r2_list)), 4),
        "std_r2": round(float(np.std(r2_list)), 4),
        "min_r2": round(float(np.min(r2_list)), 4),
        "max_r2": round(float(np.max(r2_list)), 4)
    }

def run_calibration_experiments(relative_depth: np.ndarray, reference_dsm: np.ndarray) -> Dict[str, Any]:
    """
    Given relative depth (0-1) and ground-truth DEM, simulate different regression techniques 
    to see if they would improve over standard linear regression (without changing app/calibration.py).
    """
    mask = np.isfinite(relative_depth) & np.isfinite(reference_dsm) & (reference_dsm > -1000) & (reference_dsm < 9000)
    x = relative_depth[mask].astype(np.float64)
    y = reference_dsm[mask].astype(np.float64)
    
    if len(x) < 10:
        return {"error": "Insufficient valid points for regression experiments."}
        
    experiments = {}
    
    # 1. Standard Linear (Baseline)
    slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
    y_pred_lin = x * slope + intercept
    ss_res_lin = np.sum((y - y_pred_lin) ** 2)
    ss_tot_lin = np.sum((y - np.mean(y)) ** 2)
    experiments["standard_linear"] = {
        "r_squared": round(1 - (ss_res_lin / ss_tot_lin), 4) if ss_tot_lin > 0 else 0.0,
        "rmse": round(np.sqrt(np.mean((y - y_pred_lin)**2)), 2),
        "slope": round(slope, 2),
        "intercept": round(intercept, 2)
    }
    
    # 2. Inverted Linear (if Depth Anything output is inverted)
    # i.e. closer = smaller depth value
    slope_inv, intercept_inv, _, _, _ = stats.linregress(1.0 - x, y)
    y_pred_inv = (1.0 - x) * slope_inv + intercept_inv
    ss_res_inv = np.sum((y - y_pred_inv) ** 2)
    experiments["inverted_linear"] = {
        "r_squared": round(1 - (ss_res_inv / ss_tot_lin), 4) if ss_tot_lin > 0 else 0.0,
        "rmse": round(np.sqrt(np.mean((y - y_pred_inv)**2)), 2),
    }
    
    # 3. Robust Linear Regression (Huber / RANSAC)
    # We will just do a simple trimming of the top and bottom 5% outliers of the residual
    diff_lin = y - y_pred_lin
    p5, p95 = np.percentile(diff_lin, 5), np.percentile(diff_lin, 95)
    inlier_mask = (diff_lin >= p5) & (diff_lin <= p95)
    
    if np.sum(inlier_mask) > 10:
        slope_rob, intercept_rob, _, _, _ = stats.linregress(x[inlier_mask], y[inlier_mask])
        y_pred_rob = x * slope_rob + intercept_rob
        ss_res_rob = np.sum((y - y_pred_rob) ** 2)
        experiments["trimmed_linear"] = {
            "r_squared": round(1 - (ss_res_rob / ss_tot_lin), 4) if ss_tot_lin > 0 else 0.0,
            "rmse": round(np.sqrt(np.mean((y - y_pred_rob)**2)), 2)
        }
    
    # 4. Polynomial Degree 2
    poly_coeffs = np.polyfit(x, y, 2)
    y_pred_poly = np.polyval(poly_coeffs, x)
    ss_res_poly = np.sum((y - y_pred_poly) ** 2)
    experiments["polynomial_deg2"] = {
        "r_squared": round(1 - (ss_res_poly / ss_tot_lin), 4) if ss_tot_lin > 0 else 0.0,
        "rmse": round(np.sqrt(np.mean((y - y_pred_poly)**2)), 2)
    }
    
    return experiments

def evaluate_terrain_masking(
    predicted_dsm: np.ndarray,
    reference_dsm: np.ndarray,
    land_cover_mask: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates DSM accuracy across different semantic land-cover masks.
    land_cover_mask classes typically: 0: ground, 1: veg, 2: water, 3: built-up, 4: shadow
    """
    valid_mask = np.isfinite(predicted_dsm) & np.isfinite(reference_dsm) & (reference_dsm > -1000)
    
    masks = {
        "All Valid Pixels": valid_mask,
        "Terrain-Only": valid_mask & (land_cover_mask == 0),
        "Terrain + Vegetation": valid_mask & ((land_cover_mask == 0) | (land_cover_mask == 1)),
        "Excluding Buildings": valid_mask & (land_cover_mask != 3),
        "Excluding Water": valid_mask & (land_cover_mask != 2),
        "Excluding Strong Shadows": valid_mask & (land_cover_mask != 4)
    }
    
    results = {}
    for name, m in masks.items():
        if np.sum(m) < 100:
            results[name] = {"rmse": None, "mae": None, "r_squared": None, "samples": np.sum(m)}
            continue
            
        d_val = predicted_dsm[m]
        r_val = reference_dsm[m]
        
        diff = d_val - r_val
        rmse = np.sqrt(np.mean(diff**2))
        mae = np.mean(np.abs(diff))
        
        ss_res = np.sum(diff**2)
        ss_tot = np.sum((r_val - np.mean(r_val))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        results[name] = {
            "rmse": round(float(rmse), 2),
            "mae": round(float(mae), 2),
            "r_squared": round(float(r2), 4),
            "samples": int(np.sum(m))
        }
        
    return results

def evaluate_depth_preprocessing(
    rel_depth: np.ndarray,
    reference_dsm: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates different depth preprocessing filters prior to linear regression.
    """
    from scipy.ndimage import gaussian_filter
    import cv2
    
    mask = np.isfinite(rel_depth) & np.isfinite(reference_dsm) & (reference_dsm > -1000)
    if np.sum(mask) < 100:
        return {}
        
    results = {}
    
    def score_depth(d):
        x = d[mask].astype(np.float64)
        y = reference_dsm[mask].astype(np.float64)
        slope, intercept, _, _, _ = stats.linregress(x, y)
        y_pred = x * slope + intercept
        
        diff = y - y_pred
        ss_res = np.sum(diff**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = np.sqrt(np.mean(diff**2))
        mae = np.mean(np.abs(diff))
        
        return {"rmse": round(rmse, 2), "mae": round(mae, 2), "r_squared": round(r2, 4)}
        
    # 1. Raw
    results["Raw Normalized"] = score_depth(rel_depth)
    
    # 2. Inverted
    results["Inverted Normalized"] = score_depth(1.0 - rel_depth)
    
    # 3. Light Smoothing (Gaussian)
    results["Light Smoothing (Gaussian)"] = score_depth(gaussian_filter(rel_depth, sigma=1.0))
    
    # 4. Edge-Preserving (Bilateral)
    # Convert to float32 for OpenCV
    d32 = rel_depth.astype(np.float32)
    bilateral = cv2.bilateralFilter(d32, d=9, sigmaColor=0.1, sigmaSpace=7)
    results["Edge-Preserving (Bilateral)"] = score_depth(bilateral)
    
    return results

def generate_error_map(predicted_dsm: np.ndarray, reference_dsm: np.ndarray, out_path: str):
    """
    Generates and saves a visual error map (Predicted - Reference).
    """
    import matplotlib.pyplot as plt
    
    mask = np.isfinite(predicted_dsm) & np.isfinite(reference_dsm) & (reference_dsm > -1000)
    diff = np.full_like(predicted_dsm, np.nan)
    diff[mask] = predicted_dsm[mask] - reference_dsm[mask]
    
    plt.figure(figsize=(10, 8))
    # Colormap: blue for negative error (under-predicted), red for positive error (over-predicted)
    vmax = np.nanpercentile(np.abs(diff), 95)
    plt.imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(label='Error (m)')
    plt.title('DSM Error Map (Predicted - Reference)')
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

