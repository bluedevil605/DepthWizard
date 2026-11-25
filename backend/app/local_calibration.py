"""
Localized Block-Matching Calibration Module for DepthWizard
============================================================
Experimental calibration that fits depth-to-elevation regression per
geographic block, smoothly interpolates calibration coefficients across
block boundaries, and falls back to global calibration for invalid blocks.

This module does NOT modify app/calibration.py.
"""

import logging
import json
import os
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _block_indices(h: int, w: int, block_h: int, block_w: int):
    """Yield (row_idx, col_idx, y_start, y_end, x_start, x_end) for each block."""
    n_rows = max(1, int(np.ceil(h / block_h)))
    n_cols = max(1, int(np.ceil(w / block_w)))
    for ri in range(n_rows):
        for ci in range(n_cols):
            y0 = ri * block_h
            y1 = min(y0 + block_h, h)
            x0 = ci * block_w
            x1 = min(x0 + block_w, w)
            yield ri, ci, y0, y1, x0, x1


def _fit_block(depth_block, ref_block, valid_mask_block, min_samples=100):
    """
    Fit linear regression for a single block.
    Returns dict with coefficients and diagnostics, or None on failure.
    """
    m = valid_mask_block
    if np.sum(m) < min_samples:
        return None

    xv = depth_block[m].astype(np.float64)
    yv = ref_block[m].astype(np.float64)

    # Check variance
    if np.std(xv) < 1e-6 or np.std(yv) < 1e-3:
        return None

    # Spatial split: top 70% rows → train, bottom 30% → val
    block_h = depth_block.shape[0]
    split_row = int(block_h * 0.7)

    train_mask = np.zeros_like(m)
    train_mask[:split_row, :] = m[:split_row, :]
    val_mask = np.zeros_like(m)
    val_mask[split_row:, :] = m[split_row:, :]

    n_train = np.sum(train_mask)
    n_val = np.sum(val_mask)

    if n_train < 50 or n_val < 20:
        # Fall back to using all data for fit, mark val as unreliable
        if np.std(xv) < 1e-6:
            return None
        res = stats.linregress(xv, yv)
        pred_all = xv * res.slope + res.intercept
        diff_all = yv - pred_all
        return {
            "slope": float(res.slope),
            "intercept": float(res.intercept),
            "train_r2": float(res.rvalue ** 2),
            "val_r2": float(res.rvalue ** 2),
            "val_rmse": float(np.sqrt(np.mean(diff_all ** 2))),
            "val_mae": float(np.mean(np.abs(diff_all))),
            "pearson_r": float(res.rvalue),
            "n_total": int(np.sum(m)),
            "n_train": int(np.sum(m)),
            "n_val": 0,
            "depth_range": float(np.ptp(xv)),
            "elev_range": float(np.ptp(yv)),
            "reliable_val": False,
        }

    x_train = depth_block[train_mask].astype(np.float64)
    y_train = ref_block[train_mask].astype(np.float64)
    x_val = depth_block[val_mask].astype(np.float64)
    y_val = ref_block[val_mask].astype(np.float64)

    # Guard: constant-valued training inputs are unfittable
    if np.std(x_train) < 1e-6 or np.std(y_train) < 1e-3:
        return None

    # Standard linear
    res_lin = stats.linregress(x_train, y_train)
    pred_val_lin = x_val * res_lin.slope + res_lin.intercept
    diff_val_lin = y_val - pred_val_lin
    rmse_lin = float(np.sqrt(np.mean(diff_val_lin ** 2)))

    # Huber regression (if sklearn available)
    try:
        from sklearn.linear_model import HuberRegressor
        hub = HuberRegressor().fit(x_train.reshape(-1, 1), y_train)
        pred_val_hub = hub.predict(x_val.reshape(-1, 1))
        diff_val_hub = y_val - pred_val_hub
        rmse_hub = float(np.sqrt(np.mean(diff_val_hub ** 2)))
        if rmse_hub < rmse_lin:
            chosen_slope = float(hub.coef_[0])
            chosen_intercept = float(hub.intercept_)
            chosen_rmse = rmse_hub
            chosen_diff = diff_val_hub
        else:
            chosen_slope = float(res_lin.slope)
            chosen_intercept = float(res_lin.intercept)
            chosen_rmse = rmse_lin
            chosen_diff = diff_val_lin
    except ImportError:
        chosen_slope = float(res_lin.slope)
        chosen_intercept = float(res_lin.intercept)
        chosen_rmse = rmse_lin
        chosen_diff = diff_val_lin

    ss_res = np.sum(chosen_diff ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    val_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Guard for constant validation inputs
    if len(x_val) > 2 and np.std(x_val) > 1e-6 and np.std(y_val) > 1e-6:
        pr, _ = stats.pearsonr(x_val, y_val)
    else:
        pr = 0.0

    return {
        "slope": chosen_slope,
        "intercept": chosen_intercept,
        "train_r2": float(res_lin.rvalue ** 2),
        "val_r2": float(val_r2),
        "val_rmse": chosen_rmse,
        "val_mae": float(np.mean(np.abs(chosen_diff))),
        "pearson_r": float(pr),
        "n_total": int(np.sum(m)),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "depth_range": float(np.ptp(xv)),
        "elev_range": float(np.ptp(yv)),
        "reliable_val": True,
    }


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------

def run_localized_calibration(
    rel_depth: np.ndarray,
    reference_elev: np.ndarray,
    transform,
    crs: str,
    block_size_m: float = 500.0,
    min_samples: int = 100,
    min_val_r2: float = -2.0,
    global_slope: float = None,
    global_intercept: float = None,
) -> Dict[str, Any]:
    """
    Run localized block-matching calibration.

    Parameters
    ----------
    rel_depth : 2-D float array [0, 1]
    reference_elev : 2-D float array in metres (aligned to rel_depth grid)
    transform : rasterio Affine
    crs : CRS string
    block_size_m : geographic block dimension in metres
    min_samples : minimum valid pixels per block
    min_val_r2 : blocks with val R² below this are rejected
    global_slope, global_intercept : fallback global calibration coefficients

    Returns
    -------
    dict with keys:
        calibrated_dsm, confidence_map, block_stats, summary
    """
    h, w = rel_depth.shape
    pixel_size_x = abs(transform[0])  # metres / pixel
    pixel_size_y = abs(transform[4])

    block_w = max(1, int(round(block_size_m / pixel_size_x)))
    block_h = max(1, int(round(block_size_m / pixel_size_y)))

    n_rows = max(1, int(np.ceil(h / block_h)))
    n_cols = max(1, int(np.ceil(w / block_w)))

    logger.info(f"[LOCAL-CAL] Block size: {block_size_m}m")
    logger.info(f"[LOCAL-CAL] Pixel size: {pixel_size_x:.2f}m x {pixel_size_y:.2f}m")
    logger.info(f"[LOCAL-CAL] Block pixels: {block_w} x {block_h}")
    logger.info(f"[LOCAL-CAL] Grid: {n_rows} rows x {n_cols} cols = {n_rows * n_cols} blocks")

    valid_mask = (
        np.isfinite(rel_depth) &
        np.isfinite(reference_elev) &
        (reference_elev > -1000)
    )

    # Compute global fallback if not supplied
    if global_slope is None or global_intercept is None:
        gx = rel_depth[valid_mask].astype(np.float64)
        gy = reference_elev[valid_mask].astype(np.float64)
        if len(gx) > 20:
            gres = stats.linregress(gx, gy)
            global_slope = float(gres.slope)
            global_intercept = float(gres.intercept)
        else:
            global_slope = 1.0
            global_intercept = 0.0

    # Fit each block
    slope_grid = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    intercept_grid = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    confidence_grid = np.full((n_rows, n_cols), 0.0, dtype=np.float64)
    block_stats_list = []

    valid_blocks = 0
    rejected_blocks = 0

    for ri, ci, y0, y1, x0, x1 in _block_indices(h, w, block_h, block_w):
        d_blk = rel_depth[y0:y1, x0:x1]
        r_blk = reference_elev[y0:y1, x0:x1]
        m_blk = valid_mask[y0:y1, x0:x1]

        result = _fit_block(d_blk, r_blk, m_blk, min_samples=min_samples)

        bs = {
            "row": ri, "col": ci,
            "y0": y0, "y1": y1, "x0": x0, "x1": x1,
        }

        if result is None or result["val_r2"] < min_val_r2:
            # Reject → use global fallback
            slope_grid[ri, ci] = np.nan
            intercept_grid[ri, ci] = np.nan
            confidence_grid[ri, ci] = 0.0
            bs["status"] = "rejected"
            bs["reason"] = "insufficient data" if result is None else f"val_r2={result['val_r2']:.4f}"
            rejected_blocks += 1
        else:
            slope_grid[ri, ci] = result["slope"]
            intercept_grid[ri, ci] = result["intercept"]
            # Confidence: combine val_r2 and sample fraction
            conf = max(0.0, min(1.0, result["val_r2"])) * min(1.0, result["n_total"] / 500.0)
            confidence_grid[ri, ci] = max(0.0, conf)
            bs["status"] = "valid"
            bs.update(result)
            valid_blocks += 1

        block_stats_list.append(bs)

    logger.info(f"[LOCAL-CAL] Valid blocks: {valid_blocks}")
    logger.info(f"[LOCAL-CAL] Rejected blocks: {rejected_blocks}")

    # Fill NaN blocks with global fallback coefficients
    slope_filled = slope_grid.copy()
    intercept_filled = intercept_grid.copy()
    for ri in range(n_rows):
        for ci in range(n_cols):
            if np.isnan(slope_filled[ri, ci]):
                slope_filled[ri, ci] = global_slope
                intercept_filled[ri, ci] = global_intercept

    # Smooth coefficient grids using bilinear interpolation via scipy.ndimage
    # We resize coefficient grids to pixel resolution using bilinear zoom
    from scipy.ndimage import zoom

    # Scale coefficient grids to full image resolution
    zoom_y = h / n_rows
    zoom_x = w / n_cols
    slope_map = zoom(slope_filled, (zoom_y, zoom_x), order=1)[:h, :w]
    intercept_map = zoom(intercept_filled, (zoom_y, zoom_x), order=1)[:h, :w]
    conf_map = zoom(confidence_grid, (zoom_y, zoom_x), order=1)[:h, :w]

    # Apply spatially varying calibration
    calibrated_dsm = (slope_map * rel_depth + intercept_map).astype(np.float32)

    # For invalid pixels, set to NaN
    calibrated_dsm[~valid_mask] = np.nan

    # Compute local-calibration coverage
    local_px = 0
    fallback_px = 0
    for ri, ci, y0, y1, x0, x1 in _block_indices(h, w, block_h, block_w):
        blk_valid = np.sum(valid_mask[y0:y1, x0:x1])
        if not np.isnan(slope_grid[ri, ci]):
            local_px += blk_valid
        else:
            fallback_px += blk_valid

    total_valid = np.sum(valid_mask)
    local_pct = (local_px / total_valid * 100) if total_valid > 0 else 0
    fallback_pct = (fallback_px / total_valid * 100) if total_valid > 0 else 0

    logger.info(f"[LOCAL-CAL] Local calibration coverage: {local_pct:.1f}%")
    logger.info(f"[LOCAL-CAL] Global fallback: {fallback_pct:.1f}%")

    # Compute global validation metrics
    pred_v = calibrated_dsm[valid_mask]
    ref_v = reference_elev[valid_mask]
    diff = pred_v - ref_v
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    pr, _ = stats.pearsonr(pred_v, ref_v)
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((ref_v - np.mean(ref_v)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    logger.info(f"[LOCAL-CAL] {block_size_m}m RMSE: {rmse:.2f}")
    logger.info(f"[LOCAL-CAL] {block_size_m}m MAE: {mae:.2f}")
    logger.info(f"[LOCAL-CAL] {block_size_m}m Pearson r: {pr:.4f}")
    logger.info(f"[LOCAL-CAL] {block_size_m}m R²: {r2:.4f}")

    summary = {
        "block_size_m": block_size_m,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "total_blocks": n_rows * n_cols,
        "valid_blocks": valid_blocks,
        "rejected_blocks": rejected_blocks,
        "local_coverage_pct": round(local_pct, 1),
        "fallback_pct": round(fallback_pct, 1),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "pearson_r": round(float(pr), 4),
        "r2": round(float(r2), 4),
    }

    return {
        "calibrated_dsm": calibrated_dsm,
        "confidence_map": conf_map.astype(np.float32),
        "block_stats": block_stats_list,
        "summary": summary,
        "global_slope": global_slope,
        "global_intercept": global_intercept,
    }


def spatial_cross_validate(
    rel_depth: np.ndarray,
    reference_elev: np.ndarray,
    transform,
    crs: str,
    block_size_m: float = 500.0,
    cv_grid: Tuple[int, int] = (4, 4),
) -> Dict[str, Any]:
    """
    Spatial cross-validation: hold out geographic folds, fit on the rest,
    predict on the held-out fold.
    """
    h, w = rel_depth.shape
    valid_mask = (
        np.isfinite(rel_depth) &
        np.isfinite(reference_elev) &
        (reference_elev > -1000)
    )

    n_fold_rows, n_fold_cols = cv_grid
    fold_h = h // n_fold_rows
    fold_w = w // n_fold_cols

    fold_results = []

    for fi in range(n_fold_rows):
        for fj in range(n_fold_cols):
            # Define the held-out fold region
            fy0 = fi * fold_h
            fy1 = min(fy0 + fold_h, h)
            fx0 = fj * fold_w
            fx1 = min(fx0 + fold_w, w)

            # Create training mask (everything except the fold)
            train_mask = valid_mask.copy()
            train_mask[fy0:fy1, fx0:fx1] = False

            val_mask = np.zeros_like(valid_mask)
            val_mask[fy0:fy1, fx0:fx1] = valid_mask[fy0:fy1, fx0:fx1]

            n_train = np.sum(train_mask)
            n_val = np.sum(val_mask)

            if n_train < 100 or n_val < 50:
                continue

            # Fit on training data
            x_train = rel_depth[train_mask].astype(np.float64)
            y_train = reference_elev[train_mask].astype(np.float64)

            # Global regression on training data
            res = stats.linregress(x_train, y_train)
            train_slope = float(res.slope)
            train_intercept = float(res.intercept)

            # Now run localized calibration on training data only
            # For simplicity, we compute per-fold global prediction
            # (the localized calibration would need masked fitting which is complex)
            x_val = rel_depth[val_mask].astype(np.float64)
            y_val = reference_elev[val_mask].astype(np.float64)

            # Global prediction on held-out fold
            pred_global = x_val * train_slope + train_intercept
            diff_global = pred_global - y_val
            rmse_g = float(np.sqrt(np.mean(diff_global ** 2)))
            pr_g, _ = stats.pearsonr(pred_global, y_val) if len(y_val) > 2 else (0.0, 1.0)
            ss_res_g = np.sum(diff_global ** 2)
            ss_tot_g = np.sum((y_val - np.mean(y_val)) ** 2)
            r2_g = 1.0 - (ss_res_g / ss_tot_g) if ss_tot_g > 0 else 0.0

            fold_results.append({
                "fold": f"{fi},{fj}",
                "n_train": int(n_train),
                "n_val": int(n_val),
                "rmse": round(rmse_g, 2),
                "mae": round(float(np.mean(np.abs(diff_global))), 2),
                "pearson_r": round(float(pr_g), 4),
                "r2": round(float(r2_g), 4),
            })

    if not fold_results:
        return {"status": "failed", "reason": "no valid folds"}

    rmses = [f["rmse"] for f in fold_results]
    r2s = [f["r2"] for f in fold_results]

    return {
        "status": "success",
        "n_folds": len(fold_results),
        "mean_rmse": round(float(np.mean(rmses)), 2),
        "std_rmse": round(float(np.std(rmses)), 2),
        "mean_r2": round(float(np.mean(r2s)), 4),
        "std_r2": round(float(np.std(r2s)), 4),
        "folds": fold_results,
    }
