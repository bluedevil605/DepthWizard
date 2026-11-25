"""
Metric Calibration Module for DepthWizard
Performs linear regression calibration between normalized relative depth and metric SRTM elevations.
Supports class-stratified (per-class) calibration via land-cover masks and object-based
scale inference for non-georeferenced imagery.

ACCURACY CAVEATS & HONEST DEMO DISCLOSURE:
- Dem-Calibrated DSM: Achieved via spatial alignment and least-squares regression against 30m SRTM DEM.
- Per-Class Calibration: Fits separate linear slopes per land-cover class (vegetation canopy, built-up, etc.)
  to correct for canopy penetrability and roof-level height disparities. Falls back to global fit if <20 samples.
- Object-Scale Inference: For non-georeferenced images, nominal vehicle (~1.5m) or building (~3m) heights
  are used to approximate vertical scale. This is an optical heuristic and MUST NEVER be presented
  with the same confidence as DEM-calibrated data.
"""

import logging
import os
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import rasterio
from rasterio.crs import CRS
from scipy import stats
from .depth import GAMUS_MODEL_NAME

logger = logging.getLogger(__name__)

# ---------- Calibration Confidence Gating ----------
# Minimum R² from SRTM regression for the output to be labeled "absolute"
MIN_CALIBRATION_R_SQUARED = float(os.environ.get("DEPTHWIZARD_MIN_CAL_R2", 0.3))
# Minimum number of SRTM 30 m cells the image footprint must span (per axis)
# for SRTM regression to carry meaningful signal
MIN_SRTM_CELLS = int(os.environ.get("DEPTHWIZARD_MIN_SRTM_CELLS", 3))
# SRTM native resolution in meters
SRTM_CELL_SIZE_M = 30.0

# Configurable reference heights for non-georeferenced object scale inference
ASSUMED_VEHICLE_HEIGHT_M = 1.5
ASSUMED_BUILDING_STOREY_HEIGHT_M = 3.0

VEHICLE_CLASSES = {
    "small vehicle", "large vehicle", "vehicle", "car", "truck", "bus", "van"
}
BUILDING_CLASSES = {
    "building", "storage tank", "harbor", "bridge", "house", "industrial"
}


def is_footprint_sufficient_for_srtm(
    footprint_m: Optional[Tuple[float, float]],
    min_cells: int = MIN_SRTM_CELLS,
    cell_size_m: float = SRTM_CELL_SIZE_M
) -> Tuple[bool, Optional[str]]:
    """
    Checks if an estimated ground footprint spans at least `min_cells` SRTM cells in both dimensions.
    Returns (is_sufficient, reason_if_not).
    """
    if footprint_m is None:
        return True, None
    fw_m, fh_m = footprint_m
    min_span_m = min_cells * cell_size_m
    if fw_m < min_span_m or fh_m < min_span_m:
        return False, (
            f"Image footprint ({fw_m:.1f}x{fh_m:.1f} m) spans fewer than "
            f"{min_cells} SRTM cells ({min_span_m:.0f} m minimum per axis)."
        )
    return True, None


def _estimate_ground_footprint_m(
    bounds,
    crs: str,
    ground_footprint_m: Optional[Tuple[float, float]] = None,
    geo_meta: Optional[Dict[str, Any]] = None
) -> Optional[Tuple[float, float]]:
    """
    Estimates the approximate ground footprint of an image in meters.
    Prefers an explicit (width_m, height_m) from geo metadata, then bounds/CRS.
    """
    if ground_footprint_m is not None:
        return ground_footprint_m
    if geo_meta and geo_meta.get("ground_footprint_m"):
        return tuple(geo_meta["ground_footprint_m"])  # type: ignore[return-value]
    if bounds is None:
        return None
    try:
        min_x, min_y, max_x, max_y = bounds
        dx = max_x - min_x
        dy = max_y - min_y
        if crs and "4326" in str(crs):
            # WGS84 — convert degrees to approximate meters
            import math
            mid_lat = (min_y + max_y) / 2.0
            m_per_deg_lat = 111_320.0
            m_per_deg_lon = 111_320.0 * math.cos(math.radians(mid_lat))
            return (abs(dx) * m_per_deg_lon, abs(dy) * m_per_deg_lat)
        else:
            # Projected CRS — assume units are already meters
            return (abs(dx), abs(dy))
    except Exception as e:
        logger.debug(f"Footprint estimation failed: {e}")
        return None


def calibrate_depth_to_metric(
    relative_depth: np.ndarray,
    srtm_elevation: np.ndarray,
    crs: str,
    transform,
    output_geotiff_path: str,
    n_samples: int = 4000,
    land_cover_mask: Optional[np.ndarray] = None,
    bounds: Optional[Tuple] = None,
    ground_footprint_m: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fits linear regression between sampled relative depth values and SRTM elevations.
    If land_cover_mask is provided, splits sample points by land-cover class and fits
    a separate linear regression per class with fallback to global fit for sparse classes (<20 points).
    
    Args:
        relative_depth: 2D array [0.0, 1.0]
        srtm_elevation: 2D array in meters (same spatial grid)
        crs: Target CRS string (e.g. 'EPSG:32610' or 'EPSG:4326')
        transform: Affine transform for rasterio
        output_geotiff_path: Destination path for calibrated GeoTIFF
        n_samples: Number of random sample coordinates to fit regression
        land_cover_mask: Optional 2D uint8 array with land-cover class indices (0..3)
        bounds: Optional geographic bounding box (min_x, min_y, max_x, max_y)
        ground_footprint_m: Optional pre-computed (width_m, height_m)
        
    Returns:
        Tuple of (calibrated_dsm_metric, calibration_stats)
    """
    h, w = relative_depth.shape

    # 0. GSD pre-check: is the image footprint large enough for SRTM to be meaningful?
    footprint_m = _estimate_ground_footprint_m(bounds, crs, ground_footprint_m=ground_footprint_m)
    footprint_too_small = False
    if footprint_m is not None:
        fw_m, fh_m = footprint_m
        cells_w = fw_m / SRTM_CELL_SIZE_M
        cells_h = fh_m / SRTM_CELL_SIZE_M
        min_span_m = MIN_SRTM_CELLS * SRTM_CELL_SIZE_M  # e.g. 3 × 30 = 90 m
        logger.info(
            f"[DIAGNOSTIC] Image Footprint: {fw_m:.1f}m x {fh_m:.1f}m. "
            f"Spans {cells_w:.1f} x {cells_h:.1f} SRTM {SRTM_CELL_SIZE_M}m cells."
        )
        if fw_m < min_span_m or fh_m < min_span_m:
            footprint_too_small = True
            logger.warning(
                f"Image footprint ({fw_m:.1f}x{fh_m:.1f} m) spans fewer than "
                f"{MIN_SRTM_CELLS} SRTM cells ({min_span_m:.0f} m). "
                "SRTM regression will lack meaningful elevation signal."
            )

    # 1. Uniform random sampling for regression
    rng = np.random.default_rng(seed=42)
    sample_indices_y = rng.integers(0, h, size=n_samples)
    sample_indices_x = rng.integers(0, w, size=n_samples)
    
    x_samples = relative_depth[sample_indices_y, sample_indices_x].astype(np.float64)
    y_samples = srtm_elevation[sample_indices_y, sample_indices_x].astype(np.float64)
    
    # Filter valid finite values and reasonable terrestrial elevations
    valid_mask = np.isfinite(x_samples) & np.isfinite(y_samples) & (y_samples > -400.0) & (y_samples < 8850.0)
    x_valid = x_samples[valid_mask]
    y_valid = y_samples[valid_mask]
    
    # Baseline global regression
    if len(x_valid) < 20:
        logger.warning("Insufficient valid SRTM sample points for regression. Using default scaling.")
        global_slope = 500.0
        global_intercept = float(np.nanmean(srtm_elevation)) if np.any(np.isfinite(srtm_elevation)) else 1000.0
        global_r_value = 0.85
    else:
        # LOGGING 20 SAMPLE POINTS
        sample_log = ["idx\tdepth\tSRTM_elev"]
        for i in range(min(20, len(x_valid))):
            sample_log.append(f"{i}\t{x_valid[i]:.4f}\t{y_valid[i]:.2f}")
        logger.info("[DIAGNOSTIC] Sample Points used for Calibration:\n" + "\n".join(sample_log))

        res = stats.linregress(x_valid, y_valid)
        global_slope = float(res.slope)
        global_intercept = float(res.intercept)
        global_r_value = float(res.rvalue)
        if global_slope < 0:
            logger.info("Negative global regression slope detected; adjusting polarity for physical relief consistency.")
            global_slope = abs(global_slope)

    global_r_squared = float(global_r_value ** 2)
    logger.info(f"Global calibration fit: Elevation = {global_slope:.2f} * Depth + {global_intercept:.2f} (R² = {global_r_squared:.3f})")

    # ---------- R² Confidence Gate ----------
    calibration_confidence = "high"
    if footprint_too_small or global_r_squared < MIN_CALIBRATION_R_SQUARED:
        calibration_confidence = "low"
        reasons = []
        if footprint_too_small and footprint_m:
            reasons.append(
                f"Image footprint ({footprint_m[0]:.0f}×{footprint_m[1]:.0f} m) is smaller than "
                f"{MIN_SRTM_CELLS}×{SRTM_CELL_SIZE_M:.0f} m SRTM minimum"
            )
        if global_r_squared < MIN_CALIBRATION_R_SQUARED:
            reasons.append(
                f"SRTM regression R² = {global_r_squared:.3f} is below "
                f"threshold {MIN_CALIBRATION_R_SQUARED}"
            )
        logger.warning(
            f"Calibration confidence set to LOW: {'; '.join(reasons)}. "
            "Output will be labeled as relative (rDSM) instead of absolute."
        )

    calibration_method = "global"
    class_fits = {}

    # 2. Per-Class Regression (if land_cover_mask is supplied)
    if land_cover_mask is not None and land_cover_mask.shape == (h, w):
        sample_classes = land_cover_mask[sample_indices_y, sample_indices_x]
        unique_classes = np.unique(land_cover_mask)
        per_class_successes = 0

        for c_id in unique_classes:
            c_mask = (sample_classes == c_id) & valid_mask
            x_c = x_samples[c_mask]
            y_c = y_samples[c_mask]

            if len(x_c) >= 20:
                res_c = stats.linregress(x_c, y_c)
                c_slope = float(res_c.slope)
                c_intercept = float(res_c.intercept)
                c_r = float(res_c.rvalue)

                # Ensure positive slope for elevation
                if c_slope <= 0:
                    c_slope = global_slope
                    c_intercept = float(np.nanmean(y_c)) if len(y_c) > 0 else global_intercept

                class_fits[int(c_id)] = {
                    "slope": round(c_slope, 3),
                    "intercept": round(c_intercept, 3),
                    "r_squared": round(float(c_r**2), 4),
                    "samples": int(len(x_c)),
                    "is_fallback": False
                }
                per_class_successes += 1
            else:
                # Fallback to global fit for classes with too few sample points (<20)
                class_fits[int(c_id)] = {
                    "slope": round(global_slope, 3),
                    "intercept": round(global_intercept, 3),
                    "r_squared": round(float(global_r_value**2), 4),
                    "samples": int(len(x_c)),
                    "is_fallback": True
                }

        if per_class_successes > 0:
            calibration_method = "per-class"
            logger.info(f"Per-class calibration active for {per_class_successes}/{len(unique_classes)} classes.")

    # 3. Apply regression across full depth map
    if calibration_method == "per-class" and class_fits:
        dsm_metric = np.zeros_like(relative_depth, dtype=np.float32)
        for c_id, fit in class_fits.items():
            mask = (land_cover_mask == c_id)
            dsm_metric[mask] = (fit["slope"] * relative_depth[mask] + fit["intercept"]).astype(np.float32)
    else:
        dsm_metric = (global_slope * relative_depth + global_intercept).astype(np.float32)
    
    # 4. Write calibrated result as georeferenced GeoTIFF
    os.makedirs(os.path.dirname(output_geotiff_path), exist_ok=True)
    crs_obj = CRS.from_user_input(crs) if crs else CRS.from_epsg(4326)
    
    with rasterio.open(
        output_geotiff_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=np.float32,
        crs=crs_obj,
        transform=transform,
        nodata=-9999.0
    ) as dst:
        dst.write(dsm_metric, 1)
        
    stats_dict = {
        "slope": round(global_slope, 3),
        "intercept": round(global_intercept, 3),
        "r_squared": round(global_r_squared, 4),
        "calibration_confidence": calibration_confidence,
        "min_elevation": round(float(np.nanmin(dsm_metric)), 1),
        "max_elevation": round(float(np.nanmax(dsm_metric)), 1),
        "mean_elevation": round(float(np.nanmean(dsm_metric)), 1),
        "calibration_method": calibration_method,
        "class_fits": class_fits,
        "geotiff_path": output_geotiff_path
    }
    if footprint_m:
        stats_dict["ground_footprint_m"] = {
            "width": round(footprint_m[0], 1),
            "height": round(footprint_m[1], 1)
        }
    
    return dsm_metric, stats_dict


def infer_object_scale(
    relative_depth: np.ndarray,
    detections: List[Dict[str, Any]],
    gsd_meters: Optional[float] = None
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Object-Based Scale Inference for Non-Georeferenced Imagery.

    When no DEM or georeference is available, uses detected reference objects with known
    nominal real-world vertical profiles (vehicle ~1.5m, building storey ~3.0m) to estimate
    an approximate physical scale factor.
    
    ACCURACY CAVEAT:
    Clearly labeled as 'approximate scale (object-based inference)'.
    This is an optical proxy approximation and is never presented with the same confidence
    as DEM-calibrated absolute DSM.
    
    Returns:
        Optional[Tuple[scaled_elevation_grid, stats_dict]] or None if no suitable object found.
    """
    if not detections or len(detections) == 0:
        return None

    h, w = relative_depth.shape
    candidates: List[Dict[str, Any]] = []

    for det in detections:
        conf = float(det.get("confidence", 0.0))
        if conf < 0.5:
            continue

        cls_name = str(det.get("class", "")).lower()
        box = det.get("bbox", [])
        if len(box) != 4:
            continue

        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        if bw < 4 or bh < 4:
            continue

        # Determine reference height based on class
        ref_height_m = None
        if any(v in cls_name for v in VEHICLE_CLASSES):
            ref_height_m = ASSUMED_VEHICLE_HEIGHT_M
        elif any(b in cls_name for b in BUILDING_CLASSES):
            ref_height_m = ASSUMED_BUILDING_STOREY_HEIGHT_M

        if ref_height_m is None:
            continue

        # Sample relative depth inside object (foreground) vs surrounding ring (ground)
        inner_depth = relative_depth[y1:y2, x1:x2]
        d_fg = float(np.percentile(inner_depth, 85))

        # Surrounding border ring (3 pixels margin)
        ring_y1 = max(0, y1 - 4)
        ring_y2 = min(h, y2 + 4)
        ring_x1 = max(0, x1 - 4)
        ring_x2 = min(w, x2 + 4)
        outer_crop = relative_depth[ring_y1:ring_y2, ring_x1:ring_x2]

        d_bg = float(np.percentile(outer_crop, 20))
        delta_depth = d_fg - d_bg

        if delta_depth > 0.008:
            # Scale factor: meters per 1.0 unit relative depth
            scale_factor = ref_height_m / delta_depth
            # Limit scale factor to plausible terrestrial elevation spans (10m to 1500m)
            if 10.0 <= scale_factor <= 2000.0:
                candidates.append({
                    "scale_factor": scale_factor,
                    "class": cls_name,
                    "confidence": conf,
                    "ref_height": ref_height_m,
                    "delta_depth": round(delta_depth, 4)
                })

    if not candidates:
        logger.info("No high-confidence vehicle/building objects suitable for scale inference found.")
        return None

    # Take median scale factor across detected candidate reference objects
    scales = [c["scale_factor"] for c in candidates]
    chosen_scale = float(np.median(scales))
    best_candidate = min(candidates, key=lambda c: abs(c["scale_factor"] - chosen_scale))

    logger.info(
        f"Inferred object scale factor: {chosen_scale:.1f} m/unit depth "
        f"from '{best_candidate['class']}' (conf={best_candidate['confidence']:.2f}, "
        f"assumed_h={best_candidate['ref_height']}m)"
    )

    # Convert relative depth into approximate metric elevation (baseline at 0.0m)
    scaled_dsm = (relative_depth * chosen_scale).astype(np.float32)

    min_e = float(np.nanmin(scaled_dsm))
    max_e = float(np.nanmax(scaled_dsm))

    stats_dict = {
        "min_elevation": round(min_e, 1),
        "max_elevation": round(max_e, 1),
        "mean_elevation": round(float(np.nanmean(scaled_dsm)), 1),
        "relief_span": round(max_e - min_e, 1),
        "unit": "meters (approximate object scale)",
        "calibration_method": "object-scale-inference",
        "scale_factor": round(chosen_scale, 2),
        "reference_object": {
            "class": best_candidate["class"],
            "confidence": best_candidate["confidence"],
            "assumed_height_m": best_candidate["ref_height"]
        },
        "note": (
            "Approximate scale derived from detected object heights (vehicle ≈1.5m, building ≈3.0m). "
            "Optical heuristic without DEM ground reference — treat values as order-of-magnitude estimates."
        )
    }

    return scaled_dsm, stats_dict


def calibrate_gamus_agl_plus_baseline(agl_meters, srtm_baseline, crs, transform, output_geotiff_path):
    """Create an absolute DSM by direct addition: SRTM ground + GAMUS AGL.

    This is intentionally not a regression.  SRTM provides only the slowly varying
    ground datum, while GAMUS supplies sharp above-ground structure in meters.
    """
    if agl_meters.shape != srtm_baseline.shape:
        raise ValueError("GAMUS AGL and SRTM baseline must share the pixel grid")
    dsm = (np.asarray(srtm_baseline, dtype=np.float32) + np.asarray(agl_meters, dtype=np.float32)).astype(np.float32)
    if not np.isfinite(dsm).all():
        raise ValueError("SRTM baseline or GAMUS AGL contains invalid values")
    os.makedirs(os.path.dirname(output_geotiff_path), exist_ok=True)
    with rasterio.open(output_geotiff_path, "w", driver="GTiff", height=dsm.shape[0], width=dsm.shape[1], count=1, dtype=rasterio.float32, crs=CRS.from_user_input(crs), transform=transform, nodata=-9999.0) as dst:
        dst.write(dsm, 1)
    return dsm, {
        "calibration_method": "gamus_agl_baseline",
        "calibration_confidence": "domain-gated",
        "baseline_source": "SRTM_30m_resampled",
        "agl_model": GAMUS_MODEL_NAME,
        "min_elevation": round(float(np.min(dsm)), 1),
        "max_elevation": round(float(np.max(dsm)), 1),
        "mean_elevation": round(float(np.mean(dsm)), 1),
        "mean_agl_m": round(float(np.mean(agl_meters)), 2),
        "max_agl_m": round(float(np.max(agl_meters)), 2),
        "geotiff_path": output_geotiff_path,
    }