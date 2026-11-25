"""
Nadir / Domain-Suitability Check for DepthWizard

Lightweight classical-CV heuristic that determines whether an uploaded image
is likely a top-down (nadir) aerial/satellite view vs. a ground-level or oblique
perspective photo. Runs in <100 ms on any image size using only
OpenCV + NumPy — no heavy model loading.

The score gates the main pipeline: non-nadir images are rejected with a clear
warning (unless the caller sets force=True).
"""

import logging
import time
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable weights — must sum to 1.0
# ---------------------------------------------------------------------------
_W_SKY = 0.45          # Sky / horizon detection (strongest single indicator)
_W_PERSPECTIVE = 0.25  # Vanishing-point / converging-line cue
_W_GEO = 0.30          # Georeference metadata weight

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
_NADIR_CONFIDENCE_THRESHOLD = 0.50  # Below this -> is_likely_nadir = False
_ANALYSIS_MAX_DIM = 512             # Downscale for speed; 512 is plenty for heuristics


def _resize_for_analysis(img_bgr: np.ndarray) -> np.ndarray:
    """Downscale image to _ANALYSIS_MAX_DIM on the longer edge for fast CV ops."""
    h, w = img_bgr.shape[:2]
    if max(h, w) <= _ANALYSIS_MAX_DIM:
        return img_bgr
    scale = _ANALYSIS_MAX_DIM / float(max(h, w))
    new_w = max(16, int(round(w * scale)))
    new_h = max(16, int(round(h * scale)))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ===== SIGNAL 1: Sky / Horizon Detection =====

def _longest_true_run(arr: np.ndarray) -> int:
    """Returns the length of the longest contiguous True run in a boolean array."""
    if len(arr) == 0:
        return 0
    diffs = np.diff(arr.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0] + 1
    if arr[0]:
        starts = np.concatenate([[0], starts])
    if arr[-1]:
        ends = np.concatenate([ends, [len(arr)]])
    if len(starts) == 0:
        return 0
    runs = ends[:len(starts)] - starts[:len(ends)]
    return int(np.max(runs)) if len(runs) > 0 else 0


def _score_sky_presence(img_bgr: np.ndarray) -> tuple:
    """
    Detect a large contiguous sky-like region in the upper portion of the frame.

    Returns (nadir_score, sky_fraction, reasons) where:
        nadir_score: 1.0 = no sky (nadir-like), 0.0 = heavy sky (ground-level).
        sky_fraction: fraction of the upper region that is sky-like.
    """
    h, w = img_bgr.shape[:2]
    reasons: List[str] = []

    # Analyze upper 30% of the frame
    upper_cutoff = max(1, int(h * 0.30))
    upper_region = img_bgr[:upper_cutoff, :]

    # Convert to HSV
    hsv = cv2.cvtColor(upper_region, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]       # 0-179 in OpenCV
    sat = hsv[:, :, 1]       # 0-255
    val = hsv[:, :, 2]       # 0-255

    # Sky-like pixel criteria:
    # 1. Blue sky: hue 90-135, moderate saturation, high brightness
    blue_sky_mask = (
        (hue >= 90) & (hue <= 135) &
        (sat >= 25) & (sat <= 220) &
        (val >= 110)
    )
    # 2. Overcast/white-sky: very low saturation + high value
    white_sky_mask = (
        (sat < 35) &
        (val >= 175)
    )
    # 3. Sunset / warm sky: orange/pink hues with high value
    warm_sky_mask = (
        ((hue <= 20) | (hue >= 165)) &
        (sat >= 30) &
        (val >= 150)
    )
    # Texture / uniformity check: sky typically has very low standard deviation
    gray_upper = cv2.cvtColor(upper_region, cv2.COLOR_BGR2GRAY)
    upper_std = float(np.std(gray_upper))
    upper_mean = float(np.mean(gray_upper))
    is_uniform_bright = (upper_std < 22.0) and (upper_mean > 130.0)

    # If upper region has high texture variance (rocks, mountains, quarry terraces),
    # bright pixels are textured terrain rather than overcast sky.
    if upper_std > 28.0:
        sky_mask = blue_sky_mask
    else:
        sky_mask = blue_sky_mask | white_sky_mask | warm_sky_mask

    total_pixels = upper_region.shape[0] * upper_region.shape[1]
    sky_pixels = int(np.count_nonzero(sky_mask))
    sky_fraction = sky_pixels / max(1, total_pixels)

    # Horizon edge detection: strong horizontal edge in upper 50%
    horizon_score = 0.0
    upper_half = img_bgr[:int(h * 0.50), :]
    gray_half = cv2.cvtColor(upper_half, cv2.COLOR_BGR2GRAY)
    sobel_y = cv2.Sobel(gray_half, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel = np.abs(sobel_y)
    row_edge_strength = np.mean(abs_sobel, axis=1)

    # Horizon edge detection: strong horizontal edge in upper 50%
    # An edge is only a "horizon" if sky-like pixels exist above it.
    # Otherwise, horizontal edges are simply roads, fields, or rooftops in nadir views.
    if sky_fraction > 0.12 and len(row_edge_strength) > 0:
        strong_edge_threshold = np.percentile(abs_sobel, 95) * 0.5
        strong_rows = np.where(row_edge_strength > strong_edge_threshold)[0]
        if len(strong_rows) > 0:
            for row_idx in strong_rows[:5]:
                edge_row = abs_sobel[row_idx]
                above_thresh = edge_row > strong_edge_threshold
                longest_run = _longest_true_run(above_thresh)
                if longest_run > w * 0.35:
                    horizon_score = min(1.0, longest_run / (w * 0.5))
                    reasons.append(
                        f"Detected distinct horizon line across row {row_idx} "
                        f"({longest_run}/{w} px = {longest_run / w:.0%})"
                    )
                    break

    # Score calculation based on sky fraction
    if sky_fraction > 0.40 or (is_uniform_bright and sky_fraction > 0.25):
        nadir_score = 0.0
        reasons.append(f"Large sky region detected in upper frame ({sky_fraction:.0%} sky pixels)")
    elif sky_fraction > 0.25:
        nadir_score = 0.2
        reasons.append(f"Moderate sky region detected ({sky_fraction:.0%} sky pixels)")
    elif sky_fraction > 0.10:
        nadir_score = 0.5
        reasons.append(f"Minor sky-like region ({sky_fraction:.0%})")
    else:
        nadir_score = 1.0
        reasons.append(f"No significant sky detected in upper frame ({sky_fraction:.0%})")

    # Horizon presence further penalizes nadir score if sky is present
    if horizon_score > 0.3 and sky_fraction > 0.12:
        nadir_score = min(nadir_score, 0.15)

    return nadir_score, sky_fraction, reasons


# ===== SIGNAL 2: Vanishing-Point / Perspective Cue =====

def _score_perspective_cues(img_bgr: np.ndarray) -> tuple:
    """
    Check for converging perspective lines and oblique edge angles.

    Nadir aerial imagery tends to have axis-aligned / orthogonal edges (roads,
    building footprints, agricultural fields). Ground-level or oblique photos
    show strong converging diagonals toward a central vanishing point.

    Returns (nadir_score, oblique_ratio, reasons).
    """
    reasons: List[str] = []
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=max(25, min(h, w) // 10),
        maxLineGap=10
    )

    if lines is None or len(lines) < 5:
        reasons.append("Too few distinct line segments for perspective analysis")
        return 0.8, 0.0, reasons  # Ambiguous -> neutral nadir lean

    axis_aligned = 0
    oblique = 0
    tolerance_deg = 15.0

    # Check for converging diagonals:
    # In ground perspective, lines on left half tilt up-right (dx > 0, dy < 0),
    # lines on right half tilt up-left (dx > 0, dy > 0).
    converging_count = 0
    center_x = w / 2.0

    for line in lines:
        pts = np.asarray(line).ravel()
        if len(pts) < 4:
            continue
        x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
        # Ensure x1 <= x2 for consistent slope orientation
        if x1 > x2:
            x1, y1, x2, y2 = x2, y2, x1, y1

        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue

        angle = abs(np.degrees(np.arctan2(abs(dy), abs(dx))))

        # Axis-aligned: ~0 deg (horizontal) or ~90 deg (vertical)
        if angle < tolerance_deg:
            axis_aligned += 1
        elif abs(angle - 90) < tolerance_deg:
            axis_aligned += 1
        else:
            oblique += 1
            # Check for convergence toward upper-center
            mid_x = (x1 + x2) / 2.0
            mid_y = (y1 + y2) / 2.0
            if mid_y > h * 0.25:  # lower three-quarters of image
                # Left side lines sloping toward center top (y decreases as x increases -> dy < 0)
                if mid_x < center_x and dy < -5:
                    converging_count += 1
                # Right side lines sloping toward center top (y increases as x increases -> dy > 0)
                elif mid_x > center_x and dy > 5:
                    converging_count += 1

    counted = axis_aligned + oblique
    oblique_ratio = oblique / max(1, counted)

    if converging_count >= 4 and oblique_ratio > 0.40:
        nadir_score = 0.10
        reasons.append(
            f"Strong perspective convergence detected ({converging_count} converging line segments, "
            f"{oblique_ratio:.0%} oblique): characteristic of ground-level perspective"
        )
    elif oblique_ratio > 0.55:
        nadir_score = 0.20
        reasons.append(
            f"High oblique line ratio ({oblique_ratio:.0%}): suggests non-nadir perspective"
        )
    elif oblique_ratio > 0.35:
        nadir_score = 0.50
        reasons.append(f"Moderate oblique line ratio ({oblique_ratio:.0%})")
    else:
        nadir_score = 1.0
        reasons.append(
            f"Grid-like / orthogonal edges dominate ({1 - oblique_ratio:.0%} axis-aligned)"
        )

    return nadir_score, oblique_ratio, reasons


# ===== SIGNAL 3: EXIF / Drone Pitch Inspection =====

def _score_exif_metadata(image: Image.Image) -> tuple:
    """
    Inspect EXIF metadata for camera pitch or drone gimbal orientation.
    Returns (score_or_none, reasons).
    """
    reasons: List[str] = []
    try:
        exif = image.getexif()
        if not exif:
            return None, reasons

        # Check for DJI gimbal pitch or camera orientation in MakerNote / tags
        for tag_id, value in exif.items():
            str_val = str(value)
            # Look for GimbalPitchDegree or FlightPitchDegree in drone EXIF
            if "GimbalPitchDegree" in str_val or "gimbal_pitch" in str_val.lower():
                try:
                    # Pitch is usually -90 for nadir, 0 for horizontal
                    import re
                    match = re.search(r"[-+]?\d*\.\d+|\d+", str_val)
                    if match:
                        pitch = float(match.group(0))
                        if pitch <= -75:
                            reasons.append(f"EXIF drone gimbal pitch indicates nadir camera angle ({pitch:.1f} deg)")
                            return 1.0, reasons
                        elif pitch >= -35:
                            reasons.append(f"EXIF drone gimbal pitch indicates oblique/ground angle ({pitch:.1f} deg)")
                            return 0.1, reasons
                except Exception:
                    pass
    except Exception:
        pass

    return None, reasons


# ===== Main Entry Point =====

def assess_nadir_likelihood(
    image: Image.Image,
    geo_meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Assess whether an image is likely a nadir (top-down) aerial/satellite view.

    Combines classical-CV heuristic signals and metadata:
        1. Georeference metadata (GeoTIFF CRS/transform): immediate short-circuit.
        2. Sky / Horizon detection (weight 0.45).
        3. Vanishing-point / perspective cues (weight 0.25).
        4. Spatial metadata / EXIF (weight 0.30).

    Args:
        image: PIL RGB image to assess.
        geo_meta: Dict from inspect_georeference(), or None.

    Returns:
        Dict with keys:
            - is_likely_nadir (bool): True if confidence >= 0.50.
            - confidence (float): Combined weighted score in [0.0, 1.0].
            - reasons (List[str]): Human-readable explanations.
            - sky_fraction (float): Fraction of upper frame that is sky-like.
            - oblique_line_ratio (float): Fraction of detected edges that are oblique.
            - component_scores (dict): Breakdown of sub-scores.
            - elapsed_ms (float): Wall-clock time for this check.
    """
    t0 = time.perf_counter()
    all_reasons: List[str] = []

    # 1. Short-circuit for verified georeferenced satellite/aerial data (GeoTIFF)
    if geo_meta and geo_meta.get("is_georeferenced"):
        crs = geo_meta.get("crs", "unknown")
        method = geo_meta.get("method", "geotiff")
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "is_likely_nadir": True,
            "confidence": 0.98,
            "reasons": [
                f"Georeferenced input detected ({method}, CRS: {crs}) — "
                "authenticated mapping-grade nadir aerial/satellite imagery."
            ],
            "sky_fraction": 0.0,
            "oblique_line_ratio": 0.0,
            "component_scores": {
                "sky": 1.0,
                "perspective": 1.0,
                "georeference": 1.0,
            },
            "elapsed_ms": elapsed_ms,
        }

    # Convert PIL -> OpenCV BGR
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Downscale for high speed (<100ms)
    img_small = _resize_for_analysis(img_bgr)

    # --- Signal 1: Sky / Horizon ---
    sky_score, sky_fraction, sky_reasons = _score_sky_presence(img_small)
    all_reasons.extend(sky_reasons)

    # --- Signal 2: Perspective cues ---
    perspective_score, oblique_ratio, persp_reasons = _score_perspective_cues(img_small)
    all_reasons.extend(persp_reasons)

    # --- Signal 3: EXIF / Drone pitch metadata ---
    exif_score, exif_reasons = _score_exif_metadata(image)
    all_reasons.extend(exif_reasons)

    if exif_score is not None:
        geo_score = exif_score
        confidence = (
            _W_SKY * sky_score +
            _W_PERSPECTIVE * perspective_score +
            _W_GEO * geo_score
        )
    else:
        geo_score = 0.0
        # Re-normalize visual weights when metadata is unavailable
        w_total = _W_SKY + _W_PERSPECTIVE
        confidence = (
            (_W_SKY / w_total) * sky_score +
            (_W_PERSPECTIVE / w_total) * perspective_score
        )
        all_reasons.append("No georeference or drone gimbal metadata present (visual scoring normalized)")

    confidence = round(float(np.clip(confidence, 0.0, 1.0)), 4)
    is_likely_nadir = confidence >= _NADIR_CONFIDENCE_THRESHOLD

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    result = {
        "is_likely_nadir": is_likely_nadir,
        "confidence": confidence,
        "reasons": all_reasons,
        "sky_fraction": round(float(sky_fraction), 4),
        "oblique_line_ratio": round(float(oblique_ratio), 4),
        "component_scores": {
            "sky": round(float(sky_score), 3),
            "perspective": round(float(perspective_score), 3),
            "georeference": round(float(geo_score), 3),
        },
        "elapsed_ms": elapsed_ms,
    }

    logger.info(
        f"[NADIR CHECK] is_likely_nadir={is_likely_nadir}, confidence={confidence:.3f}, "
        f"sky={sky_score:.2f}(w={_W_SKY}), perspective={perspective_score:.2f}(w={_W_PERSPECTIVE}), "
        f"geo={geo_score:.2f}(w={_W_GEO}) — {elapsed_ms:.1f}ms"
    )
    return result
