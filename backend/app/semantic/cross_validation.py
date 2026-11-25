"""
Land-Cover Cross-Validation Module for Aerial Object Detections.

Cross-validates detected objects against RGB land-cover masks (Vegetation, Water,
Built-up, Shadow) to suppress or penalize contextually implausible false positives
(e.g., ships detected in suburban residential neighborhoods or on rooftops).
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .land_cover import (
    CLASS_SHADOW,
    CLASS_WATER,
    CLASS_VEGETATION,
    CLASS_BUILT_UP,
    CLASS_NAMES
)

logger = logging.getLogger(__name__)

# Configurable environment toggles
ENABLE_LAND_COVER_CROSSVAL = os.environ.get("DEPTHWIZARD_ENABLE_LAND_COVER_CROSSVAL", "true").lower() in ("true", "1", "yes")
DISCARD_MISMATCHED_DETECTIONS = os.environ.get("DEPTHWIZARD_DISCARD_MISMATCHED_DETECTIONS", "true").lower() in ("true", "1", "yes")
MISMATCH_PENALTY_FACTOR = float(os.environ.get("DEPTHWIZARD_MISMATCH_PENALTY", 0.20))
SHIP_WATER_THRESHOLD = float(os.environ.get("DEPTHWIZARD_SHIP_WATER_THRESHOLD", 0.50))

# Semantic class groupings
SHIP_CLASSES = {"ship", "boat", "vessel"}
VEHICLE_CLASSES = {"small vehicle", "large vehicle", "vehicle", "car", "truck", "bus", "van"}
BUILDING_CLASSES = {"building", "storage tank", "harbor", "bridge", "house", "industrial"}


def calculate_bbox_land_cover_overlap(
    bbox: List[int],
    land_cover_mask: np.ndarray
) -> Dict[str, float]:
    """
    Computes the fraction of pixels within the bounding box [x1, y1, x2, y2]
    belonging to each land-cover class (vegetation, water, built_up, shadow).
    
    Returns:
        dict: {"vegetation": float, "water": float, "built_up": float, "shadow": float}
        with values in range [0.0, 1.0].
    """
    h, w = land_cover_mask.shape
    x1, y1, x2, y2 = bbox
    
    # Clip coordinates to image boundary
    cx1 = max(0, min(w, x1))
    cy1 = max(0, min(h, y1))
    cx2 = max(0, min(w, x2))
    cy2 = max(0, min(h, y2))
    
    if cx2 <= cx1 or cy2 <= cy1:
        return {CLASS_NAMES[cid]: 0.0 for cid in (CLASS_VEGETATION, CLASS_WATER, CLASS_BUILT_UP, CLASS_SHADOW)}
        
    crop = land_cover_mask[cy1:cy2, cx1:cx2]
    total_pixels = float(crop.size)
    if total_pixels == 0:
        return {CLASS_NAMES[cid]: 0.0 for cid in (CLASS_VEGETATION, CLASS_WATER, CLASS_BUILT_UP, CLASS_SHADOW)}
        
    return {
        CLASS_NAMES[cid]: round(float(np.sum(crop == cid)) / total_pixels, 3)
        for cid in (CLASS_VEGETATION, CLASS_WATER, CLASS_BUILT_UP, CLASS_SHADOW)
    }


def evaluate_detection_plausibility(
    cls_name: str,
    overlap: Dict[str, float],
    ship_water_threshold: float = SHIP_WATER_THRESHOLD
) -> Tuple[bool, Optional[str]]:
    """
    Evaluates whether an object detection class is physically plausible given
    its land-cover class overlap fractions.
    
    Returns:
        (is_plausible: bool, failure_reason: Optional[str])
    """
    name_clean = cls_name.lower().strip()
    water_frac = overlap.get("water", 0.0)
    veg_frac = overlap.get("vegetation", 0.0)
    built_frac = overlap.get("built_up", 0.0)

    # 1. Ship / Boat: Requires >50% overlap with water class
    if any(s in name_clean for s in SHIP_CLASSES):
        if water_frac < ship_water_threshold:
            return False, (
                f"land-cover mismatch (water overlap {water_frac*100:.1f}% "
                f"< {ship_water_threshold*100:.0f}% required)"
            )
        return True, None

    # 2. Vehicles: Should overlap primarily with built-up/bare
    # (built_up >= vegetation and built_up >= water)
    if any(v in name_clean for v in VEHICLE_CLASSES):
        if built_frac < veg_frac or built_frac < water_frac:
            if water_frac >= veg_frac:
                return False, (
                    f"land-cover mismatch (vehicle dominated by water: "
                    f"{water_frac*100:.1f}% water vs {built_frac*100:.1f}% built-up)"
                )
            return False, (
                f"land-cover mismatch (vehicle dominated by vegetation: "
                f"{veg_frac*100:.1f}% vegetation vs {built_frac*100:.1f}% built-up)"
            )
        return True, None

    # 3. Buildings / Tanks / Harbor structures: Should overlap primarily with built-up
    if any(b in name_clean for b in BUILDING_CLASSES):
        if name_clean == "harbor":
            return True, None
        if built_frac < veg_frac or built_frac < water_frac:
            return False, (
                f"land-cover mismatch (structure not primarily built-up: "
                f"{built_frac*100:.1f}% built-up, {veg_frac*100:.1f}% vegetation, "
                f"{water_frac*100:.1f}% water)"
            )
        return True, None

    # 4. Any class without defined rules passes through unchanged
    return True, None


def cross_validate_detections_with_land_cover(
    detections: List[Dict[str, Any]],
    land_cover_mask: Optional[np.ndarray],
    enabled: bool = ENABLE_LAND_COVER_CROSSVAL,
    discard_mismatched: bool = DISCARD_MISMATCHED_DETECTIONS,
    penalty_factor: float = MISMATCH_PENALTY_FACTOR,
    ship_water_thresh: float = SHIP_WATER_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Validates detected aerial objects against land-cover segmentation.
    Attaches both raw and adjusted confidences for frontend transparency.
    
    Args:
        detections: List of detection dicts (class, confidence, bbox, ...)
        land_cover_mask: 2D uint8 numpy array of shape (H, W) with classes 0..3
        enabled: Toggle to enable/disable cross-validation entirely
        discard_mismatched: If True, drops mismatched detections; if False, downgrades confidence
        penalty_factor: Factor to multiply confidence when mismatched (default 0.20)
        ship_water_thresh: Required water fraction for ship detections (default 0.50)
        
    Returns:
        List of validated and confidence-adjusted detection dictionaries.
    """
    if not detections:
        return []

    # If cross-validation disabled or no mask available, preserve raw confidences and return
    if not enabled or land_cover_mask is None:
        logger.info("Land-cover cross-validation is disabled or mask unavailable. Passing detections through.")
        for det in detections:
            det["raw_confidence"] = det.get("raw_confidence", det.get("confidence", 0.0))
            det["cross_val_status"] = "disabled"
        return detections

    validated_detections: List[Dict[str, Any]] = []

    for det in detections:
        raw_conf = det.get("raw_confidence", det.get("confidence", 0.0))
        det["raw_confidence"] = round(float(raw_conf), 3)

        # Building/house detections derived from the land-cover mask itself are
        # already semantically grounded — cross-validating them against the same
        # mask via their axis-aligned bbox would be circular and incorrect
        # (bboxes often straddle vegetation surrounding a rooftop). Skip them.
        if det.get("source") == "land_cover":
            det["cross_val_status"] = "passed"
            det["confidence"] = round(float(raw_conf), 3)
            validated_detections.append(det)
            continue

        box = det.get("bbox", [])
        if len(box) != 4:
            validated_detections.append(det)
            continue

        overlap = calculate_bbox_land_cover_overlap(box, land_cover_mask)
        det["land_cover_overlap"] = overlap

        cls_name = det.get("class", "")
        is_plausible, reason = evaluate_detection_plausibility(
            cls_name=cls_name,
            overlap=overlap,
            ship_water_threshold=ship_water_thresh
        )

        if is_plausible:
            det["cross_val_status"] = "passed"
            det["confidence"] = round(float(raw_conf), 3)
            validated_detections.append(det)
        else:
            # Plausibility rule violated
            logger.info(
                f"[CROSS-VAL MISMATCH] {cls_name} (conf={raw_conf:.2f}): {reason}. "
                f"Action: {'DROP' if discard_mismatched else f'PENALIZE ({penalty_factor}x)'}"
            )
            if discard_mismatched:
                # Discard detection entirely
                continue
            else:
                # Downgrade confidence sharply and record reason
                adjusted_conf = round(max(0.01, float(raw_conf) * penalty_factor), 3)
                det["confidence"] = adjusted_conf
                det["cross_val_status"] = "mismatch"
                det["adjustment_reason"] = reason
                validated_detections.append(det)

    return validated_detections
