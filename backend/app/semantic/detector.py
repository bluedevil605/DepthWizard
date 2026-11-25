"""
YOLO Object Detection Module for Aerial & Remote-Sensing Imagery.

Uses Ultralytics YOLOv8-OBB (Oriented Bounding Box) pretrained on DOTA (DOTAv1 aerial dataset),
falling back gracefully to standard yolov8n.pt (COCO) or offline mock detection if unavailable.

Prioritized aerial classes:
- vehicle ('small vehicle', 'large vehicle')
- building / industrial structures ('storage tank', 'harbor', 'bridge')
- ship
- plane
"""

import os
import logging
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Global detector instance cache
_YOLO_MODEL = None
_IS_OBB = False


def _count_detections_by_class(detections: List[Dict[str, Any]]) -> Dict[str, int]:
    """Helper function to count detections by class name."""
    counts = {}
    for det in detections:
        cls = det.get("class", "unknown")
        counts[cls] = counts.get(cls, 0) + 1
    return dict(sorted(counts.items()))

# Global COCO (YOLOv8n) detector cache used as a supplementary vehicle/person detector
# for ground-mixed / residential drone imagery where the DOTA aerial classes lack
# building-level detail (cars, trucks, buses, people).
_COCO_MODEL = None

# Default configurable confidence thresholds
# 0.25 default: lowered from 0.35 to capture genuine aerial objects (vehicles, structures)
# that often score in the 0.25-0.35 confidence range on complex nadir backgrounds.
# Implausible false positives are filtered downstream by land-cover cross-validation
# (e.g. ship requiring water) and size-sanity filters.
DEFAULT_YOLO_CONF = float(os.environ.get("DEPTHWIZARD_YOLO_CONF", 0.25))
FORCED_NONNADIR_YOLO_CONF = 0.50

# Model selection configuration
# Set DETECTION_MODEL_VERSION env var (or change default constant below):
#   "v2" -> depthwizard_v2_obb.pt (Fine-tuned on DOTAv1 for 30 epochs: mAP50 0.620, mAP50-95 0.448)
#   "v1" / "original" / "baseline" -> yolov8n-obb.pt (Original baseline DOTA checkpoint)
# You can also set a direct filename or path via DETECTION_MODEL_VERSION or DETECTION_MODEL_CHECKPOINT.
DETECTION_MODEL_VERSION = os.environ.get("DETECTION_MODEL_VERSION", "v2").strip()

MODEL_WEIGHT_FILENAMES = {
    "v2": "depthwizard_v2_obb.pt",
    "v1": "yolov8n-obb.pt",
    "original": "yolov8n-obb.pt",
    "baseline": "yolov8n-obb.pt",
}


def _find_weights_path(model_version: Optional[str] = None) -> str:
    """
    Finds best available weights file locally based on configuration or returns default model name.
    Supports version aliases ("v2", "v1", "original") or direct checkpoint filenames/paths.
    """
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Explicit checkpoint path takes highest precedence if provided
    env_checkpoint = os.environ.get("DETECTION_MODEL_CHECKPOINT", "").strip()
    if env_checkpoint:
        if os.path.isabs(env_checkpoint) and os.path.exists(env_checkpoint):
            return env_checkpoint
        candidate = os.path.join(backend_dir, env_checkpoint)
        if os.path.exists(candidate):
            return candidate

    version = (model_version or os.environ.get("DETECTION_MODEL_VERSION", DETECTION_MODEL_VERSION)).strip()
    filename = MODEL_WEIGHT_FILENAMES.get(version.lower(), version)

    # 1. Check if filename is an absolute path that exists
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename

    # 2. Check project-level backend directory
    candidate = os.path.join(backend_dir, filename)
    if os.path.exists(candidate):
        return candidate

    # 3. Check current working directory
    if os.path.exists(filename):
        return filename

    # Graceful fallback: check for yolov8n-obb.pt in backend directory
    fallback = os.path.join(backend_dir, "yolov8n-obb.pt")
    if os.path.exists(fallback):
        logger.warning(
            f"Requested weights '{filename}' (version='{version}') not found. "
            f"Falling back to baseline '{fallback}'."
        )
        return fallback

    return filename


def get_yolo_detector():
    """
    Lazy loader for YOLO detector.
    Prefers depthwizard_v2_obb.pt (fine-tuned DOTA aerial model) or yolov8n-obb.pt based on
    DETECTION_MODEL_VERSION, with fallback to yolov8n.pt (COCO).
    Prints model name, checkpoint path, task, and supported classes at startup.
    Validates that model.task reports 'obb'.
    """
    global _YOLO_MODEL, _IS_OBB
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL, _IS_OBB

    try:
        from ultralytics import YOLO

        obb_weights = _find_weights_path()
        logger.info(f"Initializing YOLO detector from weights path: {obb_weights}")
        try:
            _YOLO_MODEL = YOLO(obb_weights)
            _IS_OBB = True

            # Sanity check: confirm model.task is 'obb'
            task = getattr(_YOLO_MODEL, "task", None)
            if task != "obb":
                logger.error(
                    f"CRITICAL SANITY CHECK FAILED: Model task '{task}' is not 'obb'! "
                    f"Oriented bounding box heads are required for aerial detection."
                )
            else:
                logger.info(f"SANITY CHECK PASSED: model.task correctly reports 'obb'.")

            classes_dict = _YOLO_MODEL.names or {}
            classes_summary = ", ".join([f"{k}:{v}" for k, v in sorted(classes_dict.items())])
            logger.info(
                f"SUCCESS: YOLOv8-OBB aerial model loaded from '{obb_weights}'. "
                f"Task='{_YOLO_MODEL.task}', Checkpoint='{obb_weights}'. "
                f"Supported aerial classes ({len(classes_dict)}): [{classes_summary}]"
            )
            return _YOLO_MODEL, _IS_OBB
        except Exception as obb_err:
            logger.warning(
                f"Failed loading aerial OBB model '{obb_weights}': {obb_err}. "
                "Falling back to standard yolov8n.pt."
            )
            _YOLO_MODEL = YOLO("yolov8n.pt")
            _IS_OBB = False
            classes_dict = _YOLO_MODEL.names or {}
            logger.warning(
                "CRITICAL WARNING: Loaded standard yolov8n.pt (COCO-pretrained weights). "
                "COCO is trained on ground-level perspective photos, not nadir aerial/satellite views. "
                "Detection quality on remote sensing imagery is a known limitation of this placeholder model."
            )
            logger.info(f"COCO classes ({len(classes_dict)}): {list(classes_dict.values())[:10]}...")
            return _YOLO_MODEL, _IS_OBB

    except Exception as err:
        logger.error(f"Unable to initialize YOLO model: {err}. Object detection will be disabled.")
        _YOLO_MODEL = None
        _IS_OBB = False
        return None, False


def _generate_tiles(img_w: int, img_h: int, tile_size: int = 640, overlap: float = 0.35):
    """
    Splits image dimensions into overlapping window bounding boxes (x1, y1, x2, y2).
    Overlap default is ~35% (stride = int(tile_size * 0.65) = 416px).
    """
    if img_w <= tile_size and img_h <= tile_size:
        return [(0, 0, img_w, img_h)]

    stride = max(int(tile_size * (1.0 - overlap)), 64)

    # Compute horizontal start positions
    x_starts = []
    x = 0
    while x + tile_size < img_w:
        x_starts.append(x)
        x += stride
    x_starts.append(max(0, img_w - tile_size))
    x_starts = sorted(list(set(x_starts)))

    # Compute vertical start positions
    y_starts = []
    y = 0
    while y + tile_size < img_h:
        y_starts.append(y)
        y += stride
    y_starts.append(max(0, img_h - tile_size))
    y_starts = sorted(list(set(y_starts)))

    tiles = []
    for y1 in y_starts:
        for x1 in x_starts:
            x2 = min(img_w, x1 + tile_size)
            y2 = min(img_h, y1 + tile_size)
            tiles.append((x1, y1, x2, y2))
    return tiles


def _calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Computes Intersection over Union (IoU) for two axis-aligned bounding boxes [x1, y1, x2, y2]."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    box1_area = max(0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    box2_area = max(0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / float(union_area)


def _apply_nms(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.45,
    max_detections: int = 100,
    class_specific_thresholds: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """
    Applies class-aware Non-Maximum Suppression (NMS) across tile boundaries
    to remove duplicate detections originating from overlapping regions.
    
    DIAGNOSTIC TRACE: This function logs when called to confirm it's in the execution path.
    
    Args:
        detections: List of detection dictionaries with "bbox", "class", "confidence"
        iou_threshold: Default IoU threshold for suppression (0.45 = suppress if overlap > 45%)
        max_detections: Maximum number of detections to keep after NMS
        class_specific_thresholds: Optional dict mapping class names to custom IoU thresholds
            (e.g., {"building": 0.20} for more aggressive building merging)
    
    Returns:
        Filtered list of detections after NMS
    """
    logger.warning(
        f"🔍 _apply_nms() CALLED with {len(detections)} input detections, "
        f"IoU={iou_threshold}, class_thresholds={class_specific_thresholds}"
    )
    
    if not detections:
        logger.warning("🔍 _apply_nms() returning empty list (no input detections)")
        return []

    # Sort descending by detection confidence
    sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
    kept_dets: List[Dict[str, Any]] = []

    for det in sorted_dets:
        box = det["bbox"]
        cls_name = det["class"]
        
        # Use class-specific threshold if provided, otherwise use default
        threshold = iou_threshold
        if class_specific_thresholds and cls_name in class_specific_thresholds:
            threshold = class_specific_thresholds[cls_name]
        
        suppress = False
        for kept in kept_dets:
            if kept["class"] == cls_name:
                iou = _calculate_iou(box, kept["bbox"])
                if iou > threshold:
                    suppress = True
                    break
        if not suppress:
            kept_dets.append(det)
            if len(kept_dets) >= max_detections:
                break

    logger.warning(
        f"🔍 _apply_nms() RETURNING {len(kept_dets)} detections "
        f"(from {len(detections)} inputs)"
    )
    return kept_dets


def _apply_wbf(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.30,
    max_detections: int = 100,
    skip_box_threshold: float = 0.0001
) -> List[Dict[str, Any]]:
    """
    Applies Weighted Box Fusion (WBF) for cross-tile detection merging.
    
    DIAGNOSTIC TRACE: This function logs when called to confirm it's in the execution path.
    
    WBF averages overlapping bounding boxes from multiple tiles rather than just keeping
    the highest-confidence detection (NMS behavior). This produces more stable results for
    tiled aerial imagery where the same object may be detected at slightly different positions
    across tile boundaries.
    
    Algorithm:
    1. Cluster detections of the same class with IoU > threshold
    2. For each cluster, compute weighted average of bbox coordinates (weight = confidence)
    3. Average confidence across cluster members
    4. Return fused boxes sorted by confidence
    
    Args:
        detections: List of detection dictionaries with "bbox", "class", "confidence"
        iou_threshold: IoU threshold for clustering (default 0.30 = merge if overlap > 30%)
        max_detections: Maximum number of fused boxes to return
        skip_box_threshold: Minimum confidence to consider (filters near-zero confidence)
    
    Returns:
        List of fused detection dictionaries
    """
    logger.warning(f"🔍 _apply_wbf() CALLED with {len(detections)} input detections, IoU={iou_threshold}")
    
    if not detections:
        logger.warning("🔍 _apply_wbf() returning empty list (no input detections)")
        return []
    
    # Filter out extremely low-confidence detections
    detections = [d for d in detections if d.get("confidence", 0.0) >= skip_box_threshold]
    if not detections:
        logger.warning("🔍 _apply_wbf() returning empty list (all filtered by skip_box_threshold)")
        return []
    
    # Group detections by class
    class_groups: Dict[str, List[Dict[str, Any]]] = {}
    for det in detections:
        cls_name = det["class"]
        if cls_name not in class_groups:
            class_groups[cls_name] = []
        class_groups[cls_name].append(det)
    
    # Process each class independently (continued in _wbf_fuse_class_detections)
    fused_results: List[Dict[str, Any]] = []
    for cls_name, cls_dets in class_groups.items():
        fused_results.extend(_wbf_fuse_class_detections(cls_name, cls_dets, iou_threshold))
    
    # Sort by confidence descending and cap at max_detections
    fused_results.sort(key=lambda d: d["confidence"], reverse=True)
    final_result = fused_results[:max_detections]
    
    logger.warning(
        f"🔍 _apply_wbf() RETURNING {len(final_result)} fused detections "
        f"(from {len(detections)} inputs)"
    )
    return final_result


# -----------------------------------------------------------------------
# Class-level size / shape sanity filter
# -----------------------------------------------------------------------

# Expected bounding-box properties for DOTAv1 aerial classes.
# area_range: (min_fraction, max_fraction) of total image area
# aspect_range: (min, max) of width/height ratio
_CLASS_SIZE_EXPECTATIONS: Dict[str, Dict[str, Any]] = {
    "swimming pool":   {"area_range": (0.001,  0.08), "aspect_range": (0.3, 3.0)},
    "small vehicle":   {"area_range": (0.0001, 0.02), "aspect_range": (0.25, 4.0)},
    "large vehicle":   {"area_range": (0.0002, 0.04), "aspect_range": (0.2, 5.0)},
    "storage tank":    {"area_range": (0.0005, 0.05), "aspect_range": (0.5, 2.0)},
    "plane":           {"area_range": (0.001,  0.10), "aspect_range": (0.3, 3.5)},
    "ship":            {"area_range": (0.0005, 0.10), "aspect_range": (0.15, 6.0)},
    "helicopter":      {"area_range": (0.0003, 0.04), "aspect_range": (0.3, 3.5)},
    # COCO ground-level classes detected on residential/drone imagery.
    # Small area ranges so an implausibly huge 'car' (actually a rooftop) gets flagged.
    "person":          {"area_range": (0.00001, 0.006), "aspect_range": (0.15, 1.8)},
    "bicycle":         {"area_range": (0.00002, 0.008), "aspect_range": (0.3, 3.0)},
    "car":             {"area_range": (0.00003, 0.015), "aspect_range": (0.4, 2.6)},
    "motorcycle":      {"area_range": (0.00001, 0.006), "aspect_range": (0.4, 2.5)},
    "bus":             {"area_range": (0.00008, 0.04), "aspect_range": (0.2, 3.5)},
    "truck":           {"area_range": (0.0001,  0.03), "aspect_range": (0.3, 3.0)},
    "boat":            {"area_range": (0.00005, 0.04), "aspect_range": (0.15, 4.0)},
}

_SIZE_CONFIDENCE_PENALTY = 0.20  # Subtracted from confidence for implausible detections


def _wbf_fuse_class_detections(
    cls_name: str,
    cls_dets: List[Dict[str, Any]],
    iou_threshold: float
) -> List[Dict[str, Any]]:
    """Helper function for WBF: fuses detections within a single class."""
    fused_results: List[Dict[str, Any]] = []
    merged_flags = [False] * len(cls_dets)
    
    for i, det_i in enumerate(cls_dets):
        if merged_flags[i]:
            continue
        
        # Start a new cluster with det_i
        cluster = [det_i]
        merged_flags[i] = True
        
        # Find all detections that overlap with any member of this cluster
        for j, det_j in enumerate(cls_dets):
            if merged_flags[j] or j == i:
                continue
            
            # Check if det_j overlaps with the primary detection
            iou = _calculate_iou(det_i["bbox"], det_j["bbox"])
            if iou > iou_threshold:
                cluster.append(det_j)
                merged_flags[j] = True
        
        # Fuse the cluster using weighted averaging
        if len(cluster) == 1:
            fused_results.append(cluster[0])
        else:
            # Weighted box fusion
            total_conf = sum(d["confidence"] for d in cluster)
            weights = [d["confidence"] / total_conf for d in cluster]
            
            fused_x1 = sum(w * d["bbox"][0] for w, d in zip(weights, cluster))
            fused_y1 = sum(w * d["bbox"][1] for w, d in zip(weights, cluster))
            fused_x2 = sum(w * d["bbox"][2] for w, d in zip(weights, cluster))
            fused_y2 = sum(w * d["bbox"][3] for w, d in zip(weights, cluster))
            avg_conf = total_conf / len(cluster)
            
            fused_det = {
                "class": cls_name,
                "confidence": round(avg_conf, 3),
                "bbox": [int(round(fused_x1)), int(round(fused_y1)), 
                        int(round(fused_x2)), int(round(fused_y2))],
                "fusion_count": len(cluster)
            }
            
            # Average OBB coords if present
            if "obb_coords" in cluster[0]:
                fused_obb = []
                for pt_idx in range(4):
                    avg_x = sum(w * d["obb_coords"][pt_idx][0] for w, d in zip(weights, cluster))
                    avg_y = sum(w * d["obb_coords"][pt_idx][1] for w, d in zip(weights, cluster))
                    fused_obb.append([int(round(avg_x)), int(round(avg_y))])
                fused_det["obb_coords"] = fused_obb
            
            fused_results.append(fused_det)
    
    return fused_results


def _log_duplicate_sanity_check(
    raw_dets: List[Dict[str, Any]],
    merged_dets: List[Dict[str, Any]],
    final_dets: List[Dict[str, Any]],
    img_w: int,
    img_h: int
) -> None:
    """
    Logs warnings if post-merge detection density suggests duplicates are still present.
    
    Heuristic checks:
    1. If merge ratio < 2.0 for scenes with many raw detections, cross-tile merging may be ineffective
    2. If many detections of the same class have near-identical bounding boxes, flag as likely duplicates
    3. If detection density exceeds typical aerial scene patterns, warn user
    """
    if not raw_dets or not merged_dets:
        return
    
    merge_ratio = len(raw_dets) / len(merged_dets)
    img_area_mpx = (img_w * img_h) / 1_000_000
    
    # Check 1: Low merge ratio with high raw count suggests ineffective NMS/WBF
    if len(raw_dets) >= 30 and merge_ratio < 2.0:
        logger.warning(
            f"⚠️  DUPLICATE DETECTION WARNING: Low merge ratio ({merge_ratio:.1f}x) with {len(raw_dets)} "
            f"raw detections suggests cross-tile merging may be ineffective. Consider lowering cross_tile_iou "
            f"threshold (current: see logs above) or switching merge strategy (NMS ↔ WBF)."
        )
    
    # Check 2: Detect near-identical bounding boxes within same class (strong duplicate signal)
    class_groups: Dict[str, List[Dict[str, Any]]] = {}
    for det in merged_dets:
        cls = det["class"]
        if cls not in class_groups:
            class_groups[cls] = []
        class_groups[cls].append(det)
    
    for cls_name, cls_dets in class_groups.items():
        if len(cls_dets) < 3:
            continue
        
        # Count pairs with very high IoU (>0.80 = nearly identical)
        near_duplicate_pairs = 0
        for i, det_i in enumerate(cls_dets):
            for det_j in cls_dets[i+1:]:
                iou = _calculate_iou(det_i["bbox"], det_j["bbox"])
                if iou > 0.80:
                    near_duplicate_pairs += 1
        
        if near_duplicate_pairs >= 2:
            logger.warning(
                f"⚠️  DUPLICATE DETECTION WARNING: Found {near_duplicate_pairs} near-identical '{cls_name}' "
                f"bounding box pairs (IoU > 0.80) after merging. This strongly suggests tile-boundary duplicates "
                f"are not being merged. Review cross_tile_iou threshold and merge_strategy."
            )
    
    # Check 3: Detection density sanity check (heuristic thresholds for typical aerial scenes)
    # Typical drone/satellite imagery: 5-30 significant objects per megapixel
    # (residential: ~10-20 buildings/mpx, industrial: ~5-15 structures/mpx, rural: ~2-8 objects/mpx)
    detection_density = len(final_dets) / max(0.1, img_area_mpx)
    
    if detection_density > 50 and len(final_dets) > 20:
        logger.warning(
            f"⚠️  HIGH DETECTION DENSITY: {len(final_dets)} final detections in {img_area_mpx:.1f} megapixels "
            f"({detection_density:.0f} objects/mpx) exceeds typical aerial scene density (~5-30/mpx). "
            f"This may indicate unmerged duplicates or genuine high-density scene (urban residential)."
        )


def _apply_size_sanity_filter(
    detections: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    tile_size: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Flag detections whose bounding-box size or aspect ratio falls outside the
    expected range for that class in typical nadir aerial imagery.

    Flagged detections have their confidence reduced by _SIZE_CONFIDENCE_PENALTY
    and receive a ``size_sanity: "flagged"`` marker.

    Args:
        tile_size: When supplied (tiled inference), object-size expectations are
            computed against the tile scene scale instead of the full image. On a
            4000px image tiled into 640px windows, a 40px car is 6.7e-5 of the full
            image (below the 'small vehicle' area floor) but a healthy 0.4% of the
            tile — so full-image scaling would wrongly destroy real small-object
            detections produced by tiled inference.
    """
    # With tiled inference, detection boxes live at tile scale, so the expected
    # area fractions must be computed against the tile scene scale. Otherwise every
    # small vehicle/car in a large image would be flagged as "too small".
    if tile_size and tile_size > 0:
        img_area = max(1, tile_size * tile_size)
    else:
        img_area = max(1, img_w * img_h)
    filtered: List[Dict[str, Any]] = []

    for det in detections:
        cls_name = det["class"]
        x1, y1, x2, y2 = det["bbox"]
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        box_area = box_w * box_h
        area_frac = box_area / img_area
        aspect = box_w / box_h

        expectations = _CLASS_SIZE_EXPECTATIONS.get(cls_name)
        if not expectations:
            expectations = _CLASS_SIZE_EXPECTATIONS.get(cls_name.lower().replace("-", " "))
        if not expectations:
            expectations = _CLASS_SIZE_EXPECTATIONS.get(cls_name.lower().replace(" ", "-"))
        flagged = False
        if expectations:
            a_lo, a_hi = expectations["area_range"]
            ar_lo, ar_hi = expectations["aspect_range"]
            if area_frac < a_lo or area_frac > a_hi:
                flagged = True
            if aspect < ar_lo or aspect > ar_hi:
                flagged = True

        if flagged:
            det = dict(det)  # shallow copy to avoid mutating original
            det["confidence"] = round(max(0.0, det["confidence"] - _SIZE_CONFIDENCE_PENALTY), 3)
            det["size_sanity"] = "flagged"
            logger.debug(
                f"Size-sanity flagged: {cls_name} area_frac={area_frac:.4f} "
                f"aspect={aspect:.2f} — confidence reduced to {det['confidence']}"
            )
        else:
            det["size_sanity"] = "ok"

        filtered.append(det)

    return filtered


def detect_objects(
    image: Image.Image,
    conf_threshold: Optional[float] = None,
    max_detections: int = 100,
    tile_size: int = 640,
    tile_overlap: float = 0.35,
    nadir_confidence: Optional[float] = None,
    merge_strategy: str = "wbf",
    cross_tile_iou: float = 0.25,
    disable_tiling: bool = False,  # DIAGNOSTIC: Run single-pass inference (no tiling)
    diagnostic_mode: bool = False,  # DIAGNOSTIC: Enable extra verbose logging
    gsd_m: Optional[float] = None  # Ground sample distance in meters/pixel (e.g. 0.0247)
) -> List[Dict[str, Any]]:
    """
    Runs YOLO aerial object detection on an optical RGB image using tiled inference
    and cross-tile merging (NMS or WBF).
    
    Args:
        image: PIL Image in RGB format
        conf_threshold: Minimum detection confidence (default 0.25)
        max_detections: Maximum detections to return after merging
        tile_size: Size of square tile windows (default 640)
        tile_overlap: Fractional overlap between adjacent tiles (default 0.35 = ~35%)
        nadir_confidence: Optional nadir angle confidence score to attach to results
        merge_strategy: Merging algorithm - "nms" (greedy suppression) or "wbf" (weighted box fusion)
        cross_tile_iou: IoU threshold for cross-tile merging (default 0.25, lower = more aggressive merging)
        disable_tiling: DIAGNOSTIC FLAG - If True, runs single-pass inference on full image (no tiling)
        diagnostic_mode: DIAGNOSTIC FLAG - If True, enables extra verbose logging for debugging
        gsd_m: Ground sample distance in meters/pixel. When < 0.10 (fine drone imagery),
               adaptive tiling automatically switches to single-pass inference to avoid
               DOTA scale mismatch (e.g. car misclassified as ship in tile crops).
        
    Returns:
        List of dicts: [
            {
                "class": str,
                "confidence": float,
                "bbox": [x1, y1, x2, y2],
                "obb_coords": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] (if available),
                "fusion_count": int (only for WBF - number of boxes merged)
            }
        ]
    """
    if conf_threshold is None:
        conf_threshold = DEFAULT_YOLO_CONF

    if image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    model, is_obb = get_yolo_detector()

    if model is None:
        return []

    # ADAPTIVE TILING: For fine-GSD drone imagery (GSD < 10 cm/px), objects like
    # vehicles are physically large in pixels (e.g. 200+ px). When cropped into a
    # 640x640 tile, they fill 30%+ of the tile and lose surrounding context (roads,
    # dirt), causing the DOTA satellite-trained model to misclassify them (e.g.
    # car → ship). Single-pass inference resizes the full image so objects shrink
    # to ~30-40px — matching DOTA's training distribution — and retains context.
    _ADAPTIVE_GSD_THRESHOLD_M = 0.10  # 10 cm/px
    if gsd_m is not None and gsd_m < _ADAPTIVE_GSD_THRESHOLD_M and not disable_tiling:
        logger.info(
            f"[ADAPTIVE TILING] GSD={gsd_m*100:.2f} cm/px < {_ADAPTIVE_GSD_THRESHOLD_M*100:.0f} cm/px threshold. "
            f"Switching to single-pass inference to avoid DOTA scale mismatch "
            f"(fine drone imagery: objects are too large in tiles for satellite-trained model)."
        )
        disable_tiling = True

    # DIAGNOSTIC MODE: Single-pass inference (no tiling)
    if disable_tiling:
        logger.warning(
            "=" * 80 + "\n"
            "🔍 DIAGNOSTIC MODE: disable_tiling=True - Running single-pass inference\n"
            f"   Image: {w}x{h}px, Model: {'OBB aerial' if is_obb else 'AABB COCO'}, "
            f"Conf: {conf_threshold}\n"
            "   NOTE: Ultralytics YOLO applies internal NMS with default iou=0.45\n"
            "         No agnostic_nms parameter is explicitly set\n" +
            "=" * 80
        )
        
        # Pre-scale image to max 1024px for sub-second CPU inference
        # Ultralytics tensor conversion on raw 4K images takes ~17s on CPU; pre-scaling drops this to <0.7s
        max_dim = 1024
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            iw = max(16, int(round(w * scale)))
            ih = max(16, int(round(h * scale)))
            infer_img = image.resize((iw, ih), Image.Resampling.BILINEAR)
            sx = w / float(iw)
            sy = h / float(ih)
        else:
            infer_img = image
            sx = 1.0
            sy = 1.0

        # Run YOLO on scaled image in one pass
        # NOTE: Ultralytics applies NMS internally - we cannot disable it or inspect pre-NMS detections
        results = model(infer_img, conf=conf_threshold, verbose=False)
        
        raw_count = 0
        raw_class_breakdown = {}
        if results and len(results) > 0:
            res = results[0]
            if is_obb and hasattr(res, "obb") and res.obb is not None:
                raw_count = len(res.obb)
                if raw_count > 0:
                    classes = res.obb.cls.cpu().numpy().astype(int)
                    names = model.names or {}
                    for cls_id in classes:
                        cls_name = names.get(int(cls_id), f"class_{cls_id}")
                        raw_class_breakdown[cls_name] = raw_class_breakdown.get(cls_name, 0) + 1
            elif hasattr(res, "boxes") and res.boxes is not None:
                raw_count = len(res.boxes)
                if raw_count > 0:
                    classes = res.boxes.cls.cpu().numpy().astype(int)
                    names = model.names or {}
                    for cls_id in classes:
                        cls_name = names.get(int(cls_id), f"class_{cls_id}")
                        raw_class_breakdown[cls_name] = raw_class_breakdown.get(cls_name, 0) + 1
        
        logger.warning(
            f"🔍 SINGLE-PASS RAW OUTPUT: {raw_count} detection(s) from YOLO\n"
            f"   (This is AFTER Ultralytics' built-in NMS with iou=0.45)\n"
            f"   (This is BEFORE any custom cross-tile merge/NMS logic)\n"
            f"   Raw class breakdown: {dict(sorted(raw_class_breakdown.items()))}"
        )
        
        # Parse detections and scale coordinates back to full image
        single_pass_detections = []
        names = model.names or {}
        
        if results and len(results) > 0:
            res = results[0]
            
            if is_obb and hasattr(res, "obb") and res.obb is not None and len(res.obb) > 0:
                obb = res.obb
                classes = obb.cls.cpu().numpy().astype(int)
                confs = obb.conf.cpu().numpy().astype(float)
                xyxy = obb.xyxy.cpu().numpy().astype(float)
                xyxyxyxy = obb.xyxyxyxy.cpu().numpy().astype(float)
                
                for i in range(len(classes)):
                    cls_id = int(classes[i])
                    cls_name = names.get(cls_id, f"class_{cls_id}")
                    conf_val = round(float(confs[i]), 3)
                    
                    x1, y1, x2, y2 = xyxy[i]
                    gx1 = int(max(0, min(w - 1, round(x1 * sx))))
                    gy1 = int(max(0, min(h - 1, round(y1 * sy))))
                    gx2 = int(max(0, min(w - 1, round(x2 * sx))))
                    gy2 = int(max(0, min(h - 1, round(y2 * sy))))
                    corners = [
                        [int(max(0, min(w - 1, round(pt[0] * sx)))),
                         int(max(0, min(h - 1, round(pt[1] * sy))))]
                        for pt in xyxyxyxy[i]
                    ]
                    
                    single_pass_detections.append({
                        "class": cls_name,
                        "confidence": conf_val,
                        "bbox": [gx1, gy1, gx2, gy2],
                        "obb_coords": corners
                    })
            
            elif hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
                boxes = res.boxes
                classes = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy().astype(float)
                xyxy = boxes.xyxy.cpu().numpy().astype(float)
                
                for i in range(len(classes)):
                    cls_id = int(classes[i])
                    cls_name = names.get(cls_id, f"class_{cls_id}")
                    conf_val = round(float(confs[i]), 3)
                    
                    x1, y1, x2, y2 = xyxy[i]
                    gx1 = int(max(0, min(w - 1, round(x1 * sx))))
                    gy1 = int(max(0, min(h - 1, round(y1 * sy))))
                    gx2 = int(max(0, min(w - 1, round(x2 * sx))))
                    gy2 = int(max(0, min(h - 1, round(y2 * sy))))
                    single_pass_detections.append({
                        "class": cls_name,
                        "confidence": conf_val,
                        "bbox": [gx1, gy1, gx2, gy2],
                        "obb_coords": [[gx1, gy1], [gx2, gy1], 
                                      [gx2, gy2], [gx1, gy2]]
                    })
        
        # Count detections by class
        class_counts = {}
        for det in single_pass_detections:
            cls = det["class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        logger.warning(
            f"🔍 SINGLE-PASS FINAL: {len(single_pass_detections)} detection(s)\n"
            f"   Class breakdown: {dict(sorted(class_counts.items()))}\n" +
            "=" * 80 + "\n"
            "🔍 DIAGNOSTIC RESULT: If this count is ~2-3 buildings but tiled mode shows ~15,\n"
            "   the problem is in TILING/MERGE. If this ALSO shows ~15 buildings, the problem\n"
            "   is that Ultralytics' built-in NMS is insufficient or disabled.\n" +
            "=" * 80
        )
        
        # Attach nadir_confidence if provided
        if nadir_confidence is not None:
            for d in single_pass_detections:
                d["nadir_confidence"] = round(float(nadir_confidence), 4)
        
        return single_pass_detections

    # NORMAL MODE: Tiled inference
    # 1. Generate overlapping tiles for fine-grained aerial detection
    tiles = _generate_tiles(w, h, tile_size=tile_size, overlap=tile_overlap)
    
    if diagnostic_mode:
        logger.warning(
            "=" * 80 + "\n"
            f"🔍 TILED MODE: Starting YOLO ({'OBB aerial' if is_obb else 'AABB COCO'})\n"
            f"   Image: {w}x{h}px\n"
            f"   Tiles: {len(tiles)} tiles with {tile_overlap*100:.0f}% overlap\n"
            f"   Conf threshold: {conf_threshold}\n"
            f"   Merge strategy: {merge_strategy.upper()}, IoU: {cross_tile_iou}\n" +
            "=" * 80
        )
    else:
        logger.info(
            f"🔍 TILED MODE: Starting YOLO ({'OBB aerial' if is_obb else 'AABB COCO'}) "
            f"on {w}x{h} image across {len(tiles)} tile(s) (conf_thresh={conf_threshold})"
        )

    raw_candidates: List[Dict[str, Any]] = []
    names = model.names or {}
    tile_detection_counts = []  # DIAGNOSTIC: Track per-tile detection counts

    try:
        for tile_idx, (tx1, ty1, tx2, ty2) in enumerate(tiles):
            tile_w = tx2 - tx1
            tile_h = ty2 - ty1
            tile_crop = image.crop((tx1, ty1, tx2, ty2))

            # Run inference on tile
            results = model(tile_crop, conf=conf_threshold, verbose=False)
            if not results:
                tile_detection_counts.append(0)  # DIAGNOSTIC
                continue

            res = results[0]
            tile_det_count = 0  # DIAGNOSTIC: Count detections from this tile

            if is_obb and hasattr(res, "obb") and res.obb is not None and len(res.obb) > 0:
                # Handle Oriented Bounding Boxes (DOTAv1)
                obb = res.obb
                classes = obb.cls.cpu().numpy().astype(int)
                confs = obb.conf.cpu().numpy().astype(float)
                xyxy = obb.xyxy.cpu().numpy().astype(float)
                xyxyxyxy = obb.xyxyxyxy.cpu().numpy().astype(float)

                for i in range(len(classes)):
                    cls_id = int(classes[i])
                    cls_name = names.get(cls_id, f"class_{cls_id}")
                    conf_val = round(float(confs[i]), 3)

                    # Translate tile coordinates to full-image coordinates
                    lx1, ly1, lx2, ly2 = xyxy[i]
                    gx1 = int(max(0, min(w - 1, round(lx1 + tx1))))
                    gy1 = int(max(0, min(h - 1, round(ly1 + ty1))))
                    gx2 = int(max(0, min(w - 1, round(lx2 + tx1))))
                    gy2 = int(max(0, min(h - 1, round(ly2 + ty1))))

                    corners = [
                        [
                            int(max(0, min(w - 1, round(pt[0] + tx1)))),
                            int(max(0, min(h - 1, round(pt[1] + ty1))))
                        ]
                        for pt in xyxyxyxy[i]
                    ]

                    raw_candidates.append({
                        "class": cls_name,
                        "confidence": conf_val,
                        "bbox": [gx1, gy1, gx2, gy2],
                        "obb_coords": corners
                    })
                    tile_det_count += 1  # DIAGNOSTIC

            elif hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
                # Handle standard Axis-Aligned Bounding Boxes (COCO fallback)
                boxes = res.boxes
                classes = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy().astype(float)
                xyxy = boxes.xyxy.cpu().numpy().astype(float)

                for i in range(len(classes)):
                    cls_id = int(classes[i])
                    cls_name = names.get(cls_id, f"class_{cls_id}")
                    conf_val = round(float(confs[i]), 3)

                    lx1, ly1, lx2, ly2 = xyxy[i]
                    gx1 = int(max(0, min(w - 1, round(lx1 + tx1))))
                    gy1 = int(max(0, min(h - 1, round(ly1 + ty1))))
                    gx2 = int(max(0, min(w - 1, round(lx2 + tx1))))
                    gy2 = int(max(0, min(h - 1, round(ly2 + ty1))))

                    raw_candidates.append({
                        "class": cls_name,
                        "confidence": conf_val,
                        "bbox": [gx1, gy1, gx2, gy2],
                        "obb_coords": [
                            [gx1, gy1],
                            [gx2, gy1],
                            [gx2, gy2],
                            [gx1, gy2]
                        ]
                    })
                    tile_det_count += 1  # DIAGNOSTIC
            
            tile_detection_counts.append(tile_det_count)  # DIAGNOSTIC

        # Count raw detections by class
        raw_class_counts = _count_detections_by_class(raw_candidates)
        
        # Log per-tile detection stats
        if diagnostic_mode:
            logger.warning(
                f"🔍 TILED RAW DETECTIONS: Total {len(raw_candidates)} detections from {len(tiles)} tile(s)\n"
                f"   Per-tile counts: {tile_detection_counts}\n"
                f"   Raw class breakdown: {raw_class_counts}"
            )
        else:
            logger.warning(
                f"🔍 TILED RAW DETECTIONS: {len(raw_candidates)} total from {len(tiles)} tiles\n"
                f"   Raw class breakdown: {raw_class_counts}"
            )

        # 2. Apply cross-tile merging to eliminate duplicate detections from overlapping tile boundaries
        # CRITICAL: This operates on full-image coordinates (translation already applied in step 1)
        logger.warning(f"🔍 CALLING MERGE FUNCTION: {merge_strategy.upper()} with IoU={cross_tile_iou}")
        
        if merge_strategy.lower() == "wbf":
            merged_detections = _apply_wbf(
                raw_candidates, 
                iou_threshold=cross_tile_iou, 
                max_detections=max_detections
            )
            merge_method = "WBF"
        else:
            # Use NMS with lower threshold for cross-tile merging than default within-tile NMS
            # Buildings get even more aggressive merging to handle tile-boundary splits
            merged_detections = _apply_nms(
                raw_candidates,
                iou_threshold=cross_tile_iou,
                max_detections=max_detections,
                class_specific_thresholds={"building": max(0.15, cross_tile_iou * 0.6)}
            )
            merge_method = "NMS"
        
        logger.warning(
            f"🔍 MERGE FUNCTION RETURNED: {len(merged_detections)} detections "
            f"({len(raw_candidates)} → {len(merged_detections)})"
        )

        # 3. Apply class-level size/shape sanity filter
        # Tile-scale reference so real small vehicles/cars survive (see _apply_size_sanity_filter)
        final_detections = _apply_size_sanity_filter(merged_detections, w, h, tile_size=tile_size)

        # Remove detections whose confidence dropped to 0 after sanity penalty
        final_detections = [d for d in final_detections if d["confidence"] > 0.0]

        # Attach nadir_confidence if provided
        if nadir_confidence is not None:
            for d in final_detections:
                d["nadir_confidence"] = round(float(nadir_confidence), 4)

        # 4. Compute merge ratio and log sanity warnings
        merge_ratio = len(raw_candidates) / max(1, len(merged_detections))
        reduction_pct = int(100 * (1.0 - len(merged_detections) / max(1, len(raw_candidates))))
        
        # Count merged and final detections by class
        merged_class_counts = _count_detections_by_class(merged_detections)
        final_class_counts = _count_detections_by_class(final_detections)
        
        logger.warning(
            "=" * 80 + "\n"
            f"🔍 TILED INFERENCE COMPLETE ({merge_method}, IoU={cross_tile_iou}):\n"
            f"   {len(raw_candidates)} raw → {len(merged_detections)} merged ({reduction_pct}% reduction, "
            f"ratio={merge_ratio:.1f}x) → {len(final_detections)} final\n"
            f"   Merged class breakdown: {merged_class_counts}\n"
            f"   Final class breakdown: {final_class_counts}\n" +
            "=" * 80
        )
        
        # Sanity check: warn if merge ratio is suspiciously low (many duplicates still present)
        _log_duplicate_sanity_check(raw_candidates, merged_detections, final_detections, w, h)

        return final_detections

    except Exception as err:
        logger.warning(f"Error during YOLO object detection inference: {err}. Continuing with empty detections.")
        return []


def get_coco_detector():
    """
    Lazy loader for the supplementary COCO (YOLOv8n) detector used to find vehicles,
    people, and other ground-level classes on mixed/residential drone imagery.

    Returns (model, classes_dict) or (None, {}) if unavailable.
    """
    global _COCO_MODEL
    if _COCO_MODEL is not None:
        return _COCO_MODEL, (_COCO_MODEL.names or {})

    try:
        from ultralytics import YOLO
        weights = _find_coco_weights_path()
        logger.info(f"Initializing supplementary COCO detector from: {weights}")
        _COCO_MODEL = YOLO(weights)
        names = _COCO_MODEL.names or {}
        logger.info(
            f"SUCCESS: COCO detector loaded (task='{_COCO_MODEL.task}', "
            f"classes={len(names)}): {list(names.values())[:8]}..."
        )
        return _COCO_MODEL, names
    except Exception as err:
        logger.warning(f"Supplementary COCO detector unavailable: {err}. Cars/persons will not be detected.")
        _COCO_MODEL = None
        return None, {}


def _find_coco_weights_path() -> str:
    """Finds the locally cached yolov8n.pt (COCO) weights, or returns the plain name."""
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for candidate in (
        os.path.join(backend_dir, "yolov8n.pt"),
        os.path.join(os.getcwd(), "yolov8n.pt"),
        "yolov8n.pt",
    ):
        if os.path.exists(candidate):
            return candidate
    return "yolov8n.pt"


# COCO class ids we care about for aerial vehicle / person detection
_COCO_INTEREST_CLASSES = {
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
    8,   # boat
}


def detect_coco_objects(
    image: Image.Image,
    conf_threshold: Optional[float] = None,
    max_detections: int = 80,
    tile_size: int = 640,
    tile_overlap: float = 0.35,
    nadir_confidence: Optional[float] = None,
    gsd_m: Optional[float] = None  # Ground sample distance in meters/pixel
) -> List[Dict[str, Any]]:
    """
    Runs the supplementary COCO (YOLOv8n) detector on the image to find vehicles,
    people, and other ground-level objects frequently missed by the DOTA aerial model.

    Uses tiled inference exactly like the OBB aerial detector: the full image is split
    into overlapping 640px tiles so small cars/trucks/persons stay at a detectable scale.
    A single full-image pass would downscale a 4000x3000 photo to 640x640, destroying
    sub-40px objects (the exact reason the previous version returned 0 vehicles).

    Returns detections in the same dict format as `detect_objects`:
        {"class", "confidence", "bbox", "obb_coords", "source": "coco"}
    """
    if conf_threshold is None:
        conf_threshold = DEFAULT_YOLO_CONF

    model, names = get_coco_detector()
    if model is None or not names:
        return []

    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size

    # For fine-GSD drone imagery (<10 cm/px) or large images (>=2048px), single-pass on 1024px
    # matches COCO object scales (~40-80px per vehicle) and runs in ~0.5s instead of 35s.
    use_single_pass = (gsd_m is not None and gsd_m < 0.10) or max(w, h) >= 2048
    if use_single_pass:
        max_dim = 1024
        scale = max_dim / float(max(w, h))
        iw = max(16, int(round(w * scale)))
        ih = max(16, int(round(h * scale)))
        infer_img = image.resize((iw, ih), Image.Resampling.BILINEAR)
        sx = w / float(iw)
        sy = h / float(ih)

        results = model(infer_img, conf=conf_threshold, verbose=False)
        single_dets: List[Dict[str, Any]] = []
        if results and len(results) > 0 and hasattr(results[0], "boxes") and results[0].boxes is not None:
            boxes = results[0].boxes
            classes = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy().astype(float)
            xyxy = boxes.xyxy.cpu().numpy().astype(float)
            for i in range(len(classes)):
                cls_id = int(classes[i])
                if cls_id not in _COCO_INTEREST_CLASSES:
                    continue
                x1, y1, x2, y2 = xyxy[i]
                gx1 = int(max(0, min(w - 1, round(x1 * sx))))
                gy1 = int(max(0, min(h - 1, round(y1 * sy))))
                gx2 = int(max(0, min(w - 1, round(x2 * sx))))
                gy2 = int(max(0, min(h - 1, round(y2 * sy))))
                if (gx2 - gx1) < 6 or (gy2 - gy1) < 6:
                    continue
                single_dets.append({
                    "class": names.get(cls_id, f"class_{cls_id}"),
                    "confidence": round(float(confs[i]), 3),
                    "bbox": [gx1, gy1, gx2, gy2],
                    "obb_coords": [[gx1, gy1], [gx2, gy1], [gx2, gy2], [gx1, gy2]],
                    "source": "coco",
                })
        if nadir_confidence is not None:
            for d in single_dets:
                d["nadir_confidence"] = round(float(nadir_confidence), 4)
        return single_dets

    # Standard tiled mode with capped tile count
    effective_tile_size = max(tile_size, int(round(max(w, h) / 3.0)))
    tiles = _generate_tiles(w, h, tile_size=effective_tile_size, overlap=0.20)
    raw_candidates: List[Dict[str, Any]] = []

    try:
        for (tx1, ty1, tx2, ty2) in tiles:
            tile_img = image.crop((tx1, ty1, tx2, ty2))
            results = model(tile_img, conf=conf_threshold, verbose=False)
            if not results:
                continue

            res = results[0]
            if not hasattr(res, "boxes") or res.boxes is None or len(res.boxes) == 0:
                continue

            boxes = res.boxes
            classes = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy().astype(float)
            xyxy = boxes.xyxy.cpu().numpy().astype(float)

            for i in range(len(classes)):
                cls_id = int(classes[i])
                if cls_id not in _COCO_INTEREST_CLASSES:
                    continue
                conf_val = float(confs[i])

                # Translate tile-local coords to full-image coords
                lx1, ly1, lx2, ly2 = xyxy[i]
                gx1 = int(max(0, min(w - 1, round(lx1 + tx1))))
                gy1 = int(max(0, min(h - 1, round(ly1 + ty1))))
                gx2 = int(max(0, min(w - 1, round(lx2 + tx1))))
                gy2 = int(max(0, min(h - 1, round(ly2 + ty1))))

                # Only keep reasonably sized blobs (ignore sub-6px speckle)
                if (gx2 - gx1) < 6 or (gy2 - gy1) < 6:
                    continue

                raw_candidates.append({
                    "class": names.get(cls_id, f"class_{cls_id}"),
                    "confidence": round(conf_val, 3),
                    "bbox": [gx1, gy1, gx2, gy2],
                    "obb_coords": [[gx1, gy1], [gx2, gy1], [gx2, gy2], [gx1, gy2]],
                    "source": "coco",
                })
    except Exception as err:
        logger.warning(f"Tiled COCO detection inference error: {err}. Continuing with empty detections.")
        return []

    # Cross-tile NMS to eliminate duplicate detections in overlapping regions
    nms_detections = _apply_nms(raw_candidates, iou_threshold=0.45, max_detections=max_detections)

    # Class-level size sanity (flags implausibly-huge 'car' boxes that are really rooftops)
    # Tile-scale reference so real vehicles/cars survive (see _apply_size_sanity_filter)
    final_detections = _apply_size_sanity_filter(nms_detections, w, h, tile_size=tile_size)

    # Drop detections whose confidence was driven to 0 by the sanity penalty
    final_detections = [d for d in final_detections if d["confidence"] > 0.0]

    if nadir_confidence is not None:
        for d in final_detections:
            d["nadir_confidence"] = round(float(nadir_confidence), 4)

    logger.info(
        f"Tiled COCO supplementary detection: {len(raw_candidates)} candidate box(es) → "
        f"{len(nms_detections)} after NMS → {len(final_detections)} after size-sanity filter."
    )
    return final_detections


def detect_buildings_from_mask(
    land_cover_mask: np.ndarray,
    min_area_px: int = 40,
    max_regions: int = 400,
    nadir_confidence: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Detects buildings/houses directly from the built-up land-cover mask via
    connected-component footprint extraction. This catches residential rooftops
    that no pretrained aerial YOLO class labels (DOTA has no 'house' class).

    Returns detection dicts with source="land_cover".
    """
    if land_cover_mask is None:
        return []

    # Local import to avoid a circular import at module load time
    from .land_cover import extract_building_footprints

    footprints = extract_building_footprints(
        land_cover_mask=land_cover_mask,
        min_area_px=min_area_px,
        max_regions=max_regions,
    )
    if nadir_confidence is not None:
        for d in footprints:
            d["nadir_confidence"] = round(float(nadir_confidence), 4)
    return footprints
