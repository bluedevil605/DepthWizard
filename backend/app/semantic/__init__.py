"""
Semantic Layer for DepthWizard.
Provides aerial object detection (YOLOv8-OBB) and RGB land-cover differentiation (ExG + KMeans).
"""

from .detector import (
    detect_objects,
    get_yolo_detector,
    get_coco_detector,
    detect_coco_objects,
    detect_buildings_from_mask,
    _apply_nms,
    DEFAULT_YOLO_CONF,
    FORCED_NONNADIR_YOLO_CONF
)
from .land_cover import (
    differentiate_land_cover,
    extract_building_footprints,
    CLASS_NAMES,
    CLASS_COLORS,
    CLASS_SHADOW,
    CLASS_WATER,
    CLASS_VEGETATION,
    CLASS_BUILT_UP
)
from .cross_validation import (
    cross_validate_detections_with_land_cover,
    calculate_bbox_land_cover_overlap,
    evaluate_detection_plausibility,
    ENABLE_LAND_COVER_CROSSVAL,
    DISCARD_MISMATCHED_DETECTIONS,
    MISMATCH_PENALTY_FACTOR,
    SHIP_WATER_THRESHOLD
)
from .heatmap import (
    generate_detection_heatmap,
    CLASS_HEATMAP_COLORS,
    get_class_color_rgb,
    rgb_to_hex
)

__all__ = [
    "detect_objects",
    "get_yolo_detector",
    "get_coco_detector",
    "detect_coco_objects",
    "detect_buildings_from_mask",
    "_apply_nms",
    "DEFAULT_YOLO_CONF",
    "FORCED_NONNADIR_YOLO_CONF",
    "differentiate_land_cover",
    "extract_building_footprints",
    "CLASS_NAMES",
    "CLASS_COLORS",
    "CLASS_SHADOW",
    "CLASS_WATER",
    "CLASS_VEGETATION",
    "CLASS_BUILT_UP",
    "cross_validate_detections_with_land_cover",
    "calculate_bbox_land_cover_overlap",
    "evaluate_detection_plausibility",
    "ENABLE_LAND_COVER_CROSSVAL",
    "DISCARD_MISMATCHED_DETECTIONS",
    "MISMATCH_PENALTY_FACTOR",
    "SHIP_WATER_THRESHOLD",
    "generate_detection_heatmap",
    "CLASS_HEATMAP_COLORS",
    "get_class_color_rgb",
    "rgb_to_hex"
]
