"""
Detection Heatmap Overlay Module for DepthWizard.

Renders an at-a-glance categorical spatial heatmap of all cross-validated surviving
detections, color-coded by class using a distinct categorical palette, with per-detection
opacity proportional to confidence score, composited onto a dark neutral background.
"""

import os
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

# Dark slate neutral background (RGB)
HEATMAP_BACKGROUND_COLOR = (18, 22, 32)

# High-contrast categorical color palette for all 15 DOTAv1 aerial classes
# plus supplementary COCO classes. Visually distinguishable from Land Cover
# (which uses water=blue, veg=green, built_up=gray, shadow=black).
CLASS_HEATMAP_COLORS: Dict[str, Tuple[int, int, int]] = {
    # 15 DOTA Aerial Classes
    "plane": (0, 240, 255),              # Electric Cyan (#00f0ff)
    "ship": (255, 127, 14),              # Vivid Amber / Orange (#ff7f0e)
    "storage tank": (148, 103, 189),     # Amethyst Purple (#9467bd)
    "baseball diamond": (227, 119, 194), # Rose Pink (#e377c2)
    "tennis court": (127, 236, 54),      # Electric Chartreuse (#7fec36)
    "basketball court": (255, 87, 34),   # Bright Vermilion (#ff5722)
    "ground track field": (214, 39, 40), # Crimson Red (#d62728)
    "harbor": (51, 102, 255),            # Cobalt / Royal Blue (#3366ff)
    "bridge": (255, 225, 25),            # Vivid Yellow (#ffe119)
    "large vehicle": (240, 50, 230),     # Magenta (#f032e6)
    "small vehicle": (66, 212, 244),     # Bright Sky Blue (#42d4f4)
    "helicopter": (70, 153, 144),        # Teal / Dark Mint (#469990)
    "roundabout": (145, 30, 180),        # Violet / Purple (#911eb4)
    "soccer ball field": (60, 180, 75),  # Kelly Green (#3cb44b)
    "swimming pool": (0, 229, 255),      # Turquoise (#00e5ff)

    # Supplementary Ground / COCO Classes
    "car": (66, 212, 244),               # Bright Sky Blue (#42d4f4)
    "truck": (240, 50, 230),             # Magenta (#f032e6)
    "bus": (250, 190, 212),              # Soft Rose (#fabed4)
    "person": (220, 190, 255),           # Lavender (#dcbeff)
    "boat": (255, 158, 74),              # Amber Orange (#ff9e4a)
    "bicycle": (170, 255, 195),          # Pale Mint Green (#aaffc3)
    "motorcycle": (255, 250, 200),       # Pale Yellow (#fffac8)
    "building": (191, 239, 69),          # Lime Olive (#bfef45)
    "house": (191, 239, 69),             # Lime Olive (#bfef45)
}

# Extended categorical fallback palette for any unrecognized custom class
_FALLBACK_PALETTE: List[Tuple[int, int, int]] = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (67, 99, 216),
    (245, 130, 49), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (188, 246, 12), (250, 190, 212), (0, 128, 128), (230, 190, 255),
    (154, 99, 36), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    (128, 128, 0), (255, 215, 180), (0, 0, 117), (169, 169, 169)
]


def get_class_color_rgb(class_name: str) -> Tuple[int, int, int]:
    """Returns RGB tuple for a class name with deterministic fallback."""
    key = class_name.lower().strip()
    if key in CLASS_HEATMAP_COLORS:
        return CLASS_HEATMAP_COLORS[key]
    # Deterministic hash fallback
    idx = abs(hash(key)) % len(_FALLBACK_PALETTE)
    return _FALLBACK_PALETTE[idx]


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Converts (R, G, B) tuple to '#rrggbb' hex string."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def generate_detection_heatmap(
    image_size: Tuple[int, int],
    detections: List[Dict[str, Any]],
    output_png_path: str
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Renders an oriented categorical heatmap overlay on a dark neutral background.

    Args:
        image_size: (width, height) in pixels
        detections: Surviving cross-validated detections from the detection pipeline
        output_png_path: Filesystem destination path for the generated PNG

    Returns:
        Tuple of:
            - output_png_path (str)
            - legend (List[Dict]): Compact list of only the classes present in this image:
              [{"class": str, "color": str (hex), "count": int}, ...]
    """
    w, h = image_size
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image dimensions: {w}x{h}")

    # 1. Initialize canvas with dark neutral slate background (RGB)
    canvas = np.full((h, w, 3), HEATMAP_BACKGROUND_COLOR, dtype=np.uint8)

    # 2. Sort detections by confidence ascending so higher-confidence survivors draw on top
    sorted_dets = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)))

    # Track present classes for compact legend
    class_counts: Dict[str, int] = {}

    for det in sorted_dets:
        cls = det.get("class", "unknown")
        class_counts[cls] = class_counts.get(cls, 0) + 1

        conf = float(det.get("confidence", 0.5))
        color = get_class_color_rgb(cls)

        # 3. Per-detection opacity proportional to confidence score
        # Confidence floors at ~0.25 after filtering; map [0.25, 1.0] -> [0.35, 0.85]
        alpha = 0.35 + 0.50 * min(1.0, max(0.0, (conf - 0.25) / 0.75))

        # 4. Extract polygon vertices (oriented bounding box or fallback bbox)
        pts: Optional[np.ndarray] = None
        if "obb_coords" in det and det["obb_coords"] and len(det["obb_coords"]) >= 3:
            try:
                pts = np.array(det["obb_coords"], dtype=np.int32)
            except Exception:
                pts = None

        if pts is None or pts.shape[0] < 3:
            bbox = det.get("bbox")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)

        if pts is None or pts.shape[0] < 3:
            continue

        # Clamp points to image bounds
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

        # 5. Composite filled shape via ROI alpha blending
        x, y, bw, bh = cv2.boundingRect(pts)
        if bw <= 0 or bh <= 0:
            continue

        roi = canvas[y:y + bh, x:x + bw]
        poly_mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [pts - [x, y]], 255)

        mask_bool = poly_mask > 0
        if np.any(mask_bool):
            roi[mask_bool] = (
                roi[mask_bool] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha
            ).astype(np.uint8)

        # 6. Draw crisp anti-aliased border (thickness 2px)
        border_color = tuple(int(min(255, c * 1.15 + 20)) for c in color)
        cv2.polylines(
            canvas,
            [pts],
            isClosed=True,
            color=border_color,
            thickness=2,
            lineType=cv2.LINE_AA
        )

    # 7. Generate compact dynamic legend (only classes present in this image)
    # Sorted by count descending
    legend: List[Dict[str, Any]] = []
    for cls_name, count in sorted(class_counts.items(), key=lambda item: -item[1]):
        col = get_class_color_rgb(cls_name)
        legend.append({
            "class": cls_name,
            "color": rgb_to_hex(col),
            "count": count
        })

    # 8. Save output PNG
    os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)
    vis_pil = Image.fromarray(canvas, mode="RGB")
    vis_pil.save(output_png_path, "PNG", optimize=True)

    logger.info(
        f"[DETECTION HEATMAP] Generated heatmap at '{output_png_path}' "
        f"({w}x{h}, {len(detections)} objects, {len(legend)} distinct classes)."
    )

    return output_png_path, legend
