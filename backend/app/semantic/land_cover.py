"""
RGB Land-Cover Differentiator Module for Remote-Sensing Imagery.

Computes Excess Green Index (ExG = 2*G - R - B) per pixel as a vegetation proxy,
then clusters RGB + ExG features using K-means (k=4) to produce a 4-class land-cover mask:
- 0: Shadow / Unclassified (Black)
- 1: Water (Blue)
- 2: Vegetation (Green)
- 3: Built-up / Bare earth (Gray)

Outputs:
- Full-resolution class-index array (uint8)
- Color-coded PNG visualization
"""

import os
import logging
from typing import Tuple, Dict, List, Optional, Any
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

try:
    from scipy import ndimage as _ndimage
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _ndimage = None
    _HAVE_SCIPY = False

logger = logging.getLogger(__name__)

# Canonical Land-Cover Classes
CLASS_SHADOW = 0
CLASS_WATER = 1
CLASS_VEGETATION = 2
CLASS_BUILT_UP = 3

CLASS_NAMES = {
    CLASS_SHADOW: "shadow",
    CLASS_WATER: "water",
    CLASS_VEGETATION: "vegetation",
    CLASS_BUILT_UP: "built_up"
}

# Color palette: green=vegetation, blue=water, gray=built-up, black=shadow
CLASS_COLORS = {
    CLASS_SHADOW: np.array([20, 20, 20], dtype=np.uint8),       # Black
    CLASS_WATER: np.array([30, 100, 220], dtype=np.uint8),      # Blue
    CLASS_VEGETATION: np.array([34, 180, 60], dtype=np.uint8),  # Green
    CLASS_BUILT_UP: np.array([165, 165, 165], dtype=np.uint8),  # Gray
}


def _assign_clusters_to_classes(centers: np.ndarray) -> Dict[int, int]:
    """
    Maps 4 un-ordered KMeans cluster centers (shape: [4, 4] with [R, G, B, ExG])
    to the 4 canonical semantic classes: Shadow, Water, Vegetation, Built-up.
    
    Returns:
        dict: {cluster_idx: canonical_class_id}
    """
    # centers columns: 0: R, 1: G, 2: B, 3: ExG (all normalized 0..1 except ExG which is [-2..2])
    r = centers[:, 0]
    g = centers[:, 1]
    b = centers[:, 2]
    exg = centers[:, 3]

    # Perceived brightness / luminance: Y = 0.299*R + 0.587*G + 0.114*B
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    cluster_ids = list(range(4))
    mapping = {}

    # 1. Darkest cluster -> Shadow / Unclassified
    shadow_cluster = min(cluster_ids, key=lambda i: luminance[i])
    mapping[shadow_cluster] = CLASS_SHADOW
    remaining = [i for i in cluster_ids if i != shadow_cluster]

    # 2. Cluster with highest Excess Green -> Vegetation
    veg_cluster = max(remaining, key=lambda i: exg[i])
    mapping[veg_cluster] = CLASS_VEGETATION
    remaining = [i for i in remaining if i != veg_cluster]

    # 3. Between remaining 2 clusters:
    # Water has higher blue prominence (B - R) and lower luminance compared to built-up.
    # Terrestrial optical water typically has blue dominance or low luminance (<0.35).
    # Bright red/tan soil, concrete or shingles (R > B + 0.05 and luminance >= 0.35) are NEVER water.
    def water_score(i):
        blue_dominance = b[i] - r[i]
        return blue_dominance - luminance[i] * 1.5

    def is_plausible_water(i):
        blue_dominance = b[i] - r[i]
        # Bright red-dominated clusters (shingles, concrete, soil) are NEVER water
        if r[i] > b[i] + 0.05 and luminance[i] > 0.35:
            return False
        # Water: blue not weaker than red by more than 0.05, with relatively low luminance
        if blue_dominance >= -0.05 and luminance[i] < 0.40:
            return True
        return False

    plausible_water = [i for i in remaining if is_plausible_water(i)]

    if plausible_water:
        water_cluster = max(plausible_water, key=water_score)
        mapping[water_cluster] = CLASS_WATER
        remaining = [i for i in remaining if i != water_cluster]
    else:
        # No cluster qualifies as water — assign all remaining to built-up / bare earth
        water_cluster = None

    # 4. Remaining cluster(s) -> Built-up / Bare
    for rem_id in remaining:
        mapping[rem_id] = CLASS_BUILT_UP

    return mapping


def differentiate_land_cover(
    image: Image.Image,
    output_png_path: str
) -> Tuple[np.ndarray, str, Dict[str, float]]:
    """
    Computes ExG and performs K-means land-cover differentiation.
    
    Args:
        image: Optical RGB PIL Image
        output_png_path: Destination path for color-coded visualization PNG
        
    Returns:
        Tuple of:
            - class_mask (np.ndarray of shape (H, W), dtype uint8 with values 0..3)
            - output_png_path (str)
            - class_percentages (dict of {"vegetation": %, "water": %, "built_up": %, "shadow": %})
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    orig_w, orig_h = image.size

    # Scale image to max 1024px for sub-second semantic clustering
    # Land-cover classification does not need gigapixel granularity; 1024px preserves
    # every tree, building, and road while running 100x faster.
    max_dim = 1024
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / float(max(orig_w, orig_h))
        proc_w = max(16, int(round(orig_w * scale)))
        proc_h = max(16, int(round(orig_h * scale)))
        proc_img = image.resize((proc_w, proc_h), Image.Resampling.BILINEAR)
    else:
        proc_w, proc_h = orig_w, orig_h
        proc_img = image

    rgb_arr = np.array(proc_img, dtype=np.float32) / 255.0  # [proc_h, proc_w, 3] in 0.0..1.0

    r = rgb_arr[..., 0]
    g = rgb_arr[..., 1]
    b = rgb_arr[..., 2]

    # 1. Compute Excess Green Index: 2*G - R - B
    exg = 2.0 * g - r - b

    # 2. Prepare feature array [proc_h*proc_w, 4]: [R, G, B, ExG]
    features = np.stack([r, g, b, exg], axis=-1).reshape(-1, 4)

    # 3. Fit K-Means (k=4)
    # Using MiniBatchKMeans with a capped sample size (max 30,000 vectors) for sub-second convergence
    kmeans = MiniBatchKMeans(
        n_clusters=4,
        random_state=42,
        batch_size=4096,
        n_init=3,
        max_iter=50
    )

    total_pixels = proc_h * proc_w
    if total_pixels > 30000:
        step = max(1, total_pixels // 30000)
        sample_features = features[::step]
        kmeans.fit(sample_features)
        raw_labels = kmeans.predict(features)
    else:
        raw_labels = kmeans.fit_predict(features)

    # 4. Map clusters to physical semantic classes
    cluster_mapping = _assign_clusters_to_classes(kmeans.cluster_centers_)
    
    # Remap flat labels to canonical class IDs
    canonical_labels = np.zeros_like(raw_labels, dtype=np.uint8)
    for c_idx, class_id in cluster_mapping.items():
        canonical_labels[raw_labels == c_idx] = class_id

    sub_mask = canonical_labels.reshape((proc_h, proc_w))

    # 5. Restore full resolution if needed
    if (proc_w, proc_h) != (orig_w, orig_h):
        mask_pil = Image.fromarray(sub_mask)
        class_mask = np.array(
            mask_pil.resize((orig_w, orig_h), Image.Resampling.NEAREST),
            dtype=np.uint8
        )
    else:
        class_mask = sub_mask

    # 6. Calculate class distributions
    total_count = float(class_mask.size)
    percentages = {
        CLASS_NAMES[cid]: round(float(np.sum(class_mask == cid)) / total_count * 100.0, 1)
        for cid in (CLASS_VEGETATION, CLASS_WATER, CLASS_BUILT_UP, CLASS_SHADOW)
    }

    # 7. Generate color-coded PNG visualization (saved at processing resolution for fast upload/rendering)
    color_img = np.zeros((proc_h, proc_w, 3), dtype=np.uint8)
    for cid, rgb in CLASS_COLORS.items():
        color_img[sub_mask == cid] = rgb

    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    vis_pil = Image.fromarray(color_img, mode="RGB")
    vis_pil.save(output_png_path, "PNG", optimize=False)

    logger.info(
        f"Land-cover differentiation completed: "
        f"Veg={percentages['vegetation']}%, Water={percentages['water']}%, "
        f"Built-up={percentages['built_up']}%, Shadow={percentages['shadow']}%"
    )

    return class_mask, output_png_path, percentages


def extract_building_footprints(
    land_cover_mask: np.ndarray,
    min_area_px: int = 40,
    max_regions: int = 400
) -> List[Dict[str, Any]]:
    """
    Extracts discrete building / rooftop footprints from the built-up land-cover mask
    using connected-component analysis.

    Each built-up connected component with a plausible building footprint (area and
    compact aspect ratio) becomes a 'building'/'house' detection with an axis-aligned
    bounding box, centroid, and pixel area.

    Args:
        land_cover_mask: 2D uint8 array of land-cover class ids (0=shadow,1=water,
            2=vegetation,3=built_up).
        min_area_px: Minimum component pixel area to be considered a real building
            (suppresses speckle / single-pixel noise).
        max_regions: Cap on the number of building footprints returned (largest first).

    Returns:
        List of detection dicts suitable for the frontend detector overlay:
            {
                "class": "house",
                "confidence": float,       # area-normalized plausibility 0..1
                "bbox": [x1, y1, x2, y2],
                "obb_coords": [[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
                "area_px": int,
                "source": "land_cover"
            }
    """
    if land_cover_mask is None:
        return []
    mask = np.asarray(land_cover_mask)
    if mask.ndim != 2:
        return []
    if mask.size == 0:
        return []

    built_up = (mask == CLASS_BUILT_UP).astype(np.uint8)
    h, w = built_up.shape
    total_px = float(h * w)

    # Break thin road / laneway connections between adjacent rooftops so individual
    # houses separate into discrete footprints. A small morphological opening (erode
    # then dilate) removes 1-2px bridge pixels that link houses via roads.
    if _HAVE_SCIPY and _ndimage is not None:
        struct = np.ones((3, 3), dtype=np.uint8)
        built_up = _ndimage.binary_opening(built_up, structure=struct).astype(np.uint8)

    # Label connected 8-connected components
    if _HAVE_SCIPY and _ndimage is not None:
        labeled, n_labels = _ndimage.label(built_up, structure=np.ones((3, 3)))
    else:
        # Minimal fallback without scipy: no connected-components, treat whole mask
        labeled = None
        n_labels = 0

    footprints: List[Dict[str, Any]] = []

    if _HAVE_SCIPY and labeled is not None and n_labels > 0:
        # Component sizes in pixels
        sizes = _ndimage.sum(built_up, labeled, range(1, n_labels + 1))
        slices = _ndimage.find_objects(labeled)

        # Sort components largest-to-smallest, keep the ones that look like buildings
        order = sorted(range(n_labels), key=lambda i: sizes[i], reverse=True)
        for idx in order:
            area = float(sizes[idx])
            if area < min_area_px:
                continue
            sl = slices[idx]
            if sl is None:
                continue
            y1, y2 = sl[0].start, sl[0].stop
            x1, x2 = sl[1].start, sl[1].stop
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            # Plausible building aspect ratio: not an entire linear road strip
            aspect = max(bw, bh) / float(max(1, min(bw, bh)))
            if aspect > 12.0:
                # Likely a road / field edge — skip overly elongated slivers
                continue
            # A real house footprint should have a fairly high fill ratio of its bbox
            fill_ratio = area / float(bw * bh)
            if fill_ratio < 0.12:
                continue
            # Skip footprints that span a huge fraction of the frame — these are
            # regional built-up merges (or the whole scene), not individual buildings.
            bbox_area_frac = (bw * bh) / total_px
            if bbox_area_frac > 0.25:
                continue

            # Normalized plausibility confidence from area + fill ratio (0.5..0.99)
            area_fraction = area / total_px
            conf = round(min(0.99, 0.55 + 0.20 * min(1.0, area_fraction * 2000.0) + 0.20 * fill_ratio), 3)

            footprints.append({
                "class": "building",
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "obb_coords": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "area_px": int(area),
                "source": "land_cover",
            })
            if len(footprints) >= max_regions:
                break
    else:
        # Fallback: no scipy — skip building extraction (log and return empty)
        logger.warning("Building footprint extraction skipped: scipy.ndimage unavailable.")

    logger.info(
        f"Building footprint extraction: {len(footprints)} building(s) detected "
        f"from built-up land-cover mask ({land_cover_mask.sum() if False else int(np.sum(built_up))} px built-up)."
    )
    return footprints
