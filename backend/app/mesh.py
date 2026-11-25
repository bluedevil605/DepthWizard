"""
3D Mesh Generation & Texture Draping Module for DepthWizard
Generates a triangulated heightfield 3D mesh, drapes the optical RGB imagery as UV texture,
and exports a self-contained GLB file for Three.js / React-Three-Fiber rendering.

PRODUCTION SYSTEM IMPROVEMENT NOTES:
- Continuous Level-of-Detail (LOD): For large-scale terrain, production systems use
  quadtree-based multiresolution LOD streaming (e.g. Cesium 3D Tiles, Quantized Mesh / tin-terrain).
- Texture Compression: Textures in production should be compressed into KTX2/Basis Universal GPU textures
  with mipmaps to minimize VRAM footprint on mobile and web clients.
- Normal Mapping: Calculate high-frequency normal maps from full-resolution depth to preserve fine micro-relief
  (creeks, boulders, rooftop edges) without requiring billions of geometric polygons.
"""

import logging
import os
import json
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from PIL import Image
import trimesh

logger = logging.getLogger(__name__)


def _extrude_buildings(
    elev_resampled: np.ndarray,
    buildings: List[Dict[str, Any]],
    orig_w: int,
    orig_h: int,
    built_up_mask: Optional[np.ndarray] = None,
    is_relative: bool = False,
    relief_span_ref: float = 1.0,
) -> np.ndarray:
    """Raise building footprints as crisp flat-roofed blocks with smooth wall transitions.

    Uses the built-up mask to precisely define rooftop cells and creates smooth 1-2
    cell feathered transitions at building edges, so buildings look like recognizable
    structures rather than fuzzy stretched ramps.

    Returns:
        Modified elevation array.
    """
    mesh_h, mesh_w = elev_resampled.shape
    sx = mesh_w / float(max(1, orig_w))
    sy = mesh_h / float(max(1, orig_h))

    # Resize built-up mask to mesh resolution for crisp rooftop carving
    carve_mask = None
    if built_up_mask is not None:
        try:
            bm = np.asarray(built_up_mask)
            if bm.ndim == 2 and bm.size > 0:
                bm_pil = Image.fromarray(bm.astype(np.uint8))
                carve_mask = np.array(
                    bm_pil.resize((mesh_w, mesh_h), Image.Resampling.NEAREST),
                    dtype=bool,
                )
        except Exception:
            carve_mask = None

    min_elev = float(np.nanmin(elev_resampled))
    max_elev = float(np.nanmax(elev_resampled))
    local_span = max(1e-4, max_elev - min_elev)

    # Conservative offset: just enough to make buildings visually pop above terrain.
    # 2% of local relief for relative mode (already exaggerated), avoids stretching.
    # 5% for absolute metric mode; minimum floor of 0.15 units keeps tiny houses visible.
    offset = max(float(local_span) * 0.02, 0.15) if is_relative else max(float(local_span) * 0.05, 0.15)

    extruded = elev_resampled.astype(np.float32).copy()

    def _bbox_to_mesh(box):
        x1, y1, x2, y2 = box
        gx1 = max(0, min(mesh_w - 1, int(round(x1 * sx))))
        gy1 = max(0, min(mesh_h - 1, int(round(y1 * sy))))
        gx2 = max(gx1 + 1, min(mesh_w, int(max(gx1 + 1, round(x2 * sx)))))
        gy2 = max(gy1 + 1, min(mesh_h, int(max(gy1 + 1, round(y2 * sy)))))
        return gx1, gy1, gx2, gy2

    for det in buildings:
        box = det.get("bbox")
        if not box or len(box) != 4:
            continue
        gx1, gy1, gx2, gy2 = _bbox_to_mesh(box)
        sub = extruded[gy1:gy2, gx1:gx2]
        if sub.size == 0:
            continue

        # Ground level from surrounding ring
        ry1, ry2 = max(0, gy1 - 3), min(mesh_h, gy2 + 3)
        rx1, rx2 = max(0, gx1 - 3), min(mesh_w, gx2 + 3)
        ring = extruded[ry1:ry2, rx1:rx2]
        ground_level = float(np.percentile(ring, 15)) if ring.size else float(np.percentile(sub, 15))
        roof_peak = float(np.percentile(sub, 90))
        roof_height = max(roof_peak, ground_level) + offset

        bw, bh = gx2 - gx1, gy2 - gy1

        if carve_mask is not None and carve_mask.shape == (mesh_h, mesh_w):
            # ── Mask-based: raise only built-up cells, smooth wall feather at edges ──
            region_roof = carve_mask[gy1:gy2, gx1:gx2]
            raised = np.where(region_roof, roof_height, sub)
            # Smooth 1-cell border ramp for natural-looking walls
            if bh > 2:
                ramp = ground_level + (roof_height - ground_level) * 0.3
                raised[0, :] = np.where(region_roof[0, :], raised[0, :], ramp)
                raised[-1, :] = np.where(region_roof[-1, :], raised[-1, :], ramp)
            if bw > 2:
                ramp = ground_level + (roof_height - ground_level) * 0.3
                raised[:, 0] = np.where(region_roof[:, 0], raised[:, 0], ramp)
                raised[:, -1] = np.where(region_roof[:, -1], raised[:, -1], ramp)
            extruded[gy1:gy2, gx1:gx2] = raised
        elif bw > 2 and bh > 2:
            # ── Fallback: bbox-based with smooth 1-cell wall ramp ──
            region = sub.copy()
            region[1:-1, 1:-1] = roof_height
            ramp = ground_level + (roof_height - ground_level) * 0.3
            region[0, :] = ramp
            region[-1, :] = ramp
            region[:, 0] = np.minimum(region[:, 0], ramp)
            region[:, -1] = np.minimum(region[:, -1], ramp)
            extruded[gy1:gy2, gx1:gx2] = region
        else:
            extruded[gy1:gy2, gx1:gx2] = roof_height

    logger.info(
        f"Building extrusion applied: {len(buildings)} footprint(s) raised on "
        f"{mesh_w}x{mesh_h} terrain (offset={offset:.3f}, mask={'yes' if carve_mask is not None else 'no'})."
    )
    return extruded


def generate_terrain_mesh(
    elevation_grid: np.ndarray,
    texture_image: Image.Image,
    output_glb_path: str,
    max_resolution: int = 512,
    vertical_exaggeration: float = 1.0,
    is_relative: bool = False,
    relative_exaggeration: float = 18.0,
    buildings: Optional[List[Dict[str, Any]]] = None,
    built_up_mask: Optional[np.ndarray] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Constructs a 3D textured terrain mesh from an elevation grid and an RGB texture,
    exporting it as a single binary .glb file.
    
    Args:
        elevation_grid: 2D numpy array with elevations (metric or relative 0..1).
        texture_image: Original RGB PIL image.
        output_glb_path: Target path for the .glb file.
        max_resolution: Maximum grid dimension along the longer edge (e.g. 256 or 512).
        vertical_exaggeration: Multiplier for elevation display relief.
        is_relative: True if unreferenced relative depth (rDSM).
        relative_exaggeration: Configurable artificial vertical exaggeration multiplier
            applied ONLY to relative-depth meshes (default 18.0 — gives target_max_height
            of 22.5 scene units; use percentile clipping to prevent spikes; range 8.0–24.0).
        
    Returns:
        Tuple of (glb_path, metadata_dict)
    """
    os.makedirs(os.path.dirname(output_glb_path), exist_ok=True)
    orig_h, orig_w = elevation_grid.shape

    # 1. Downsample elevation grid to max_resolution for smooth 60fps WebGL
    scale = min(max_resolution / float(orig_w), max_resolution / float(orig_h), 1.0)
    mesh_w = int(round(orig_w * scale))
    mesh_h = int(round(orig_h * scale))
    mesh_w = max(mesh_w, 16)
    mesh_h = max(mesh_h, 16)

    # Use PIL for clean anti-aliased bicubic downsampling of elevation
    elev_pil = Image.fromarray(elevation_grid.astype(np.float32))
    elev_resampled = np.array(
        elev_pil.resize((mesh_w, mesh_h), Image.Resampling.BICUBIC),
        dtype=np.float32
    )

    # Building extrusion is intentionally skipped — Depth Anything already encodes
    # building heights as natural depth relief. Independent extrusion creates
    # flat-roofed blocks that fight the real depth signal and produces spikes.
    # The raw depth + mild relative_exaggeration gives recognizable buildings
    # with natural boundaries and no aggressive geometry manipulation.

    # Multi-scale unsharp mask: amplify depth gradients at structural boundaries
    # (building walls, roof edges, terrain ridgelines) without inventing geometry.
    # Two complementary scales:
    #   Fine  (sigma=1.0, alpha=1.2): pixel-level edge crispness — building outlines, road edges
    #   Coarse(sigma=4.0, alpha=0.6): structural mass contrast — building tops vs ground plane
    # Flat areas (roads, open ground) have near-zero gradient and are barely affected.
    try:
        from scipy.ndimage import gaussian_filter as _gf
        _fine_blur   = _gf(elev_resampled, sigma=1.0)
        _coarse_blur = _gf(elev_resampled, sigma=4.0)
        elev_resampled = (elev_resampled
                          + 1.2 * (elev_resampled - _fine_blur)
                          + 0.6 * (elev_resampled - _coarse_blur))
        logger.info("[MESH] Multi-scale unsharp mask applied (fine=1.0/1.2x, coarse=4.0/0.6x)")
    except Exception as _sharpen_err:
        logger.debug(f"Unsharp mask skipped: {_sharpen_err}")

    # Calculate elevation statistics
    min_elev = float(np.nanmin(elev_resampled))
    max_elev = float(np.nanmax(elev_resampled))
    mean_elev = float(np.nanmean(elev_resampled))
    relief_span = max_elev - min_elev

    # 2. Coordinate normalization for WebGL viewing
    # Horizontal extent: [-aspect_x, +aspect_x] and [-aspect_z, +aspect_z]
    aspect_ratio = mesh_w / float(mesh_h)
    scene_width = 10.0 if aspect_ratio >= 1.0 else 10.0 * aspect_ratio
    scene_depth = 10.0 / aspect_ratio if aspect_ratio >= 1.0 else 10.0

    xs = np.linspace(-scene_width / 2.0, scene_width / 2.0, mesh_w, dtype=np.float32)
    zs = np.linspace(-scene_depth / 2.0, scene_depth / 2.0, mesh_h, dtype=np.float32)
    xv, zv = np.meshgrid(xs, zs)

    # Vertical elevation scaling for WebGL viewing.
    # ---------------------------------------------------------------
    # RELATIVE rDSM PATH
    # ---------------------------------------------------------------
    # Depth Anything emits normalized disparities in [0, 1]. Bicubic downsampling
    # compresses the range further. We:
    #   1. Clip to [p1, p99] to remove isolated outlier spikes while preserving
    #      sharp structural discontinuities (roof edges are NOT outliers — they are
    #      concentrated groups of pixels at a consistent depth offset from terrain).
    #   2. Map the clipped range to [0, target_max_height] in scene units.
    #   3. Do NOT re-normalize after applying the scale.
    if is_relative:
        target_max_height = 12.5 * (relative_exaggeration / 10.0) * vertical_exaggeration

        # Step 1: percentile-clip to suppress isolated spikes
        p1_val = float(np.percentile(elev_resampled, 1))
        p99_val = float(np.percentile(elev_resampled, 99))
        elev_clipped = np.clip(elev_resampled, p1_val, p99_val)
        clipped_span = max(p99_val - p1_val, 1e-4)

        # Step 2: scale to scene height — no additional renormalization
        yv = ((elev_clipped - p1_val) / clipped_span) * target_max_height

        # Step 3: diagnostic log of final mesh Y amplitude (inspect in backend logs)
        logger.info(
            f"[MESH-Z rDSM] depth grid p1={p1_val:.4f} p99={p99_val:.4f} "
            f"clipped_span={clipped_span:.4f} target_max_height={target_max_height:.3f} "
            f"relative_exaggeration={relative_exaggeration}"
        )
        logger.info(
            f"[MESH-Z rDSM] final vertex Y: "
            f"min={float(yv.min()):.4f} max={float(yv.max()):.4f} "
            f"p5={float(np.percentile(yv, 5)):.4f} "
            f"p50={float(np.percentile(yv, 50)):.4f} "
            f"p95={float(np.percentile(yv, 95)):.4f} "
            f"p99={float(np.percentile(yv, 99)):.4f}"
        )
    else:
        # Physical metric scaling: scene_width is 10 units across the terrain footprint.
        # target_max_height=8.0 gives visually punchy relief (St Helens 900m → 8 scene units)
        # without over-exaggerating flat coastal plains.
        target_max_height = 8.0 * vertical_exaggeration
        if relief_span > 1e-4:
            yv = ((elev_resampled - min_elev) / relief_span) * target_max_height
        else:
            yv = np.zeros_like(elev_resampled)

    # 3. Build vertex array: (N, 3) where columns are [X, Y, Z]
    # In Three.js: Y is up, X is right, Z is toward viewer
    vertices = np.column_stack([xv.ravel(), yv.ravel(), zv.ravel()])

    # 4. Compute UV coordinates: (N, 2)
    # UV (0,0) is bottom-left, (1,1) is top-right
    u = np.linspace(0.0, 1.0, mesh_w, dtype=np.float32)
    v = np.linspace(1.0, 0.0, mesh_h, dtype=np.float32)  # Inverted Y for image texture alignment
    uv_u, uv_v = np.meshgrid(u, v)
    uvs = np.column_stack([uv_u.ravel(), uv_v.ravel()])

    # 5. Build triangular faces: 2 triangles per quad
    # Grid indexing: index = row * mesh_w + col
    cols = np.arange(mesh_w - 1)
    rows = np.arange(mesh_h - 1)
    col_grid, row_grid = np.meshgrid(cols, rows)
    
    p0 = (row_grid * mesh_w + col_grid).ravel()
    p1 = (row_grid * mesh_w + (col_grid + 1)).ravel()
    p2 = ((row_grid + 1) * mesh_w + col_grid).ravel()
    p3 = ((row_grid + 1) * mesh_w + (col_grid + 1)).ravel()

    # Two triangles: (p0, p2, p1) and (p1, p2, p3)
    faces_t1 = np.column_stack([p0, p2, p1])
    faces_t2 = np.column_stack([p1, p2, p3])
    faces = np.vstack([faces_t1, faces_t2])

    # 6. Attach draped texture visual
    # Resize texture to max 4096x4096 for maximum visual crispness.
    # 4K texture with 1024-resolution mesh gives sharp photo-realistic draping.
    tex = texture_image.convert("RGB")
    tex.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
    
    visual = trimesh.visual.TextureVisuals(uv=uvs, image=tex)
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=visual,
        process=False,
        validate=False
    )

    # 7. Export GLB
    glb_bytes = trimesh.exchange.gltf.export_glb(mesh)
    with open(output_glb_path, "wb") as f:
        f.write(glb_bytes)

    # 8. Save companion JSON metadata for client-side raycasting and terrain inspection
    meta_path = output_glb_path.replace(".glb", "_meta.json")
    meta_dict = {
        "image_width": orig_w,
        "image_height": orig_h,
        "grid_width": mesh_w,
        "grid_height": mesh_h,
        "scene_width": float(scene_width),
        "scene_depth": float(scene_depth),
        "min_elevation": round(min_elev, 2),
        "max_elevation": round(max_elev, 2),
        "mean_elevation": round(mean_elev, 2),
        "relief_span": round(relief_span, 2),
        "unit": "meters" if not is_relative else "normalized_relative",
        "is_relative": is_relative,
        "target_max_height": round(float(target_max_height), 2),
        "relative_exaggeration": round(float(relative_exaggeration), 2) if is_relative else 1.0,
        # Sampling down to 64x64 matrix for instant client elevation raycasting lookup
        "elevation_samples": elev_resampled[::max(1, mesh_h // 64), ::max(1, mesh_w // 64)].tolist()
    }
    with open(meta_path, "w") as f:
        json.dump(meta_dict, f)

    logger.info(f"Terrain mesh exported to {output_glb_path} ({len(vertices)} vertices, {len(faces)} faces)")
    return output_glb_path, meta_dict
