"""
Unit and Integration Tests for DepthWizard Semantic Layer
Tests YOLOv8-OBB aerial object detection, RGB land-cover differentiation,
per-class SRTM DEM calibration, and object-based scale inference.
"""

import os
import sys
import pytest
from PIL import Image
import numpy as np

# Ensure paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
for p in (WORKSPACE_DIR, BASE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from app.semantic import detect_objects, differentiate_land_cover, CLASS_NAMES
    from app.calibration import calibrate_depth_to_metric, infer_object_scale
    from app.main import app
except ModuleNotFoundError:
    from backend.app.semantic import detect_objects, differentiate_land_cover, CLASS_NAMES
    from backend.app.calibration import calibrate_depth_to_metric, infer_object_scale
    from backend.app.main import app


def test_land_cover_differentiator(tmp_path):
    """Verifies Excess Green Index and 4-class KMeans land-cover segmentation."""
    h, w = 120, 120
    img_arr = np.zeros((h, w, 3), dtype=np.uint8)
    img_arr[:60, :60] = [34, 160, 45]     # green vegetation
    img_arr[:60, 60:] = [20, 80, 200]     # water
    img_arr[60:, :60] = [170, 170, 170]   # built-up
    img_arr[60:, 60:] = [10, 10, 10]      # shadow
    
    pil_img = Image.fromarray(img_arr, mode="RGB")
    out_png = str(tmp_path / "test_lc.png")
    
    mask, png_path, percentages = differentiate_land_cover(pil_img, out_png)
    
    assert os.path.exists(png_path)
    assert mask.shape == (h, w)
    assert mask.dtype == np.uint8
    assert set(percentages.keys()) == {"vegetation", "water", "built_up", "shadow"}
    # Vegetation quadrant should have high vegetation percentage
    assert percentages["vegetation"] > 10.0


def test_yolo_object_detection():
    """Verifies YOLO detection pipeline runs gracefully on an RGB image."""
    test_img = Image.new("RGB", (256, 256), color=(100, 120, 140))
    detections = detect_objects(test_img, conf_threshold=0.1)
    
    assert isinstance(detections, list)
    for det in detections:
        assert "class" in det
        assert "confidence" in det
        assert "bbox" in det
        assert len(det["bbox"]) == 4


def test_per_class_calibration(tmp_path):
    """Verifies that calibrate_depth_to_metric fits separate linear regressions per land-cover class."""
    h, w = 64, 64
    rel_depth = np.linspace(0.1, 0.9, h * w).reshape((h, w)).astype(np.float32)
    # Synthetic SRTM: Vegetation has elevation 1000 + 400*depth, Built-up has 1200 + 600*depth
    land_cover_mask = np.zeros((h, w), dtype=np.uint8)
    land_cover_mask[:32, :] = 2  # vegetation
    land_cover_mask[32:, :] = 3  # built_up

    srtm_elev = np.zeros((h, w), dtype=np.float32)
    srtm_elev[:32, :] = 1000.0 + 400.0 * rel_depth[:32, :]
    srtm_elev[32:, :] = 1200.0 + 600.0 * rel_depth[32:, :]

    out_tif = str(tmp_path / "calibrated_per_class.tif")
    from rasterio.transform import from_origin
    transform = from_origin(500000, 5100000, 10, 10)

    cal_dsm, cal_stats = calibrate_depth_to_metric(
        relative_depth=rel_depth,
        srtm_elevation=srtm_elev,
        crs="EPSG:32610",
        transform=transform,
        output_geotiff_path=out_tif,
        n_samples=1000,
        land_cover_mask=land_cover_mask
    )

    assert os.path.exists(out_tif)
    assert cal_stats["calibration_method"] == "per-class"
    assert "class_fits" in cal_stats
    assert cal_dsm.shape == (h, w)


def test_object_scale_inference():
    """Verifies object-based scale inference when a vehicle or building is detected."""
    h, w = 100, 100
    # Create flat ground with depth = 0.2 and a vehicle at [40, 40, 60, 60] with depth = 0.35 (delta = 0.15)
    rel_depth = np.full((h, w), 0.2, dtype=np.float32)
    rel_depth[40:60, 40:60] = 0.35

    mock_detections = [
        {
            "class": "small vehicle",
            "confidence": 0.85,
            "bbox": [40, 40, 60, 60]
        }
    ]

    res = infer_object_scale(rel_depth, mock_detections)
    assert res is not None
    scaled_dsm, stats = res

    assert stats["calibration_method"] == "object-scale-inference"
    assert "scale_factor" in stats
    # Expected scale factor approx 1.5m / 0.15 ≈ 10.0 m/unit
    assert 5.0 <= stats["scale_factor"] <= 20.0
    assert "approximate" in stats["note"].lower()


def test_upload_api_response_schema():
    """Integration test checking that /api/process-sample response contains the new semantic fields."""
    from fastapi.testclient import TestClient
    client = TestClient(app)

    res = client.post("/api/process-sample/alpine_drone")
    assert res.status_code == 200
    data = res.json()

    # Verify backward compatibility
    assert "glb_url" in data
    assert "preview_url" in data
    assert "stats" in data
    assert "mode" in data

    # Verify new semantic layer fields
    assert "detections" in data
    assert isinstance(data["detections"], list)
    assert "land_cover_mask_url" in data
    assert data["land_cover_mask_url"] is not None
    assert "calibration_method" in data
    assert data["calibration_method"] in ("global", "per-class", "object-scale-inference")
