import os
import sys
import io
import time
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

# Ensure both workspace root and backend directory are in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
for p in (WORKSPACE_DIR, BASE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from app.validation.nadir_check import assess_nadir_likelihood
    from validation.nadir_check import assess_nadir_likelihood as assess_nadir_top_level
    from app.semantic.detector import (
        _apply_size_sanity_filter,
        detect_objects,
        DEFAULT_YOLO_CONF,
        FORCED_NONNADIR_YOLO_CONF
    )
    from app.main import app
except ModuleNotFoundError:
    from backend.app.validation.nadir_check import assess_nadir_likelihood
    from backend.validation.nadir_check import assess_nadir_likelihood as assess_nadir_top_level
    from backend.app.semantic.detector import (
        _apply_size_sanity_filter,
        detect_objects,
        DEFAULT_YOLO_CONF,
        FORCED_NONNADIR_YOLO_CONF
    )
    from backend.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_top_level_import_compatibility():
    """Verify backend/validation/nadir_check.py can be imported directly as specified."""
    assert callable(assess_nadir_top_level)


def test_nadir_geotiff_short_circuit():
    """Georeferenced mapping-grade imagery should immediately short-circuit to high nadir confidence."""
    # Create simple image with mock georeference metadata
    img = Image.new("RGB", (256, 256), color=(100, 150, 80))
    geo_meta = {
        "is_georeferenced": True,
        "crs": "EPSG:32610",
        "method": "geotiff",
        "transform": None
    }
    result = assess_nadir_likelihood(img, geo_meta)
    assert result["is_likely_nadir"] is True
    assert result["confidence"] >= 0.95
    assert any("Georeferenced" in r for r in result["reasons"])
    assert result["elapsed_ms"] < 200.0


def test_nadir_aerial_grid_passes():
    """A top-down ortho view with earth tones and orthogonal lines should pass nadir check."""
    # Create 512x512 image of agricultural / urban grid (earth tones, no sky)
    arr = np.full((512, 512, 3), 90, dtype=np.uint8)
    # Add green fields
    arr[50:200, 50:200] = [40, 110, 45]
    arr[250:450, 50:200] = [60, 130, 50]
    # Add grid roads (horizontal & vertical)
    arr[210:230, :] = [180, 180, 180]
    arr[:, 210:230] = [180, 180, 180]
    img = Image.fromarray(arr, mode="RGB")

    result = assess_nadir_likelihood(img)
    assert result["is_likely_nadir"] is True
    assert result["confidence"] >= 0.50
    assert result["sky_fraction"] < 0.10
    assert "elapsed_ms" in result
    assert result["elapsed_ms"] < 500.0


def test_ground_level_photo_with_sky_rejected():
    """Ground-level photo with prominent blue sky in upper portion must be rejected."""
    # 512x512 image: top 45% is blue sky, bottom is ground/road
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    # Blue sky: RGB (100, 170, 240) -> BGR (240, 170, 100) -> HSV hue in ~105
    arr[:220, :] = [100, 170, 240]
    # Ground: gray asphalt / dirt
    arr[220:, :] = [70, 65, 60]
    img = Image.fromarray(arr, mode="RGB")

    result = assess_nadir_likelihood(img)
    assert result["is_likely_nadir"] is False
    assert result["confidence"] < 0.50
    assert result["sky_fraction"] > 0.25
    assert any("sky" in r.lower() for r in result["reasons"])


def test_ground_level_converging_lines_rejected():
    """Oblique ground-level photo with strong converging diagonals toward vanishing point."""
    import cv2
    arr = np.full((512, 512, 3), 70, dtype=np.uint8)
    # Draw perspective road / corridor converging to center horizon (256, 120)
    cv2.line(arr, (0, 511), (250, 120), (220, 220, 220), 4)
    cv2.line(arr, (511, 511), (262, 120), (220, 220, 220), 4)
    # Draw additional perspective lines (e.g. wall diagonals)
    cv2.line(arr, (30, 400), (245, 125), (200, 200, 200), 3)
    cv2.line(arr, (480, 400), (267, 125), (200, 200, 200), 3)
    # Add sky/horizon
    arr[:120, :] = [120, 180, 230]

    img = Image.fromarray(arr, mode="RGB")
    result = assess_nadir_likelihood(img)
    assert result["is_likely_nadir"] is False
    assert result["confidence"] < 0.50


def test_size_sanity_filter_normal_vs_flagged():
    """Size sanity filter should flag implausible bounding boxes and preserve plausible ones."""
    img_w, img_h = 1000, 1000  # area = 1,000,000

    detections = [
        # Normal swimming pool: 50x30 = 1500 px = 0.0015 fraction, aspect 1.67
        {
            "class": "swimming pool",
            "confidence": 0.85,
            "bbox": [100, 100, 150, 130]
        },
        # Implausible swimming pool: 600x500 = 300,000 px = 30% of the entire image!
        {
            "class": "swimming-pool",
            "confidence": 0.75,
            "bbox": [50, 50, 650, 550]
        },
        # Normal small vehicle: 30x15 = 450 px = 0.00045 fraction, aspect 2.0
        {
            "class": "small vehicle",
            "confidence": 0.90,
            "bbox": [200, 200, 230, 215]
        },
        # Distorted small vehicle with aspect ratio 15:1 (impossible nadir vehicle)
        {
            "class": "small-vehicle",
            "confidence": 0.70,
            "bbox": [300, 300, 600, 320]
        }
    ]

    filtered = _apply_size_sanity_filter(detections, img_w, img_h)
    assert len(filtered) == 4

    # Plausible pool
    assert filtered[0]["size_sanity"] == "ok"
    assert filtered[0]["confidence"] == 0.85

    # Implausibly giant pool
    assert filtered[1]["size_sanity"] == "flagged"
    assert filtered[1]["confidence"] < 0.75

    # Plausible small vehicle
    assert filtered[2]["size_sanity"] == "ok"
    assert filtered[2]["confidence"] == 0.90

    # Distorted vehicle
    assert filtered[3]["size_sanity"] == "flagged"
    assert filtered[3]["confidence"] < 0.70


def test_upload_api_nadir_warning_and_force(client):
    """Test that /api/upload returns warning for ground-level image and processes on force=True."""
    # Create ground-level image with blue sky
    arr = np.zeros((400, 400, 3), dtype=np.uint8)
    arr[:180, :] = [100, 170, 240]  # blue sky
    arr[180:, :] = [80, 75, 70]     # ground
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    # 1. Normal upload (without force) -> should be intercepted with status="warning"
    files = {"file": ("ground_shot.jpg", buf, "image/jpeg")}
    res = client.post("/api/upload", files=files)
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "warning"
    assert data["proceed_anyway_available"] is True
    assert "nadir_assessment" in data
    assert data["nadir_assessment"]["is_likely_nadir"] is False
    assert "doesn't appear to be a top-down" in data["message"]

    # 2. Upload with force=True -> should proceed through full pipeline
    buf.seek(0)
    files = {"file": ("ground_shot.jpg", buf, "image/jpeg")}
    res_forced = client.post("/api/upload", files=files, data={"force": "true"})
    assert res_forced.status_code == 200
    data_forced = res_forced.json()

    assert data_forced["status"] == "forced_warning"
    assert "glb_url" in data_forced
    assert "preview_url" in data_forced
    assert "nadir_assessment" in data_forced
    assert data_forced["nadir_assessment"]["confidence"] < 0.50
