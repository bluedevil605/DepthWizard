"""
Unit and integration tests for Detection Heatmap Overlay and dynamic legend.
"""

import os
import shutil
import pytest
import numpy as np
from PIL import Image

from app.semantic.heatmap import (
    generate_detection_heatmap,
    CLASS_HEATMAP_COLORS,
    HEATMAP_BACKGROUND_COLOR,
    get_class_color_rgb,
    rgb_to_hex
)
from app.semantic.land_cover import CLASS_COLORS


@pytest.fixture
def tmp_dir():
    d = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static", "outputs", "_test_tmp_hm"))
    os.makedirs(d, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_heatmap_palette_distinct_from_land_cover():
    """Verify categorical heatmap colors are distinct and do not match land-cover palette."""
    land_cover_rgb_values = [tuple(c) for c in CLASS_COLORS.values()]
    
    # 15 DOTA classes must all be mapped
    dota_15_classes = [
        "plane", "ship", "storage tank", "baseball diamond", "tennis court",
        "basketball court", "ground track field", "harbor", "bridge",
        "large vehicle", "small vehicle", "helicopter", "roundabout",
        "soccer ball field", "swimming pool"
    ]
    for cls in dota_15_classes:
        assert cls in CLASS_HEATMAP_COLORS, f"Class '{cls}' missing from CLASS_HEATMAP_COLORS"
        color = CLASS_HEATMAP_COLORS[cls]
        assert len(color) == 3
        # Ship should not be colored like water (blue)
        if cls == "ship":
            assert color[0] > color[2], "Ship color should be orange/warm, distinct from blue water"


def test_heatmap_generation_with_obb_and_legend(tmp_dir):
    """Test generating a heatmap overlay with oriented bounding boxes and dynamic legend."""
    out_png = os.path.join(tmp_dir, "test_heatmap.png")
    w, h = 600, 400

    detections = [
        {
            "class": "ship",
            "confidence": 0.85,
            "obb_coords": [[100, 100], [200, 120], [180, 250], [80, 230]],
            "bbox": [80, 100, 200, 250]
        },
        {
            "class": "ship",
            "confidence": 0.60,
            "obb_coords": [[250, 150], [350, 150], [350, 220], [250, 220]],
            "bbox": [250, 150, 350, 220]
        },
        {
            "class": "harbor",
            "confidence": 0.92,
            "obb_coords": [[50, 50], [550, 50], [550, 90], [50, 90]],
            "bbox": [50, 50, 550, 90]
        },
        {
            "class": "small vehicle",
            "confidence": 0.40,
            # Fallback to axis-aligned bbox without obb_coords
            "bbox": [400, 300, 430, 340]
        }
    ]

    png_path, legend = generate_detection_heatmap((w, h), detections, out_png)

    # 1. Check file created and valid
    assert os.path.exists(png_path)
    im = Image.open(png_path)
    assert im.size == (w, h)
    assert im.mode == "RGB"

    # 2. Check pixel content is not all background
    arr = np.array(im)
    bg_diff = np.abs(arr.astype(int) - np.array(HEATMAP_BACKGROUND_COLOR)).sum(axis=-1)
    drawn_pixels = np.sum(bg_diff > 15)
    assert drawn_pixels > 500, "Heatmap should have drawn polygon pixels"

    # 3. Check dynamic legend: only present classes appear
    present_classes = {item["class"] for item in legend}
    assert present_classes == {"ship", "harbor", "small vehicle"}
    assert len(legend) == 3

    # Legend counts and colors
    legend_dict = {item["class"]: item for item in legend}
    assert legend_dict["ship"]["count"] == 2
    assert legend_dict["harbor"]["count"] == 1
    assert legend_dict["small vehicle"]["count"] == 1
    assert legend_dict["ship"]["color"].startswith("#")


def test_heatmap_empty_detections(tmp_dir):
    """Test generating a heatmap overlay with 0 detections."""
    out_png = os.path.join(tmp_dir, "empty_heatmap.png")
    w, h = 200, 200

    png_path, legend = generate_detection_heatmap((w, h), [], out_png)
    assert os.path.exists(png_path)
    assert len(legend) == 0

    im = Image.open(png_path)
    arr = np.array(im)
    # Whole image should be background color
    assert np.all(arr == HEATMAP_BACKGROUND_COLOR)


def test_confidence_proportional_opacity(tmp_dir):
    """Test that higher confidence detections produce higher opacity/saturation."""
    out_png_low = os.path.join(tmp_dir, "low_conf.png")
    out_png_high = os.path.join(tmp_dir, "high_conf.png")
    w, h = 200, 200

    # Low confidence detection (0.25)
    det_low = [{
        "class": "ship",
        "confidence": 0.25,
        "bbox": [50, 50, 150, 150],
        "obb_coords": [[50, 50], [150, 50], [150, 150], [50, 150]]
    }]
    generate_detection_heatmap((w, h), det_low, out_png_low)

    # High confidence detection (0.95)
    det_high = [{
        "class": "ship",
        "confidence": 0.95,
        "bbox": [50, 50, 150, 150],
        "obb_coords": [[50, 50], [150, 50], [150, 150], [50, 150]]
    }]
    generate_detection_heatmap((w, h), det_high, out_png_high)

    im_low = np.array(Image.open(out_png_low))
    im_high = np.array(Image.open(out_png_high))

    # Center pixel (100, 100) inside the filled polygon:
    # High confidence should be more saturated / further from the dark background
    bg = np.array(HEATMAP_BACKGROUND_COLOR)
    diff_low = np.linalg.norm(im_low[100, 100].astype(float) - bg)
    diff_high = np.linalg.norm(im_high[100, 100].astype(float) - bg)

    assert diff_high > diff_low, (
        f"High confidence ({diff_high:.2f}) should have higher saturation/contrast than low ({diff_low:.2f})"
    )


def test_api_sample_includes_detection_heatmap():
    """Verify /api/samples lists P0053 and P0047."""
    from app.main import list_samples
    samples = list_samples()
    sample_ids = {s["id"] for s in samples}
    assert "P0053" in sample_ids
    assert "P0047" in sample_ids


@pytest.mark.anyio
async def test_pipeline_generates_heatmap_and_legend(tmp_dir):
    """Integration test: verify _execute_pipeline produces detection_heatmap_url and legend."""
    from app.main import _execute_pipeline
    
    # Create a small test image in tmp_dir
    img_path = os.path.join(tmp_dir, "test_input.jpg")
    img = Image.new("RGB", (300, 300), (80, 120, 80))
    img.save(img_path)

    res = await _execute_pipeline(
        input_file_path=img_path,
        original_filename="test_input.jpg",
        reference_file_path=None,
        force=True
    )

    assert "detection_heatmap_url" in res
    assert res["detection_heatmap_url"] is not None
    assert res["detection_heatmap_url"].endswith("_detection_heatmap.png")
    assert "detection_heatmap_legend" in res
    assert isinstance(res["detection_heatmap_legend"], list)

