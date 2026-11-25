"""
Tests for SRTM calibration confidence gating and land-cover detection cross-validation.
"""

import os
import sys
import pytest
import numpy as np
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
for p in (WORKSPACE_DIR, BASE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from app.calibration import (
        is_footprint_sufficient_for_srtm,
        calibrate_depth_to_metric,
        MIN_CALIBRATION_R_SQUARED,
        MIN_SRTM_CELLS,
        SRTM_CELL_SIZE_M,
        _estimate_ground_footprint_m,
    )
    from app.geo import _extract_drone_flight_telemetry
    from app.semantic.cross_validation import cross_validate_detections_with_land_cover
    from app.semantic.land_cover import (
        _assign_clusters_to_classes,
        CLASS_SHADOW,
        CLASS_WATER,
        CLASS_VEGETATION,
        CLASS_BUILT_UP,
    )
except ModuleNotFoundError:
    from backend.app.calibration import (
        is_footprint_sufficient_for_srtm,
        calibrate_depth_to_metric,
        MIN_CALIBRATION_R_SQUARED,
        MIN_SRTM_CELLS,
        SRTM_CELL_SIZE_M,
        _estimate_ground_footprint_m,
    )
    from backend.app.geo import _extract_drone_flight_telemetry
    from backend.app.semantic.cross_validation import cross_validate_detections_with_land_cover
    from backend.app.semantic.land_cover import (
        _assign_clusters_to_classes,
        CLASS_SHADOW,
        CLASS_WATER,
        CLASS_VEGETATION,
        CLASS_BUILT_UP,
    )


def _affine():
    from rasterio.transform import from_origin
    return from_origin(500000, 5100000, 10, 10)


# ---------------------------------------------------------------------------
# Footprint / SRTM pre-check
# ---------------------------------------------------------------------------

def test_footprint_precheck_skips_when_fewer_than_three_cells():
    min_span = MIN_SRTM_CELLS * SRTM_CELL_SIZE_M  # 90 m
    ok, reason = is_footprint_sufficient_for_srtm((50.0, 80.0))
    assert ok is False
    assert reason is not None
    assert "fewer than" in reason

    ok_wide, _ = is_footprint_sufficient_for_srtm((min_span + 1.0, 40.0))
    assert ok_wide is False

    ok_both, reason_both = is_footprint_sufficient_for_srtm((120.0, 150.0))
    assert ok_both is True
    assert reason_both is None


def test_estimate_ground_footprint_prefers_geo_meta():
    fp = _estimate_ground_footprint_m(
        bounds=(-122.1, 37.4, -122.0, 37.5),
        crs="EPSG:4326",
        geo_meta={"ground_footprint_m": (42.0, 31.0)},
    )
    assert fp == (42.0, 31.0)


def test_default_nadir_gsd_footprint_is_not_kilometer_scale(tmp_path):
    img_path = str(tmp_path / "plain.jpg")
    Image.new("RGB", (4000, 3000), color=(120, 110, 100)).save(img_path, "JPEG")
    width_m, height_m = _extract_drone_flight_telemetry(img_path, 4000, 3000)
    # 4000 * 0.025 m = 100 m, 3000 * 0.025 m = 75 m
    assert 50.0 <= width_m <= 200.0
    assert 40.0 <= height_m <= 150.0


def test_calibrate_marks_small_footprint_as_low_confidence(tmp_path):
    h, w = 32, 32
    rel = np.linspace(0.1, 0.9, h * w).reshape((h, w)).astype(np.float32)
    srtm = (1000.0 + 400.0 * rel).astype(np.float32)
    out_tif = str(tmp_path / "small_fp.tif")
    _, stats = calibrate_depth_to_metric(
        relative_depth=rel,
        srtm_elevation=srtm,
        crs="EPSG:32610",
        transform=_affine(),
        output_geotiff_path=out_tif,
        n_samples=400,
        ground_footprint_m=(40.0, 55.0),
    )
    assert stats["calibration_confidence"] == "low"


# ---------------------------------------------------------------------------
# R² gating
# ---------------------------------------------------------------------------

def test_r2_threshold_gating_low_confidence(tmp_path):
    rng = np.random.default_rng(0)
    h, w = 48, 48
    rel = rng.random((h, w)).astype(np.float32)
    srtm = rng.uniform(800.0, 900.0, size=(h, w)).astype(np.float32)
    out_tif = str(tmp_path / "uncorrelated.tif")
    _, stats = calibrate_depth_to_metric(
        relative_depth=rel,
        srtm_elevation=srtm,
        crs="EPSG:32610",
        transform=_affine(),
        output_geotiff_path=out_tif,
        n_samples=800,
        ground_footprint_m=(250.0, 250.0),
    )
    assert stats["r_squared"] < MIN_CALIBRATION_R_SQUARED
    assert stats["calibration_confidence"] == "low"


def test_r2_threshold_gating_high_confidence(tmp_path):
    h, w = 48, 48
    rel = np.linspace(0.05, 0.95, h * w).reshape((h, w)).astype(np.float32)
    srtm = (1200.0 + 500.0 * rel).astype(np.float32)
    out_tif = str(tmp_path / "correlated.tif")
    _, stats = calibrate_depth_to_metric(
        relative_depth=rel,
        srtm_elevation=srtm,
        crs="EPSG:32610",
        transform=_affine(),
        output_geotiff_path=out_tif,
        n_samples=800,
        ground_footprint_m=(250.0, 250.0),
    )
    assert stats["r_squared"] >= MIN_CALIBRATION_R_SQUARED
    assert stats["calibration_confidence"] == "high"


# ---------------------------------------------------------------------------
# Land-cover cluster mapping
# ---------------------------------------------------------------------------

def test_bright_red_rooftop_cluster_is_not_water():
    # [R, G, B, ExG]
    centers = np.array(
        [
            [0.05, 0.05, 0.05, -0.05],  # shadow
            [0.78, 0.42, 0.32, -0.26],  # bright red shingles
            [0.18, 0.62, 0.16, 0.90],   # vegetation
            [0.55, 0.54, 0.52, -0.01],  # concrete / built-up
        ],
        dtype=np.float64,
    )
    mapping = _assign_clusters_to_classes(centers)
    assert mapping[1] == CLASS_BUILT_UP
    assert CLASS_WATER not in mapping.values()


# ---------------------------------------------------------------------------
# Cross-validation rules
# ---------------------------------------------------------------------------

def _mask(h=80, w=80, fill=CLASS_BUILT_UP):
    return np.full((h, w), fill, dtype=np.uint8)


def test_ship_on_land_is_downgraded_by_penalty():
    mask = _mask(fill=CLASS_BUILT_UP)
    dets = [{"class": "Ship", "confidence": 0.61, "bbox": [10, 10, 40, 40]}]
    out = cross_validate_detections_with_land_cover(dets, mask, enabled=True, discard_mismatched=False)
    assert len(out) == 1
    assert out[0]["raw_confidence"] == 0.61
    assert out[0]["confidence"] == round(0.61 * 0.20, 3)
    assert out[0]["cross_val_status"] == "mismatch"
    assert "land-cover mismatch" in out[0]["adjustment_reason"]
    assert "water" in out[0]["adjustment_reason"].lower()


def test_ship_on_water_passes():
    mask = _mask(fill=CLASS_WATER)
    dets = [{"class": "ship", "confidence": 0.72, "bbox": [5, 5, 50, 50]}]
    out = cross_validate_detections_with_land_cover(dets, mask)
    assert out[0]["cross_val_status"] == "passed"
    assert out[0]["confidence"] == 0.72


def test_vehicle_on_built_up_passes():
    mask = _mask(fill=CLASS_BUILT_UP)
    dets = [{"class": "small vehicle", "confidence": 0.88, "bbox": [12, 12, 30, 28]}]
    out = cross_validate_detections_with_land_cover(dets, mask)
    assert out[0]["cross_val_status"] == "passed"
    assert out[0]["confidence"] == 0.88


def test_vehicle_in_deep_water_is_penalized():
    mask = _mask(fill=CLASS_WATER)
    dets = [{"class": "car", "confidence": 0.70, "bbox": [8, 8, 28, 28]}]
    out = cross_validate_detections_with_land_cover(dets, mask, discard_mismatched=False)
    assert out[0]["cross_val_status"] == "mismatch"
    assert out[0]["confidence"] == round(0.70 * 0.20, 3)


def test_discard_mismatched_detections_toggle():
    mask = _mask(fill=CLASS_BUILT_UP)
    dets = [{"class": "ship", "confidence": 0.61, "bbox": [10, 10, 40, 40]}]
    out = cross_validate_detections_with_land_cover(
        dets, mask, enabled=True, discard_mismatched=True
    )
    assert out == []


def test_disable_cross_validation_preserves_raw_confidence():
    mask = _mask(fill=CLASS_BUILT_UP)
    dets = [{"class": "ship", "confidence": 0.61, "bbox": [10, 10, 40, 40]}]
    out = cross_validate_detections_with_land_cover(dets, mask, enabled=False)
    assert len(out) == 1
    assert out[0]["confidence"] == 0.61
    assert out[0]["cross_val_status"] == "disabled"
