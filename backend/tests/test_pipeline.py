"""
Unit and Integration Tests for DepthWizard Backend Pipeline
"""

import os
import sys
import pytest
from PIL import Image
import numpy as np
import rasterio

# Ensure both workspace root and backend directory are in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
for p in (WORKSPACE_DIR, BASE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from app.sample_generator import generate_sample_datasets
    from app.geo import inspect_georeference, fetch_or_mock_srtm
    from app.depth import estimate_relative_depth, generate_depth_preview_png
    from app.calibration import calibrate_depth_to_metric
    from app.mesh import generate_terrain_mesh
    from app.validation import validate_against_reference
except ModuleNotFoundError:
    from backend.app.sample_generator import generate_sample_datasets
    from backend.app.geo import inspect_georeference, fetch_or_mock_srtm
    from backend.app.depth import estimate_relative_depth, generate_depth_preview_png
    from backend.app.calibration import calibrate_depth_to_metric
    from backend.app.mesh import generate_terrain_mesh
    from backend.app.validation import validate_against_reference



@pytest.fixture(scope="session")
def sample_files():
    return generate_sample_datasets()


def test_sample_datasets_generated(sample_files):
    assert os.path.exists(sample_files["msh_ortho"])
    assert os.path.exists(sample_files["msh_lidar"])
    assert os.path.exists(sample_files["drone_jpg"])


def test_georeference_detection(sample_files):
    # Mount St. Helens GeoTIFF should be recognized as georeferenced
    geo_info = inspect_georeference(sample_files["msh_ortho"])
    assert geo_info["is_georeferenced"] is True
    assert "32610" in str(geo_info["crs"])
    assert geo_info["bounds"] is not None

    # Alpine drone JPG should not be georeferenced
    drone_info = inspect_georeference(sample_files["drone_jpg"])
    assert drone_info["is_georeferenced"] is False


def test_depth_estimation(sample_files):
    img = Image.open(sample_files["drone_jpg"])
    rel_depth = estimate_relative_depth(img)

    assert isinstance(rel_depth, np.ndarray)
    assert rel_depth.shape == (512, 512)
    assert rel_depth.min() >= 0.0
    assert rel_depth.max() <= 1.0


def test_preview_generation(tmp_path):
    dummy_depth = np.linspace(0, 1, 100 * 100).reshape((100, 100)).astype(np.float32)
    out_png = str(tmp_path / "preview.png")
    generate_depth_preview_png(dummy_depth, out_png)
    assert os.path.exists(out_png)


def test_srtm_and_calibration(sample_files, tmp_path):
    geo_info = inspect_georeference(sample_files["msh_ortho"])
    w, h = geo_info["width"], geo_info["height"]
    
    # 1. Fetch SRTM
    srtm_grid = fetch_or_mock_srtm(geo_info["bounds"], w, h, geo_info["crs"])
    assert srtm_grid.shape == (h, w)
    assert srtm_grid.mean() > 1000.0  # realistic mountain elevation

    # 2. Mock relative depth
    img = Image.open(sample_files["msh_ortho"])
    rel_depth = estimate_relative_depth(img)

    # 3. Calibrate to metric GeoTIFF
    out_tif = str(tmp_path / "calibrated_dsm.tif")
    calibrated_dsm, cal_stats = calibrate_depth_to_metric(
        relative_depth=rel_depth,
        srtm_elevation=srtm_grid,
        crs=geo_info["crs"],
        transform=geo_info["transform"],
        output_geotiff_path=out_tif
    )

    assert os.path.exists(out_tif)
    assert calibrated_dsm.shape == (h, w)
    assert cal_stats["r_squared"] >= 0.0
    assert cal_stats["min_elevation"] > 0.0


def test_mesh_glb_export(sample_files, tmp_path):
    img = Image.open(sample_files["msh_ortho"])
    dummy_elev = np.ones((128, 128), dtype=np.float32) * 1500.0
    out_glb = str(tmp_path / "terrain.glb")

    glb_path, meta = generate_terrain_mesh(
        elevation_grid=dummy_elev,
        texture_image=img,
        output_glb_path=out_glb,
        max_resolution=64,
        is_relative=False
    )

    assert os.path.exists(glb_path)
    assert os.path.getsize(glb_path) > 1000  # Valid binary GLB
    assert "min_elevation" in meta


def test_validation_metrics(sample_files):
    with rasterio.open(sample_files["msh_lidar"]) as src:
        lidar = src.read(1)
        target_crs = src.crs.to_string()
        target_transform = src.transform

    # Add small known bias (+5m)
    test_dsm = lidar + 5.0
    val_res = validate_against_reference(
        generated_dsm=test_dsm,
        reference_file_path=sample_files["msh_lidar"],
        target_crs=target_crs,
        target_transform=target_transform
    )

    assert val_res["rmse"] == pytest.approx(5.0, abs=0.5)
    assert val_res["mae"] == pytest.approx(5.0, abs=0.5)
    assert val_res["pearson_r"] > 0.99
