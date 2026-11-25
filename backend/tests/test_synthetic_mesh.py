import os
import sys
import json
import pytest
import numpy as np
from PIL import Image
import trimesh

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.mesh import generate_terrain_mesh

def create_mock_texture(w, h):
    return Image.new('RGB', (w, h), color=(100, 150, 100))

def run_mesh_test(elevation_grid, tmp_path, is_relative=False, vertical_exag=1.0, ground_footprint_m=None):
    os.makedirs(tmp_path, exist_ok=True)
    img = create_mock_texture(elevation_grid.shape[1], elevation_grid.shape[0])
    glb_path = str(tmp_path / "test.glb")
    
    out_glb, meta = generate_terrain_mesh(
        elevation_grid=elevation_grid,
        texture_image=img,
        output_glb_path=glb_path,
        max_resolution=32,
        vertical_exaggeration=vertical_exag,
        is_relative=is_relative,
        ground_footprint_m=ground_footprint_m
    )
    
    # Load GLB to inspect vertices
    scene = trimesh.load(out_glb, force='scene')
    geom = list(scene.geometry.values())[0]
    vertices = geom.vertices
    
    return meta, vertices, geom

def test_flat_terrain(tmp_path):
    # 3x3 flat terrain at 100m
    elev = np.full((3, 3), 100.0, dtype=np.float32)
    meta, vertices, _ = run_mesh_test(elev, tmp_path, is_relative=False, ground_footprint_m=(3.0, 3.0))
    
    # Check metadata
    assert meta["min_elevation"] == 100.0
    assert meta["max_elevation"] == 100.0
    assert meta["relief_span"] == 0.0
    assert meta["is_metric"] == True
    
    # Check geometry Z differences (in Three.js Y is up, which corresponds to the 2nd coordinate in the vertices)
    y_vals = vertices[:, 1]
    assert np.ptp(y_vals) == 0.0, f"Expected flat mesh, got Y variance {np.ptp(y_vals)}"

def test_constant_slope(tmp_path):
    elev = np.array([
        [100, 101, 102],
        [100, 101, 102],
        [100, 101, 102]
    ], dtype=np.float32)
    
    # 3m width means 3 horizontal steps -> roughly 1m step horizontally
    meta, vertices, geom = run_mesh_test(elev, tmp_path, is_relative=False, vertical_exag=1.0, ground_footprint_m=(3.0, 3.0))
    
    y_vals = vertices[:, 1]
    y_min, y_max = np.min(y_vals), np.max(y_vals)
    
    assert meta["relief_span"] == 2.0
    
    # For a slope of 2m difference with vertical_exag=1.0, the Y range should be exactly 2m * scale
    # If metric_scale_factor is 1.0 (since footprint 3.0 < 500.0), Y range should be 2.0
    assert np.isclose(meta["metric_scale_factor"], 1.0)
    assert np.isclose(y_max - y_min, 2.0), f"Metric elevation difference not preserved at 1x exaggeration. Got {y_max - y_min}, expected 2.0"
    
def test_vertical_exaggeration(tmp_path):
    elev = np.array([
        [10, 10, 10, 10],
        [10, 11, 12, 13],
        [10, 12, 14, 16],
        [10, 13, 16, 19]
    ], dtype=np.float32)
    
    meta1, vertices1, _ = run_mesh_test(elev, tmp_path / "1x", is_relative=False, vertical_exag=1.0, ground_footprint_m=(4.0, 4.0))
    y_vals1 = vertices1[:, 1]
    
    meta2, vertices2, _ = run_mesh_test(elev, tmp_path / "2x", is_relative=False, vertical_exag=2.0, ground_footprint_m=(4.0, 4.0))
    y_vals2 = vertices2[:, 1]
    
    # At 1x, range is 9m
    assert np.isclose(np.max(y_vals1) - np.min(y_vals1), 9.0)
    
    # At 2x, range should be exactly double
    assert np.isclose(np.max(y_vals2) - np.min(y_vals2), 18.0)

def test_nodata_handling(tmp_path):
    elev = np.array([
        [100, 100, 100],
        [100, -9999, 100],
        [100, 100, 100]
    ], dtype=np.float32)
    
    meta, vertices, _ = run_mesh_test(elev, tmp_path, is_relative=False, ground_footprint_m=(3.0, 3.0))
    
    y_vals = vertices[:, 1]
    
    # NoData (-9999) should be replaced with the valid min (100 in this case)
    # so there should be no huge negative spikes
    assert np.min(y_vals) >= 0.0
    assert meta["min_elevation"] == 100.0

def test_hill_and_valley(tmp_path):
    hill = np.array([
        [100, 101, 100],
        [101, 105, 101],
        [100, 101, 100]
    ], dtype=np.float32)
    
    valley = np.array([
        [105, 101, 105],
        [101, 100, 101],
        [105, 101, 105]
    ], dtype=np.float32)
    
    m_hill, v_hill, _ = run_mesh_test(hill, tmp_path / "hill", is_relative=False, ground_footprint_m=(3.0, 3.0))
    m_valley, v_valley, _ = run_mesh_test(valley, tmp_path / "valley", is_relative=False, ground_footprint_m=(3.0, 3.0))
    
    assert m_hill["max_elevation"] == 105.0
    assert m_valley["min_elevation"] == 100.0
    
    # Y range for hill is 5.0
    assert np.isclose(np.ptp(v_hill[:, 1]), 5.0)
    # Y range for valley is 5.0
    assert np.isclose(np.ptp(v_valley[:, 1]), 5.0)

def test_y_orientation(tmp_path):
    elev = np.array([
        [100, 100, 100],
        [100, 100, 100],
        [150, 150, 150]
    ], dtype=np.float32)
    
    m, v, geom = run_mesh_test(elev, tmp_path, is_relative=False, ground_footprint_m=(3.0, 3.0))
    
    # The bottom of the image (index 2) is 150.
    # In mesh space, WebGL maps image space such that Z grows towards the viewer (positive Z is bottom of image).
    # This ensures that the texture draping aligns correctly.
    # Let's check the highest Y vertices and find their Z coordinate.
    y_vals = v[:, 1]
    z_vals = v[:, 2]
    
    max_y_mask = y_vals > 49.0
    # The highest Y vertices should be at the positive Z boundary
    assert np.all(z_vals[max_y_mask] > 0)


def test_relative_exaggeration_uses_single_display_scale(tmp_path):
    elev = np.array([
        [0.00, 0.20, 0.40],
        [0.10, 0.50, 0.70],
        [0.30, 0.60, 1.00],
    ], dtype=np.float32)

    img = Image.new("RGB", (3, 3), color=(50, 50, 50))
    glb_a = str(tmp_path / "relative_1_25.glb")
    glb_b = str(tmp_path / "relative_2_00.glb")

    _, meta_a = generate_terrain_mesh(
        elevation_grid=elev,
        texture_image=img,
        output_glb_path=glb_a,
        max_resolution=32,
        vertical_exaggeration=1.0,
        is_relative=True,
        relative_exaggeration=1.25,
    )
    _, meta_b = generate_terrain_mesh(
        elevation_grid=elev,
        texture_image=img,
        output_glb_path=glb_b,
        max_resolution=32,
        vertical_exaggeration=1.0,
        is_relative=True,
        relative_exaggeration=2.0,
    )

    assert meta_a["relative_exaggeration"] == pytest.approx(1.25)
    assert meta_b["relative_exaggeration"] == pytest.approx(2.0)
    assert meta_b["relative_exaggeration"] / meta_a["relative_exaggeration"] == pytest.approx(2.0 / 1.25, rel=1e-3)


def test_relative_scene_exports_independent_building_node(tmp_path):
    elev = np.zeros((32, 32), dtype=np.float32)
    texture = Image.new("RGB", (32, 32), color=(120, 120, 120))
    glb_path = str(tmp_path / "scene_with_building.glb")
    generate_terrain_mesh(
        elevation_grid=elev,
        texture_image=texture,
        output_glb_path=glb_path,
        max_resolution=32,
        is_relative=True,
        relative_exaggeration=1.25,
        structure_objects=[{
            "id": "Building_001",
            "footprint": [[8, 8], [24, 8], [24, 24], [8, 24]],
            "local_ground": 0.0,
            "roof_elevation": 0.2,
        }],
    )

    scene = trimesh.load(glb_path, force="scene")
    assert "Terrain" in scene.geometry
    assert "Building_001" in scene.geometry
    terrain = scene.geometry["Terrain"]
    building = scene.geometry["Building_001"]
    assert np.ptp(terrain.vertices[:, 1]) == pytest.approx(0.0)
    assert np.ptp(building.vertices[:, 1]) == pytest.approx(0.625)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
