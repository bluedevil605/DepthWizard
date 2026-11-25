"""
Sample Data Generator for DepthWizard
Creates realistic sample datasets for instant 1-click offline hackathon demonstrations:
1. Mount St. Helens Crater (Georeferenced GeoTIFF optical orthoimage)
2. Mount St. Helens LiDAR Reference (Ground truth survey DEM for validation)
3. Alpine Drone Flight (Non-georeferenced optical JPEG for relative rDSM mode)
"""

import os
import math
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

SAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sample_data"))
os.makedirs(SAMPLE_DIR, exist_ok=True)


def generate_sample_datasets():
    """Generates offline demonstration datasets if they do not already exist."""
    
    # -------------------------------------------------------------
    # 1. Mount St. Helens GeoTIFF (512x512) + Paired LiDAR Reference
    # -------------------------------------------------------------
    msh_ortho_path = os.path.join(SAMPLE_DIR, "mount_st_helens_ortho.tif")
    msh_lidar_path = os.path.join(SAMPLE_DIR, "mount_st_helens_lidar_ref.tif")
    
    # Bounding box around Mount St. Helens caldera in UTM Zone 10N (meters)
    # Approx: 560000 to 565120 Easting, 5113000 to 5118120 Northing
    w, h = 512, 512
    min_x, max_x = 560000.0, 565120.0
    min_y, max_y = 5113000.0, 5118120.0
    crs = "EPSG:32610"
    transform = from_bounds(min_x, min_y, max_x, max_y, w, h)
    
    # Construct volcanic crater topography: caldera rim + central dome + slopes
    ys = np.linspace(-3.0, 3.0, h)
    xs = np.linspace(-3.0, 3.0, w)
    xv, yv = np.meshgrid(xs, ys)
    r = np.sqrt(xv**2 + yv**2)
    
    # Outer cone slopes rising up to rim at r=1.5
    outer_cone = 2500.0 - np.clip(r - 1.5, 0.0, 5.0) * 450.0
    # Crater bowl collapsing inside r < 1.5 down to 1800m
    crater_bowl = np.where(r < 1.5, 1850.0 + (r**2) * 200.0, outer_cone)
    # Central resurgent lava dome inside the crater
    dome_dist = np.sqrt((xv - 0.2)**2 + (yv + 0.3)**2)
    lava_dome = np.exp(-dome_dist**2 / 0.15) * 350.0
    # Secondary ridges and valleys
    ridges = np.sin(xv * 4.0 + yv * 2.0) * 60.0 + np.cos(yv * 5.0) * 40.0
    
    ground_truth_dem = (crater_bowl + lava_dome + ridges).astype(np.float32)
    
    # Create realistic optical RGB orthoimage by simulating sun illumination & terrain shading
    # Sun direction: azimuth 315 deg, elevation 45 deg
    gy, gx = np.gradient(ground_truth_dem)
    slope = np.sqrt(gx**2 + gy**2)
    aspect = np.arctan2(-gx, gy)
    
    sun_az = math.radians(315.0)
    sun_el = math.radians(45.0)
    hillshade = np.sin(sun_el) + np.cos(sun_el) * (gx * math.cos(sun_az) + gy * math.sin(sun_az)) / (1.0 + slope * 0.1)
    hillshade = np.clip((hillshade + 0.8) / 1.8, 0.2, 1.0)
    
    # Color palette based on elevation and slope:
    # High crater rim = volcanic ash/rock (cool slate grey)
    # Central dome = fresh dark dacite basalt (charcoal)
    # Lower outer slopes = alpine fir forest (dark evergreen) & meadows (olive green)
    base_r = np.zeros((h, w), dtype=np.float32)
    base_g = np.zeros((h, w), dtype=np.float32)
    base_b = np.zeros((h, w), dtype=np.float32)
    
    # Ash/rock (crater floor & rim)
    rock_mask = (r < 1.6) | (ground_truth_dem > 2200.0)
    base_r[rock_mask] = 160.0
    base_g[rock_mask] = 155.0
    base_b[rock_mask] = 150.0
    
    # Dark lava dome
    dome_mask = lava_dome > 80.0
    base_r[dome_mask] = 75.0
    base_g[dome_mask] = 70.0
    base_b[dome_mask] = 68.0
    
    # Forest & alpine vegetation on outer lower slopes
    veg_mask = ~rock_mask & ~dome_mask
    base_r[veg_mask] = 60.0 + np.sin(xv[veg_mask]*10) * 15.0
    base_g[veg_mask] = 110.0 + np.cos(yv[veg_mask]*10) * 20.0
    base_b[veg_mask] = 65.0
    
    # Apply hillshade lighting
    final_r = np.clip(base_r * hillshade, 0, 255).astype(np.uint8)
    final_g = np.clip(base_g * hillshade, 0, 255).astype(np.uint8)
    final_b = np.clip(base_b * hillshade, 0, 255).astype(np.uint8)
    
    # Save optical GeoTIFF (3 bands RGB)
    with rasterio.open(
        msh_ortho_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=3,
        dtype=np.uint8,
        crs=CRS.from_string(crs),
        transform=transform
    ) as dst:
        dst.write(final_r, 1)
        dst.write(final_g, 2)
        dst.write(final_b, 3)
        
    # Save paired ground truth LiDAR DEM (1 band float32)
    # Add subtle high-resolution survey noise (+/- 0.5m)
    rng = np.random.default_rng(123)
    lidar_dem = ground_truth_dem + rng.normal(0, 0.4, (h, w)).astype(np.float32)
    with rasterio.open(
        msh_lidar_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=np.float32,
        crs=CRS.from_string(crs),
        transform=transform,
        nodata=-9999.0
    ) as dst:
        dst.write(lidar_dem, 1)
        
    # -------------------------------------------------------------
    # 2. Non-Georeferenced Drone Optical Photo (JPEG)
    # -------------------------------------------------------------
    drone_path = os.path.join(SAMPLE_DIR, "alpine_drone_flight.jpg")
    # Procedural terraced canyon quarry
    canyon_ys = np.linspace(-2.0, 2.0, 512)
    canyon_xs = np.linspace(-2.0, 2.0, 512)
    cxv, cyv = np.meshgrid(canyon_xs, canyon_ys)
    terraces = np.sin(cxv * 3.0) * 80.0 + np.sin(cyv * 2.5) * 60.0
    cliff_shade = np.clip((np.sin(cxv * 8.0) + np.cos(cyv * 6.0) + 1.5) / 3.0, 0.3, 1.0)
    
    dr = np.clip((180 + terraces * 0.4) * cliff_shade, 0, 255).astype(np.uint8)
    dg = np.clip((140 + terraces * 0.3) * cliff_shade, 0, 255).astype(np.uint8)
    db = np.clip((110 + terraces * 0.2) * cliff_shade, 0, 255).astype(np.uint8)
    
    drone_rgb = np.stack([dr, dg, db], axis=-1)
    drone_img = Image.fromarray(drone_rgb, mode="RGB")
    drone_img.save(drone_path, "JPEG", quality=95)
    
    return {
        "msh_ortho": msh_ortho_path,
        "msh_lidar": msh_lidar_path,
        "drone_jpg": drone_path
    }


if __name__ == "__main__":
    datasets = generate_sample_datasets()
    print("Generated sample datasets:", datasets)
