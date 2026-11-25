import os
import sys
import numpy as np
import rasterio

# Ensure backend can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from app.geo import inspect_georeference
from app.main import _extract_rgb_pil
from app.depth import estimate_relative_depth
from app.semantic import differentiate_land_cover
from validation.dsm_evaluation import (
    evaluate_multiscale, 
    evaluate_spatial_cross_validation, 
    run_calibration_experiments,
    evaluate_terrain_masking,
    evaluate_depth_preprocessing,
    generate_error_map
)

def run_experiments():
    sample_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/sample_data/mount_st_helens_ortho.tif"))
    ref_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/sample_data/mount_st_helens_lidar_ref.tif"))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/static/outputs/diagnostics"))
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 80)
    print("DSM ESTIMATION & VALIDATION PIPELINE: EXPERIMENT RUNNER")
    print("Target: Mount St. Helens (Hilly/Forested)")
    print("=" * 80)

    # 1. Load optical data & georeference
    print("\n[1] Fetching target extent and loading LiDAR reference...")
    geo_meta = inspect_georeference(sample_path)
    crs = geo_meta['crs']
    transform = geo_meta['transform']
    w, h = geo_meta['width'], geo_meta['height']
    
    if not os.path.exists(ref_path):
        print(f"ERROR: High-resolution LiDAR reference not found at {ref_path}")
        return
        
    # Read aligned reference directly (we assume the LiDAR has been prepared matching the optical TIFF)
    # But wait! We should use rasterio.warp.reproject to be safe.
    aligned_ref = np.zeros((h, w), dtype=np.float32)
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS
    with rasterio.open(ref_path) as ref_src:
        ref_data = ref_src.read(1).astype(np.float32)
        reproject(
            source=ref_data,
            destination=aligned_ref,
            src_transform=ref_src.transform,
            src_crs=ref_src.crs,
            dst_transform=transform,
            dst_crs=CRS.from_user_input(crs),
            resampling=Resampling.bilinear
        )
        if ref_src.nodata is not None:
            aligned_ref[aligned_ref == ref_src.nodata] = np.nan
    
    # 2. Depth & Semantic Processing
    print("\n[2] Estimating relative depth and land cover (ExG/KMeans)...")
    pil_rgb = _extract_rgb_pil(sample_path)
    rel_depth = estimate_relative_depth(pil_rgb)
    
    # Normalizing just in case
    d_min, d_max = np.nanmin(rel_depth), np.nanmax(rel_depth)
    rel_depth = (rel_depth - d_min) / (d_max - d_min + 1e-8)
    
    # Semantic mask
    lc_mask, _, _ = differentiate_land_cover(pil_rgb, os.path.join(out_dir, "dummy_lc.png"))
    
    # Global Linear Calibration
    mask = np.isfinite(rel_depth) & np.isfinite(aligned_ref) & (aligned_ref > -1000)
    x = rel_depth[mask]
    y = aligned_ref[mask]
    
    from scipy import stats
    slope, intercept, _, _, _ = stats.linregress(x, y)
    predicted_dsm = rel_depth * slope + intercept
    
    # 3. Multi-Scale Evaluation
    print("\n[3] Running Multi-Scale Validation...")
    ms_results = evaluate_multiscale(predicted_dsm, ref_path, crs, transform, scales=[1, 5, 10])
    print(f"\n--- Terrain Performance Table ---")
    print(f"{'Scale':<10} | {'RMSE (m)':<10} | {'MAE (m)':<10} | {'R² Score':<10} | {'Coverage':<10}")
    print("-" * 65)
    for scale in [1, 5, 10]:
        res = ms_results.get(f"{scale}x", {})
        print(f"{scale}x{scale:<8} | {res.get('rmse', 'N/A'):<10} | {res.get('mae', 'N/A'):<10} | {res.get('r_squared', 'N/A'):<10} | {res.get('valid_percentage', 0)}%")
        
    # 4. Spatial Cross-Validation
    print("\n[4] Running Spatial Cross-Validation (4x4 blocks)...")
    cv_results = evaluate_spatial_cross_validation(predicted_dsm, ref_path, crs, transform, grid_size=(4, 4))
    if cv_results.get("status") == "success":
        print(f"Blocks evaluated: {cv_results['num_valid_blocks']} / 16")
        print(f"Mean RMSE: {cv_results['mean_rmse']}m ± {cv_results['std_rmse']}m")
        print(f"Mean R²:   {cv_results['mean_r2']} ± {cv_results['std_r2']}")
        print(f"R² Range:  [{cv_results['min_r2']}, {cv_results['max_r2']}]")
        
    # 5. Depth Preprocessing Experiments
    print("\n[5] Running Depth Preprocessing Experiments...")
    pre_results = evaluate_depth_preprocessing(rel_depth, aligned_ref)
    
    # 6. Terrain Masking Experiments
    print("\n[6] Running Terrain Masking Validation...")
    mask_results = evaluate_terrain_masking(predicted_dsm, aligned_ref, lc_mask)
    
    # 7. Calibration Experiments
    print("\n[7] Running Calibration Strategy Experiments...")
    calib_results = run_calibration_experiments(rel_depth, aligned_ref)
    
    # Print unified Experiment Comparison Table
    print(f"\n--- Experiment Comparison Table ---")
    print(f"{'Experiment Type':<25} | {'Strategy/Mask':<25} | {'RMSE (m)':<10} | {'R² Score':<10}")
    print("-" * 80)
    for strategy, metrics in pre_results.items():
        if strategy == "error": continue
        print(f"{'Depth Preprocessing':<25} | {strategy:<25} | {metrics.get('rmse', 'N/A'):<10} | {metrics.get('r_squared', 'N/A'):<10}")
    
    for strategy, metrics in mask_results.items():
        if strategy == "error": continue
        print(f"{'Terrain Masking':<25} | {strategy:<25} | {metrics.get('rmse', 'N/A'):<10} | {metrics.get('r_squared', 'N/A'):<10}")

    for strategy, metrics in calib_results.items():
        if strategy == "error": continue
        print(f"{'Robust Calibration':<25} | {strategy:<25} | {metrics.get('rmse', 'N/A'):<10} | {metrics.get('r_squared', 'N/A'):<10}")
        
    # 8. Error Mapping
    print("\n[8] Generating Visual Error Maps...")
    err_map_path = os.path.join(out_dir, "error_map.png")
    generate_error_map(predicted_dsm, aligned_ref, err_map_path)
    print(f"Saved error map to {err_map_path}")
    
    print(f"\n--- Final Benchmark Table ---")
    print(f"{'Landscape':<15} | {'RMSE (m)':<10} | {'MAE (m)':<10} | {'Correlation':<12} | {'R²':<8} | {'Coverage':<10}")
    print("-" * 75)
    
    # We only have Hilly/Forested (Mount St. Helens)
    s1x = ms_results.get("1x", {})
    from scipy import stats
    pr, _ = stats.pearsonr(x, y)
    print(f"{'Forested/Hilly':<15} | {s1x.get('rmse', 'N/A'):<10} | {s1x.get('mae', 'N/A'):<10} | {round(pr, 4):<12} | {s1x.get('r_squared', 'N/A'):<8} | {s1x.get('valid_percentage', 0):<5}%")
    print(f"{'Urban':<15} | {'NOT AVAILABLE':<10} | {'NOT AVAILABLE':<10} | {'NOT AVAILABLE':<12} | {'N/A':<8} | {'N/A':<5}")
    print(f"{'Sparse':<15} | {'NOT AVAILABLE':<10} | {'NOT AVAILABLE':<10} | {'NOT AVAILABLE':<12} | {'N/A':<8} | {'N/A':<5}")

    print("\n================================================================================")
    print("EVALUATION COMPLETE")
    print("================================================================================")

if __name__ == "__main__":
    run_experiments()
