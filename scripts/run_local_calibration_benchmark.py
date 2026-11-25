"""
Localized Block-Matching Calibration Benchmark
===============================================
Runs the full 21-phase benchmark: multi-scale block calibration, spatial
cross-validation, slope/elevation/vegetation analysis, error maps, and
a final comparison against the global baseline.

Outputs:
  - localized_calibration_report.json
  - error map PNGs
  - console tables
"""

import os, sys, json
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))
from app.geo import inspect_georeference
from app.main import _extract_rgb_pil
from app.depth import estimate_relative_depth
from app.semantic import differentiate_land_cover
from app.local_calibration import run_localized_calibration, spatial_cross_validate

# Paths
SAMPLE   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/sample_data/mount_st_helens_ortho.tif"))
REF      = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/sample_data/mount_st_helens_lidar_ref.tif"))
OUT_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/static/outputs/diagnostics"))


def _metrics(pred, ref, mask):
    """Compute RMSE, MAE, Pearson r, R² on masked arrays."""
    p = pred[mask].astype(np.float64)
    r = ref[mask].astype(np.float64)
    if len(p) < 10:
        return {"rmse": None, "mae": None, "pearson_r": None, "r2": None, "n": int(len(p))}
    d = p - r
    rmse = float(np.sqrt(np.mean(d ** 2)))
    mae  = float(np.mean(np.abs(d)))
    pr, _ = stats.pearsonr(p, r)
    ss_res = np.sum(d ** 2)
    ss_tot = np.sum((r - np.mean(r)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"rmse": round(rmse, 2), "mae": round(mae, 2),
            "pearson_r": round(float(pr), 4), "r2": round(float(r2), 4), "n": int(len(p))}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 80)
    print("LOCALIZED BLOCK-MATCHING CALIBRATION BENCHMARK")
    print("=" * 80)

    # ── Phase 1-2: Load & Align ──
    print("\n[PHASE 1-2] Loading data and aligning reference...")
    geo = inspect_georeference(SAMPLE)
    crs, transform = geo["crs"], geo["transform"]
    w, h = geo["width"], geo["height"]

    aligned_ref = np.zeros((h, w), dtype=np.float32)
    with rasterio.open(REF) as src:
        raw = src.read(1).astype(np.float32)
        reproject(source=raw, destination=aligned_ref,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=CRS.from_user_input(crs),
                  resampling=Resampling.bilinear)
        if src.nodata is not None:
            aligned_ref[aligned_ref == src.nodata] = np.nan

    pil_rgb = _extract_rgb_pil(SAMPLE)
    rel_depth = estimate_relative_depth(pil_rgb)
    d_min, d_max = np.nanmin(rel_depth), np.nanmax(rel_depth)
    rel_depth = (rel_depth - d_min) / (d_max - d_min + 1e-8)

    mask = np.isfinite(rel_depth) & np.isfinite(aligned_ref) & (aligned_ref > -1000)

    # ── Global Baseline ──
    print("\n[BASELINE] Computing global linear calibration...")
    x_all, y_all = rel_depth[mask].astype(np.float64), aligned_ref[mask].astype(np.float64)
    gres = stats.linregress(x_all, y_all)
    global_slope, global_intercept = float(gres.slope), float(gres.intercept)
    global_dsm = (rel_depth * global_slope + global_intercept).astype(np.float32)
    bl = _metrics(global_dsm, aligned_ref, mask)
    print(f"  Global RMSE={bl['rmse']}  MAE={bl['mae']}  r={bl['pearson_r']}  R²={bl['r2']}")

    report = {"baseline": bl}

    # ── Phase 11: Multi-Scale Experiment (250m / 500m / 1000m) ──
    print("\n[PHASE 11] Multi-scale localized calibration experiments...")
    experiments = {}
    best_scale, best_rmse = None, 1e9

    for block_m in [250, 500, 1000]:
        print(f"\n--- Block size: {block_m}m ---")
        res = run_localized_calibration(
            rel_depth, aligned_ref, transform, crs,
            block_size_m=float(block_m),
            global_slope=global_slope, global_intercept=global_intercept,
        )
        s = res["summary"]
        experiments[f"{block_m}m"] = s
        print(f"  Blocks: {s['total_blocks']} (valid={s['valid_blocks']}, rejected={s['rejected_blocks']})")
        print(f"  Coverage: {s['local_coverage_pct']}% local, {s['fallback_pct']}% fallback")
        print(f"  RMSE={s['rmse']}  MAE={s['mae']}  r={s['pearson_r']}  R²={s['r2']}")

        if s["rmse"] < best_rmse:
            best_rmse = s["rmse"]
            best_scale = block_m
            best_dsm = res["calibrated_dsm"].copy()
            best_conf = res["confidence_map"].copy()

    report["experiments"] = experiments

    # ── Phase 12: Spatial Cross-Validation ──
    print("\n[PHASE 12] Spatial cross-validation (4×4 folds)...")
    cv = spatial_cross_validate(rel_depth, aligned_ref, transform, crs,
                                block_size_m=float(best_scale), cv_grid=(4, 4))
    if cv["status"] == "success":
        print(f"  Folds: {cv['n_folds']}")
        print(f"  Spatial CV RMSE: {cv['mean_rmse']} ± {cv['std_rmse']}")
        print(f"  Spatial CV R²:   {cv['mean_r2']} ± {cv['std_r2']}")
    report["spatial_cv"] = cv

    # ── Phase 13: Elevation Range Check ──
    print("\n[PHASE 13] Elevation range comparison...")
    ref_stats = {"min": round(float(np.nanmin(y_all)), 1),
                 "max": round(float(np.nanmax(y_all)), 1),
                 "mean": round(float(np.nanmean(y_all)), 1),
                 "median": round(float(np.nanmedian(y_all)), 1)}
    gl_vals = global_dsm[mask]
    gl_stats = {"min": round(float(np.min(gl_vals)), 1),
                "max": round(float(np.max(gl_vals)), 1),
                "mean": round(float(np.mean(gl_vals)), 1),
                "median": round(float(np.median(gl_vals)), 1)}
    loc_vals = best_dsm[mask]
    loc_stats = {"min": round(float(np.nanmin(loc_vals)), 1),
                 "max": round(float(np.nanmax(loc_vals)), 1),
                 "mean": round(float(np.nanmean(loc_vals)), 1),
                 "median": round(float(np.nanmedian(loc_vals)), 1)}
    print(f"  Reference : {ref_stats}")
    print(f"  Global    : {gl_stats}")
    print(f"  Localized : {loc_stats}")
    report["elevation_range"] = {"reference": ref_stats, "global": gl_stats, "localized": loc_stats}

    # ── Phase 14: Slope Analysis ──
    print("\n[PHASE 14] Slope-dependent error analysis...")
    dy, dx = np.gradient(aligned_ref, abs(transform[4]), abs(transform[0]))
    slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    slope_report = {}
    for smin, smax in [(0, 10), (10, 20), (20, 30), (30, 45), (45, 90)]:
        sm = mask & (slope_deg >= smin) & (slope_deg < smax)
        mg = _metrics(global_dsm, aligned_ref, sm)
        ml = _metrics(best_dsm, aligned_ref, sm)
        tag = f"{smin}-{smax}deg"
        slope_report[tag] = {"global": mg, "localized": ml}
        print(f"  {tag:>10}: Global RMSE={mg['rmse']}  Local RMSE={ml['rmse']}")

    report["slope_analysis"] = slope_report

    # ── Phase 15: Vegetation Analysis ──
    print("\n[PHASE 15] Vegetation / terrain masking analysis...")
    lc_mask, _, _ = differentiate_land_cover(pil_rgb, os.path.join(OUT_DIR, "lc_bench.png"))
    veg_report = {}
    for name, cmask in [("Terrain-only", lc_mask == 0), ("Vegetation-only", lc_mask == 1),
                          ("Built-up", lc_mask == 3), ("All", np.ones_like(mask, dtype=bool))]:
        m_c = mask & cmask
        mg = _metrics(global_dsm, aligned_ref, m_c)
        ml = _metrics(best_dsm, aligned_ref, m_c)
        veg_report[name] = {"global": mg, "localized": ml}
        print(f"  {name:>18}: Global r={mg['pearson_r']}  Local r={ml['pearson_r']}")

    report["vegetation_analysis"] = veg_report

    # ── Phase 16: Error Maps ──
    print("\n[PHASE 16] Generating error maps...")
    # Global error
    ge = np.full_like(global_dsm, np.nan)
    ge[mask] = global_dsm[mask] - aligned_ref[mask]
    # Local error
    le = np.full_like(best_dsm, np.nan)
    le[mask] = best_dsm[mask] - aligned_ref[mask]

    vmax = max(np.nanpercentile(np.abs(ge), 95), 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(ge, cmap='RdBu_r', vmin=-vmax, vmax=vmax); axes[0].set_title("Global Error")
    axes[1].imshow(le, cmap='RdBu_r', vmin=-vmax, vmax=vmax); axes[1].set_title(f"Local {best_scale}m Error")
    imp = np.abs(ge) - np.abs(le)
    axes[2].imshow(imp, cmap='RdYlGn', vmin=-100, vmax=100); axes[2].set_title("Improvement (green=better)")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "global_vs_local_error.png"), dpi=150)
    plt.close()

    # Confidence map
    plt.figure(figsize=(8, 6))
    plt.imshow(best_conf, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Confidence')
    plt.title(f'Local Calibration Confidence ({best_scale}m blocks)')
    plt.axis('off')
    plt.savefig(os.path.join(OUT_DIR, "local_confidence_map.png"), dpi=150)
    plt.close()
    print(f"  Saved error maps and confidence map to {OUT_DIR}")

    # ── Phase 17: Benchmark Table ──
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK TABLE")
    print("=" * 80)
    print(f"{'Method':<20} | {'Block':>6} | {'RMSE':>8} | {'MAE':>8} | {'r':>8} | {'R²':>8}")
    print("-" * 70)
    print(f"{'Global baseline':<20} | {'—':>6} | {bl['rmse']:>8} | {bl['mae']:>8} | {bl['pearson_r']:>8} | {bl['r2']:>8}")
    for bm in [250, 500, 1000]:
        e = experiments[f"{bm}m"]
        print(f"{'Local calibration':<20} | {f'{bm}m':>6} | {e['rmse']:>8} | {e['mae']:>8} | {e['pearson_r']:>8} | {e['r2']:>8}")

    # ── Phase 19: Success Criteria ──
    best_exp = experiments[f"{best_scale}m"]
    rmse_improve = (bl['rmse'] - best_exp['rmse']) / bl['rmse'] * 100
    mae_improve  = (bl['mae']  - best_exp['mae'])  / bl['mae']  * 100
    r_reached_05 = best_exp['pearson_r'] >= 0.50

    print(f"\n{'Best block size:':<30} {best_scale}m")
    print(f"{'RMSE improvement:':<30} {rmse_improve:.1f}%")
    print(f"{'MAE improvement:':<30} {mae_improve:.1f}%")
    print(f"{'Pearson r >= 0.50?':<30} {'YES' if r_reached_05 else 'NO'} (r={best_exp['pearson_r']})")

    report["best_scale"] = best_scale
    report["rmse_improvement_pct"] = round(rmse_improve, 1)
    report["mae_improvement_pct"] = round(mae_improve, 1)
    report["pearson_r_reached_050"] = r_reached_05
    report["confidence"] = {"mode": "Relative rDSM", "confidence": "LOW"} if not r_reached_05 else {"mode": "Localized DSM", "confidence": "MODERATE"}

    # ── Save JSON report ──
    json_path = os.path.join(OUT_DIR, "localized_calibration_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to: {json_path}")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
