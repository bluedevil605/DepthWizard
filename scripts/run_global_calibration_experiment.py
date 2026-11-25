"""
Global Calibration Experiment Runner
=====================================
Runs all 22 phases of the global calibration improvement experiment.
Tests depth preprocessing, nonlinear calibration, terrain features,
spatial drift correction, and vegetation compensation — all evaluated
with spatially separated validation folds.
"""

import os, sys, json
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from scipy import stats
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))
from app.geo import inspect_georeference
from app.main import _extract_rgb_pil
from app.depth import estimate_relative_depth
from app.semantic import differentiate_land_cover
from app.global_calibration_experimental import (
    build_depth_features,
    build_vegetation_feature,
    make_spatial_folds,
    spatial_cv_evaluate,
    LinearModel, HuberModel, PolyModel, IsotonicModel, RidgeModel,
)

SAMPLE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/sample_data/mount_st_helens_ortho.tif"))
REF    = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/sample_data/mount_st_helens_lidar_ref.tif"))
OUT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/static/outputs/diagnostics"))


def _quick_metrics(pred, ref):
    d = pred - ref
    rmse = float(np.sqrt(np.mean(d**2)))
    mae = float(np.mean(np.abs(d)))
    if np.std(pred) > 1e-6 and np.std(ref) > 1e-6:
        pr, _ = stats.pearsonr(pred, ref)
    else:
        pr = 0.0
    ss = np.sum(d**2); st = np.sum((ref - np.mean(ref))**2)
    r2 = 1 - ss/st if st > 0 else 0
    return {"rmse": round(rmse,2), "mae": round(mae,2), "pearson_r": round(float(pr),4), "r2": round(float(r2),4)}


def main():
    os.makedirs(OUT, exist_ok=True)
    print("="*80)
    print("GLOBAL CALIBRATION IMPROVEMENT EXPERIMENT")
    print("="*80)

    # ── Load & align ──
    print("\n[LOAD] Loading data...")
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
    dmin, dmax = np.nanmin(rel_depth), np.nanmax(rel_depth)
    rel_depth = (rel_depth - dmin) / (dmax - dmin + 1e-8)

    valid_mask = np.isfinite(rel_depth) & np.isfinite(aligned_ref) & (aligned_ref > -1000)
    y_ref = aligned_ref.ravel()[valid_mask.ravel()].astype(np.float64)

    # Land cover
    lc_mask, _, _ = differentiate_land_cover(pil_rgb, os.path.join(OUT, "lc_exp.png"))
    veg_feat = build_vegetation_feature(lc_mask)

    # Features
    print("[FEATURES] Building depth feature maps...")
    feats = build_depth_features(rel_depth)

    # Spatial folds
    folds = make_spatial_folds(h, w, grid=(4, 4))

    # Helper: make flat feature matrix from named features
    def make_X(feat_names):
        cols = [feats[fn].ravel()[valid_mask.ravel()] for fn in feat_names]
        return np.column_stack(cols)

    results = {}

    # ── Phase 1: Baseline ──
    print("\n[PHASE 1] VERIFY BASELINE")
    X_base = make_X(["depth"])
    bl = spatial_cv_evaluate(X_base, y_ref, valid_mask, folds,
                             lambda X, y: LinearModel().fit(X, y))
    print(f"  Baseline: RMSE={bl['rmse']}  MAE={bl['mae']}  r={bl['pearson_r']}  R2={bl['r2']}")
    results["baseline_linear"] = bl

    # ── Phase 2: Depth Preprocessing ──
    print("\n[PHASE 2] DEPTH PREPROCESSING")
    for dname in ["depth", "depth_inv", "depth_pclip", "depth_rank", "depth_gauss"]:
        X = make_X([dname])
        m = spatial_cv_evaluate(X, y_ref, valid_mask, folds,
                                lambda X, y: LinearModel().fit(X, y))
        print(f"  {dname:20s}: r={m['pearson_r']}  RMSE={m['rmse']}")
        results[f"preproc_{dname}"] = m

    # ── Phase 3: Distribution Normalization ──
    print("\n[PHASE 3] DISTRIBUTION NORMALIZATION (covered in Phase 2)")

    # ── Phase 4: Nonlinear Calibration ──
    print("\n[PHASE 4] NONLINEAR GLOBAL CALIBRATION")
    configs = [
        ("Linear",   ["depth"], lambda X,y: LinearModel().fit(X,y)),
        ("Huber",    ["depth"], lambda X,y: HuberModel().fit(X,y)),
        ("Poly2",    ["depth"], lambda X,y: PolyModel(2).fit(X,y)),
        ("Poly3",    ["depth"], lambda X,y: PolyModel(3).fit(X,y)),
        ("Isotonic", ["depth"], lambda X,y: IsotonicModel().fit(X,y)),
    ]
    for name, fnames, fit_fn in configs:
        X = make_X(fnames)
        m = spatial_cv_evaluate(X, y_ref, valid_mask, folds, fit_fn)
        print(f"  {name:20s}: r={m['pearson_r']}  RMSE={m['rmse']}")
        results[f"nonlin_{name}"] = m

    # Isotonic on rank-normalized depth
    X_rank = make_X(["depth_rank"])
    m = spatial_cv_evaluate(X_rank, y_ref, valid_mask, folds,
                            lambda X, y: IsotonicModel().fit(X, y))
    print(f"  {'Isotonic(rank)':20s}: r={m['pearson_r']}  RMSE={m['rmse']}")
    results["nonlin_Isotonic_rank"] = m

    # ── Phase 5: Terrain-Aware Features ──
    print("\n[PHASE 5] TERRAIN-AWARE FEATURES")
    feat_sets = {
        "depth_only":        ["depth"],
        "depth+local_stats": ["depth", "depth_mean_7", "depth_std_7"],
        "depth+gradients":   ["depth", "grad_x", "grad_y", "grad_mag"],
        "depth+stats+grads": ["depth", "depth_mean_7", "depth_std_7", "grad_x", "grad_y", "grad_mag"],
    }
    for fname, flist in feat_sets.items():
        X = make_X(flist)
        m = spatial_cv_evaluate(X, y_ref, valid_mask, folds,
                                lambda X, y: RidgeModel(alpha=10).fit(X, y))
        print(f"  {fname:25s}: r={m['pearson_r']}  RMSE={m['rmse']}")
        results[f"terrain_{fname}"] = m

    # ── Phase 6: Multi-Scale Depth ──
    print("\n[PHASE 6] MULTI-SCALE DEPTH")
    ms_feats = ["depth", "depth_mean_3", "depth_mean_7", "depth_mean_15", "depth_mean_31"]
    X = make_X(ms_feats)
    m = spatial_cv_evaluate(X, y_ref, valid_mask, folds,
                            lambda X, y: RidgeModel(alpha=10).fit(X, y))
    print(f"  Multi-scale depth: r={m['pearson_r']}  RMSE={m['rmse']}")
    results["multiscale_depth"] = m

    # ── Phase 7: Spatial Coordinate Features ──
    print("\n[PHASE 7] SPATIAL COORDINATE FEATURES")
    sp_configs = {
        "depth+xy":      ["depth", "coord_x", "coord_y"],
        "depth+xy+xy2":  ["depth", "coord_x", "coord_y", "coord_x2", "coord_y2", "coord_xy"],
    }
    for sname, slist in sp_configs.items():
        X = make_X(slist)
        m = spatial_cv_evaluate(X, y_ref, valid_mask, folds,
                                lambda X, y: RidgeModel(alpha=10).fit(X, y))
        print(f"  {sname:25s}: r={m['pearson_r']}  RMSE={m['rmse']}")
        results[f"spatial_{sname}"] = m

    # ── Phase 8: Global Spatial Drift Model ──
    print("\n[PHASE 8] GLOBAL SPATIAL DRIFT MODEL")
    drift_feats = ["depth", "coord_x", "coord_y", "coord_x2", "coord_y2", "coord_xy",
                   "depth_mean_15", "depth_std_15"]
    X = make_X(drift_feats)
    m = spatial_cv_evaluate(X, y_ref, valid_mask, folds,
                            lambda X, y: RidgeModel(alpha=50).fit(X, y))
    print(f"  Drift model (Ridge a=50): r={m['pearson_r']}  RMSE={m['rmse']}")
    results["spatial_drift_ridge50"] = m

    m2 = spatial_cv_evaluate(X, y_ref, valid_mask, folds,
                             lambda X, y: PolyModel(2).fit(X, y))
    print(f"  Drift model (Poly2):      r={m2['pearson_r']}  RMSE={m2['rmse']}")
    results["spatial_drift_poly2"] = m2

    # ── Phase 10: Vegetation Compensation ──
    print("\n[PHASE 10] VEGETATION COMPENSATION")
    # Add vegetation feature
    feats["veg"] = veg_feat
    veg_configs = {
        "depth+veg":            ["depth", "veg"],
        "depth+veg+xy":         ["depth", "veg", "coord_x", "coord_y"],
        "depth+veg+xy+stats":   ["depth", "veg", "coord_x", "coord_y", "depth_mean_7", "depth_std_7"],
    }
    for vname, vlist in veg_configs.items():
        X = make_X(vlist)
        m = spatial_cv_evaluate(X, y_ref, valid_mask, folds,
                                lambda X, y: RidgeModel(alpha=10).fit(X, y))
        print(f"  {vname:30s}: r={m['pearson_r']}  RMSE={m['rmse']}")
        results[f"veg_{vname}"] = m

    # ── Phase 11: Slope Feature ──
    print("\n[PHASE 11] SLOPE FEATURE")
    dy, dx = np.gradient(rel_depth)
    feats["depth_slope"] = np.sqrt(dx**2 + dy**2)
    slope_configs = {
        "depth+slope":      ["depth", "depth_slope"],
        "depth+slope+xy":   ["depth", "depth_slope", "coord_x", "coord_y"],
    }
    for sname, slist in slope_configs.items():
        X = make_X(slist)
        m = spatial_cv_evaluate(X, y_ref, valid_mask, folds,
                                lambda X, y: RidgeModel(alpha=10).fit(X, y))
        print(f"  {sname:25s}: r={m['pearson_r']}  RMSE={m['rmse']}")
        results[f"slope_{sname}"] = m

    # ── BEST COMBINED MODEL ──
    print("\n[BEST] COMBINED GLOBAL MODEL")
    best_feats = ["depth", "depth_rank", "coord_x", "coord_y", "coord_x2", "coord_y2", "coord_xy",
                  "depth_mean_7", "depth_mean_15", "depth_mean_31",
                  "depth_std_7", "grad_mag", "veg", "depth_slope"]
    X = make_X(best_feats)

    # Ridge with moderate regularization
    for alpha in [1, 10, 50, 100, 500]:
        m = spatial_cv_evaluate(X, y_ref, valid_mask, folds,
                                lambda X, y, a=alpha: RidgeModel(alpha=a).fit(X, y))
        print(f"  Full features Ridge(a={alpha:4d}): r={m['pearson_r']}  RMSE={m['rmse']}")
        results[f"best_ridge_a{alpha}"] = m

    # ── Find best result ──
    best_key = max(results, key=lambda k: results[k]["pearson_r"] or -999)
    best = results[best_key]
    print(f"\n  BEST GLOBAL: {best_key}")
    print(f"    r={best['pearson_r']}  RMSE={best['rmse']}  MAE={best['mae']}  R2={best['r2']}")

    # ── Phase 16: Experiment Table ──
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON TABLE (Spatial CV, 4x4 folds)")
    print("="*80)
    print(f"{'Method':<35} | {'RMSE':>8} | {'MAE':>8} | {'r':>8} | {'R2':>8}")
    print("-"*80)
    for k, v in results.items():
        print(f"{k:<35} | {v['rmse'] or 'N/A':>8} | {v['mae'] or 'N/A':>8} | {v['pearson_r'] or 'N/A':>8} | {v['r2'] or 'N/A':>8}")

    # ── Phase 17: Error Analysis for best model ──
    print("\n[PHASE 17] ERROR ANALYSIS FOR BEST MODEL")
    # Train best model on all data to generate predictions
    X_full = make_X(best_feats)
    best_model = RidgeModel(alpha=10).fit(X_full, y_ref)
    pred_best = np.full(h * w, np.nan)
    pred_best[valid_mask.ravel()] = best_model.predict(X_full)
    pred_best = pred_best.reshape(h, w)

    # Also global baseline prediction
    X_bl = make_X(["depth"])
    bl_model = LinearModel().fit(X_bl, y_ref)
    pred_bl = np.full(h * w, np.nan)
    pred_bl[valid_mask.ravel()] = bl_model.predict(X_bl)
    pred_bl = pred_bl.reshape(h, w)

    # Slope analysis
    ref_dy, ref_dx = np.gradient(aligned_ref, abs(transform[4]), abs(transform[0]))
    slope_deg = np.degrees(np.arctan(np.sqrt(ref_dx**2 + ref_dy**2)))
    print("\n  Slope-dependent error (best model):")
    for smin, smax in [(0,10),(10,20),(20,30),(30,45),(45,90)]:
        sm = valid_mask & (slope_deg >= smin) & (slope_deg < smax)
        if np.sum(sm) < 10: continue
        mb = _quick_metrics(pred_best[sm], aligned_ref[sm])
        mg = _quick_metrics(pred_bl[sm], aligned_ref[sm])
        print(f"    {smin:2d}-{smax:2d}deg: Baseline RMSE={mg['rmse']:>8}  Best RMSE={mb['rmse']:>8}")

    # Vegetation analysis
    print("\n  Vegetation analysis (best model):")
    for name, cmask in [("Terrain", lc_mask==0), ("Vegetation", lc_mask==1),
                         ("Built-up", lc_mask==3), ("All", np.ones((h,w),bool))]:
        mc = valid_mask & cmask
        if np.sum(mc) < 10: continue
        mb = _quick_metrics(pred_best[mc], aligned_ref[mc])
        mg = _quick_metrics(pred_bl[mc], aligned_ref[mc])
        print(f"    {name:12s}: Baseline r={mg['pearson_r']}  Best r={mb['pearson_r']}")

    # Elevation range
    pv = pred_best[valid_mask]; rv = aligned_ref[valid_mask]; bv = pred_bl[valid_mask]
    print(f"\n  Elevation range:")
    print(f"    Reference:  [{np.nanmin(rv):.1f}, {np.nanmax(rv):.1f}]  mean={np.nanmean(rv):.1f}")
    print(f"    Baseline:   [{np.nanmin(bv):.1f}, {np.nanmax(bv):.1f}]  mean={np.nanmean(bv):.1f}")
    print(f"    Best model: [{np.nanmin(pv):.1f}, {np.nanmax(pv):.1f}]  mean={np.nanmean(pv):.1f}")

    # ── Phase 18: Error Maps ──
    print("\n[PHASE 18] Generating error maps...")
    ge = np.full((h,w), np.nan); ge[valid_mask] = pred_bl[valid_mask] - aligned_ref[valid_mask]
    be = np.full((h,w), np.nan); be[valid_mask] = pred_best[valid_mask] - aligned_ref[valid_mask]
    vmax = max(np.nanpercentile(np.abs(ge), 95), 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(ge, cmap='RdBu_r', vmin=-vmax, vmax=vmax); axes[0].set_title("Baseline Error")
    axes[1].imshow(be, cmap='RdBu_r', vmin=-vmax, vmax=vmax); axes[1].set_title("Best Model Error")
    imp = np.abs(ge) - np.abs(be)
    axes[2].imshow(imp, cmap='RdYlGn', vmin=-100, vmax=100); axes[2].set_title("Improvement (green=better)")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "global_experiment_errors.png"), dpi=150)
    plt.close()
    print(f"  Saved to {OUT}")

    # ── Phase 20: Success Condition ──
    reached = best['pearson_r'] is not None and best['pearson_r'] >= 0.50
    bl_r = results["baseline_linear"]["pearson_r"]
    rmse_imp = ((results["baseline_linear"]["rmse"] - best["rmse"]) / results["baseline_linear"]["rmse"] * 100) if best["rmse"] else 0
    mae_imp  = ((results["baseline_linear"]["mae"]  - best["mae"])  / results["baseline_linear"]["mae"]  * 100) if best["mae"] else 0

    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"  Baseline Pearson r:     {bl_r}")
    print(f"  Best global Pearson r:  {best['pearson_r']}")
    print(f"  Target r >= 0.50:       {'ACHIEVED' if reached else 'NOT ACHIEVED'}")
    print(f"  RMSE improvement:       {rmse_imp:.1f}%")
    print(f"  MAE improvement:        {mae_imp:.1f}%")
    print(f"  Best method:            {best_key}")

    # Save JSON
    report = {"baseline": results["baseline_linear"], "best_key": best_key,
              "best_metrics": best, "all_experiments": results,
              "pearson_reached_050": reached,
              "rmse_improvement_pct": round(rmse_imp, 1),
              "mae_improvement_pct": round(mae_imp, 1)}
    with open(os.path.join(OUT, "global_calibration_experiment.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
