"""
Global Calibration Experimental Module
=======================================
Tests advanced global calibration strategies to improve Pearson r toward >= 0.50
WITHOUT using localized/block calibration. All experiments use spatially separated
validation to prevent data leakage.

Does NOT modify app/calibration.py.
"""

import logging
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter, uniform_filter
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def build_depth_features(rel_depth: np.ndarray) -> Dict[str, np.ndarray]:
    """Generate depth-derived feature maps."""
    h, w = rel_depth.shape
    features = {"depth": rel_depth.copy()}

    # Inverted depth
    features["depth_inv"] = 1.0 - rel_depth

    # Percentile-clipped depth (clip to 2nd-98th percentile, re-normalize)
    p2, p98 = np.nanpercentile(rel_depth, [2, 98])
    clipped = np.clip(rel_depth, p2, p98)
    features["depth_pclip"] = (clipped - p2) / (p98 - p2 + 1e-8)

    # Rank/quantile normalized depth
    flat = rel_depth.ravel()
    order = np.argsort(flat)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.linspace(0, 1, len(flat))
    features["depth_rank"] = ranks.reshape(h, w)

    # Gaussian smoothed
    features["depth_gauss"] = gaussian_filter(rel_depth, sigma=2.0)

    # Local mean and std at multiple scales
    for k in [3, 7, 15, 31]:
        lmean = uniform_filter(rel_depth, size=k)
        lsq = uniform_filter(rel_depth ** 2, size=k)
        lstd = np.sqrt(np.maximum(lsq - lmean ** 2, 0))
        features[f"depth_mean_{k}"] = lmean
        features[f"depth_std_{k}"] = lstd

    # Gradients
    dy, dx = np.gradient(rel_depth)
    features["grad_x"] = dx
    features["grad_y"] = dy
    features["grad_mag"] = np.sqrt(dx ** 2 + dy ** 2)

    # Spatial coordinates (normalized 0-1)
    yy, xx = np.mgrid[0:h, 0:w]
    features["coord_x"] = xx.astype(np.float64) / max(w - 1, 1)
    features["coord_y"] = yy.astype(np.float64) / max(h - 1, 1)
    features["coord_x2"] = features["coord_x"] ** 2
    features["coord_y2"] = features["coord_y"] ** 2
    features["coord_xy"] = features["coord_x"] * features["coord_y"]

    return features


def build_vegetation_feature(lc_mask: np.ndarray) -> np.ndarray:
    """Binary vegetation indicator from land-cover mask (class 1 = vegetation)."""
    return (lc_mask == 1).astype(np.float64)


# ---------------------------------------------------------------------------
# Spatial CV Infrastructure
# ---------------------------------------------------------------------------

def make_spatial_folds(h: int, w: int, grid: Tuple[int, int] = (4, 4)):
    """
    Generate spatial fold masks.
    Returns list of (train_mask_2d, val_mask_2d) pairs.
    """
    nr, nc = grid
    fold_h = h // nr
    fold_w = w // nc
    folds = []
    for fi in range(nr):
        for fj in range(nc):
            val_mask = np.zeros((h, w), dtype=bool)
            y0 = fi * fold_h
            y1 = min(y0 + fold_h, h)
            x0 = fj * fold_w
            x1 = min(x0 + fold_w, w)
            val_mask[y0:y1, x0:x1] = True
            train_mask = ~val_mask
            folds.append((train_mask, val_mask))
    return folds


def spatial_cv_evaluate(X_all, y_all, valid_mask, folds, fit_fn):
    """
    Run spatial cross-validation with a given fit function.
    fit_fn(X_train, y_train) -> model with .predict(X)
    Returns dict of mean/std metrics.
    """
    fold_metrics = []
    for train_2d, val_2d in folds:
        train_m = valid_mask & train_2d
        val_m = valid_mask & val_2d

        n_train = np.sum(train_m)
        n_val = np.sum(val_m)
        if n_train < 200 or n_val < 50:
            continue

        X_tr = X_all[train_m.ravel()]
        y_tr = y_all[train_m.ravel()]
        X_va = X_all[val_m.ravel()]
        y_va = y_all[val_m.ravel()]

        try:
            model = fit_fn(X_tr, y_tr)
            pred = model.predict(X_va)
        except Exception:
            continue

        diff = pred - y_va
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))
        if np.std(pred) > 1e-6 and np.std(y_va) > 1e-6:
            pr, _ = stats.pearsonr(pred, y_va)
        else:
            pr = 0.0
        ss_res = np.sum(diff ** 2)
        ss_tot = np.sum((y_va - np.mean(y_va)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        fold_metrics.append({"rmse": rmse, "mae": mae, "pearson_r": float(pr), "r2": float(r2)})

    if not fold_metrics:
        return {"rmse": None, "mae": None, "pearson_r": None, "r2": None, "n_folds": 0}

    return {
        "rmse": round(float(np.mean([m["rmse"] for m in fold_metrics])), 2),
        "mae": round(float(np.mean([m["mae"] for m in fold_metrics])), 2),
        "pearson_r": round(float(np.mean([m["pearson_r"] for m in fold_metrics])), 4),
        "r2": round(float(np.mean([m["r2"] for m in fold_metrics])), 4),
        "std_rmse": round(float(np.std([m["rmse"] for m in fold_metrics])), 2),
        "std_r": round(float(np.std([m["pearson_r"] for m in fold_metrics])), 4),
        "n_folds": len(fold_metrics),
    }


# ---------------------------------------------------------------------------
# Model wrappers (sklearn-compatible .predict interface)
# ---------------------------------------------------------------------------

class LinearModel:
    def __init__(self): self.coef_ = None; self.intercept_ = 0
    def fit(self, X, y):
        from sklearn.linear_model import LinearRegression
        m = LinearRegression().fit(X, y)
        self.coef_ = m.coef_; self.intercept_ = m.intercept_
        return self
    def predict(self, X):
        return X @ self.coef_ + self.intercept_

class HuberModel:
    def __init__(self): self.m = None
    def fit(self, X, y):
        from sklearn.linear_model import HuberRegressor
        self.m = HuberRegressor(max_iter=200).fit(X, y)
        return self
    def predict(self, X):
        return self.m.predict(X)

class PolyModel:
    def __init__(self, deg=2): self.deg = deg; self.pipe = None
    def fit(self, X, y):
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        self.pipe = make_pipeline(PolynomialFeatures(self.deg, include_bias=False), Ridge(alpha=1.0))
        self.pipe.fit(X, y)
        return self
    def predict(self, X):
        return self.pipe.predict(X)

class IsotonicModel:
    """Isotonic regression on 1-D depth."""
    def __init__(self): self.iso = None
    def fit(self, X, y):
        from sklearn.isotonic import IsotonicRegression
        self.iso = IsotonicRegression(out_of_bounds='clip').fit(X[:, 0], y)
        return self
    def predict(self, X):
        return self.iso.predict(X[:, 0])

class RidgeModel:
    def __init__(self, alpha=10.0): self.alpha = alpha; self.m = None
    def fit(self, X, y):
        from sklearn.linear_model import Ridge
        self.m = Ridge(alpha=self.alpha).fit(X, y)
        return self
    def predict(self, X):
        return self.m.predict(X)
