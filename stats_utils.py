# ============================================================
# STATISTICAL UTILITIES
# ============================================================
"""
Bootstrap confidence intervals and paired significance tests.

Used by experiments.py to report results with proper uncertainty
quantification, and to compare pairs of models on the same test set.
"""

import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(y_true, y_pred):
    """Return RMSE, MAE, R2 as a dict."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'R2': float(r2_score(y_true, y_pred)),
    }


def bootstrap_metrics(y_true, y_pred, n_boot=1000, seed=42, ci=0.95):
    """
    Bootstrap point estimates and CIs for RMSE / MAE / R2.

    Returns a dict like:
        {'RMSE': {'point': 0.99, 'lo': 0.94, 'hi': 1.04}, ...}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    rmse, mae, r2 = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        rmse.append(np.sqrt(mean_squared_error(yt, yp)))
        mae.append(mean_absolute_error(yt, yp))
        # R2 is undefined when all yt are equal; guard.
        if np.var(yt) > 0:
            r2.append(r2_score(yt, yp))

    alpha = (1 - ci) / 2
    point = metrics(y_true, y_pred)
    return {
        'RMSE': {
            'point': point['RMSE'],
            'lo': float(np.quantile(rmse, alpha)),
            'hi': float(np.quantile(rmse, 1 - alpha)),
        },
        'MAE': {
            'point': point['MAE'],
            'lo': float(np.quantile(mae, alpha)),
            'hi': float(np.quantile(mae, 1 - alpha)),
        },
        'R2': {
            'point': point['R2'],
            'lo': float(np.quantile(r2, alpha)) if r2 else float('nan'),
            'hi': float(np.quantile(r2, 1 - alpha)) if r2 else float('nan'),
        },
    }


def paired_significance(y_true, y_pred_a, y_pred_b, n_boot=1000, seed=42):
    """
    Paired comparison of two models' predictions on the same test set.

    Returns:
        - delta_mae: MAE_a - MAE_b (positive => B better)
        - delta_rmse: RMSE_a - RMSE_b
        - wilcoxon_p: paired Wilcoxon signed-rank p-value on per-sample |error|
        - bootstrap_ci_delta_mae: 95% CI for the MAE difference (paired bootstrap)
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    err_a = np.abs(y_pred_a - y_true)
    err_b = np.abs(y_pred_b - y_true)

    delta_mae = float(np.mean(err_a) - np.mean(err_b))
    delta_rmse = float(
        np.sqrt(np.mean((y_pred_a - y_true) ** 2))
        - np.sqrt(np.mean((y_pred_b - y_true) ** 2))
    )

    # Paired Wilcoxon on per-sample absolute errors.
    # Drop ties (zero diffs) per scipy convention.
    diff = err_a - err_b
    nonzero = diff[diff != 0]
    if len(nonzero) > 0:
        w_stat, w_p = scipy_stats.wilcoxon(err_a, err_b, zero_method='wilcox')
        w_p = float(w_p)
    else:
        w_p = 1.0

    # Paired bootstrap CI for delta_mae.
    n = len(y_true)
    rng = np.random.default_rng(seed)
    boot_deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_deltas.append(np.mean(err_a[idx]) - np.mean(err_b[idx]))

    return {
        'delta_mae': delta_mae,
        'delta_rmse': delta_rmse,
        'wilcoxon_p': w_p,
        'ci_lo_delta_mae': float(np.quantile(boot_deltas, 0.025)),
        'ci_hi_delta_mae': float(np.quantile(boot_deltas, 0.975)),
    }


def format_metric(stats_dict, key):
    """Format a metric with CI as 'point [lo, hi]'."""
    s = stats_dict[key]
    return f"{s['point']:.4f} [{s['lo']:.4f}, {s['hi']:.4f}]"


if __name__ == '__main__':
    # Smoke test
    rng = np.random.default_rng(0)
    y_true = rng.uniform(1, 10, 200)
    y_pred = y_true + rng.normal(0, 0.5, 200)
    y_pred_b = y_true + rng.normal(0, 0.7, 200)

    print("Metrics:", metrics(y_true, y_pred))
    print("Bootstrap:")
    bs = bootstrap_metrics(y_true, y_pred, n_boot=200)
    for k in bs:
        print(f"  {k}: {format_metric(bs, k)}")
    print("Paired test (A=better, B=worse):")
    print(paired_significance(y_true, y_pred, y_pred_b, n_boot=200))
