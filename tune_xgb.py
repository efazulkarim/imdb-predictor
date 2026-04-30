# ============================================================
# OPTUNA HYPERPARAMETER SEARCH FOR THE SBERT+XGBoost SYSTEM
# ============================================================
"""
Hyperparameter search over the XGBoost head sitting on top of cached
SBERT (mean-pooled) embeddings + the 19 hand-crafted structural
features.

Protocol:
    - Single 80/20 train/test split (random_state=42)
    - Within train, an internal 80/20 train/val split is used both
      for early stopping and as the Optuna objective
    - 100 trials with TPE sampler
    - Final reported metrics: held-out test RMSE/MAE/R^2 with
      bootstrap 95% CIs at the best hyperparameters

Cached SBERT embeddings are reused (no re-encoding).

Usage:
    python tune_xgb.py             # 100 trials, default seed
    python tune_xgb.py --trials 50 # custom trial budget
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import optuna

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from config import RANDOM_STATE, SBERT_MODEL_NAME
from data_loader import load_dataset
from experiments import _get_or_build_embeddings_multi, _ensure_dir, RESULTS_DIR
import stats_utils


def _build_xy(idx_train, idx_val, idx_test, embeddings, features_df, ratings):
    """Return scaled feature matrices for train/val/test."""
    median = features_df.iloc[idx_train].median(numeric_only=True)
    Xn_tr = features_df.iloc[idx_train].fillna(median)
    Xn_va = features_df.iloc[idx_val].fillna(median)
    Xn_te = features_df.iloc[idx_test].fillna(median)
    scaler = StandardScaler().fit(Xn_tr)
    Xn_tr_s = scaler.transform(Xn_tr)
    Xn_va_s = scaler.transform(Xn_va)
    Xn_te_s = scaler.transform(Xn_te)

    X_tr = np.hstack([embeddings[idx_train], Xn_tr_s])
    X_va = np.hstack([embeddings[idx_val], Xn_va_s])
    X_te = np.hstack([embeddings[idx_test], Xn_te_s])
    return X_tr, X_va, X_te


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trials', type=int, default=100)
    p.add_argument('--seed', type=int, default=RANDOM_STATE)
    args = p.parse_args()

    _ensure_dir(RESULTS_DIR)

    print("=" * 70)
    print(f"  OPTUNA XGBoost SEARCH (trials={args.trials}, seed={args.seed})")
    print("=" * 70)

    # 1. Load + embeddings.
    (scripts_text, ratings, features_df, _, _, _, scripts_text_sbert) = load_dataset()
    embeddings = _get_or_build_embeddings_multi(
        scripts_text_sbert, SBERT_MODEL_NAME, poolings=['mean'],
    )['mean']
    n = len(scripts_text)

    # 2. Splits.
    indices = np.arange(n)
    idx_trainval, idx_test = train_test_split(
        indices, test_size=0.20, random_state=args.seed,
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.20, random_state=args.seed,
    )
    print(f"\nSplits: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")

    X_tr, X_va, X_te = _build_xy(idx_train, idx_val, idx_test,
                                 embeddings, features_df, ratings)
    y_tr = ratings[idx_train]
    y_va = ratings[idx_val]
    y_te = ratings[idx_test]

    # 3. Optuna objective.
    def objective(trial):
        params = {
            'n_estimators':     2000,
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth':        trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.5, 10.0, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'gamma':            trial.suggest_float('gamma', 1e-4, 5.0, log=True),
            'random_state': args.seed,
            'n_jobs': -1,
            'verbosity': 0,
            'tree_method': 'hist',
            'early_stopping_rounds': 30,
        }
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        pred = np.clip(model.predict(X_va), 1.0, 10.0)
        return float(np.sqrt(mean_squared_error(y_va, pred)))

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print(f"\nBest validation RMSE: {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"   {k}: {v}")

    # 4. Refit best on train+val and report on test.
    best = dict(study.best_params)
    best.update({
        'n_estimators': 2000,
        'random_state': args.seed,
        'n_jobs': -1,
        'verbosity': 0,
        'tree_method': 'hist',
        'early_stopping_rounds': 30,
    })

    # Combine train+val for final fit, use a small slice of train for early stopping.
    X_trval = np.vstack([X_tr, X_va])
    y_trval = np.concatenate([y_tr, y_va])
    idx_fit, idx_es = train_test_split(
        np.arange(len(X_trval)), test_size=0.10, random_state=args.seed,
    )
    final = XGBRegressor(**best)
    final.fit(
        X_trval[idx_fit], y_trval[idx_fit],
        eval_set=[(X_trval[idx_es], y_trval[idx_es])],
        verbose=False,
    )

    pred_te = np.clip(final.predict(X_te), 1.0, 10.0)

    # 5. Bootstrap CIs on test metrics.
    print("\n" + "-" * 70)
    print("  TEST METRICS (95% bootstrap CIs)")
    print("-" * 70)
    bs = stats_utils.bootstrap_metrics(y_te, pred_te, n_boot=1000)
    for k in ['RMSE', 'MAE', 'R2']:
        print(f"  {k}: {stats_utils.format_metric(bs, k)}")

    # 6. Save.
    out = {
        'best_params': study.best_params,
        'best_val_rmse': float(study.best_value),
        'test_metrics_bootstrap': bs,
        'n_trials': args.trials,
        'seed': args.seed,
        'split_sizes': {'train': len(idx_train), 'val': len(idx_val), 'test': len(idx_test)},
        'all_trials': [
            {'value': t.value, 'params': t.params}
            for t in study.trials if t.value is not None
        ],
    }
    out_path = os.path.join(RESULTS_DIR, 'tune_xgb_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved {out_path}")

    np.savez(
        os.path.join(RESULTS_DIR, 'tune_xgb_predictions.npz'),
        y_test=y_te,
        pred_test=pred_te,
    )


if __name__ == '__main__':
    main()
