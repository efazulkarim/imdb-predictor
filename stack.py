# ============================================================
# STACKED ENSEMBLE
# ============================================================
"""
Train a Ridge meta-regressor on top of the base models' out-of-fold
predictions to test whether stacking buys additional R^2 over any
single base model.

Protocol:
    - 5-fold KFold over the corpus.
    - In each outer fold:
        * outer test = held-out test rows
        * outer train = remaining rows
        * On outer train, do an inner KFold (k=5) to produce
          out-of-fold predictions for every base model. These OOF
          preds (one per row of outer train) are the meta-features.
        * Refit each base model on ALL outer train rows and predict
          on outer test. These are the meta-features at test time.
        * Train Ridge on (OOF preds -> y_outer_train), apply to
          (test base preds -> y_outer_test).
    - Aggregate metrics across outer folds; pool errors for paired
      Wilcoxon vs the headline SBERT model.

Cached SBERT embeddings are reused; nothing here re-runs SBERT.

Usage:
    python stack.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# UTF-8 stdout (Windows console)
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from config import RANDOM_STATE, SBERT_MODEL_NAME
from data_loader import load_dataset
import baselines
import stats_utils
from experiments import (
    POOLING_STRATEGIES,
    _get_or_build_embeddings_multi,
    _ensure_dir,
    RESULTS_DIR,
)


BASE_MODELS = ['ols_metadata', 'ols_structural', 'tfidf_xgboost', 'sbert_xgboost']


def _train_sbert_xgb(X_train, y_train, X_val, y_val, X_test, random_state):
    """SBERT system without sample weights (matches the headline config)."""
    model = XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=random_state, n_jobs=-1, verbosity=0,
        tree_method='hist', early_stopping_rounds=20,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return np.clip(model.predict(X_test), 1.0, 10.0)


def _build_features(features_df, idx, scaler=None):
    """Median-impute then standardize on a precomputed scaler (or fit)."""
    sub = features_df.iloc[idx]
    if scaler is None:
        median = sub.median(numeric_only=True)
        sub = sub.fillna(median)
        scaler = StandardScaler().fit(sub)
        return scaler.transform(sub), scaler, median
    raise NotImplementedError


def _predict_base_models(
    train_idx, val_idx, predict_idx,
    scripts_text, ratings, features_df, embeddings_mean,
    random_state,
):
    """
    Train every base model on train_idx (with val_idx for early-stopping
    where applicable) and return predictions on predict_idx.

    Returns dict {model_name: np.ndarray of length len(predict_idx)}.
    """
    y_train = ratings[train_idx]

    # Median-impute + scale numerical features (fit on train).
    median = features_df.iloc[train_idx].median(numeric_only=True)
    Xn_train = features_df.iloc[train_idx].fillna(median)
    Xn_pred = features_df.iloc[predict_idx].fillna(median)
    scaler_full = StandardScaler().fit(Xn_train)
    Xn_train_s = scaler_full.transform(Xn_train)
    Xn_pred_s = scaler_full.transform(Xn_pred)

    # Metadata-only scaling subset
    META_COLS = ['year', 'movie_length', 'decade_encoded']
    median_meta = features_df.iloc[train_idx][META_COLS].median(numeric_only=True)
    Xm_train = features_df.iloc[train_idx][META_COLS].fillna(median_meta)
    Xm_pred = features_df.iloc[predict_idx][META_COLS].fillna(median_meta)
    scaler_meta = StandardScaler().fit(Xm_train)
    Xm_train_s = scaler_meta.transform(Xm_train)
    Xm_pred_s = scaler_meta.transform(Xm_pred)

    out = {}

    # ols_metadata
    from sklearn.linear_model import LinearRegression
    m = LinearRegression().fit(Xm_train_s, y_train)
    out['ols_metadata'] = np.clip(m.predict(Xm_pred_s), 1.0, 10.0)

    # ols_structural
    m = LinearRegression().fit(Xn_train_s, y_train)
    out['ols_structural'] = np.clip(m.predict(Xn_pred_s), 1.0, 10.0)

    # tfidf_xgboost
    out['tfidf_xgboost'] = baselines.tfidf_xgboost(
        [scripts_text[i] for i in train_idx],
        [scripts_text[i] for i in predict_idx],
        y_train,
        sample_weight=None,
        random_state=random_state,
    )

    # sbert_xgboost (no weights, mean pooling)
    Xs_train = np.hstack([embeddings_mean[train_idx], Xn_train_s])
    Xs_pred = np.hstack([embeddings_mean[predict_idx], Xn_pred_s])
    Xs_val = np.hstack([embeddings_mean[val_idx],
                        scaler_full.transform(features_df.iloc[val_idx].fillna(median))])
    out['sbert_xgboost'] = _train_sbert_xgb(
        Xs_train, y_train, Xs_val, ratings[val_idx], Xs_pred,
        random_state=random_state,
    )

    return out


def main():
    _ensure_dir(RESULTS_DIR)
    print("=" * 70)
    print("  STACKED ENSEMBLE (Ridge over base models, 5-fold outer CV)")
    print("=" * 70)

    # 1. Load data + cached SBERT embeddings (mean pooling).
    (scripts_text, ratings, features_df, _, _, _, scripts_text_sbert) = load_dataset()
    n = len(scripts_text)
    print(f"\nLoaded {n} scripts.")

    embeddings = _get_or_build_embeddings_multi(
        scripts_text_sbert, SBERT_MODEL_NAME, poolings=['mean'],
    )['mean']
    print(f"Embeddings shape: {embeddings.shape}\n")

    # 2. Outer 5-fold CV.
    outer_kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    indices = np.arange(n)

    pooled_y_true = []
    pooled_preds = {name: [] for name in BASE_MODELS + ['stacked']}
    fold_metrics = {name: [] for name in BASE_MODELS + ['stacked']}

    for fold, (outer_train_idx, outer_test_idx) in enumerate(outer_kf.split(indices), 1):
        print(f"--- Outer fold {fold}/5: train={len(outer_train_idx)}, test={len(outer_test_idx)}")

        # Inner CV for OOF predictions on outer_train_idx.
        inner_kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + fold)
        oof_preds = {name: np.zeros(len(outer_train_idx)) for name in BASE_MODELS}

        for inner_fold, (inner_tr_pos, inner_val_pos) in enumerate(inner_kf.split(outer_train_idx), 1):
            inner_tr_idx = outer_train_idx[inner_tr_pos]
            inner_val_idx = outer_train_idx[inner_val_pos]

            # Carve out a small validation set from inner_tr for early stopping.
            inner_tr_fit, inner_tr_es = train_test_split(
                inner_tr_idx, test_size=0.15,
                random_state=RANDOM_STATE + fold * 10 + inner_fold,
            )

            preds = _predict_base_models(
                train_idx=inner_tr_fit,
                val_idx=inner_tr_es,
                predict_idx=inner_val_idx,
                scripts_text=scripts_text,
                ratings=ratings,
                features_df=features_df,
                embeddings_mean=embeddings,
                random_state=RANDOM_STATE + fold * 10 + inner_fold,
            )
            for name, p in preds.items():
                oof_preds[name][inner_val_pos] = p

        # Now refit each base model on ALL of outer_train_idx and predict on outer_test_idx.
        # Reserve a small validation slice for early stopping.
        outer_fit_idx, outer_es_idx = train_test_split(
            outer_train_idx, test_size=0.15,
            random_state=RANDOM_STATE + 100 + fold,
        )
        test_base_preds = _predict_base_models(
            train_idx=outer_fit_idx,
            val_idx=outer_es_idx,
            predict_idx=outer_test_idx,
            scripts_text=scripts_text,
            ratings=ratings,
            features_df=features_df,
            embeddings_mean=embeddings,
            random_state=RANDOM_STATE + 100 + fold,
        )

        # Train Ridge meta-regressor on (oof_preds -> y_outer_train).
        meta_train = np.column_stack([oof_preds[m] for m in BASE_MODELS])
        meta_test = np.column_stack([test_base_preds[m] for m in BASE_MODELS])
        y_outer_train = ratings[outer_train_idx]
        y_outer_test = ratings[outer_test_idx]

        meta = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        meta.fit(meta_train, y_outer_train)
        stacked_preds = np.clip(meta.predict(meta_test), 1.0, 10.0)

        # Per-fold metrics.
        for name in BASE_MODELS:
            m = stats_utils.metrics(y_outer_test, test_base_preds[name])
            fold_metrics[name].append(m)
            pooled_preds[name].append(test_base_preds[name])

        m_stack = stats_utils.metrics(y_outer_test, stacked_preds)
        fold_metrics['stacked'].append(m_stack)
        pooled_preds['stacked'].append(stacked_preds)
        pooled_y_true.append(y_outer_test)

        print(f"   stacked (this fold): RMSE={m_stack['RMSE']:.4f}  R2={m_stack['R2']:.4f}  "
              f"meta_coefs={dict(zip(BASE_MODELS, np.round(meta.coef_, 3).tolist()))}")

    # Aggregate.
    print("\n" + "=" * 70)
    print("  CV SUMMARY (mean ± std across 5 outer folds)")
    print("=" * 70)
    cv_summary = {}
    for name in BASE_MODELS + ['stacked']:
        rmse = np.array([m['RMSE'] for m in fold_metrics[name]])
        mae = np.array([m['MAE'] for m in fold_metrics[name]])
        r2 = np.array([m['R2'] for m in fold_metrics[name]])
        cv_summary[name] = {
            'RMSE_mean': float(rmse.mean()), 'RMSE_std': float(rmse.std(ddof=1)),
            'MAE_mean': float(mae.mean()), 'MAE_std': float(mae.std(ddof=1)),
            'R2_mean': float(r2.mean()), 'R2_std': float(r2.std(ddof=1)),
            'rmse_per_fold': rmse.tolist(),
        }
        print(f"  {name:18s} RMSE={rmse.mean():.4f}±{rmse.std(ddof=1):.4f}  "
              f"MAE={mae.mean():.4f}±{mae.std(ddof=1):.4f}  "
              f"R2={r2.mean():.4f}±{r2.std(ddof=1):.4f}")

    # Pooled paired Wilcoxon: stacked vs sbert_xgboost.
    pooled_y_arr = np.concatenate(pooled_y_true)
    pooled_stack = np.concatenate(pooled_preds['stacked'])
    pooled_sbert = np.concatenate(pooled_preds['sbert_xgboost'])
    sig = stats_utils.paired_significance(pooled_y_arr, pooled_sbert, pooled_stack)
    print(f"\n  stacked vs sbert_xgboost (pooled, n={len(pooled_y_arr)}):")
    print(f"    delta_MAE = {sig['delta_mae']:+.4f}  "
          f"95% CI [{sig['ci_lo_delta_mae']:+.4f}, {sig['ci_hi_delta_mae']:+.4f}]")
    print(f"    Wilcoxon p = {sig['wilcoxon_p']:.2e}")
    print(f"    Winner: {'stacked' if sig['delta_mae'] > 0 else 'sbert_xgboost'}")

    # Save.
    out = {
        'cv_summary': cv_summary,
        'stacked_vs_sbert_pooled': sig,
    }
    with open(os.path.join(RESULTS_DIR, 'stacking_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved {RESULTS_DIR}/stacking_results.json")


if __name__ == '__main__':
    main()
