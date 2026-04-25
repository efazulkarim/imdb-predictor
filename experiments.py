# ============================================================
# UNIFIED EXPERIMENT RUNNER
# ============================================================
"""
Runs the full comparison table for the paper:

    1. predict_mean
    2. ols_metadata
    3. ols_structural
    4. tfidf_xgboost
    5. sbert_xgboost          (the main system)

For each model, reports test RMSE / MAE / R2 with 95% bootstrap CIs
and pairs each baseline against the SBERT system with a paired
Wilcoxon test on per-sample absolute errors.

Caches SBERT embeddings to disk so subsequent runs are seconds
instead of minutes. Outputs:
    - results/comparison_table.json  (machine-readable)
    - results/comparison_table.md    (human-readable, paper-ready)
    - results/predictions.npz        (per-model predictions, for later analysis)

Usage:
    python experiments.py
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import (
    TEST_SIZE, VALIDATION_SIZE, RANDOM_STATE,
    SBERT_MODEL_NAME, SBERT_EMBEDDING_DIM, CHUNK_SIZE, CHUNK_OVERLAP,
)
from data_loader import load_dataset
from preprocessing import SBERT_PREPROCESSING_VERSION
from trainer import compute_sample_weights, embed_scripts_sbert
import baselines
import stats_utils


RESULTS_DIR = 'results'
# Cache filename includes model name (sanitized) so that switching SBERT
# encoders does not collide caches.
EMBEDDINGS_CACHE_TEMPLATE = 'results/sbert_embeddings_{model}_{pooling}.npz'

# Pooling strategies to evaluate. The first one in the list is treated
# as the "headline" SBERT system; the others appear as ablations.
POOLING_STRATEGIES = ['mean', 'max', 'weighted_norm']


def _safe_filename(s):
    """Make a string safe to embed in a filename."""
    return s.replace('/', '_').replace('\\', '_').replace(':', '_')


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _split_indices(n_samples, ratings, seed=RANDOM_STATE):
    """Reproduce the trainer's 70/15/15 split, returning index arrays."""
    indices = np.arange(n_samples)
    temp_size = TEST_SIZE + VALIDATION_SIZE
    idx_train, idx_temp, _, y_temp = train_test_split(
        indices, ratings, test_size=temp_size, random_state=seed
    )
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp, y_temp, test_size=0.5, random_state=seed
    )
    return idx_train, idx_val, idx_test


def _get_or_build_embeddings(scripts_text, model_name, pooling='mean', force=False):
    """
    Compute (or load cached) SBERT embeddings for ALL scripts.

    Cache key is (model_name, n_scripts, chunk_size, overlap, pooling,
    preprocessing_version). If any field changes, we recompute.
    """
    cache_path = EMBEDDINGS_CACHE_TEMPLATE.format(
        model=_safe_filename(model_name), pooling=pooling
    )
    cache_key = {
        'model_name': model_name,
        'n_scripts': len(scripts_text),
        'chunk_size': CHUNK_SIZE,
        'chunk_overlap': CHUNK_OVERLAP,
        'pooling': pooling,
        'preprocessing_version': SBERT_PREPROCESSING_VERSION,
    }

    if not force and os.path.exists(cache_path):
        try:
            cached = np.load(cache_path, allow_pickle=True)
            cached_meta = json.loads(str(cached['meta']))
            if cached_meta == cache_key:
                print(f"   [cache] loaded SBERT embeddings ({model_name}, {pooling}) from {cache_path}")
                return cached['embeddings']
            else:
                print(f"   [cache] stale ({pooling}), recomputing")
        except Exception as e:
            print(f"   [cache] could not load {cache_path} ({e}), recomputing")

    print(f"   Loading SBERT model '{model_name}' for pooling='{pooling}'...")
    from sentence_transformers import SentenceTransformer
    sbert_model = SentenceTransformer(model_name)

    print(f"   Embedding {len(scripts_text)} scripts (pooling={pooling})...")
    t0 = time.time()
    embeddings = embed_scripts_sbert(scripts_text, sbert_model, pooling=pooling)
    print(f"   Done in {time.time() - t0:.1f}s. Shape: {embeddings.shape}")

    _ensure_dir(RESULTS_DIR)
    np.savez(cache_path,
             embeddings=embeddings,
             meta=json.dumps(cache_key))
    print(f"   [cache] saved to {cache_path}")
    return embeddings


def run_sbert_xgboost(
    embeddings, features_df, y, idx_train, idx_val, idx_test,
    sample_weight=None, random_state=RANDOM_STATE,
):
    """
    The main system: SBERT embeddings + 19 structural features + XGBoost.

    Mirrors the configuration in trainer.py but with early stopping
    on the validation set to prevent overfitting.
    """
    X_emb_train = embeddings[idx_train]
    X_emb_val = embeddings[idx_val]
    X_emb_test = embeddings[idx_test]

    # Median-impute then scale numerical features
    median = features_df.iloc[idx_train].median(numeric_only=True)
    X_num_train = features_df.iloc[idx_train].fillna(median)
    X_num_val = features_df.iloc[idx_val].fillna(median)
    X_num_test = features_df.iloc[idx_test].fillna(median)

    scaler = StandardScaler()
    X_num_train_s = scaler.fit_transform(X_num_train)
    X_num_val_s = scaler.transform(X_num_val)
    X_num_test_s = scaler.transform(X_num_test)

    X_train = np.hstack([X_emb_train, X_num_train_s])
    X_val = np.hstack([X_emb_val, X_num_val_s])
    X_test = np.hstack([X_emb_test, X_num_test_s])

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
        tree_method='hist',
        early_stopping_rounds=20,
    )
    fit_kwargs = {'eval_set': [(X_val, y[idx_val])], 'verbose': False}
    if sample_weight is not None:
        fit_kwargs['sample_weight'] = sample_weight
    model.fit(X_train, y[idx_train], **fit_kwargs)

    preds_test = np.clip(model.predict(X_test), 1.0, 10.0)
    preds_val = np.clip(model.predict(X_val), 1.0, 10.0)
    return {
        'test_preds': preds_test,
        'val_preds': preds_val,
        'best_iteration': int(getattr(model, 'best_iteration', model.n_estimators)),
    }


def _model_order():
    """Canonical row order for the comparison table."""
    order = ['predict_mean', 'ols_metadata', 'ols_structural',
             'tfidf_xgboost', 'sbert_xgboost']
    for pool in POOLING_STRATEGIES[1:]:
        order.append(f'sbert_xgboost_{pool}')
    order.append('sbert_xgboost_noweight')
    return order


def _run_one_split(
    scripts_text, scripts_text_sbert, ratings, features_df,
    embeddings_by_pool,
    idx_train, idx_val, idx_test,
    random_state=RANDOM_STATE,
):
    """
    Train every model on (idx_train + idx_val) and return test predictions.

    Returns:
        dict {model_name: np.ndarray of test predictions}
    """
    y_train = ratings[idx_train]

    # Sample weights for the training portion (used by TF-IDF and SBERT systems
    # for fair comparison with the legacy main pipeline).
    sample_weights = compute_sample_weights(y_train)

    # Baselines (slice text & feature DF the same way).
    texts_train = [scripts_text[i] for i in idx_train]
    texts_test = [scripts_text[i] for i in idx_test]
    features_train = features_df.iloc[idx_train].reset_index(drop=True)
    features_test = features_df.iloc[idx_test].reset_index(drop=True)

    all_preds = baselines.run_all_baselines(
        texts_train=texts_train,
        texts_test=texts_test,
        features_train=features_train,
        features_test=features_test,
        y_train=y_train,
        sample_weight=sample_weights,
        random_state=random_state,
    )

    # SBERT main system, one variant per pooling strategy.
    for pool in POOLING_STRATEGIES:
        name = f'sbert_xgboost_{pool}' if pool != POOLING_STRATEGIES[0] else 'sbert_xgboost'
        sbert_out = run_sbert_xgboost(
            embeddings_by_pool[pool], features_df, ratings,
            idx_train, idx_val, idx_test,
            sample_weight=sample_weights,
            random_state=random_state,
        )
        all_preds[name] = sbert_out['test_preds']

    # Ablation: SBERT (mean pooling) trained WITHOUT sample weights, to
    # quantify the effect of the rating-bucket reweighting.
    sbert_nw = run_sbert_xgboost(
        embeddings_by_pool[POOLING_STRATEGIES[0]], features_df, ratings,
        idx_train, idx_val, idx_test,
        sample_weight=None,
        random_state=random_state,
    )
    all_preds['sbert_xgboost_noweight'] = sbert_nw['test_preds']

    return all_preds


def _write_markdown(md_path, bs_results, sig_results, header_lines):
    """Write a paper-ready markdown comparison table."""
    md_lines = list(header_lines)
    md_lines.append("\n## Test Metrics (point estimate [95% bootstrap CI])\n")
    md_lines.append("| Model | RMSE | MAE | R² |")
    md_lines.append("|---|---|---|---|")
    for name in _model_order():
        if name not in bs_results:
            continue
        bs = bs_results[name]
        md_lines.append(
            f"| {name} | "
            f"{bs['RMSE']['point']:.4f} [{bs['RMSE']['lo']:.4f}, {bs['RMSE']['hi']:.4f}] | "
            f"{bs['MAE']['point']:.4f} [{bs['MAE']['lo']:.4f}, {bs['MAE']['hi']:.4f}] | "
            f"{bs['R2']['point']:.4f} [{bs['R2']['lo']:.4f}, {bs['R2']['hi']:.4f}] |"
        )

    if sig_results:
        md_lines.append("\n## Paired Comparison vs SBERT+XGBoost\n")
        md_lines.append("| Baseline | ΔMAE (baseline − SBERT) | 95% CI | Wilcoxon p |")
        md_lines.append("|---|---|---|---|")
        for name in _model_order():
            if name == 'sbert_xgboost' or name not in sig_results:
                continue
            sig = sig_results[name]
            md_lines.append(
                f"| {name} | {sig['delta_mae']:+.4f} | "
                f"[{sig['ci_lo_delta_mae']:+.4f}, {sig['ci_hi_delta_mae']:+.4f}] | "
                f"{sig['wilcoxon_p']:.2e} |"
            )
        md_lines.append("\n_Positive ΔMAE means SBERT has lower error (SBERT wins)._\n")

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))


def main_single(sbert_model_name=SBERT_MODEL_NAME, output_suffix=''):
    """Single 70/15/15 split run (paper headline numbers + bootstrap CIs)."""
    _ensure_dir(RESULTS_DIR)

    print("=" * 70)
    print("  EXPERIMENT RUNNER  (single 70/15/15 split)")
    print(f"  SBERT model: {sbert_model_name}")
    print("=" * 70)

    # 1. Load data
    (scripts_text, ratings, features_df, movie_names, script_files,
     _, scripts_text_sbert) = load_dataset()
    n = len(scripts_text)
    print(f"\nLoaded {n} scripts.")

    # 2. Train/val/test split
    idx_train, idx_val, idx_test = _split_indices(n, ratings)
    y_test = ratings[idx_test]
    print(f"Split: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")

    # 3. Pre-compute (or load) SBERT embeddings for every pooling strategy.
    print("\n" + "-" * 70)
    print("  SBERT EMBEDDINGS (one set per pooling strategy)")
    print("-" * 70)
    embeddings_by_pool = {
        pool: _get_or_build_embeddings(scripts_text_sbert, sbert_model_name, pooling=pool)
        for pool in POOLING_STRATEGIES
    }

    # 4. Train all models on this split.
    all_preds = _run_one_split(
        scripts_text, scripts_text_sbert, ratings, features_df,
        embeddings_by_pool,
        idx_train, idx_val, idx_test,
    )

    # 5. Bootstrap CIs.
    print("\n" + "-" * 70)
    print("  BOOTSTRAP METRICS (95% CI, n_boot=1000)")
    print("-" * 70)
    bs_results = {}
    for name, preds in all_preds.items():
        bs = stats_utils.bootstrap_metrics(y_test, preds, n_boot=1000)
        bs_results[name] = bs
        print(f"\n  {name}")
        print(f"    RMSE: {stats_utils.format_metric(bs, 'RMSE')}")
        print(f"    MAE:  {stats_utils.format_metric(bs, 'MAE')}")
        print(f"    R2:   {stats_utils.format_metric(bs, 'R2')}")

    # 6. Paired significance vs SBERT.
    print("\n" + "-" * 70)
    print("  PAIRED SIGNIFICANCE vs SBERT+XGBoost")
    print("-" * 70)
    sig_results = {}
    sbert_preds = all_preds['sbert_xgboost']
    for name, preds in all_preds.items():
        if name == 'sbert_xgboost':
            continue
        sig = stats_utils.paired_significance(y_test, preds, sbert_preds)
        sig_results[name] = sig
        winner = 'SBERT' if sig['delta_mae'] > 0 else name
        print(f"\n  {name} vs sbert_xgboost")
        print(f"    delta_MAE = {sig['delta_mae']:+.4f}  "
              f"(95% CI [{sig['ci_lo_delta_mae']:+.4f}, {sig['ci_hi_delta_mae']:+.4f}])")
        print(f"    Wilcoxon p = {sig['wilcoxon_p']:.2e}")
        print(f"    Winner: {winner}")

    # 7. Save.
    print("\n" + "-" * 70)
    print("  SAVING ARTIFACTS")
    print("-" * 70)
    out_json = {
        'mode': 'single',
        'split': {
            'random_state': RANDOM_STATE,
            'n_train': len(idx_train),
            'n_val': len(idx_val),
            'n_test': len(idx_test),
        },
        'sbert_model': sbert_model_name,
        'metrics': bs_results,
        'paired_vs_sbert': sig_results,
    }
    json_path = os.path.join(RESULTS_DIR, f'comparison_table{output_suffix}.json')
    with open(json_path, 'w') as f:
        json.dump(out_json, f, indent=2)
    print(f"   Saved {json_path}")

    np.savez(
        os.path.join(RESULTS_DIR, f'predictions{output_suffix}.npz'),
        y_test=y_test,
        **{k: v for k, v in all_preds.items()},
    )
    print(f"   Saved {RESULTS_DIR}/predictions{output_suffix}.npz")

    header = [
        "# Comparison Table (single split)\n",
        f"Test set n={len(idx_test)}, random_state={RANDOM_STATE}, "
        f"SBERT={sbert_model_name}\n",
    ]
    md_path = os.path.join(RESULTS_DIR, f'comparison_table{output_suffix}.md')
    _write_markdown(md_path, bs_results, sig_results, header)
    print(f"   Saved {md_path}")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


def main_cv(k=5, seed=RANDOM_STATE, sbert_model_name=SBERT_MODEL_NAME, output_suffix=''):
    """
    K-fold cross-validated evaluation.

    For each fold:
        - 1/k of the data becomes the test set
        - From the remaining (k-1)/k, we carve off ~15% as val for early stopping
        - All models are trained and evaluated independently per fold

    Then we report mean ± std of test metrics across folds, and aggregate
    the per-sample paired errors across all folds for one combined Wilcoxon
    test versus SBERT+XGBoost.
    """
    _ensure_dir(RESULTS_DIR)

    print("=" * 70)
    print(f"  EXPERIMENT RUNNER  ({k}-fold cross-validation)")
    print(f"  SBERT model: {sbert_model_name}")
    print("=" * 70)

    # 1. Load data + embeddings (shared across folds).
    (scripts_text, ratings, features_df, movie_names, script_files,
     _, scripts_text_sbert) = load_dataset()
    n = len(scripts_text)
    print(f"\nLoaded {n} scripts.")

    print("\n" + "-" * 70)
    print("  SBERT EMBEDDINGS (one set per pooling strategy)")
    print("-" * 70)
    embeddings_by_pool = {
        pool: _get_or_build_embeddings(scripts_text_sbert, sbert_model_name, pooling=pool)
        for pool in POOLING_STRATEGIES
    }

    # 2. Run each fold.
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    indices = np.arange(n)

    fold_metrics = {}      # {model: list of {RMSE, MAE, R2} per fold}
    pooled_y_true = []     # concatenated test labels across all folds
    pooled_preds = {}      # {model: concatenated test predictions across folds}

    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(indices), start=1):
        # Internal val split: ~15% of total = 18.75% of (train+val)
        idx_train, idx_val = train_test_split(
            train_val_idx,
            test_size=0.1875,
            random_state=seed + fold_idx,
        )
        idx_test = test_idx

        print(f"\n--- Fold {fold_idx}/{k}: "
              f"train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")

        all_preds = _run_one_split(
            scripts_text, scripts_text_sbert, ratings, features_df,
            embeddings_by_pool,
            idx_train, idx_val, idx_test,
            random_state=seed + fold_idx,
        )

        y_test = ratings[idx_test]
        pooled_y_true.append(y_test)
        for name, preds in all_preds.items():
            m = stats_utils.metrics(y_test, preds)
            fold_metrics.setdefault(name, []).append(m)
            pooled_preds.setdefault(name, []).append(preds)

        # Per-fold quick summary
        print(f"   Fold {fold_idx} test RMSE:")
        for name in _model_order():
            if name in fold_metrics:
                print(f"      {name:30s} {fold_metrics[name][-1]['RMSE']:.4f}")

    # 3. Aggregate per-model: mean ± std across folds.
    print("\n" + "-" * 70)
    print(f"  CV SUMMARY (mean ± std across {k} folds)")
    print("-" * 70)

    cv_summary = {}
    for name in _model_order():
        if name not in fold_metrics:
            continue
        arr = fold_metrics[name]
        rmse = np.array([m['RMSE'] for m in arr])
        mae = np.array([m['MAE'] for m in arr])
        r2 = np.array([m['R2'] for m in arr])
        cv_summary[name] = {
            'RMSE': {'mean': float(rmse.mean()), 'std': float(rmse.std(ddof=1)),
                     'per_fold': rmse.tolist()},
            'MAE': {'mean': float(mae.mean()), 'std': float(mae.std(ddof=1)),
                    'per_fold': mae.tolist()},
            'R2': {'mean': float(r2.mean()), 'std': float(r2.std(ddof=1)),
                   'per_fold': r2.tolist()},
        }
        print(f"\n  {name}")
        print(f"    RMSE: {rmse.mean():.4f} ± {rmse.std(ddof=1):.4f}  "
              f"(folds: {', '.join(f'{x:.4f}' for x in rmse)})")
        print(f"    MAE:  {mae.mean():.4f} ± {mae.std(ddof=1):.4f}")
        print(f"    R2:   {r2.mean():.4f} ± {r2.std(ddof=1):.4f}")

    # 4. Paired significance on pooled per-sample errors.
    pooled_y_true_arr = np.concatenate(pooled_y_true)
    pooled_preds_arr = {
        name: np.concatenate(parts) for name, parts in pooled_preds.items()
    }

    print("\n" + "-" * 70)
    print("  PAIRED SIGNIFICANCE vs SBERT+XGBoost (pooled across folds)")
    print("-" * 70)
    sig_results = {}
    if 'sbert_xgboost' in pooled_preds_arr:
        sbert_preds_pooled = pooled_preds_arr['sbert_xgboost']
        for name, preds in pooled_preds_arr.items():
            if name == 'sbert_xgboost':
                continue
            sig = stats_utils.paired_significance(pooled_y_true_arr, preds, sbert_preds_pooled)
            sig_results[name] = sig
            winner = 'SBERT' if sig['delta_mae'] > 0 else name
            print(f"\n  {name} vs sbert_xgboost")
            print(f"    delta_MAE = {sig['delta_mae']:+.4f}  "
                  f"(95% CI [{sig['ci_lo_delta_mae']:+.4f}, {sig['ci_hi_delta_mae']:+.4f}])")
            print(f"    Wilcoxon p = {sig['wilcoxon_p']:.2e}")
            print(f"    Winner: {winner}")

    # 5. Save.
    print("\n" + "-" * 70)
    print("  SAVING ARTIFACTS")
    print("-" * 70)
    out_json = {
        'mode': 'cv',
        'k': k,
        'random_state': seed,
        'sbert_model': sbert_model_name,
        'cv_summary': cv_summary,
        'paired_vs_sbert_pooled': sig_results,
    }
    json_path = os.path.join(RESULTS_DIR, f'cv_table{output_suffix}.json')
    with open(json_path, 'w') as f:
        json.dump(out_json, f, indent=2)
    print(f"   Saved {json_path}")

    # Markdown CV summary table.
    md_lines = [f"# Comparison Table ({k}-fold CV)\n",
                f"SBERT={sbert_model_name}, seed={seed}\n",
                "\n## Test Metrics (mean ± std across folds)\n",
                "| Model | RMSE | MAE | R² |",
                "|---|---|---|---|"]
    for name in _model_order():
        if name not in cv_summary:
            continue
        s = cv_summary[name]
        md_lines.append(
            f"| {name} | "
            f"{s['RMSE']['mean']:.4f} ± {s['RMSE']['std']:.4f} | "
            f"{s['MAE']['mean']:.4f} ± {s['MAE']['std']:.4f} | "
            f"{s['R2']['mean']:.4f} ± {s['R2']['std']:.4f} |"
        )

    if sig_results:
        md_lines.append("\n## Pooled Paired Comparison vs SBERT+XGBoost\n")
        md_lines.append("| Baseline | ΔMAE (baseline − SBERT) | 95% CI | Wilcoxon p |")
        md_lines.append("|---|---|---|---|")
        for name in _model_order():
            if name == 'sbert_xgboost' or name not in sig_results:
                continue
            sig = sig_results[name]
            md_lines.append(
                f"| {name} | {sig['delta_mae']:+.4f} | "
                f"[{sig['ci_lo_delta_mae']:+.4f}, {sig['ci_hi_delta_mae']:+.4f}] | "
                f"{sig['wilcoxon_p']:.2e} |"
            )
        md_lines.append("\n_Positive ΔMAE means SBERT has lower error (SBERT wins)._\n")

    md_path = os.path.join(RESULTS_DIR, f'cv_table{output_suffix}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))
    print(f"   Saved {md_path}")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="IMDb predictor experiment runner")
    parser.add_argument('--cv', type=int, default=0,
                        help="Run k-fold CV with this many folds (e.g. --cv 5). "
                             "Default 0 = single 70/15/15 split.")
    parser.add_argument('--seed', type=int, default=RANDOM_STATE,
                        help="Random seed for splitting.")
    parser.add_argument('--sbert-model', type=str, default=SBERT_MODEL_NAME,
                        help="SBERT model name (e.g. 'all-MiniLM-L6-v2', "
                             "'all-mpnet-base-v2'). Default: config.SBERT_MODEL_NAME.")
    parser.add_argument('--suffix', type=str, default='',
                        help="Suffix appended to output filenames so multiple "
                             "configurations don't overwrite each other "
                             "(e.g. '_mpnet').")
    args = parser.parse_args()

    if args.cv and args.cv > 1:
        main_cv(k=args.cv, seed=args.seed,
                sbert_model_name=args.sbert_model,
                output_suffix=args.suffix)
    else:
        main_single(sbert_model_name=args.sbert_model,
                    output_suffix=args.suffix)


if __name__ == '__main__':
    main()
