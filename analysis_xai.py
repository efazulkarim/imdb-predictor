"""
Interpretability and temporal-shift analysis (read-only).

Produces the numbers behind the paper's explainability subsection:

  A. Gain attribution of the shipped model (imdb_model.pkl) split into
     384 SBERT dimensions / 16 structural statistics / 3 metadata columns.
  B. Block-permutation importance under the chronological protocol, where
     a whole feature block is shuffled and the holdout loss increase is
     measured.
  C. Per-year holdout error and calibration (slope, intercept, prediction
     dispersion) to characterise temporal target compression.
  D. Wilcoxon signed-rank statistics reported properly: W, the number of
     non-tied pairs, and the matched-pairs rank-biserial effect size.

Writes results/xai_analysis.json.

Usage:
    python analysis_xai.py
"""

import json
import pickle

import numpy as np
from scipy import stats as scipy_stats
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from leakage_eval import build, _prep_meta, HOLDOUT_FROM, SEED

RESULTS_DIR = 'results'
STRUCTURAL_16 = [
    'char_count', 'word_count', 'line_count', 'avg_word_length',
    'unique_word_ratio', 'long_word_ratio', 'sentence_count',
    'avg_sentence_length', 'sentence_length_std', 'dialogue_density',
    'unique_characters', 'exclamation_ratio', 'question_ratio',
    'action_density', 'scene_count', 'words_per_scene',
]


# ------------------------------------------------------------------
def gain_attribution():
    """A. Split the shipped model's total gain across feature blocks."""
    pkg = pickle.load(open('imdb_model.pkl', 'rb'))
    fi = np.asarray(pkg['model'].feature_importances_, dtype=float)
    names = ([f'sbert_{i}' for i in range(384)]
             + list(pkg['scaler'].feature_names_in_))
    assert len(names) == len(fi)

    sbert = fi[:384]
    numeric = fi[384:]
    struct = numeric[:16]
    meta = numeric[16:]

    per_feature = {names[384 + j]: float(numeric[j]) for j in range(len(numeric))}
    order = np.argsort(-fi)
    return {
        'model': pkg['model_name'],
        'n_features': len(fi),
        'block_gain': {
            'sbert_384': float(sbert.sum()),
            'structural_16': float(struct.sum()),
            'metadata_3': float(meta.sum()),
        },
        'sbert_dim_stats': {
            'max': float(sbert.max()),
            'mean': float(sbert.mean()),
            'top10_sum': float(np.sort(sbert)[-10:].sum()),
        },
        'numeric_feature_gain': per_feature,
        'top20_overall': [{'feature': names[i], 'gain': float(fi[i])}
                          for i in order[:20]],
        'rank_of_numeric': {names[384 + j]: int(list(order).index(384 + j) + 1)
                            for j in range(len(numeric))},
    }


# ------------------------------------------------------------------
def chrono_fit(emb, meta, y, year):
    tr_pool = np.where(year < HOLDOUT_FROM)[0]
    te = np.where(year >= HOLDOUT_FROM)[0]
    yr_tr = year[tr_pool]
    cut = np.quantile(yr_tr, 0.85)
    va = tr_pool[yr_tr >= cut]
    tr = tr_pool[yr_tr < cut]

    Mtr, Mte, Mva = _prep_meta(meta, tr, te, va)
    Xtr = np.hstack([emb[tr], Mtr])
    Xva = np.hstack([emb[va], Mva])
    Xte = np.hstack([emb[te], Mte])

    mdl = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                       random_state=SEED, n_jobs=-1, verbosity=0,
                       tree_method='hist', early_stopping_rounds=20)
    mdl.fit(Xtr, y[tr], eval_set=[(Xva, y[va])], verbose=False)
    return mdl, Xte, te, tr, va


def block_permutation(mdl, Xte, y_te, n_repeats=20):
    """B. Shuffle a block of columns, measure the holdout MAE increase."""
    base = float(np.mean(np.abs(np.clip(mdl.predict(Xte), 1, 10) - y_te)))
    blocks = {
        'sbert_384': list(range(384)),
        'year': [384],
        'movie_length': [385],
        'decade_encoded': [386],
        'metadata_3': [384, 385, 386],
    }
    out = {'baseline_mae': base}
    rng = np.random.default_rng(SEED)
    for name, cols in blocks.items():
        deltas = []
        for _ in range(n_repeats):
            Xp = Xte.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, cols] = Xp[perm][:, cols]
            mae = float(np.mean(np.abs(np.clip(mdl.predict(Xp), 1, 10) - y_te)))
            deltas.append(mae - base)
        out[name] = {'delta_mae_mean': float(np.mean(deltas)),
                     'delta_mae_sd': float(np.std(deltas, ddof=1))}
        print(f"   permute {name:16s} dMAE = {np.mean(deltas):+.4f} "
              f"± {np.std(deltas, ddof=1):.4f}")
    return out


def wilcoxon_full(err_a, err_b):
    """D. W statistic, non-tied pair count, p, rank-biserial effect size."""
    d = err_a - err_b
    nz = d[d != 0]
    w, p = scipy_stats.wilcoxon(err_a, err_b, zero_method='wilcox')
    ranks = scipy_stats.rankdata(np.abs(nz))
    r_plus = ranks[nz > 0].sum()
    r_minus = ranks[nz < 0].sum()
    total = r_plus + r_minus
    return {'W': float(w), 'n_pairs': int(len(err_a)),
            'n_nonzero': int(len(nz)), 'p': float(p),
            'rank_biserial': float((r_plus - r_minus) / total) if total else 0.0}


def main():
    emb, meta, y, year, fam_id, fam, title = build()

    print("\nA. Gain attribution of the shipped model")
    gain = gain_attribution()
    b = gain['block_gain']
    print(f"   SBERT(384) {b['sbert_384']*100:.1f}%  "
          f"structural(16) {b['structural_16']*100:.1f}%  "
          f"metadata(3) {b['metadata_3']*100:.1f}%")

    print("\nB. Block-permutation importance, chronological holdout")
    mdl, Xte, te, tr, va = chrono_fit(emb, meta, y, year)
    y_te = y[te]
    perm = block_permutation(mdl, Xte, y_te)

    print("\nC. Temporal behaviour of the holdout")
    pred = np.clip(mdl.predict(Xte), 1, 10)
    yr_te = year[te].astype(int)
    per_year = {}
    for yy in sorted(set(yr_te)):
        m = yr_te == yy
        per_year[int(yy)] = {
            'n': int(m.sum()),
            'mae': float(np.mean(np.abs(pred[m] - y_te[m]))),
            'bias': float(np.mean(pred[m] - y_te[m])),
            'true_mean': float(y_te[m].mean()),
            'pred_mean': float(pred[m].mean()),
        }
        print(f"   {yy}: n={m.sum():4d} MAE={per_year[int(yy)]['mae']:.4f} "
              f"bias={per_year[int(yy)]['bias']:+.4f} "
              f"true={y_te[m].mean():.3f} pred={pred[m].mean():.3f}")

    slope, intercept, r, p, se = scipy_stats.linregress(y_te, pred)
    calib = {
        'slope': float(slope), 'intercept': float(intercept),
        'stderr': float(se), 'r': float(r), 'p': float(p),
        'sd_true': float(y_te.std(ddof=1)),
        'sd_pred': float(pred.std(ddof=1)),
        'dispersion_ratio': float(pred.std(ddof=1) / y_te.std(ddof=1)),
        'train_sd': float(y[tr].std(ddof=1)),
        'train_mean': float(y[tr].mean()),
    }
    print(f"   calibration slope={slope:.4f} (se {se:.4f}), "
          f"intercept={intercept:.4f}")
    print(f"   sd(pred)/sd(true) = {calib['dispersion_ratio']:.4f}")

    print("\nD. Wilcoxon detail, chronological holdout")
    # SBERT-only vs SBERT+metadata on the same holdout.
    Xte_sb = emb[te]
    mdl_sb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                          random_state=SEED, n_jobs=-1, verbosity=0,
                          tree_method='hist', early_stopping_rounds=20)
    mdl_sb.fit(emb[tr], y[tr], eval_set=[(emb[va], y[va])], verbose=False)
    pred_sb = np.clip(mdl_sb.predict(Xte_sb), 1, 10)

    wil = {
        'sbert_only_vs_sbert_meta':
            wilcoxon_full(np.abs(pred_sb - y_te), np.abs(pred - y_te)),
    }
    for k, v in wil.items():
        print(f"   {k}: W={v['W']:.1f} n={v['n_pairs']} "
              f"(non-tied {v['n_nonzero']}) p={v['p']:.4g} "
              f"rank-biserial={v['rank_biserial']:+.4f}")

    np.savez('results/chrono_holdout_predictions.npz',
             y_true=y_te, pred_sbert_meta=pred, pred_sbert=pred_sb,
             year=yr_te)
    print("Saved results/chrono_holdout_predictions.npz")

    out = {'gain_attribution': gain,
           'block_permutation_chronological': perm,
           'per_year_holdout': per_year,
           'calibration_holdout': calib,
           'wilcoxon_detail': wil,
           'structural_feature_list': STRUCTURAL_16}
    with open(f'{RESULTS_DIR}/xai_analysis.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/xai_analysis.json")


if __name__ == '__main__':
    main()
