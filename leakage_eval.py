"""
Leakage-controlled evaluation (read-only w.r.t. the existing pipeline).

Three partitioning protocols are applied to exactly the same feature
matrices and the same learners, so any difference in reported skill is
attributable to the partition and nothing else:

    P1  random     : KFold(5, shuffle=True)              -- the optimistic protocol
    P2  grouped    : GroupKFold(5) over title families   -- no franchise straddles a fold
    P3  chronologic: train <= 2019, test 2020-2025       -- no future information

Representations come from artifacts already committed to the repository:
mean-pooled chunked SBERT embeddings (all-MiniLM-L6-v2, 384-d) plus the
three metadata columns. Record-level metadata is joined through the
positional map recovered by recover_alignment.py.

Writes results/leakage_eval.json and results/leakage_eval.md.

Usage:
    python leakage_eval.py
"""

import os
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBRegressor

import stats_utils
from audit_corpus import title_family

EXCEL = 'movie_lengths.xlsx'
RESULTS_DIR = 'results'
EMB_PATH = 'results/sbert_embeddings_all-MiniLM-L6-v2_mean.npz'
SEED = 42
YEAR_MIN, YEAR_MAX = 1900, 2025
HOLDOUT_FROM = 2020

META_COLS = ['year', 'movie_length', 'decade_encoded']


# ------------------------------------------------------------------
# Data assembly
# ------------------------------------------------------------------
def build():
    emb = np.load(EMB_PATH, allow_pickle=True)['embeddings'].astype(np.float64)
    align = np.load('results/alignment.npz', allow_pickle=True)
    excel_idx = align['excel_idx']
    ambiguous = align['ambiguous']

    df = pd.read_excel(EXCEL)
    df.columns = df.columns.str.strip()
    df = df.reset_index(drop=True)

    n = emb.shape[0]
    assert len(excel_idx) == n

    year = pd.to_numeric(df['Year'], errors='coerce').values[excel_idx]
    rating = pd.to_numeric(df['IMDb Rating'], errors='coerce').values[excel_idx]
    length = pd.to_numeric(df['Movie length'], errors='coerce').values[excel_idx]
    decade = df['Decade'].astype(str).values[excel_idx]
    title = df['Movie name'].astype(str).values[excel_idx]

    keep = (~ambiguous) & (year >= YEAR_MIN) & (year <= YEAR_MAX) & ~np.isnan(rating)
    print(f"Loaded records: {n}")
    print(f"  dropped, alignment-ambiguous : {int(ambiguous.sum())}")
    print(f"  dropped, year out of [{YEAR_MIN},{YEAR_MAX}] : "
          f"{int(((year < YEAR_MIN) | (year > YEAR_MAX)).sum())}")
    print(f"Analysis corpus: {int(keep.sum())}")

    emb = emb[keep]
    year = year[keep].astype(float)
    rating = rating[keep].astype(float)
    length = length[keep].astype(float)
    decade = decade[keep]
    title = title[keep]

    fam = np.array([title_family(t) for t in title], dtype=object)
    le = LabelEncoder()
    dec_enc = le.fit_transform(decade).astype(float)

    meta = pd.DataFrame({'year': year,
                         'movie_length': length,
                         'decade_encoded': dec_enc})

    uniq_fam, fam_id = np.unique(fam, return_inverse=True)
    print(f"Title families: {len(uniq_fam)} "
          f"(largest {np.bincount(fam_id).max()} members, "
          f"{int((np.bincount(fam_id) > 1).sum())} multi-member)")

    return emb, meta, rating, year, fam_id, fam, title


# ------------------------------------------------------------------
# Learners: each takes prepared train/test blocks, returns predictions
# ------------------------------------------------------------------
def _prep_meta(meta, tr, te, va=None):
    med = meta.iloc[tr].median(numeric_only=True)
    Xtr = meta.iloc[tr].fillna(med)
    Xte = meta.iloc[te].fillna(med)
    sc = StandardScaler().fit(Xtr)
    out = [sc.transform(Xtr), sc.transform(Xte)]
    if va is not None:
        out.append(sc.transform(meta.iloc[va].fillna(med)))
    return out


def m_predict_mean(emb, meta, y, tr, te, va, seed):
    return np.full(len(te), float(np.mean(y[tr])))


def m_ols_metadata(emb, meta, y, tr, te, va, seed):
    Xtr, Xte = _prep_meta(meta, tr, te)
    mdl = LinearRegression().fit(Xtr, y[tr])
    return np.clip(mdl.predict(Xte), 1.0, 10.0)


def m_ridge_sbert(emb, meta, y, tr, te, va, seed):
    sc = StandardScaler().fit(emb[tr])
    mdl = Ridge(alpha=1.5).fit(sc.transform(emb[tr]), y[tr])
    return np.clip(mdl.predict(sc.transform(emb[te])), 1.0, 10.0)


def _xgb(Xtr, ytr, Xva, yva, Xte, seed):
    mdl = XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        random_state=seed, n_jobs=-1, verbosity=0,
        tree_method='hist', early_stopping_rounds=20,
    )
    mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return np.clip(mdl.predict(Xte), 1.0, 10.0), mdl


def m_xgb_sbert(emb, meta, y, tr, te, va, seed):
    p, _ = _xgb(emb[tr], y[tr], emb[va], y[va], emb[te], seed)
    return p


def m_xgb_sbert_meta(emb, meta, y, tr, te, va, seed):
    Mtr, Mte, Mva = _prep_meta(meta, tr, te, va)
    Xtr = np.hstack([emb[tr], Mtr])
    Xva = np.hstack([emb[va], Mva])
    Xte = np.hstack([emb[te], Mte])
    p, _ = _xgb(Xtr, y[tr], Xva, y[va], Xte, seed)
    return p


MODELS = [
    ('predict_mean', m_predict_mean),
    ('ols_metadata', m_ols_metadata),
    ('ridge_sbert', m_ridge_sbert),
    ('xgb_sbert', m_xgb_sbert),
    ('xgb_sbert_meta', m_xgb_sbert_meta),
]
REFERENCE = 'xgb_sbert_meta'


# ------------------------------------------------------------------
# Protocols
# ------------------------------------------------------------------
def inner_val(tr_pool, rng, frac=0.1875):
    """Carve a validation slice out of a training pool (for early stopping)."""
    idx = np.array(tr_pool)
    rng.shuffle(idx)
    n_val = max(1, int(round(len(idx) * frac)))
    return idx[n_val:], idx[:n_val]


def run_cv(emb, meta, y, fam_id, splits, label):
    """Run every model over a list of (train_pool, test) index pairs."""
    per_fold = {name: [] for name, _ in MODELS}
    pooled_true, pooled_pred = [], {name: [] for name, _ in MODELS}
    leak_stats = []

    for k, (tr_pool, te) in enumerate(splits, start=1):
        rng = np.random.default_rng(SEED + k)
        tr, va = inner_val(tr_pool, rng)

        # How much family overlap does this partition leave behind?
        tr_fams = set(fam_id[tr_pool])
        n_leak = int(sum(1 for f in fam_id[te] if f in tr_fams))
        leak_stats.append(n_leak / len(te))

        for name, fn in MODELS:
            p = fn(emb, meta, y, tr, te, va, SEED + k)
            per_fold[name].append(stats_utils.metrics(y[te], p))
            pooled_pred[name].append(p)
        pooled_true.append(y[te])
        print(f"   [{label}] fold {k}: train={len(tr)} val={len(va)} "
              f"test={len(te)} family-overlap={leak_stats[-1]*100:.1f}%")

    pooled_true = np.concatenate(pooled_true)
    pooled_pred = {n: np.concatenate(v) for n, v in pooled_pred.items()}

    summary = {}
    for name, _ in MODELS:
        arr = per_fold[name]
        d = {}
        for m in ['RMSE', 'MAE', 'R2']:
            v = np.array([a[m] for a in arr])
            d[m] = {'mean': float(v.mean()),
                    'std': float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                    'per_fold': v.tolist()}
        d['pooled_bootstrap'] = stats_utils.bootstrap_metrics(
            pooled_true, pooled_pred[name], n_boot=1000)
        summary[name] = d

    sig = {}
    for name, _ in MODELS:
        if name == REFERENCE:
            continue
        sig[name] = stats_utils.paired_significance(
            pooled_true, pooled_pred[name], pooled_pred[REFERENCE])

    return {
        'summary': summary,
        'paired_vs_reference': sig,
        'reference': REFERENCE,
        'family_overlap_per_fold': leak_stats,
        'mean_family_overlap': float(np.mean(leak_stats)),
        'n_pooled': int(len(pooled_true)),
    }, pooled_true, pooled_pred


def run_holdout(emb, meta, y, year, fam_id, label):
    """Chronological holdout: fit on <= 2019, test on 2020-2025."""
    tr_pool = np.where(year < HOLDOUT_FROM)[0]
    te = np.where(year >= HOLDOUT_FROM)[0]

    # Validation = the most recent pre-holdout years, so early stopping is
    # itself decided without peeking forward.
    yr_tr = year[tr_pool]
    cut = np.quantile(yr_tr, 0.85)
    va = tr_pool[yr_tr >= cut]
    tr = tr_pool[yr_tr < cut]
    print(f"   [{label}] train={len(tr)} (<= {int(cut) - 1}) "
          f"val={len(va)} ({int(cut)}-2019) test={len(te)} (2020-2025)")

    tr_fams = set(fam_id[tr_pool])
    overlap = float(np.mean([f in tr_fams for f in fam_id[te]]))
    print(f"   [{label}] family overlap into holdout: {overlap*100:.1f}%")

    preds = {}
    for name, fn in MODELS:
        preds[name] = fn(emb, meta, y, tr, te, va, SEED)

    summary = {name: stats_utils.bootstrap_metrics(y[te], preds[name], n_boot=1000)
               for name, _ in MODELS}
    sig = {name: stats_utils.paired_significance(y[te], preds[name], preds[REFERENCE])
           for name, _ in MODELS if name != REFERENCE}

    return {
        'summary': summary,
        'paired_vs_reference': sig,
        'reference': REFERENCE,
        'n_train': int(len(tr)), 'n_val': int(len(va)), 'n_test': int(len(te)),
        'val_year_cut': int(cut),
        'family_overlap_holdout': overlap,
        'target_stats': {
            'train_mean': float(y[tr_pool].mean()),
            'train_sd': float(y[tr_pool].std(ddof=1)),
            'test_mean': float(y[te].mean()),
            'test_sd': float(y[te].std(ddof=1)),
        },
    }, y[te], preds


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    emb, meta, y, year, fam_id, fam, title = build()

    out = {'seed': SEED, 'n': int(len(y)),
           'embedding': 'all-MiniLM-L6-v2 mean-pooled, 384-d'}

    print("\n=== P1 random KFold(5) ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = [(tr, te) for tr, te in kf.split(np.arange(len(y)))]
    out['P1_random'], _, _ = run_cv(emb, meta, y, fam_id, splits, 'random')

    print("\n=== P2 GroupKFold(5) by title family ===")
    gkf = GroupKFold(n_splits=5)
    splits = [(tr, te) for tr, te in gkf.split(np.arange(len(y)), groups=fam_id)]
    out['P2_grouped'], _, _ = run_cv(emb, meta, y, fam_id, splits, 'grouped')

    print("\n=== P3 chronological holdout ===")
    out['P3_chronological'], y_te, preds = run_holdout(
        emb, meta, y, year, fam_id, 'chrono')

    # Target compression across the split boundary.
    print("\nTarget distribution:")
    print(f"   train <=2019 : mean={out['P3_chronological']['target_stats']['train_mean']:.4f} "
          f"sd={out['P3_chronological']['target_stats']['train_sd']:.4f}")
    print(f"   test 2020-25 : mean={out['P3_chronological']['target_stats']['test_mean']:.4f} "
          f"sd={out['P3_chronological']['target_stats']['test_sd']:.4f}")

    with open(os.path.join(RESULTS_DIR, 'leakage_eval.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/leakage_eval.json")

    # ---- markdown summary ------------------------------------------------
    lines = ["# Leakage-controlled evaluation\n",
             f"n = {out['n']}, embedding = {out['embedding']}, seed = {SEED}\n"]
    for proto, title_ in [('P1_random', 'P1 random KFold(5)'),
                          ('P2_grouped', 'P2 GroupKFold(5) by title family'),
                          ('P3_chronological', 'P3 chronological (train<=2019, test 2020-25)')]:
        lines.append(f"\n## {title_}\n")
        lines.append("| Model | RMSE | MAE | R2 |")
        lines.append("|---|---|---|---|")
        for name, _ in MODELS:
            s = out[proto]['summary'][name]
            if proto == 'P3_chronological':
                lines.append(f"| {name} | "
                             f"{s['RMSE']['point']:.4f} [{s['RMSE']['lo']:.4f}, {s['RMSE']['hi']:.4f}] | "
                             f"{s['MAE']['point']:.4f} [{s['MAE']['lo']:.4f}, {s['MAE']['hi']:.4f}] | "
                             f"{s['R2']['point']:.4f} [{s['R2']['lo']:.4f}, {s['R2']['hi']:.4f}] |")
            else:
                lines.append(f"| {name} | "
                             f"{s['RMSE']['mean']:.4f} ± {s['RMSE']['std']:.4f} | "
                             f"{s['MAE']['mean']:.4f} ± {s['MAE']['std']:.4f} | "
                             f"{s['R2']['mean']:.4f} ± {s['R2']['std']:.4f} |")
        lines.append(f"\n| Baseline | dMAE vs {REFERENCE} | 95% CI | Wilcoxon p |")
        lines.append("|---|---|---|---|")
        for name, _ in MODELS:
            if name == REFERENCE:
                continue
            g = out[proto]['paired_vs_reference'][name]
            lines.append(f"| {name} | {g['delta_mae']:+.4f} | "
                         f"[{g['ci_lo_delta_mae']:+.4f}, {g['ci_hi_delta_mae']:+.4f}] | "
                         f"{g['wilcoxon_p']:.2e} |")
    with open(os.path.join(RESULTS_DIR, 'leakage_eval.md'), 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print("Saved results/leakage_eval.md")


if __name__ == '__main__':
    main()
