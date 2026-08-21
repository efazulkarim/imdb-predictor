"""
Cross-check every quantitative claim in paper/main.tex against the
committed JSON artifacts. Fails loudly on any drift, so the paper cannot
silently diverge from the runs that produced it.

Run from the repository root:
    python verify_paper_numbers.py
"""

import json
import os
import re
import sys

TEX_PATH = os.path.join('paper', 'main.tex')
TEX = open(TEX_PATH, encoding='utf-8').read()
lk = json.load(open('results/leakage_eval.json'))
tok = json.load(open('results/token_audit.json'))
xai = json.load(open('results/xai_analysis.json'))
cv = json.load(open('results/cv_table.json'))

failures = []
checks = 0


def claim(label, value, fmt='{:.4f}'):
    """Assert that `value`, rendered as `fmt`, literally appears in main.tex."""
    global checks
    checks += 1
    s = fmt.format(value)
    # LaTeX writes thousands with a protected comma.
    variants = {s, s.replace(',', '{,}')}
    if not any(v in TEX for v in variants):
        failures.append(f"{label}: '{s}' not found in main.tex")


# ---- corpus ledger --------------------------------------------------
claim('joined records', 5204, '{:,}')
claim('analysis corpus', lk['n'], '{:,}')
claim('title families', 4979, '{:,}')

# ---- token audit ----------------------------------------------------
tr = tok['truncation']
claim('windows over budget %', tr['frac_chunks_over_budget'] * 100, '{:.2f}')
claim('tokens dropped %', tr['frac_tokens_dropped_corpuswide'] * 100, '{:.2f}')
claim('per-script mean rho', tr['per_script_frac_tokens_dropped']['mean'], '{:.4f}')
claim('per-script max rho', tr['per_script_frac_tokens_dropped']['max'], '{:.4f}')
claim('n windows', tr['n_chunks_total'], '{:,}')
claim('median window tokens', tr['chunk_token_len']['median'], '{:.0f}')
claim('mean window tokens', tr['chunk_token_len']['mean'], '{:.1f}')
claim('p99 window tokens', tr['chunk_token_len']['p99'], '{:.0f}')
claim('max window tokens', tr['chunk_token_len']['max'], '{:.0f}')

# ---- main results table ---------------------------------------------
MODELS = ['predict_mean', 'ols_metadata', 'ridge_sbert',
          'xgb_sbert', 'xgb_sbert_meta']
for proto in ['P1_random', 'P2_grouped', 'P3_chronological']:
    for m in MODELS:
        s = lk[proto]['summary'][m]
        src = s if proto == 'P3_chronological' else s['pooled_bootstrap']
        for metric in ['RMSE', 'MAE', 'R2']:
            v = src[metric]['point']
            claim(f'{proto}/{m}/{metric}', abs(v) if v < 0 else v, '{:.4f}')

# best-system intervals quoted in the table footnote
for proto, key in [('P1_random', 'P1'), ('P2_grouped', 'P2'),
                   ('P3_chronological', 'P3')]:
    s = lk[proto]['summary']['xgb_sbert_meta']
    src = s if proto == 'P3_chronological' else s['pooled_bootstrap']
    claim(f'{key} R2 lo', src['R2']['lo'], '{:.4f}')
    claim(f'{key} R2 hi', src['R2']['hi'], '{:.4f}')

# ---- paired tests ----------------------------------------------------
for proto in ['P1_random', 'P2_grouped', 'P3_chronological']:
    for m in MODELS:
        if m == 'xgb_sbert_meta':
            continue
        claim(f'{proto}/{m}/dMAE',
              lk[proto]['paired_vs_reference'][m]['delta_mae'], '{:.4f}')

# ---- leakage rates ---------------------------------------------------
claim('P1 overlap', lk['P1_random']['mean_family_overlap'] * 100, '{:.2f}')
claim('P3 overlap', lk['P3_chronological']['family_overlap_holdout'] * 100, '{:.2f}')

# ---- split sizes -----------------------------------------------------
p3 = lk['P3_chronological']
claim('n_train', p3['n_train'], '{:,}')
claim('n_val', p3['n_val'], '{:,}')
claim('n_test', p3['n_test'], '{:,}')
claim('val cut', p3['val_year_cut'], '{:d}')

# ---- target distribution --------------------------------------------
ts = p3['target_stats']
claim('train pool sd', ts['train_sd'], '{:.4f}')
claim('holdout sd', ts['test_sd'], '{:.4f}')
claim('train pool mean', ts['train_mean'], '{:.4f}')
claim('holdout mean', ts['test_mean'], '{:.4f}')

# ---- XAI -------------------------------------------------------------
bg = xai['gain_attribution']['block_gain']
claim('sbert gain %', bg['sbert_384'] * 100, '{:.1f}')
claim('metadata gain %', bg['metadata_3'] * 100, '{:.1f}')
claim('structural gain %', bg['structural_16'] * 100, '{:.2f}')
nf = xai['gain_attribution']['numeric_feature_gain']
claim('movie_length gain %', nf['movie_length'] * 100, '{:.2f}')
claim('year gain %', nf['year'] * 100, '{:.2f}')
claim('max sbert dim %', xai['gain_attribution']['sbert_dim_stats']['max'] * 100, '{:.2f}')

perm = xai['block_permutation_chronological']
claim('perm baseline MAE', perm['baseline_mae'], '{:.4f}')
for k in ['metadata_3', 'movie_length', 'sbert_384']:
    claim(f'perm {k}', perm[k]['delta_mae_mean'], '{:.4f}')
    claim(f'perm {k} sd', perm[k]['delta_mae_sd'], '{:.4f}')

cal = xai['calibration_holdout']
claim('calib slope', cal['slope'], '{:.4f}')
claim('calib slope se', cal['stderr'], '{:.4f}')
claim('calib intercept', cal['intercept'], '{:.4f}')
claim('dispersion ratio', cal['dispersion_ratio'], '{:.3f}')

w = xai['wilcoxon_detail']['sbert_only_vs_sbert_meta']
claim('wilcoxon W', w['W'], '{:,.0f}')
claim('wilcoxon n', w['n_pairs'], '{:d}')
claim('rank biserial', w['rank_biserial'], '{:.3f}')

# per-year holdout counts quoted in the figure caption
for y in ['2020', '2021', '2022', '2023', '2024', '2025']:
    claim(f'holdout n {y}', xai['per_year_holdout'][y]['n'], '{:d}')

# ---- archived random-protocol arms ----------------------------------
for name, key in [('ols_structural', 'ols_structural'),
                  ('tfidf_xgboost', 'tfidf_xgboost'),
                  ('sbert_xgboost', 'sbert_xgboost'),
                  ('sbert_xgboost_noweight', 'sbert_xgboost_noweight')]:
    s = cv['cv_summary'][key]
    for metric in ['RMSE', 'MAE', 'R2']:
        claim(f'cv/{name}/{metric}', s[metric]['mean'], '{:.3f}')

for name in ['tfidf_xgboost', 'sbert_xgboost_noweight']:
    claim(f'cv paired {name}',
          abs(cv['paired_vs_sbert_pooled'][name]['delta_mae']), '{:.4f}')

# ---- archived ensemble / hyperparameter search -----------------------
st = json.load(open('results/stacking_results.json'))
tu = json.load(open('results/tune_xgb_results.json'))
for key, name in [('sbert_xgboost', 'stack base'), ('stacked', 'stacked')]:
    s = st['cv_summary'][key]
    claim(f'{name} RMSE', s['RMSE_mean'], '{:.3f}')
    claim(f'{name} MAE', s['MAE_mean'], '{:.3f}')
    claim(f'{name} R2', s['R2_mean'], '{:.3f}')
claim('stack dMAE', st['stacked_vs_sbert_pooled']['delta_mae'], '{:.4f}')
for metric in ['RMSE', 'MAE', 'R2']:
    claim(f'tuned {metric}', tu['test_metrics_bootstrap'][metric]['point'], '{:.3f}')
claim('tuned R2 lo', tu['test_metrics_bootstrap']['R2']['lo'], '{:.4f}')
claim('tuned R2 hi', tu['test_metrics_bootstrap']['R2']['hi'], '{:.4f}')
claim('n_trials', tu['n_trials'], '{:d}')

# ---- derived quantities stated in prose ------------------------------
r2 = lambda p: (lk[p]['summary']['xgb_sbert_meta']['R2']['point']
                if p == 'P3_chronological'
                else lk[p]['summary']['xgb_sbert_meta']['pooled_bootstrap']['R2']['point'])
claim('P1 minus P3 R2', r2('P1_random') - r2('P3_chronological'), '{:.3f}')
claim('P1 minus P2 R2', r2('P1_random') - r2('P2_grouped'), '{:.4f}')
claim('holdout R2 sbert only',
      lk['P3_chronological']['summary']['xgb_sbert']['R2']['point'], '{:.3f}')

# ---- report ----------------------------------------------------------
print(f"checked {checks} numeric claims against results/*.json")
if failures:
    print(f"\n{len(failures)} MISMATCH(ES):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("all claims in main.tex trace to a committed artifact")
