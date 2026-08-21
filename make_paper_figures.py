"""
Figures for the leakage-audit paper.

All values are read from the committed JSON artifacts; nothing is
hard-coded. Output goes to figures/11_*.pdf|png so the existing figures
are left untouched.

Design notes: the three partition protocols form an ORDERED scale
(random -> grouped -> chronological, increasing strictness), so they get
a single-hue sequential ramp with monotone lightness rather than
categorical hues, plus hatching as a secondary encoding that survives
grayscale printing.

Usage:
    python make_paper_figures.py
"""

import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGDIR = 'figures'

# Sequential ramp, light -> dark, monotone lightness; hatch is the
# secondary encoding for grayscale print.
RAMP = ['#b9cfe3', '#6b9bc7', '#24486e']
HATCH = ['', '///', '...']
INK = '#1a1a1a'
MUTED = '#6b6b6b'
GRID = '#d8d8d8'
ACCENT = '#b04a1e'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'axes.edgecolor': MUTED,
    'axes.linewidth': 0.6,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'text.color': INK,
    'axes.labelcolor': INK,
    'figure.dpi': 400,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

MODELS = ['predict_mean', 'ols_metadata', 'ridge_sbert',
          'xgb_sbert', 'xgb_sbert_meta']
PRETTY = {'predict_mean': 'Mean', 'ols_metadata': 'Metadata\nOLS',
          'ridge_sbert': 'SBERT\nRidge', 'xgb_sbert': 'SBERT\nXGB',
          'xgb_sbert_meta': 'SBERT+meta\nXGB'}
PROTOS = [('P1_random', 'P1 random $k$-fold'),
          ('P2_grouped', 'P2 grouped by title family'),
          ('P3_chronological', 'P3 chronological 2020--25')]


def tidy(ax, ygrid=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)


def get(lk, proto, model, metric):
    s = lk[proto]['summary'][model]
    if proto == 'P3_chronological':
        return s[metric]['point'], s[metric]['lo'], s[metric]['hi']
    b = s['pooled_bootstrap'][metric]
    return b['point'], b['lo'], b['hi']


# ------------------------------------------------------------------
def fig_protocols(lk):
    """Skill degradation as the partition gets stricter."""
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5))
    width = 0.26
    x = np.arange(len(MODELS))

    for ax, metric, label in zip(axes, ['MAE', 'R2'],
                                 ['Test MAE (rating points)', r'Test $R^2$']):
        for j, (pk, plabel) in enumerate(PROTOS):
            pts, los, his = zip(*[get(lk, pk, m, metric) for m in MODELS])
            pts = np.array(pts)
            err = np.vstack([pts - np.array(los), np.array(his) - pts])
            ax.bar(x + (j - 1) * width, pts, width * 0.88,
                   color=RAMP[j], edgecolor=INK, linewidth=0.4,
                   hatch=HATCH[j], label=plabel,
                   yerr=err, error_kw=dict(ecolor=MUTED, elinewidth=0.6,
                                           capsize=1.4, capthick=0.6))
        ax.set_xticks(x)
        ax.set_xticklabels([PRETTY[m] for m in MODELS])
        ax.set_ylabel(label)
        tidy(ax)
        if metric == 'R2':
            ax.axhline(0, color=MUTED, linewidth=0.6)

    axes[0].legend(frameon=False, loc='upper right', handlelength=1.6,
                   borderpad=0.2, labelspacing=0.25)
    fig.tight_layout(pad=0.4)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIGDIR}/11_protocols.{ext}')
    plt.close(fig)
    print('wrote figures/11_protocols.pdf')


# ------------------------------------------------------------------
def fig_tokens(tok):
    """Where the 254-token content budget cuts the 256-word windows."""
    tr = tok['truncation']
    budget = tr['content_budget']
    fig, ax = plt.subplots(figsize=(3.4, 2.2))

    # Reconstruct a display distribution from the recorded quantiles is not
    # possible, so re-derive the histogram from the raw audit if present;
    # otherwise annotate the summary statistics directly.
    mean, med = tr['chunk_token_len']['mean'], tr['chunk_token_len']['median']
    p90, p99 = tr['chunk_token_len']['p90'], tr['chunk_token_len']['p99']
    mx = tr['chunk_token_len']['max']

    xs = [budget, med, mean, p90, p99, mx]
    labels = ['budget', 'median', 'mean', 'p90', 'p99', 'max']
    cols = [ACCENT] + [RAMP[2]] * 5

    ax.barh(range(len(xs)), xs, height=0.62,
            color=cols, edgecolor=INK, linewidth=0.4)
    ax.axvline(budget, color=ACCENT, linewidth=1.0, linestyle='--')
    ax.set_yticks(range(len(xs)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('WordPiece tokens per 256-word window')
    for i, v in enumerate(xs):
        ax.text(v + 8, i, f'{v:.0f}', va='center', fontsize=7, color=INK)
    ax.set_xlim(0, mx * 1.12)
    tidy(ax, ygrid=False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.text(0.98, 0.96,
            f"{tr['frac_chunks_over_budget']*100:.1f}% of windows truncated\n"
            f"{tr['frac_tokens_dropped_corpuswide']*100:.1f}% of tokens discarded",
            transform=ax.transAxes, ha='right', va='top', fontsize=7,
            color=ACCENT)
    fig.tight_layout(pad=0.4)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIGDIR}/12_token_budget.{ext}')
    plt.close(fig)
    print('wrote figures/12_token_budget.pdf')


# ------------------------------------------------------------------
def fig_temporal(xai):
    """Per-year bias and the compression of the predicted range."""
    py = xai['per_year_holdout']
    years = sorted(int(y) for y in py)
    true = [py[str(y)]['true_mean'] for y in years]
    pred = [py[str(y)]['pred_mean'] for y in years]
    ns = [py[str(y)]['n'] for y in years]
    cal = xai['calibration_holdout']

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.3))

    ax = axes[0]
    ax.plot(years, true, marker='o', markersize=4.5, linewidth=1.4,
            color=RAMP[2], label='observed mean')
    ax.plot(years, pred, marker='s', markersize=4.5, linewidth=1.4,
            color=ACCENT, linestyle='--', label='predicted mean')
    ax.set_xlabel('Release year (holdout)')
    ax.set_ylabel('IMDb rating')
    ax.legend(frameon=False, loc='lower right', handlelength=1.8,
              borderpad=0.2, labelspacing=0.25)
    tidy(ax)

    ax = axes[1]
    hp = np.load('results/chrono_holdout_predictions.npz')
    lo, hi = 2.0, 9.0
    ax.scatter(hp['y_true'], hp['pred_sbert_meta'], s=5, alpha=0.35,
               color=RAMP[2], linewidths=0, label='holdout film')
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=0.8, linestyle=':',
            label='identity')
    xs = np.linspace(lo, hi, 50)
    ax.plot(xs, cal['slope'] * xs + cal['intercept'], color=ACCENT,
            linewidth=1.4,
            label=f"fit, slope={cal['slope']:.3f}")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Observed rating')
    ax.set_ylabel('Predicted rating')
    ax.legend(frameon=False, loc='upper left', handlelength=1.8,
              borderpad=0.2, labelspacing=0.25)
    ax.text(0.97, 0.06,
            r'$\hat{\sigma}/\sigma=$' + f"{cal['dispersion_ratio']:.2f}",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5,
            color=INK)
    tidy(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)

    fig.tight_layout(pad=0.4)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIGDIR}/13_temporal.{ext}')
    plt.close(fig)
    print('wrote figures/13_temporal.pdf')


# ------------------------------------------------------------------
def fig_importance(xai):
    """Gain attribution vs. holdout permutation loss, side by side."""
    g = xai['gain_attribution']['block_gain']
    p = xai['block_permutation_chronological']

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.15))

    ax = axes[0]
    keys = ['sbert_384', 'metadata_3', 'structural_16']
    labels = ['SBERT (384 dims)', 'Metadata (3)', 'Structural (16)']
    vals = [g[k] * 100 for k in keys]
    ax.barh(range(3), vals, height=0.58, color=[RAMP[2], RAMP[1], RAMP[0]],
            edgecolor=INK, linewidth=0.4, hatch=['', '///', '...'])
    ax.set_yticks(range(3))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Share of total split gain (%)')
    for i, v in enumerate(vals):
        ax.text(v + 1.0, i, f'{v:.1f}', va='center', fontsize=7, color=INK)
    ax.set_xlim(0, max(vals) * 1.18)
    tidy(ax, ygrid=False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title('(a) In-sample gain, random split', loc='left')

    ax = axes[1]
    keys = ['metadata_3', 'movie_length', 'sbert_384', 'year', 'decade_encoded']
    labels = ['Metadata (3)', '  runtime', 'SBERT (384 dims)',
              '  year', '  decade']
    vals = [p[k]['delta_mae_mean'] for k in keys]
    errs = [p[k]['delta_mae_sd'] for k in keys]
    ax.barh(range(len(vals)), vals, height=0.58,
            color=[RAMP[1], RAMP[1], RAMP[2], RAMP[0], RAMP[0]],
            edgecolor=INK, linewidth=0.4,
            xerr=errs, error_kw=dict(ecolor=MUTED, elinewidth=0.6, capsize=1.4))
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(r'Holdout $\Delta$MAE when permuted')
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(v + e + 0.010, i, f'{v:+.3f}', va='center', fontsize=7, color=INK)
    ax.set_xlim(0, (max(v + e for v, e in zip(vals, errs))) * 1.34)
    tidy(ax, ygrid=False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title('(b) Permutation loss, chronological holdout', loc='left')

    fig.tight_layout(pad=0.4)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIGDIR}/14_importance.{ext}')
    plt.close(fig)
    print('wrote figures/14_importance.pdf')


def main():
    lk = json.load(open('results/leakage_eval.json'))
    tok = json.load(open('results/token_audit.json'))
    xai = json.load(open('results/xai_analysis.json'))
    fig_protocols(lk)
    fig_tokens(tok)
    fig_temporal(xai)
    fig_importance(xai)


if __name__ == '__main__':
    main()
