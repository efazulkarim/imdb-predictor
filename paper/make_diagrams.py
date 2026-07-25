# ============================================================
# ACADEMIC ARCHITECTURE DIAGRAM GENERATOR
# ============================================================
"""
Generates publication-quality architecture diagrams for the paper:
    - figures/06_pipeline.png    Scientific block diagram of the main SBERT+XGBoost pipeline
    - figures/07_stacking.png    Formal block diagram of the stacked ensemble architecture
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

# Formal Academic Palette (Slate/Monochrome/Publication style)
C_INPUT  = '#F8F9FA'
C_PREP   = '#EDF2F7'
C_MODEL  = '#E2E8F0'
C_HEAD   = '#CBD5E0'
C_OUTPUT = '#E2E8F0'
EDGE     = '#1A202C'

ARROW = dict(arrowstyle='->', color=EDGE, linewidth=1.2, shrinkA=0, shrinkB=0)


def _box(ax, xy, w, h, text, color, fs=9.5, weight='normal'):
    x, y = xy
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle='square,pad=0.02',
        linewidth=1.2, edgecolor=EDGE, facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text,
            ha='center', va='center', fontsize=fs, color=EDGE, fontweight=weight)


def _arrow(ax, p_from, p_to):
    ax.annotate('', xy=p_to, xytext=p_from, arrowprops=ARROW)


def pipeline_diagram():
    fig, ax = plt.subplots(figsize=(12, 3.4), dpi=300)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.4)
    ax.axis('off')

    boxes = {
        'raw':    ((0.20, 1.20), 1.55, 1.00, 'Raw Screenplay\nDocument ($d$)', C_INPUT, 'bold'),
        'clean':  ((2.00, 1.20), 1.55, 1.00, 'Chunk Windowing\n$L=256, O=50$', C_PREP, 'normal'),
        'sbert':  ((3.80, 1.20), 1.55, 1.00, 'SBERT Encoder\n$\\mathcal{E}(c_k) \\in \\mathbb{R}^{d}$', C_MODEL, 'bold'),
        'pool':   ((5.60, 1.95), 1.55, 0.65, 'Mean-Pooling\n$\\bar{e} = \\frac{1}{K}\\sum e_k$', C_MODEL, 'normal'),
        'feat':   ((5.60, 0.55), 1.55, 0.65, '19 Structural\nFeatures ($\\mathbf{z}$)', C_PREP, 'normal'),
        'concat': ((7.40, 1.20), 1.55, 1.00, 'Concatenation\n$\\mathbf{x} = [\\bar{e} ; \\mathbf{z}]$', C_HEAD, 'bold'),
        'xgb':    ((9.20, 1.20), 1.55, 1.00, 'XGBoost Regressor\n$f(\\mathbf{x}) = \\sum f_t$', C_MODEL, 'bold'),
        'y':      ((11.00, 1.50), 1.30, 0.40, r'$\hat{y}_d \in [1.0, 10.0]$', C_OUTPUT, 'bold'),
    }

    for _, (xy, w, h, t, c, wt) in boxes.items():
        _box(ax, xy, w, h, t, c, fs=9, weight=wt)

    def right_mid(b):
        xy, w, h, _, _, _ = b
        return (xy[0] + w, xy[1] + h / 2)

    def left_mid(b):
        xy, w, h, _, _, _ = b
        return (xy[0], xy[1] + h / 2)

    _arrow(ax, right_mid(boxes['raw']),   left_mid(boxes['clean']))
    _arrow(ax, right_mid(boxes['clean']), left_mid(boxes['sbert']))
    _arrow(ax, right_mid(boxes['sbert']), left_mid(boxes['pool']))
    _arrow(ax, (boxes['sbert'][0][0] + boxes['sbert'][2]/2, boxes['sbert'][0][1]), left_mid(boxes['feat']))
    _arrow(ax, right_mid(boxes['pool']), (boxes['concat'][0][0], boxes['concat'][0][1] + 0.75))
    _arrow(ax, right_mid(boxes['feat']), (boxes['concat'][0][0], boxes['concat'][0][1] + 0.25))
    _arrow(ax, right_mid(boxes['concat']), left_mid(boxes['xgb']))
    _arrow(ax, right_mid(boxes['xgb']), left_mid(boxes['y']))

    plt.title("End-to-End Screenplay Vectorization and Prediction Pipeline Architecture", fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, '06_pipeline.png')
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'saved {out}')


def stacking_diagram():
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    base = [
        ('OLS Metadata Baseline',                             4.50),
        ('OLS Structural Baseline',                           3.55),
        ('TF-IDF + XGBoost Baseline',                         2.60),
        ('SBERT + XGBoost Base Model',                        1.65),
        ('SBERT + LightGBM Base Model',                       0.70),
    ]

    base_boxes = []
    for label, y in base:
        xy = (0.30, y - 0.35)
        w, h = 2.7, 0.70
        _box(ax, xy, w, h, label, C_PREP, fs=8.5)
        base_boxes.append((xy, w, h))

    oof_xy = (3.80, 1.90)
    oof_w, oof_h = 2.50, 1.70
    _box(ax, oof_xy, oof_w, oof_h, 'Nested 5-Fold Cross-Validation\nOut-of-Fold Base Predictions\n$\\mathbf{M} \\in \\mathbb{R}^{n \\times 5}$', C_HEAD, fs=9.5, weight='bold')

    ridge_xy = (6.80, 2.30)
    ridge_w, ridge_h = 1.90, 0.90
    _box(ax, ridge_xy, ridge_w, ridge_h, r'Ridge Meta-Regressor' + '\n' + r'$\min_{\mathbf{w}} \|y - \mathbf{M}\mathbf{w}\|_2^2 + \alpha\|\mathbf{w}\|_2^2$', C_MODEL, fs=9, weight='bold')

    out_xy = (9.30, 2.55)
    out_w, out_h = 1.20, 0.40
    _box(ax, out_xy, out_w, out_h, r'$\hat{y}_{\text{stack}} \in [1, 10]$', C_OUTPUT, fs=10, weight='bold')

    for (xy, w, h) in base_boxes:
        _arrow(ax, (xy[0] + w, xy[1] + h / 2), (oof_xy[0], oof_xy[1] + oof_h / 2))
    _arrow(ax, (oof_xy[0] + oof_w, oof_xy[1] + oof_h / 2), (ridge_xy[0], ridge_xy[1] + ridge_h / 2))
    _arrow(ax, (ridge_xy[0] + ridge_w, ridge_xy[1] + ridge_h / 2), (out_xy[0], out_xy[1] + out_h / 2))

    ax.text(5.5, 5.15, 'Stacked Ensemble Meta-Architecture (Nested 5-Fold Protocol)', ha='center', va='center', fontsize=12, color=EDGE, weight='bold')

    plt.tight_layout()
    out = os.path.join(FIG_DIR, '07_stacking.png')
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'saved {out}')


if __name__ == '__main__':
    pipeline_diagram()
    stacking_diagram()
