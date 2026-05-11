"""
Generate architecture diagrams for the paper.

Produces:
    figures/06_pipeline.png    block diagram of the base SBERT+XGBoost pipeline
    figures/07_stacking.png    stacked-ensemble architecture (4 base models + Ridge)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use('Agg')

FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

# Color palette
C_INPUT  = '#E8F0FE'
C_PREP   = '#FCE8E6'
C_MODEL  = '#E6F4EA'
C_HEAD   = '#FEF7E0'
C_OUTPUT = '#F3E8FD'
EDGE     = '#202124'

ARROW = dict(arrowstyle='->', color=EDGE, linewidth=1.0,
             shrinkA=0, shrinkB=0)


def _box(ax, xy, w, h, text, color, fs=10):
    x, y = xy
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.08',
        linewidth=1.2, edgecolor=EDGE, facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text,
            ha='center', va='center', fontsize=fs, color=EDGE)


def _arrow(ax, p_from, p_to):
    ax.annotate('', xy=p_to, xytext=p_from, arrowprops=ARROW)


def pipeline_diagram():
    fig, ax = plt.subplots(figsize=(12, 3.4), dpi=220)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.4)
    ax.axis('off')

    # All boxes on a single horizontal lane, fork at SBERT
    boxes = {
        'raw':    ((0.20, 1.20), 1.55, 1.00, 'Raw\nScreenplay', C_INPUT),
        'clean':  ((2.00, 1.20), 1.55, 1.00, 'Clean +\nChunk\n(256 / 50)', C_PREP),
        'sbert':  ((3.80, 1.20), 1.55, 1.00, 'SBERT\nMiniLM-L6\n(384-d)', C_MODEL),
        'pool':   ((5.60, 1.95), 1.55, 0.65, 'Mean-pool\nchunks', C_MODEL),
        'feat':   ((5.60, 0.55), 1.55, 0.65, '19 structural\nfeatures', C_PREP),
        'concat': ((7.40, 1.20), 1.55, 1.00, 'Concat 403-d\n+ StdScaler', C_HEAD),
        'xgb':    ((9.20, 1.20), 1.55, 1.00, 'XGBoost\n+ early\nstop (20)', C_MODEL),
        'y':      ((11.00, 1.50), 1.20, 0.40, r'$\hat{y}\;\in\;[1,10]$', C_OUTPUT),
    }
    for _, (xy, w, h, t, c) in boxes.items():
        _box(ax, xy, w, h, t, c, fs=9.5)

    # Helper: right edge midpoint -> left edge midpoint of next
    def right_mid(b):
        xy, w, h, _, _ = b
        return (xy[0] + w, xy[1] + h / 2)

    def left_mid(b):
        xy, w, h, _, _ = b
        return (xy[0], xy[1] + h / 2)

    _arrow(ax, right_mid(boxes['raw']),   left_mid(boxes['clean']))
    _arrow(ax, right_mid(boxes['clean']), left_mid(boxes['sbert']))
    # SBERT -> pool (up branch)
    _arrow(ax, right_mid(boxes['sbert']), left_mid(boxes['pool']))
    # SBERT -> feat (down branch, going around)
    _arrow(ax, (boxes['sbert'][0][0] + boxes['sbert'][2]/2, boxes['sbert'][0][1]),
              left_mid(boxes['feat']))
    # pool -> concat
    _arrow(ax, right_mid(boxes['pool']),
              (boxes['concat'][0][0], boxes['concat'][0][1] + 0.75))
    # feat -> concat
    _arrow(ax, right_mid(boxes['feat']),
              (boxes['concat'][0][0], boxes['concat'][0][1] + 0.25))
    # concat -> xgb
    _arrow(ax, right_mid(boxes['concat']), left_mid(boxes['xgb']))
    # xgb -> y
    _arrow(ax, right_mid(boxes['xgb']), left_mid(boxes['y']))

    plt.tight_layout()
    out = os.path.join(FIG_DIR, '06_pipeline.png')
    plt.savefig(out, bbox_inches='tight', dpi=220)
    plt.close(fig)
    print(f'saved {out}')


def stacking_diagram():
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=220)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    # base models
    base = [
        ('predict_mean',                                    4.50),
        ('ols_metadata\n(year, length, decade)',            3.55),
        ('ols_structural\n(19 features)',                   2.60),
        ('tfidf_xgboost\n(8k uni+bigrams)',                 1.65),
        ('sbert_xgboost\n(no weights)',                     0.70),
    ]
    base_boxes = []
    for label, y in base:
        xy = (0.30, y - 0.35)
        w, h = 2.6, 0.70
        color = C_MODEL if 'xgboost' in label else C_PREP
        _box(ax, xy, w, h, label, color, fs=9)
        base_boxes.append((xy, w, h))

    # OOF meta-features box
    oof_xy = (3.80, 1.90)
    oof_w, oof_h = 2.50, 1.70
    _box(ax, oof_xy, oof_w, oof_h,
         'Inner 5-fold\nout-of-fold\npredictions\n(meta features)',
         C_HEAD, fs=10)

    # Ridge box
    ridge_xy = (6.80, 2.30)
    ridge_w, ridge_h = 1.90, 0.90
    _box(ax, ridge_xy, ridge_w, ridge_h, r'Ridge  $\alpha = 1.0$', C_MODEL, fs=11)

    # Output
    out_xy = (9.30, 2.55)
    out_w, out_h = 1.10, 0.40
    _box(ax, out_xy, out_w, out_h, r'$\hat{y}_{\text{stack}}$', C_OUTPUT, fs=12)

    # Arrows: each base box -> OOF
    for (xy, w, h) in base_boxes:
        _arrow(ax, (xy[0] + w, xy[1] + h / 2),
                  (oof_xy[0], oof_xy[1] + oof_h / 2))
    # OOF -> Ridge
    _arrow(ax, (oof_xy[0] + oof_w, oof_xy[1] + oof_h / 2),
              (ridge_xy[0], ridge_xy[1] + ridge_h / 2))
    # Ridge -> output
    _arrow(ax, (ridge_xy[0] + ridge_w, ridge_xy[1] + ridge_h / 2),
              (out_xy[0], out_xy[1] + out_h / 2))

    # Coefficient annotation
    ax.text(6.80, 1.60,
            'mean meta-coefs over 5 outer folds:\n'
            'sbert ≈ 0.69,  tfidf ≈ 0.38,\n'
            'ols_meta ≈ 0.27,  ols_struct ≈ −0.13',
            ha='left', va='top', fontsize=8.5, color=EDGE,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FAFAFA',
                      edgecolor='#BDBDBD'))

    # Title
    ax.text(5.5, 5.15, 'Stacked Ensemble (nested CV)',
            ha='center', va='center', fontsize=13, color=EDGE, weight='bold')

    plt.tight_layout()
    out = os.path.join(FIG_DIR, '07_stacking.png')
    plt.savefig(out, bbox_inches='tight', dpi=220)
    plt.close(fig)
    print(f'saved {out}')


if __name__ == '__main__':
    pipeline_diagram()
    stacking_diagram()
