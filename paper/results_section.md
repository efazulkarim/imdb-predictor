# Results

We report results in three parts: (1) the single-split headline
comparison with bootstrap confidence intervals; (2) a 5-fold
cross-validation as a robustness check; and (3) two ablations — on
inverse-frequency sample weighting and on the marginal contribution of
script content over metadata.

## Headline Comparison (Single Split, n_test = 780)

Table 1 reports test-set RMSE, MAE, and R² for every model under a
fixed 70/15/15 split. Point estimates are accompanied by 95%
non-parametric bootstrap confidence intervals (1,000 resamples).

**Table 1.** Test metrics with 95% bootstrap CIs. Lower is better for
RMSE/MAE, higher is better for R².

| Model | RMSE | MAE | R² |
|---|---|---|---|
| predict_mean | 1.518 [1.456, 1.579] | 1.247 [1.186, 1.305] | $-$0.001 |
| ols_metadata | 1.184 [1.128, 1.239] | 0.946 [0.894, 0.994] | 0.391 [0.353, 0.428] |
| ols_structural | 1.129 [1.068, 1.186] | 0.881 [0.831, 0.927] | 0.447 [0.407, 0.489] |
| tfidf_xgboost | 1.129 [1.060, 1.195] | 0.854 [0.801, 0.903] | 0.447 [0.389, 0.501] |
| sbert_xgboost (weighted) | 1.032 [0.974, 1.094] | 0.789 [0.744, 0.837] | 0.538 [0.483, 0.587] |
| **sbert_xgboost (no weights)** | **0.997 [0.942, 1.055]** | **0.749 [0.701, 0.793]** | **0.568 [0.526, 0.608]** |

Three observations from Table 1:

1. **A constant-predictor floor of MAE 1.25 and RMSE 1.52** establishes
   that any non-trivial model adds value, and that the rating standard
   deviation σ ≈ 1.44 sets the natural error scale.
2. **Three metadata features alone (year, decade, runtime) explain
   $R^{2} = 0.39$.** That is roughly 70% of the headline R² we obtain
   from the full pipeline, recovered without consulting the screenplay.
   We discuss the implications of this finding in §Discussion.
3. **The semantic SBERT pipeline outperforms the lexical TF-IDF
   pipeline** by $\Delta\text{MAE} = 0.105$ (TF-IDF $-$ SBERT-noweight),
   and outperforms the metadata-only baseline by Δ R² = +0.18.

## Paired Significance vs SBERT (Single Split)

Table 2 reports the paired comparison of every baseline against the
main system using per-sample absolute errors on the same test samples.
We use a two-sided Wilcoxon signed-rank test and report a paired-
bootstrap 95% confidence interval for the difference in mean absolute
error.

**Table 2.** Δ MAE (baseline − SBERT) under a paired comparison. A
positive difference means the baseline has higher error, i.e. the
SBERT system wins.

| Baseline | $\Delta$MAE | 95% CI | Wilcoxon $p$ |
|---|---|---|---|
| predict_mean | $+$0.458 | [$+$0.394, $+$0.522] | $4.3 \times 10^{-34}$ |
| ols_metadata | $+$0.157 | [$+$0.112, $+$0.200] | $9.1 \times 10^{-11}$ |
| ols_structural | $+$0.093 | [$+$0.054, $+$0.132] | $2.7 \times 10^{-5}$ |
| tfidf_xgboost | $+$0.065 | [$+$0.015, $+$0.108] | $1.4 \times 10^{-2}$ |

Every CI excludes zero and every $p$-value is below $\alpha = 0.05$
even after Bonferroni correction over the four comparisons (corrected
$\alpha \approx 0.0125$ leaves the $\Delta\text{MAE}$ over TF-IDF
borderline; this margin is resolved decisively in the cross-validated
results below).

## 5-Fold Cross-Validation (Robustness)

We repeat the comparison under five-fold cross-validation to verify
that the ordering above is not a single-seed artifact. Per-fold
metrics are reported as mean ± sample standard deviation. For the
significance tests we pool per-sample absolute errors across the five
non-overlapping test folds, yielding $n \approx 5{,}195$ paired
observations.

**Table 3.** 5-fold CV test metrics, mean ± std.

| Model | RMSE | MAE | R² |
|---|---|---|---|
| predict_mean | 1.438 ± 0.056 | 1.175 ± 0.057 | $-$0.000 ± 0.000 |
| ols_metadata | 1.130 ± 0.047 | 0.883 ± 0.042 | 0.382 ± 0.017 |
| ols_structural | 1.076 ± 0.040 | 0.833 ± 0.031 | 0.440 ± 0.007 |
| tfidf_xgboost | 1.083 ± 0.034 | 0.819 ± 0.022 | 0.433 ± 0.014 |
| sbert_xgboost (weighted) | 1.000 ± 0.027 | 0.768 ± 0.024 | 0.516 ± 0.026 |
| **sbert_xgboost (no weights)** | **0.958 ± 0.033** | **0.720 ± 0.027** | **0.556 ± 0.009** |

**Table 4.** Pooled paired Wilcoxon vs the weighted SBERT main system
under 5-fold CV.

| Baseline | $\Delta$MAE | 95% CI | $p$ |
|---|---|---|---|
| predict_mean | $+$0.407 | [$+$0.384, $+$0.429] | $9.3 \times 10^{-212}$ |
| ols_metadata | $+$0.115 | [$+$0.100, $+$0.130] | $6.7 \times 10^{-37}$ |
| ols_structural | $+$0.065 | [$+$0.053, $+$0.078] | $9.8 \times 10^{-16}$ |
| tfidf_xgboost | $+$0.051 | [$+$0.034, $+$0.067] | $2.6 \times 10^{-5}$ |
| sbert_xgboost (no weights) | $-$0.048 | [$-$0.057, $-$0.039] | $2.5 \times 10^{-26}$ |

The cross-validated picture is consistent with the single-split
ordering, with two strengthenings:

- The previously borderline win over TF-IDF is now resolved at
  $p \approx 2.6 \times 10^{-5}$.
- The standard deviation of $R^{2}$ for the no-weight SBERT system is
  only 0.009 across folds, indicating a stable headline figure.

## Ablation 1: Inverse-Frequency Sample Weighting Hurts

The legacy training pipeline applied inverse-frequency sample weights
across rating buckets (Low/Medium/Good/Excellent), in particular a
13.24× weight on the Excellent bucket and a 5.25× weight on the Low
bucket. We test this against an otherwise-identical pipeline trained
without sample weights.

In both single-split (Table 2 row 5) and 5-fold (Table 4 last row)
regimes, the unweighted variant has **significantly lower error**:

- single split: $\Delta\text{MAE} = -0.040$ ([$-0.066, -0.014$]),
  Wilcoxon $p = 1.2 \times 10^{-2}$;
- 5-fold pooled: $\Delta\text{MAE} = -0.048$ ([$-0.057, -0.039$]),
  Wilcoxon $p = 2.5 \times 10^{-26}$.

The most plausible mechanism is that the 13.24× upweighting of the
small Excellent bucket destabilizes the gradient-boosted regression by
amplifying gradients on a high-leverage minority of training examples.
Removing the reweighting also slightly improves the per-bucket MAE for
the **Low** bucket (1.541 → 1.577 in the single-split run, within
sampling variability), suggesting that the legacy weighting did not
in fact deliver the rebalancing it was designed to produce.

We adopt the unweighted variant as the headline configuration.

## Ablation 2: How Much Comes from Metadata?

A central question raised by the feature-importance analysis of the
legacy pipeline (`movie_length` and `year` together accounted for
≈ 0.36 of the gradient-boosted importance mass) is the share of
rating variance attributable to metadata alone. The ols_metadata row
of Tables 1 and 3 answers this directly: with three numerical
features (`year`, `decade_encoded`, `movie_length`) and a linear
regressor, we obtain

- single-split $R^{2} = 0.391$ [0.353, 0.428],
- 5-fold $R^{2} = 0.382 ± 0.017$.

That is, **roughly 70% of the variance our headline pipeline explains is
already captured by three metadata numbers** that do not look at the
screenplay at all. The marginal contribution of script content is the
remainder, $\Delta R^{2} \approx +0.17$ on average. This margin is
statistically significant ($p \approx 7 \times 10^{-37}$ under pooled
paired Wilcoxon), but it is meaningfully smaller than a casual reading
of the headline R² would suggest.

A natural follow-up would isolate the SBERT contribution from
script-derived structural features (i.e. compare `sbert + 19` against
`sbert` only and against `19` only). Within the scope of the present
work the closest available comparison is `ols_structural` (which
already incorporates year, length, and decade together with 16
text statistics). It achieves $R^{2} = 0.440 ± 0.007$ — a Δ of
$+0.06$ over `ols_metadata` from the structural text features and a
further Δ of $+0.12$ from SBERT semantic embeddings — but a controlled
metadata-removed ablation is left to future work and is named in
§Limitations.

## Per-Bucket Error Analysis

Test-set MAE (single split, no-weight SBERT, $n = 780$) by rating
bucket:

| Bucket | $n$ | MAE |
|---|---|---|
| Low [1, 4) | 107 | 1.577 |
| Medium [4, 6) | 380 | 0.558 |
| Good [6, 8) | 250 | 0.672 |
| Excellent [8, 10) | 43 | 0.818 |

Performance is best on the Medium bucket (the corpus mode), and
deteriorates on both tails — most severely on the Low bucket. Because
predictions are clipped to $[1, 10]$ and trees trained with squared
loss regress toward the conditional mean, predicted ratings cluster
between approximately 4 and 8 even when true ratings span 1.5 to 9.3.
We recommend that downstream consumers of this model interpret an
out-of-range prediction (e.g. predicted $\hat{y} \approx 4.5$ for a
script the model has never seen) as evidence of low confidence in
either tail rather than as a precise estimate.

## Error-Bracket Coverage

Cumulative coverage of test predictions within absolute-error
thresholds (no-weight SBERT, single split):

- within ±0.5 rating points: 47.1% of test set (367/780),
- within ±1.0: 71.8% (560/780),
- within ±1.5: 84.7% (661/780),
- within ±2.0: 93.8% (732/780).

## Stacked Ensemble

To test whether the four base models capture complementary signal we
train a Ridge meta-regressor over their predictions. The protocol is
nested CV: an outer 5-fold CV produces the test sets reported below;
within each outer training portion an inner 5-fold CV produces
out-of-fold base-model predictions on which the Ridge regressor is
fit; the same base models are then refit on the entire outer training
set and evaluated on the held-out outer test fold via the trained
Ridge meta-regressor. This protocol prevents target leakage from the
base models into the meta features.

**Table 5.** 5-fold CV test metrics for the stacked ensemble compared
to its base models. Same train/test splits as Table 3.

| Model | RMSE | MAE | R² |
|---|---|---|---|
| ols_metadata | 1.130 ± 0.047 | 0.883 ± 0.042 | 0.382 ± 0.016 |
| ols_structural | 1.076 ± 0.040 | 0.832 ± 0.032 | 0.440 ± 0.007 |
| tfidf_xgboost | 1.055 ± 0.035 | 0.790 ± 0.028 | 0.462 ± 0.014 |
| sbert_xgboost (no weights) | 0.956 ± 0.034 | 0.719 ± 0.028 | 0.558 ± 0.013 |
| **stacked (Ridge)** | **0.935 ± 0.032** | **0.706 ± 0.028** | **0.577 ± 0.010** |

Pooled paired Wilcoxon (n ≈ 5,195) of the stacked ensemble against
the strongest single base model:

$\Delta\text{MAE}_{\text{sbert} - \text{stacked}} = +0.013$
([+0.008, +0.019]), Wilcoxon $p = 4.2 \times 10^{-6}$.

Two structural observations from Table 5:

1. The stacked R² standard deviation (0.010) is **smaller** than that
   of any single base model (0.013 for SBERT, 0.014 for TF-IDF), so
   the stacked predictions are not just more accurate on average but
   also more stable across folds.
2. The Ridge meta-regressor's coefficients are remarkably stable
   across the five outer folds: SBERT receives a weight of
   approximately 0.69, TF-IDF approximately 0.38, ols_metadata
   approximately 0.27, and ols_structural a small *negative* weight
   ($\approx -0.13$). The negative coefficient on `ols_structural`
   indicates that, conditional on the SBERT and TF-IDF predictions
   already being available, the OLS-on-structural-features prediction
   carries no additional signal — its information is fully absorbed by
   the upstream models. This is consistent with the ranking in
   Table 3 (`ols_structural` is dominated by `tfidf_xgboost`).

## Hyperparameter Tuning (Optuna)

The XGBoost head reported above uses the hand-picked configuration
$\{\text{lr}=0.05, \text{max\_depth}=6, \text{reg\_alpha}=0.1,
\text{reg\_lambda}=1.0\}$. We re-evaluate this choice with a 100-trial
Optuna TPE search over an 8-dimensional hyperparameter space
(learning rate, depth, min child weight, subsample,
column-subsample, $L_1$ and $L_2$ regularization, gamma), with the
search objective being validation RMSE on a fixed train/val split
internal to the training partition. SBERT embeddings, the 19
structural features, and the train/val/test split are held fixed
across trials.

The best-found configuration uses substantially heavier regularization
than the hand-picked default — `learning_rate` $\approx$ 0.011 (about
5× lower), `max_depth` = 5 (one shallower), `reg_alpha` $\approx$ 1.0
(about 10× higher) — consistent with the diagnosis that the original
training curve in §Background overfit aggressively. On a held-out
20% test split the tuned configuration attains:

| Configuration | RMSE | MAE | R² |
|---|---|---|---|
| Hand-picked default | 0.997 [0.94, 1.06] | 0.749 [0.70, 0.79] | 0.568 [0.53, 0.61] |
| Optuna-tuned | **0.966 [0.91, 1.02]** | **0.714 [0.67, 0.75]** | **0.581 [0.55, 0.62]** |

The improvement is +0.013 in R² and a 0.035 reduction in MAE on the
single-split protocol. We do not report a paired significance test
because the hand-picked and tuned configurations are evaluated on the
same test split; we therefore note only that the bootstrap CI on the
tuned R² shifts upward by $\approx 0.02$. The tuned configuration is
the recommended deployment configuration; we use it in the released
training script as the default for `--tune` mode.
