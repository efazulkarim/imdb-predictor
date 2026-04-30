# Methodology

This section describes the prediction pipeline, the baseline suite we
compare against, and the statistical protocol used to report results.

## Problem Formulation

We treat IMDb rating prediction as a univariate regression problem. Given
a screenplay $s$ and a small set of metadata features (release year,
decade, runtime), we predict the average IMDb user rating
$\hat{y} \in [1, 10]$, with a root-mean-squared-error objective and
predictions clipped to the rating scale at inference time.

## Preprocessing

Two text variants are produced from each raw screenplay:

1. **`clean_text`** — aggressive normalization used for the TF-IDF
   baseline and as input to all hand-crafted statistical features. Steps:
   lowercasing, removal of stage directions in brackets and parentheses,
   removal of all-caps character cues at start of line, removal of
   timestamps and scene numbers, retention of only `[a-zA-Z0-9'.,!?\s]`
   characters, whitespace normalization.
2. **`clean_text_for_sbert`** — light normalization that preserves the
   linguistic surface SBERT was trained on. Stage directions, character
   cues, INT./EXT. markers, timestamps, and scene numbers are removed,
   but case, punctuation, and paragraph structure are kept intact.

The first variant degrades SBERT performance by stripping case and most
punctuation, which the model relies on for sentence-boundary detection;
the second variant therefore feeds the SBERT encoder. The aggressive
variant remains in the pipeline only for the bag-of-words baseline.

## Hand-Crafted Structural Features (n=19)

For each script we compute 19 numerical features grouped as:

- **Length / volume.** `char_count`, `word_count`, `line_count`,
  `sentence_count`.
- **Vocabulary complexity.** `avg_word_length`, `unique_word_ratio`,
  `long_word_ratio` (fraction of tokens of length ≥ 8).
- **Sentence structure.** `avg_sentence_length`, `sentence_length_std`.
- **Dialogue.** `dialogue_density` (fraction of lines that are character
  cues), `unique_characters` (count of distinct ALL-CAPS speaker names).
- **Emotional indicators.** `exclamation_ratio`, `question_ratio`
  (per-100-word).
- **Action / structure.** `action_density` (bracketed/parenthesized stage
  direction density), `scene_count` (INT./EXT. headings),
  `words_per_scene`.
- **Metadata.** `year`, `decade_encoded` (label-encoded), `movie_length`
  (minutes).

All features are imputed with the training-set median where missing,
then standardized to zero mean and unit variance using a `StandardScaler`
fit on the training fold only.

## SBERT Encoding with Chunking

We use the `all-MiniLM-L6-v2` Sentence-Transformers checkpoint, which
produces 384-dimensional document embeddings and runs in seconds on a
modern CPU. SBERT's transformer backbone has a 256-token positional
limit, while screenplays in our corpus span tens of thousands of tokens.
We therefore chunk each document with the following protocol:

1. Split the cleaned text into word lists.
2. Slide a window of 256 words with a stride of 256 − 50 = 206 words,
   so each chunk overlaps the next by 50 words for context continuity.
3. Encode each chunk independently with SBERT.
4. Mean-pool the per-chunk embeddings to produce a single
   384-dimensional document representation.

Per-document embeddings are computed once for the entire corpus and
cached on disk, keyed by `(model_name, n_scripts, chunk_size,
overlap, pooling, preprocessing_version)`. Subsequent experiments
(cross-validation, ablations, alternative regressors) reuse the cache.

## Main System

The main predictor concatenates:

- the 384-dimensional SBERT document embedding, and
- the standardized 19-dimensional hand-crafted feature vector,

into a single 403-dimensional input, then trains an XGBoost regressor
with the following configuration: `max_depth=6`, `learning_rate=0.05`,
`reg_alpha=0.1`, `reg_lambda=1.0`, `tree_method=hist`,
`n_estimators=1000` with **early stopping**
(`early_stopping_rounds=20`) on the held-out validation split.

We do **not** use inverse-frequency sample weights in the final
configuration; an ablation reported in §Results shows they degrade
performance at $p \approx 3 \times 10^{-26}$ under a paired Wilcoxon
signed-rank test on cross-validated errors.

## Baselines

To isolate the marginal contribution of each pipeline component, we
compare against four progressively stronger baselines:

1. **predict_mean** — constant predictor returning the training-set
   mean rating. Lower bound; any model that fails to beat this is
   uninformative.
2. **ols_metadata** — ordinary least squares on three metadata features
   only: `year`, `decade_encoded`, `movie_length`. Quantifies the share
   of rating variance attributable to era and runtime.
3. **ols_structural** — OLS on the full 19-dimensional structural
   feature vector (which includes the three metadata features above).
   Quantifies the contribution of hand-crafted text statistics over and
   above metadata.
4. **tfidf_xgboost** — TF-IDF vectorization with a vocabulary of 8,000
   tokens (1- and 2-grams, English stopwords removed, sublinear TF,
   $\text{min\_df}=3$, $\text{max\_df}=0.85$), fed to the same XGBoost
   regressor as the main system. Provides a lexical-only counterpart to
   semantic SBERT embeddings.

All baselines are trained and evaluated on the **same** train/validation/
test splits as the main system, with predictions clipped to $[1, 10]$.

## Splits and Cross-Validation

We report two evaluation regimes:

- **Single split (headline).** A 70 / 15 / 15 train / validation / test
  partition with `random_state=42`, used for the headline numbers and
  for early-stopping the gradient-boosted models.
- **5-fold cross-validation (robustness).** Five non-overlapping test
  folds drawn by `KFold(n_splits=5, shuffle=True, random_state=42)`.
  Within each fold's training portion, an internal 18.75% validation
  split is reserved for early stopping (yielding effective fold sizes of
  approximately 65 / 15 / 20 percent). Per-fold metrics are aggregated
  as mean ± standard deviation; per-sample errors are pooled across
  folds for the significance tests in §Results.

## Statistical Protocol

We report three metrics per model: root-mean-squared error (RMSE), mean
absolute error (MAE), and the coefficient of determination ($R^{2}$).

For the single-split regime, we attach **non-parametric 95% bootstrap
confidence intervals** (1,000 resamples with replacement) to every
metric, computed by resampling the test set indices and recomputing
each metric on each resample.

For the cross-validated regime, we report the mean ± standard deviation
of the per-fold metrics.

To compare each baseline against the main system on **the same test
samples**, we additionally report:

- the difference in mean absolute error, $\Delta\text{MAE} = \text{MAE}
  _{\text{baseline}} - \text{MAE}_{\text{main}}$;
- a paired-bootstrap 95% confidence interval for $\Delta\text{MAE}$;
- a paired **Wilcoxon signed-rank test** on the per-sample absolute
  errors, treating each test sample as a paired observation. We report
  the two-sided $p$-value with no multiple-comparison correction;
  effective significance levels remain unambiguous because the smallest
  $p$-value in our comparisons is $\sim 10^{-5}$, well below any
  Bonferroni-adjusted $\alpha$ over the four baselines tested.

All statistical utilities are implemented in `stats_utils.py` and
exercised by the `experiments.py` runner; the per-model test predictions
are released alongside the code so that any reader can reproduce the
significance calculations directly from `results/predictions.npz`.

## Stacked Ensemble

To test whether the four baseline models capture complementary signal
relative to the SBERT main system, we additionally train a Ridge
meta-regressor (`alpha=1.0`) over the predictions of four base models:
`ols_metadata`, `ols_structural`, `tfidf_xgboost`, and
`sbert_xgboost` (no class weights, mean pooling). To prevent target
leakage from the base models into the meta features we use a
**nested cross-validation** protocol:

1. Outer 5-fold KFold partitions the corpus into five disjoint test
   folds.
2. Within each outer training portion, an inner 5-fold KFold produces
   *out-of-fold* base-model predictions: each row of the outer
   training set has been predicted by base models that did not see
   that row during training.
3. The Ridge meta-regressor is fit on these out-of-fold predictions.
4. The base models are re-fit on the entire outer training set and
   used to predict the outer test fold; the trained meta-regressor
   then maps these test-time base predictions to a stacked prediction.

This protocol — implemented in the released `stack.py` script — yields
a stacked test prediction for every screenplay across the five outer
folds, comparable to the predictions of the constituent base models
on the same outer folds.

## Hyperparameter Search

We perform a 100-trial Optuna [Akiba et al., 2019] search with a
TPE sampler over the XGBoost head's hyperparameter space:
$\text{learning\_rate}\in [0.01, 0.2]$ (log scale),
$\text{max\_depth}\in \{3,\ldots,10\}$,
$\text{min\_child\_weight}\in [0.5, 10]$ (log),
$\text{subsample}\in [0.6, 1.0]$,
$\text{colsample\_bytree}\in [0.6, 1.0]$,
$\text{reg\_alpha}\in [10^{-3}, 5]$ (log),
$\text{reg\_lambda}\in [10^{-3}, 10]$ (log),
$\text{gamma}\in [10^{-4}, 5]$ (log),
with `n_estimators=2000` capped by `early_stopping_rounds=30`
on a fixed train/val split. SBERT embeddings, the 19 structural
features, and the train/test partition are held fixed across trials
(see `tune_xgb.py`). After search completion the best-found
configuration is refit on the combined training-plus-validation set
(with a 10% slice reserved for early stopping) and evaluated on the
held-out test split with bootstrap CIs.

## Reproducibility

Every numerical result in this paper is produced by a single command:

- `python experiments.py` (single split) or
  `python experiments.py --cv 5` for the headline and CV tables;
- `python stack.py` for the stacked-ensemble nested-CV protocol;
- `python tune_xgb.py --trials 100` for the hyperparameter search.

Seeds, splits, hyperparameters, and SBERT embedding cache keys are
all emitted to JSON artifacts in `results/`. Re-running the pipeline
from a fresh checkout reproduces the reported numbers to
floating-point tolerance.
