# Abstract

We study the extent to which a movie's IMDb rating can be predicted
from its screenplay alone, using a corpus of 5,195 feature-film
screenplays joined with IMDb metadata. Our predictor combines
mean-pooled chunked SBERT (`all-MiniLM-L6-v2`) embeddings of the
script with 19 hand-crafted structural features and a gradient-boosted
regressor with early stopping; a Ridge meta-regressor is then stacked
over four base models (predict-mean, OLS on metadata, TF-IDF +
XGBoost, and SBERT + XGBoost) to produce the final prediction. We
benchmark every component against the others and report 95% bootstrap
confidence intervals together with paired Wilcoxon significance
tests on per-sample errors. Under five-fold cross-validation the
stacked system attains $\text{RMSE} = 0.935 \pm 0.032$,
$\text{MAE} = 0.706 \pm 0.028$, and $R^{2} = 0.577 \pm 0.010$,
significantly improving over the strongest single base model
(SBERT + XGBoost, $R^{2} = 0.558 \pm 0.013$; paired Wilcoxon
$p \approx 4 \times 10^{-6}$) and over every other baseline
(p \le 3 \times 10^{-5}$). A 100-trial Optuna hyperparameter search
on the XGBoost head independently yields a comparable lift
($R^{2} = 0.581$ on a held-out test split), driven by markedly
heavier regularization than the hand-picked defaults. Two findings
deserve emphasis. First, **three metadata features alone explain
$R^{2} \approx 0.38$**: the marginal contribution of script content
via SBERT is $\Delta R^{2} \approx +0.18$, statistically robust but
smaller than the headline R² alone suggests. Second, the **legacy
practice of inverse-frequency sample weighting harms performance**
in this setting ($\Delta\text{MAE} = -0.048$,
$p \approx 3 \times 10^{-26}$), contradicting earlier reports. We additionally disclose two
properties of the dataset whose handling we believe is important for
the integrity of future screenplay-rating studies: an editorial
selection bias that produces structural gaps in the rating
distribution (zero records in $[4.3, 5.0)$ and $[6.0, 7.0)$), and a
strong year-rating confound consistent with survivorship bias. Our
training pipeline runs in minutes on a CPU and produces a 2 MB model
suitable for deployment without specialized hardware.

# 1. Introduction

Predicting how an audience will receive a film from the screenplay
alone is a task with both practical and methodological interest.
**Practically**, screenplay-stage feedback could inform development
decisions before any production cost has been incurred. **Method-
ologically**, the task is a stress-test for long-document language
models: a feature-film screenplay is roughly $2.5 \times 10^4$
words, well beyond the context window of standard transformer
encoders, and the target — an aggregate audience rating — depends
on factors well beyond the script itself.

Three lines of prior work address this problem. (i) Classical
text-feature pipelines treat the screenplay as a bag of words or
$n$-grams and feed the resulting sparse vectors to a regressor.
These approaches scale to long documents but ignore semantics.
(ii) Pre-trained sentence encoders such as Sentence-BERT (SBERT)
produce dense semantic embeddings; combined with chunking, they can
represent documents that exceed their native context window. (iii)
End-to-end long-document transformers such as Longformer can
process sequences up to a few thousand tokens but require
fine-tuning, GPU compute, and a regression head whose capacity must
be tuned independently of the encoder.

This paper focuses on (ii) — a frozen SBERT encoder combined with
classical engineered features and a gradient-boosted regressor —
and asks: how does this hybrid compare to *carefully chosen
non-transformer baselines*, what is the marginal contribution of
the SBERT component once metadata is accounted for, and which
training conventions in published prior work generalize to a
properly controlled comparison?

## Research questions

We organize the paper around three questions:

- **RQ1: Strength of the SBERT pipeline.** Does mean-pooled SBERT +
  XGBoost outperform classical baselines (constant predictor,
  metadata-only OLS, structural-feature OLS, TF-IDF + XGBoost) on a
  fixed train/test partition with rigorous statistical reporting?

- **RQ2: Decomposition.** What share of the headline predictive
  performance is recoverable from metadata alone (year, decade,
  runtime), and what is the marginal contribution of the script
  content via SBERT?

- **RQ3: Robustness of training conventions.** Does the legacy
  practice of inverse-frequency sample weighting across rating
  buckets actually improve performance, as commonly claimed?

## Contributions

We make five contributions:

1. **A rigorous baseline suite for screenplay rating prediction**:
   constant-mean, OLS on metadata, OLS on 19 hand-crafted
   structural features, TF-IDF + XGBoost, and SBERT + XGBoost — all
   trained on the same splits and reported with bootstrap 95% CIs
   and paired Wilcoxon significance tests.

2. **A stacked-ensemble predictor** in which a Ridge meta-regressor
   is trained on out-of-fold predictions from the four base models;
   the stacked model attains $R^{2} = 0.577 \pm 0.010$ under 5-fold
   CV, a statistically significant lift over the strongest single
   base model ($\Delta\text{MAE} = +0.013$, $p \approx 4 \times
   10^{-6}$), with a *tighter* per-fold standard deviation than any
   constituent (0.010 vs 0.013).

3. **A decomposition of the headline R² between metadata and
   script content**, showing that ≈ 70% of explained variance is
   recoverable from three metadata numbers without consulting the
   screenplay, and the screenplay's marginal contribution is
   $\Delta R^{2} \approx +0.18$ (significant at $p \ll 10^{-30}$).

4. **A negative result on inverse-frequency sample weighting** for
   regression with extreme bucket imbalance: it significantly
   *increases* error in our setting ($p \approx 3 \times 10^{-26}$
   under 5-fold pooled paired Wilcoxon), in contradiction to
   earlier reports.

5. **An honest characterization of an IMSDb-style screenplay
   corpus**, including the bimodal-by-construction rating
   distribution (with structural gaps), the strong year-rating
   confound, and the implications of these properties for any
   downstream evaluation. We release dataset summary statistics and
   per-model test predictions to enable independent
   reproduction.

A 100-trial Optuna hyperparameter search on the XGBoost head
provides an additional methodology contribution: the hand-picked
defaults common in published code are systematically
under-regularized for this task; the tuned configuration converges to
substantially heavier regularization (`reg_alpha` ≈ 1.0,
`learning_rate` ≈ 0.011, `max_depth` = 5) and improves single-split
R² from 0.568 to 0.581.

A reproducible pipeline (`experiments.py`) implements every result
reported in this paper from a single command, with cached SBERT
embeddings to ensure that repeated runs do not redo expensive
encoding.

## Roadmap

Section 2 describes the corpus and discloses its sampling
properties. Section 3 presents the pipeline, baselines, and
statistical protocol. Section 4 reports the headline comparison,
5-fold cross-validation, and two ablations. Section 5 discusses
the implications for the field. Section 6 enumerates limitations.
Section 7 concludes.
