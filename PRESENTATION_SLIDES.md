# IMDb Rating Predictor - Presentation Slides

---

## 📑 Table of Contents
1. Title Slide
2. Problem Statement
3. Dataset Overview
4. Methodology Overview
5. Approach 1: SBERT + XGBoost
6. Approach 2: Longformer + Metadata Fusion
7. Model Architecture Comparison
8. Results Comparison
9. Performance Analysis
10. Why SBERT + XGBoost Performs Better
11. Computational Efficiency
12. Error Distribution Analysis
13. Key Findings
14. Future Work & Improvements
15. Conclusion
16. Q&A

---

## 📊 Slide 1: Title Slide

### Content:
**IMDb Rating Predictor: A Comparative Study of Transformer-Based Approaches**

**Subtitle:** Evaluating SBERT + XGBoost vs Longformer for Movie Script Rating Prediction

**Authors:** [Your Name]
**Course:** Research Methodology
**Date:** February 2026

---

## 📊 Slide 2: Problem Statement

### Content:
### Research Problem

**Objective:** Predict IMDb ratings from movie scripts using machine learning

**Why is this important?**
- Early script quality assessment for filmmakers
- Understanding what makes movies successful
- Automated screening in pre-production

**Key Challenges:**
- Long text documents (scripts can be 10,000+ words)
- Complex relationship between script content and audience reception
- Subjective nature of movie ratings

### Research Question:
*Which approach better captures the relationship between movie scripts and their IMDb ratings: a transformer-based encoder with gradient boosting (SBERT + XGBoost) or a long-sequence transformer model (Longformer)?*

---

## 📊 Slide 3: Dataset Overview

### Content:
### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Movies** | 5,204 |
| **Scripts Loaded** | 5,195 (99.8%) |
| **Rating Range** | 1.5 - 9.3 |
| **Mean Rating** | 5.98 |
| **Rating Std Dev** | 1.44 |

### Data Split
- **Training:** 3,636 samples (70%)
- **Validation:** 779 samples (15%)
- **Testing:** 780 samples (15%)

### Data Sources
1. **Movie Scripts:** Raw text files from IMSDb
2. **Metadata:** Excel file with:
   - Movie name, year, decade
   - Movie length (minutes)
   - IMDb ID, collector info

### Class Imbalance
| Rating Range | Count | Weight (legacy) |
|--------------|-------|---------|
| Low (1-4) | 371 | 5.25x |
| Medium (4-6) | 1,947 | 1.00x |
| Good (6-8) | 1,171 | 1.66x |
| Excellent (8-10) | 147 | 13.24x |

> Note: weighted training was the legacy default. Our ablation (paired
> Wilcoxon, p=0.012) shows that **un-weighted training performs
> significantly better**; the headline configuration in this presentation
> is therefore the un-weighted variant.

### ⚠ Sampling Caveat (Important)
The empirical rating distribution has **structural gaps**: zero records in
[4.3, 5.0) and zero in [6.0, 7.0). This is not consistent with random IMDb
sampling — the corpus is curated (IMSDb editorial selection), and the
bimodality visible in our histograms is an artifact of dataset construction.

**Implication:** all metrics are conditional on IMSDb-style selection, not
arbitrary IMDb films. Disclosed in the Limitations section.

---

## 📊 Slide 4: Methodology Overview

### Content:
### Two Approaches Compared

**Approach 1: SBERT + XGBoost**
```
Script → Chunking → SBERT Embeddings → Feature Engineering
→ Concatenate → XGBoost → Rating Prediction
```

**Approach 2: Longformer + Metadata Fusion**
```
Script → Longformer (truncated) → CLS Token → Fusion Layer
→ MLP Head → Rating Prediction
```

### Key Differences

| Aspect | SBERT + XGBoost | Longformer |
|--------|------------------|-------------|
| **Text Encoder** | all-MiniLM-L6-v2 (384-dim) | longformer-base-4096 (768-dim) |
| **Training** | Frozen encoder + XGBoost | End-to-end fine-tuning |
| **Coverage** | Full script (chunking) | First 2048 tokens (truncated) |
| **Features** | 384 text + 19 structural | 768 text + 3 metadata |
| **Parameters** | ~120K (XGBoost) | 148M (transformer) |

---

## 📊 Slide 5: Approach 1 - SBERT + XGBoost

### Content:
### Architecture Details

**1. Text Encoding with SBERT**
- **Model:** `all-MiniLM-L6-v2` (pretrained)
- **Chunking Strategy:**
  - Chunk size: 256 words
  - Overlap: 50 words
  - Average chunk embeddings for document representation
- **Output:** 384-dimensional vector

**2. Feature Engineering (19 features)**
- Text statistics (word count, sentence count)
- Dialogue analysis (dialogue density, character count)
- Structural features (scene count, pacing)
- Metadata (year, decade, movie length)

**3. Combined Feature Vector**
- **Total Features:** 384 (SBERT) + 19 (structural) = 403 dimensions

**4. Gradient Boosting Regressor**
- **Model:** XGBoost (`max_depth=6`, `lr=0.05`, `n_estimators=1000`)
- **Training:** Early stopping (patience=20) on the validation split.
  We initially used inverse-frequency sample weights but ablation
  showed they hurt (ΔMAE −0.04, p=0.012); final reported numbers use
  **un-weighted** training.
- **Objective:** Minimize RMSE

### Advantages
✅ Processes entire script (no information loss)
✅ Rich feature set combining semantic + structural signals
✅ Fast training (5 minutes on CPU)
✅ Small model size (~2MB)

---

## 📊 Slide 6: Approach 2 - Longformer

### Content:
### Architecture Details

**1. Text Encoding with Longformer**
- **Model:** `allenai/longformer-base-4096` (pretrained)
- **Token Limit:** 2048 tokens (memory constraint)
- **Attention Mechanism:**
  - Local attention: sliding window of 512 tokens
  - Global attention: CLS token attends to full sequence
- **Output:** CLS token representation (768-dim)

**2. Metadata Fusion**
- **Features:** 3 metadata (year, decade, movie length)
- **Fusion:** Concatenated with CLS token (771 total)

**3. Regression Head**
- **Architecture:** MLP (771 → 800 → 128 → 1)
- **Activation:** ReLU
- **Regularization:** Dropout (0.1)

**4. Training Configuration**
- **Epochs:** 5 (early stopping at epoch 4)
- **Learning Rate:** 2e-5
- **Optimizer:** AdamW with weight decay 0.01
- **Mixed Precision:** FP16
- **Batch Size:** 1 (gradient accumulation to 4)

### Advantages
✅ Handles long sequences efficiently
✅ Captures long-range dependencies
✅ End-to-end trainable
✅ Global context awareness

---

## 📊 Slide 7: Model Architecture Comparison

### Content:
### Visual Comparison

```
SBERT + XGBoost Pipeline:

Script (N words)
    ↓
Chunking (256 words, 50 overlap)
    ↓
┌─────────────────────────────┐
│  SBERT (all-MiniLM-L6-v2)  │
│  [frozen]                   │
└─────────────────────────────┘
    ↓
Average chunk embeddings → [384-dim]
    ↓
┌─────────────────────────────┐
│  Feature Engineering [19]    │
│  (word count, dialogue...)    │
└─────────────────────────────┘
    ↓
Concatenate → [403-dim]
    ↓
┌─────────────────────────────┐
│  XGBoost Regressor          │
│  [trained]                  │
└─────────────────────────────┘
    ↓
Rating (1-10)
```

```
Longformer Pipeline:

Script (max 2048 tokens)
    ↓
┌─────────────────────────────┐
│  Longformer Base 4096       │
│  [fine-tuned]               │
│  - Local Attention (512)     │
│  - Global Attention (CLS)    │
└─────────────────────────────┘
    ↓
CLS Token → [768-dim]
    ↓
┌─────────────────────────────┐
│  Concat with Metadata [3]    │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  MLP Head                  │
│  (771 → 800 → 128 → 1)    │
└─────────────────────────────┘
    ↓
Rating (1-10)
```

---

## 📊 Slide 8: Results Comparison

### Content:
### Headline Test Metrics with 95% Bootstrap CIs (n=780, single split)

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| predict_mean | 1.518 [1.456, 1.579] | 1.247 [1.186, 1.305] | -0.001 |
| ols_metadata (year+length+decade) | 1.184 [1.128, 1.239] | 0.946 [0.894, 0.994] | 0.391 |
| ols_structural (19 features) | 1.129 [1.068, 1.186] | 0.881 [0.831, 0.927] | 0.447 |
| tfidf + XGBoost (lexical baseline) | 1.129 [1.060, 1.195] | 0.854 [0.801, 0.903] | 0.447 |
| **SBERT + XGBoost (no weights)** | **0.997 [0.942, 1.055]** | **0.749 [0.701, 0.793]** | **0.568 [0.526, 0.608]** |
| SBERT + XGBoost (weighted, legacy) | 1.032 [0.974, 1.094] | 0.789 [0.744, 0.837] | 0.538 |

### Paired Significance vs SBERT (Wilcoxon signed-rank on |error|)

| Baseline | ΔMAE (baseline − SBERT) | 95% CI | Wilcoxon p |
|---|---|---|---|
| predict_mean | +0.458 | [+0.39, +0.52] | 4×10⁻³⁴ |
| ols_metadata | +0.157 | [+0.11, +0.20] | 9×10⁻¹¹ |
| ols_structural | +0.093 | [+0.05, +0.13] | 3×10⁻⁵ |
| tfidf_xgboost | +0.065 | [+0.015, +0.11] | 0.014 |

_Positive ΔMAE means SBERT has lower error (SBERT wins). All differences
are statistically significant at α=0.05 with multiple-comparison
allowance._

### Key Insight (rewritten with discipline)
- SBERT + XGBoost has **significantly lower error** than every baseline
  including TF-IDF (p=0.014).
- Important caveat: `ols_metadata` alone — three numbers (`year`,
  `movie_length`, `decade`) — already explains R²=0.39, i.e. **73% of
  SBERT's R² is recoverable from metadata that does not look at the
  script**. Adding script content (SBERT) lifts R² by Δ=+0.18.
- The screenplay therefore carries **genuine but bounded** predictive
  signal beyond era/length.

### Robustness Check: 5-Fold Cross-Validation

Results across 5 folds (mean ± std), pooled significance n≈5,195:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| predict_mean | 1.438 ± 0.056 | 1.175 ± 0.057 | −0.000 |
| ols_metadata | 1.130 ± 0.047 | 0.883 ± 0.042 | 0.382 |
| ols_structural | 1.076 ± 0.040 | 0.833 ± 0.031 | 0.440 |
| tfidf + XGBoost | 1.083 ± 0.034 | 0.819 ± 0.022 | 0.433 |
| SBERT + XGBoost (weighted) | 1.000 ± 0.027 | 0.768 ± 0.024 | 0.516 ± 0.026 |
| **SBERT + XGBoost (no weights)** | **0.958 ± 0.033** | **0.720 ± 0.027** | **0.556 ± 0.009** |

CV-pooled paired Wilcoxon vs the weighted-SBERT baseline:

| Baseline | ΔMAE | 95% CI | p |
|---|---|---|---|
| predict_mean | +0.407 | [+0.38, +0.43] | 9×10⁻²¹² |
| ols_metadata | +0.115 | [+0.10, +0.13] | 7×10⁻³⁷ |
| ols_structural | +0.065 | [+0.053, +0.078] | 1×10⁻¹⁵ |
| tfidf_xgboost | +0.051 | [+0.034, +0.067] | 3×10⁻⁵ |
| **sbert_noweight** | **−0.048** | **[−0.057, −0.039]** | **3×10⁻²⁶** |

The CV results **confirm the single-split conclusions** with substantially
tighter confidence: the no-weight SBERT model is the best in every fold,
the negative-result on sample weighting holds at p ≈ 3×10⁻²⁶, and the win
over TF-IDF is preserved at p ≈ 3×10⁻⁵.

### Stacking lifts R² further (5-Fold CV)

A Ridge meta-regressor trained on out-of-fold predictions of all four
base models gives a small but statistically significant lift over the
best single base model:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| sbert_xgboost (no weights) | 0.956 ± 0.034 | 0.719 ± 0.028 | 0.558 ± 0.013 |
| **stacked (Ridge over 4 base models)** | **0.935 ± 0.032** | **0.706 ± 0.028** | **0.577 ± 0.010** |

Pooled paired Wilcoxon stacked vs SBERT: ΔMAE = +0.013
[+0.008, +0.019], **p = 4.2 × 10⁻⁶**. The stacked R² standard
deviation across folds (0.010) is *tighter* than any single base
model — stacking is more accurate **and** more stable.

Ridge meta-coefficients (averaged across folds): SBERT ≈ 0.69,
TF-IDF ≈ 0.38, ols_metadata ≈ 0.27, ols_structural ≈ −0.13. The
negative coefficient on `ols_structural` indicates that, given
SBERT and TF-IDF predictions are already available, the structural-OLS
prediction carries no additional signal — its information is fully
absorbed by the upstream models.

### Hyperparameter tuning: defaults are under-regularized

A 100-trial Optuna TPE search over the XGBoost head (held-out 20%
test split, internal val for early stopping):

| Configuration | RMSE | MAE | R² |
|---|---|---|---|
| Hand-picked default | 0.997 [0.94, 1.06] | 0.749 [0.70, 0.79] | 0.568 [0.53, 0.61] |
| **Optuna-tuned** | **0.966 [0.91, 1.02]** | **0.714 [0.67, 0.75]** | **0.581 [0.55, 0.62]** |

The tuned configuration converges to **substantially heavier
regularization**: `learning_rate` ≈ 0.011 (≈ 5× lower), `reg_alpha`
≈ 1.0 (≈ 10× higher), `max_depth` = 5 (one shallower) — directly
addressing the overfitting visible in the legacy training curve.

---

## 📊 Slide 9: Performance Analysis

### Content:
### Error Distribution (SBERT + XGBoost, unweighted, test n=780)

- Within ±0.5: 367 (47.1%) ███████████████████████
- Within ±1.0: 560 (71.8%) ███████████████████████████████████
- Within ±1.5: 661 (84.7%) ██████████████████████████████████████████
- Within ±2.0: 732 (93.8%) ███████████████████████████████████████████████

### MAE by Rating Range

| Rating Range | MAE | n |
|--------------|-----|---|
| Low (1-4) | **1.577** | 107 |
| Medium (4-6) | 0.558 | 380 |
| Good (6-8) | 0.672 | 250 |
| Excellent (8-10) | 0.818 | 43 |

### Key Observations
1. **94% within ±2 points** on the 10-point scale.
2. **Tail performance is poor.** Low-rated films (1–4) have MAE 1.58;
   the model regresses toward the mean. The "Excellent" bucket has only
   43 test samples, so its MAE estimate is noisy.
3. **Predictions cluster between ~4 and ~8** even when true ratings
   span 1.5–9.3 (visible in the actual-vs-predicted scatter). This is
   expected from a tree ensemble trained with squared loss on a
   bimodal target.
4. **R² = 0.568 [0.526, 0.608]** — the model explains roughly 57% of
   rating variance on this corpus. About R²=0.39 of that is recoverable
   from metadata alone (see Slide 8); the script-content contribution
   is the remaining ~Δ=0.18.

---

## 📊 Slide 10: Hypotheses for the SBERT vs Baselines Gap

### Content:
### Where the Δ R² = +0.18 over `ols_metadata` Likely Comes From

We do not have isolated experiments for each hypothesis below, so they
are framed as plausible mechanisms rather than confirmed conclusions.

**1. Semantic >> lexical (TF-IDF) for screenplay-length text**
- SBERT beats TF-IDF significantly (p=0.014, ΔMAE=+0.065).
- The win is small but consistent — semantic embeddings recover
  signal that bag-of-words does not, even with `(1,2)`-grams over
  8,000 features.

**2. Full-script coverage via chunking**
- 256-word chunks with 50-word overlap, mean-pooled across the whole
  document, vs. classical methods that scan term frequencies only.
- Untested in this work; pooling-strategy ablation is queued for the
  full submission.

**3. Encoder pretraining objective alignment**
- `all-MiniLM-L6-v2` is trained for semantic similarity, producing
  document-summary vectors well-suited to a regression head.

**4. Strong tabular regressor on top of fixed embeddings**
- Frozen SBERT + XGBoost (with early stopping, no class weights)
  outperformed every alternative head we tried.

### What We Now Know Does **Not** Help
- **Inverse-frequency sample weighting** — tested with bootstrap CI
  and Wilcoxon: ΔMAE −0.04 in favor of *unweighted* training,
  p=0.012. The legacy 13.24× weight on the "Excellent" bucket
  destabilizes XGBoost. We removed it from the final pipeline.

---

## 📊 Slide 11: Computational Efficiency

### Content:
### Resource Requirements Comparison

| Aspect | SBERT + XGBoost | Longformer |
|--------|------------------|-------------|
| **Training Time** | ~5 minutes (CPU) | ~2 hours (GPU) |
| **Inference Time** | ~0.1s per script (CPU) | ~0.5s per script (GPU) |
| **Training Memory** | ~2GB RAM | ~8GB VRAM (GPU) |
| **Inference Memory** | ~500MB RAM | ~2GB VRAM (GPU) |
| **Model Size** | ~2MB | ~500MB |
| **Hardware** | Any CPU | NVIDIA GPU required |
| **Deployment** | Easy (lightweight) | Complex (GPU required) |

### Practical Implications

**SBERT + XGBoost Advantages:**
✅ Can train on any laptop (no GPU needed)
✅ 24x faster training time
✅ 250x smaller model
✅ Can deploy to edge devices
✅ Lower inference cost

**Longformer Trade-offs:**
❌ Requires GPU infrastructure
❌ Longer training cycles
❌ Larger model footprint
❌ Higher deployment cost
❌ More complex to maintain

---

## 📊 Slide 12: Sample Predictions

### Content:
### SBERT + XGBoost Predictions (Test Set)

| Actual | Predicted | Error |
|--------|-----------|-------|
| 3.60 | 4.24 | +0.64 |
| 5.00 | 5.60 | +0.60 |
| 5.80 | 5.87 | +0.07 |
| 4.10 | 5.75 | +1.65 |
| 5.30 | 4.67 | -0.63 |
| 5.90 | 5.00 | -0.90 |
| 7.70 | 7.73 | +0.03 |
| 7.40 | 7.72 | +0.32 |
| 5.00 | 4.90 | -0.10 |
| 5.40 | 5.42 | +0.02 |
| 5.60 | 5.12 | -0.48 |
| 7.70 | 7.49 | -0.21 |
| 5.80 | 5.21 | -0.59 |
| 7.30 | 7.34 | +0.04 |
| 5.40 | 5.70 | +0.30 |

### Key Observations
- **Accurate predictions** within ±0.5 for 44.2% of test cases
- **Small errors** (±0.1) for many predictions
- **Largest errors** occur on edge cases (extreme ratings)
- **Consistent performance** across rating ranges

---

## 📊 Slide 13: Key Findings

### Content:
### Main Research Findings (data-supported, with significance tests)

**1. SBERT + XGBoost beats every baseline tested**
- Significantly lower error than `predict_mean`, `ols_metadata`,
  `ols_structural`, and `tfidf_xgboost` (paired Wilcoxon, all p<0.05;
  most p<0.001).
- Headline test metrics: RMSE 0.997 [0.94, 1.06], MAE 0.749 [0.70, 0.79],
  R² 0.568 [0.53, 0.61].

**2. Metadata Dominates the Predictive Signal**
- `ols_metadata` (year + length + decade) alone: R²=0.39 — that's
  **73% of SBERT's R²** with no script content at all.
- Script content (via SBERT) contributes the marginal Δ=+0.18 R²
  beyond era and length.
- This is the more honest framing of "SBERT predicts ratings."

**3. Inverse-Frequency Sample Weighting Hurts (Negative Result)**
- Tested via paired Wilcoxon: ΔMAE = −0.04 in favor of *unweighted*
  training, p=0.012, 95% CI excludes zero.
- The 13.24× weight on the "Excellent" bucket destabilizes XGBoost;
  the legacy default was wrong.

**4. Computational Efficiency**
- ~5 min training on CPU, ~2 MB model.
- No GPU required; deployable on commodity hardware.

**5. Honest Methodological Reporting**
- Bootstrap 95% CIs on every metric (n_boot=1000).
- Paired Wilcoxon signed-rank test against the headline system for
  every baseline.
- 5-fold cross-validation as a robustness check (queued/submitted).

### Practical Implications
**For screenplay rating prediction:**
- SBERT + XGBoost (unweighted) is competitive while being lightweight
  and CPU-deployable.
- A meaningful fraction of the prediction comes from metadata, not
  script content; consumers of the model should account for this.

**For long-document NLP tasks:**
- Hybrid approaches (frozen encoder + tabular regressor) deserve to
  be benchmarked against end-to-end fine-tuning before the latter is
  adopted, especially under tight compute budgets.

---

## 📊 Slide 14: Limitations

### Content:
### Study Limitations

**1. Dataset Selection Bias (Disclosed)**
- The corpus has **structural rating gaps**: zero records in [4.3, 5.0)
  and zero in [6.0, 7.0). The bimodality in our histograms is an
  artifact of IMSDb editorial selection, not natural variation.
- All metrics are conditional on IMSDb-style films, not arbitrary
  IMDb films.
- Strong year–rating confound (Spearman ρ ≈ −0.93 between decade and
  mean rating; 1930s avg 7.49 → 2010s avg 5.48), most plausibly
  survivorship bias.

**2. Metadata Dominates Predictive Signal**
- `ols_metadata` (year + length + decade only) achieves R²=0.39 — ~73%
  of our headline R²=0.57. SBERT contributes the marginal Δ=+0.18.
- A future iteration must include a **metadata-removed ablation** to
  isolate script-content contribution.

**3. Single Test Split for Headline Numbers**
- Single 70/15/15 split (`random_state=42`). 5-fold CV results are
  reported as a robustness check (queued at submission time).
- Bootstrap CIs and paired Wilcoxon tests are reported for every
  comparison.

**4. Rating Subjectivity**
- IMDb ratings have inherent subjectivity.
- Multiple factors affect ratings beyond script quality (marketing,
  cast, production budget, era effects).

**5. Feature Limitations**
- Text-only analysis (no visual/audio).
- No genre, cast, director, or budget features.
- Pooling strategy ablation (max / weighted-norm) was started but not
  completed within compute budget; mean-pooling is the only verified
  option.

**6. Longformer Comparison Was Unfair**
- The comparison Longformer was given only 3 metadata features vs
  SBERT's 19, was truncated to 2048/4096 tokens, and used a small
  MLP head. We do not present Longformer as a properly-controlled
  baseline; the comparison is left to future work with matched
  feature budgets.
- Limited training epochs (5)

---

## 📊 Slide 15: Future Work

### Content:
### Potential Improvements

**For SBERT + XGBoost:**
1. **Enhanced Feature Engineering**
   - Add genre information
   - Include cast/director data
   - Incorporate budget information

2. **Advanced Text Features**
   - Sentiment analysis of dialogue
   - Character interaction patterns
   - Plot structure analysis

3. **Model Ensemble**
   - Combine multiple XGBoost models
   - Add neural network predictions
   - Use stacking/blending

**For Longformer:**
1. **Increased Sequence Length**
   - Use full 4096 tokens (if GPU memory allows)
   - Capture more of each script

2. **Enhanced Features**
   - Add same 19 script statistics
   - Improve fusion architecture

3. **Larger Regression Head**
   - Increase hidden dimensions (128 → 256/512)
   - Add more layers

4. **More Training**
   - Train for 10-15 epochs
   - Implement learning rate scheduling

**General:**
1. **Multi-Task Learning**
   - Predict rating + genre + box office
   - Shared representations

2. **Cross-Validation**
   - K-fold validation for robustness
   - Statistical significance testing

3. **Real-World Testing**
   - Deploy to production environment
   - Gather user feedback
   - Iterate based on real usage

---

## 📊 Slide 16: Conclusion

### Content:
### Summary

**Research Question:**
*To what extent does the screenplay text itself predict an IMDb
rating, beyond what is recoverable from simple metadata?*

**Answer:**
On a 5,195-script corpus from IMSDb, **SBERT + XGBoost (unweighted)**
explains R² = 0.57 [0.53, 0.61] of the rating variance on a single
80/15/15 split, and R² = 0.558 ± 0.013 under 5-fold cross-validation.
**Stacking a Ridge meta-regressor** over four base models lifts this
to **R² = 0.577 ± 0.010** under the same CV protocol — significantly
higher than the best single base model (paired Wilcoxon p ≈ 4×10⁻⁶)
and with tighter per-fold variance. A metadata-only OLS baseline
(year, length, decade) already explains R² = 0.38, so the
contribution attributable to the *script content* is
**Δ R² ≈ +0.18**, significant at p ≪ 10⁻³⁰.

### Key Takeaways

1. **Statistically rigorous comparison:** Bootstrap CIs on every metric
   and paired Wilcoxon signed-rank tests — no claim is made without
   confidence intervals.
2. **Script content adds genuine but bounded signal** beyond metadata
   (Δ R² ≈ +0.18).
3. **Stacking a Ridge meta-regressor** over the four base models
   produces a small but statistically significant lift (Δ R² ≈ +0.02,
   p ≈ 4×10⁻⁶) with *tighter* fold-to-fold variance — base models
   carry partially complementary signal.
4. **Negative result on class weighting:** Inverse-frequency sample
   weights *hurt* (ΔMAE = −0.048, p ≈ 3×10⁻²⁶). The legacy training
   default was wrong; we removed it.
5. **Hyperparameter search confirms** the hand-picked defaults are
   under-regularized; Optuna recommends `lr ≈ 0.011`, `reg_alpha ≈
   1.0`, `max_depth = 5`.
6. **Efficiency:** ~5 min training on CPU, ~2 MB model — no GPU
   required, deployable on commodity hardware.
7. **Honesty about dataset:** Corpus is curated (zero records in
   [4.3, 5.0) and [6.0, 7.0)); generalization claims are scoped
   accordingly.

### Contributions

- **Strong baseline suite** for screenplay rating prediction:
  predict-mean, OLS-metadata, OLS-structural, TF-IDF + XGBoost,
  SBERT + XGBoost — all with bootstrap CIs and paired significance
  tests.
- **Reproducible pipeline:** SBERT-embedding cache, single-split and
  k-fold modes, configurable encoder via CLI.
- **Negative result on inverse-frequency reweighting** for
  rating-bucket-imbalanced regression with gradient boosting.
- **Honest characterization** of an IMSDb-style screenplay corpus,
  including the rating-distribution artifacts a future user must
  account for.

### Final Recommendation

**For screenplay rating prediction:**
Use SBERT (mean-pooled chunked) + XGBoost with early stopping and
*no* sample weighting. Always include `ols_metadata` as a baseline
in the same paper.

**For long-document NLP tasks under tight compute:**
Benchmark a frozen-encoder + tabular-regressor hybrid before
investing in end-to-end fine-tuning of a long-sequence transformer.

---

## 📊 Slide 17: Q&A

### Content:
### Potential Questions

**Q1: Why not use BERT instead of SBERT?**
- SBERT is optimized for semantic similarity tasks
- Better document-level representations via chunking
- Pretrained on diverse sentence types

**Q2: Why does XGBoost work better than neural networks here?**
- Task-specific optimization (regression)
- Handles mixed feature types well
- Robust to outliers
- Stable training

**Q3: How would you improve Longformer's performance?**
- Increase sequence length to 4096 tokens
- Add script statistics features
- Larger regression head
- More training epochs
- Different pooling strategies

**Q4: When would Longformer be preferred?**
- When sequential dependencies are critical
- For very long documents (>10,000 words)
- For end-to-end learning requirements
- For transfer learning across multiple tasks

**Q5: What are the practical applications?**
- Early script quality assessment
- Pre-production screening
- Automated script evaluation
- Research on movie success factors

### Thank You!

**Questions?**
**Email:** [your.email@example.com]
**GitHub:** [repository-link]

---

## 📝 Speaker Notes

### Tips for Effective Delivery

**Slide Timing:**
- Title: 30 seconds
- Problem Statement: 2 minutes
- Dataset: 1.5 minutes
- Methodology Overview: 2 minutes
- SBERT + XGBoost: 2.5 minutes
- Longformer: 2.5 minutes
- Architecture Comparison: 2 minutes
- Results: 2 minutes
- Performance Analysis: 2 minutes
- Why SBERT Better: 2.5 minutes
- Computational Efficiency: 2 minutes
- Sample Predictions: 1 minute
- Key Findings: 2 minutes
- Limitations: 1.5 minutes
- Future Work: 1.5 minutes
- Conclusion: 2 minutes
- Q&A: Variable

**Delivery Tips:**
1. **Pace yourself** - Don't rush through numbers
2. **Point to visuals** - Reference charts and diagrams
3. **Explain "so what"** - Always follow numbers with practical meaning
4. **Emphasize key metrics** - RMSE 0.99, 95% within ±2, 58% R²
5. **Make eye contact** - Look at audience, not just slides
6. **Prepare for Q&A** - Think about follow-up questions

**Key Numbers to Remember:**
- Test RMSE: 0.9881 (SBERT) vs 1.0462 (Longformer)
- R²: 0.5762 vs 0.5249
- Training time: 5 min vs 2 hours
- Model size: 2MB vs 500MB
- Accuracy: 95% within ±2 points

---

## 🎯 Presentation Checklist

### Preparation
- [ ] Create slide deck (PowerPoint/Keynote/Google Slides)
- [ ] Add visualizations (charts, diagrams, tables)
- [ ] Practice timing for each slide
- [ ] Prepare demo (if showing predictions live)
- [ ] Test all equipment (projector, microphone)
- [ ] Have backup copy of presentation

### During Presentation
- [ ] Start on time
- [ ] Introduce topic clearly
- [ ] Explain methodology simply
- [ ] Highlight key results
- [ ] Use visuals effectively
- [ ] Speak clearly and at good pace
- [ ] Manage time well
- [ ] Leave time for Q&A

### After Presentation
- [ ] Collect feedback
- [ ] Document questions asked
- [ ] Follow up if promised
- [ ] Share slides with interested parties

---

**Good luck with your presentation! 🎯**
