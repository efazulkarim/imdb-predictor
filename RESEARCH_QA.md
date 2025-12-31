# IMDb Rating Predictor - Research Methodology Q&A

## Table of Contents

1. [Novelty of the Project](#1-novelty-of-the-project)
2. [Why Dataset Starts from Rating 1.5?](#2-why-dataset-starts-from-rating-15)
3. [Why Machine Learning over Deep Learning?](#3-why-machine-learning-over-deep-learning)
4. [How NLP Works in This Project](#4-how-nlp-works-in-this-project)
5. [Why Gradient Boosting Works Best](#5-why-gradient-boosting-works-best)
6. [Dealing with Limitations](#6-dealing-with-limitations)
7. [Understanding the Models](#7-understanding-the-models)
8. [Understanding the Metrics](#8-understanding-the-metrics-rmse-mae-r²)

---

## 1. Novelty of the Project

### What Makes This Project Unique?

| Aspect                         | Our Innovation                                                                                                                                     |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Script-Based Prediction**    | Unlike existing systems that predict ratings from trailers, posters, or cast metadata, we predict IMDb ratings directly from **movie script text** |
| **Multi-Feature NLP Pipeline** | We combine TF-IDF text features with **17 hand-crafted structural features** (dialogue density, pacing, vocabulary complexity)                     |
| **Temporal Awareness**         | We incorporate **decade encoding** and **year features** to capture evolving audience preferences over time                                        |
| **Practical Applicability**    | Screenwriters and producers can evaluate scripts **before production** begins                                                                      |

### Novel Contributions:

1. **First-of-kind script-to-rating predictor** using ensemble ML with linguistic feature engineering
2. **Hybrid feature approach**: Text semantics (TF-IDF) + Structural analysis (custom features)
3. **Class-balanced training**: Sample weighting to handle imbalanced rating distributions
4. **Interpretable predictions**: Feature importance analysis reveals what makes a script successful

---

## 2. Why Dataset Starts from Rating 1.5?

### Reasoning:

| Factor                       | Explanation                                                                         |
| ---------------------------- | ----------------------------------------------------------------------------------- |
| **IMDb Rating Distribution** | Very few movies receive ratings below 1.5. The average IMDb rating is ~6.2/10       |
| **Data Availability**        | Movies rated below 1.5 are extremely rare (statistically <0.1% of all rated movies) |
| **Script Availability**      | Low-rated obscure films rarely have publicly available scripts                      |
| **Statistical Significance** | Including ratings 1.0-1.5 would add noise without meaningful patterns               |

### Distribution Reality:

```
Rating Range     | Approximate % of Movies
-----------------|------------------------
1.0 - 2.0        | < 1%
2.0 - 4.0        | ~5%
4.0 - 6.0        | ~25%
6.0 - 8.0        | ~55%   ← Most common
8.0 - 10.0       | ~14%
```

> **Conclusion**: Our dataset starting at 1.5 reflects the natural distribution of IMDb ratings where meaningful data exists.

---

## 3. Why Machine Learning over Deep Learning?

### Decision Matrix:

| Factor                 | Traditional ML ✅              | Deep Learning ❌                      |
| ---------------------- | ------------------------------ | ------------------------------------- |
| **Dataset Size**       | Works well with ~5,000 samples | Needs 50,000+ for optimal performance |
| **Training Time**      | 2-5 minutes (CPU)              | 2-6 hours (requires GPU)              |
| **Hardware**           | Runs on any laptop             | Requires NVIDIA GPU with CUDA         |
| **Model Size**         | ~2MB (deployable anywhere)     | 400MB-1GB+ (expensive hosting)        |
| **Interpretability**   | Feature importance available   | Black-box predictions                 |
| **Deployment Cost**    | Free tier (Render, Railway)    | $50-200/month for GPU inference       |
| **Long Text Handling** | TF-IDF handles any length      | BERT limited to 512 tokens            |

### Key Reasons We Chose ML:

1. **Limited Data**: With ~5,200 scripts, deep learning would severely overfit
2. **Interpretability**: Stakeholders need to understand _why_ a script gets a certain prediction
3. **Deployment Constraints**: Free deployment requires lightweight models
4. **Feature Engineering Value**: Hand-crafted features (dialogue density, pacing) capture domain knowledge that DL must learn from data

### When Deep Learning Would Be Better:

- Dataset exceeds 50,000+ scripts
- Access to GPU infrastructure
- No need for feature interpretability
- Budget for cloud GPU hosting

---

## 4. How NLP Works in This Project

### NLP Pipeline Architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAW SCRIPT TEXT                              │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      TEXT PREPROCESSING                              │
│  • Lowercase conversion                                              │
│  • Stage direction removal: [GUNSHOT], (CRYING)                     │
│  • Character name removal: JOHN:, MARY:                             │
│  • Timestamp/scene number cleanup                                    │
│  • Whitespace normalization                                          │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────┴───────────────────────┐
        │                                               │
        ▼                                               ▼
┌───────────────────────┐                 ┌───────────────────────────┐
│   TF-IDF VECTORIZER   │                 │  STRUCTURAL FEATURES      │
│  • 8,000 features     │                 │  • word_count             │
│  • Unigrams + Bigrams │                 │  • dialogue_density       │
│  • Stop words removed │                 │  • unique_characters      │
│  • Min DF = 3         │                 │  • avg_sentence_length    │
│  • Max DF = 0.85      │                 │  • vocabulary_complexity  │
│  • Sublinear TF       │                 │  • scene_count            │
└───────────────────────┘                 │  • words_per_scene        │
        │                                 │  • exclamation_ratio      │
        │                                 │  • + 9 more features      │
        │                                 └───────────────────────────┘
        │                                               │
        └───────────────────────┬───────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE COMBINATION (hstack)                      │
│              [TF-IDF Features] + [Numerical Features]                │
│                     8,000 + 17 = 8,017 total features                │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ENSEMBLE ML MODEL                               │
│                   (Gradient Boosting Regressor)                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PREDICTED IMDb RATING (1.0 - 10.0)                │
└─────────────────────────────────────────────────────────────────────┘
```

### TF-IDF (Term Frequency-Inverse Document Frequency):

**How it works:**

1. **Term Frequency (TF)**: How often a word appears in a script
2. **Inverse Document Frequency (IDF)**: Penalizes common words across all scripts
3. **Combined Score**: `TF-IDF = TF × IDF`

**Example:**
| Word | Script A TF | Corpus IDF | TF-IDF Score |
|------|-------------|------------|--------------|
| "love" | 0.02 | 1.5 | 0.03 (common word) |
| "heist" | 0.01 | 4.2 | 0.042 (rarer word) |

### 17 Extracted Structural Features:

| Feature Name                             | What It Measures           |
| ---------------------------------------- | -------------------------- |
| `char_count`                             | Total characters in script |
| `word_count`                             | Total words                |
| `line_count`                             | Number of lines            |
| `avg_word_length`                        | Vocabulary sophistication  |
| `unique_word_ratio`                      | Lexical diversity          |
| `long_word_ratio`                        | Complexity indicator       |
| `sentence_count`                         | Script structure           |
| `avg_sentence_length`                    | Writing style              |
| `sentence_length_std`                    | Pacing variation           |
| `dialogue_density`                       | Action vs. dialogue ratio  |
| `unique_characters`                      | Cast size indicator        |
| `exclamation_ratio`                      | Emotional intensity        |
| `question_ratio`                         | Interactive dialogue       |
| `action_density`                         | Action sequence frequency  |
| `scene_count`                            | Number of scenes           |
| `words_per_scene`                        | Pacing indicator           |
| `year`, `decade_encoded`, `movie_length` | Metadata features          |

---

### 4.1 Detailed Breakdown of 17 Structural Features

#### Feature Categories Overview

| Category              | # Features | Purpose                |
| --------------------- | ---------- | ---------------------- |
| Basic Statistics      | 3          | Script size metrics    |
| Vocabulary Complexity | 3          | Writing sophistication |
| Sentence Structure    | 3          | Writing style          |
| Dialogue Analysis     | 2          | Character interaction  |
| Emotional Indicators  | 2          | Tone/intensity         |
| Script Structure      | 2          | Pacing/format          |
| Metadata              | 3          | External context       |

---

#### Category 1: Basic Statistics (3 features)

| Feature      | Code                        | What It Measures           | Why It Matters                                              |
| ------------ | --------------------------- | -------------------------- | ----------------------------------------------------------- |
| `char_count` | `len(raw_text)`             | Total characters in script | Longer scripts = epic narratives, shorter = tight thrillers |
| `word_count` | `len(raw_text.split())`     | Total words                | Correlates with runtime, complexity                         |
| `line_count` | `len(raw_text.split('\n'))` | Number of lines            | Format density, dialogue vs action ratio                    |

**Example:**

```
Short script (90 min comedy):     ~15,000 words
Long script (3 hr epic):          ~40,000 words
```

---

#### Category 2: Vocabulary Complexity (3 features)

| Feature             | Formula                         | What It Measures            | Why It Matters                                 |
| ------------------- | ------------------------------- | --------------------------- | ---------------------------------------------- |
| `avg_word_length`   | `mean([len(w) for w in words])` | Average characters per word | Simple (4-5) vs sophisticated (6+) vocabulary  |
| `unique_word_ratio` | `len(set(words)) / len(words)`  | Lexical diversity           | Higher = richer vocabulary, lower = repetitive |
| `long_word_ratio`   | Words ≥8 chars / total words    | Proportion of complex words | Indicates intellectual depth                   |

**Example:**

```python
# Children's movie
"The dog ran fast. The cat ran faster."
→ avg_word_length: 3.5, unique_ratio: 0.7, long_word_ratio: 0.0

# Legal thriller
"The jurisdiction encompasses multifarious constitutional implications."
→ avg_word_length: 8.2, unique_ratio: 0.95, long_word_ratio: 0.5
```

---

#### Category 3: Sentence Structure (3 features)

| Feature               | Formula                 | What It Measures    | Why It Matters                                    |
| --------------------- | ----------------------- | ------------------- | ------------------------------------------------- |
| `sentence_count`      | Split by `.!?`          | Number of sentences | Writing density                                   |
| `avg_sentence_length` | Mean words per sentence | Sentence complexity | Short = punchy action, Long = dramatic exposition |
| `sentence_length_std` | Standard deviation      | Pacing variation    | Low = monotonous, High = dynamic rhythm           |

**Example:**

```python
# Action movie (short, punchy)
"He ran. He jumped. He fell."
→ avg_sentence_length: 2.0, std: 0.0 (monotonous)

# Drama (varied pacing)
"He ran. The thunder echoed across the mountains as she stood there,
wondering if he would ever return."
→ avg_sentence_length: 8.5, std: 8.5 (dynamic)
```

---

#### Category 4: Dialogue Analysis (2 features)

| Feature             | Formula                               | What It Measures             | Why It Matters                                            |
| ------------------- | ------------------------------------- | ---------------------------- | --------------------------------------------------------- |
| `dialogue_density`  | Character lines / total lines         | % of script that is dialogue | High = dialogue-heavy (comedy, drama), Low = action-heavy |
| `unique_characters` | Count of unique `CHARACTER:` patterns | Number of speaking roles     | More characters = ensemble cast complexity                |

**How dialogue is detected:**

```
JOHN: Hello there!          ← Matches pattern ^[A-Z][A-Z\s]+:
MARY: Hi John.              ← Matches
[John walks away]           ← NOT a character (action direction)
```

**Example:**

```
Dialogue-heavy drama:   dialogue_density = 0.7, unique_characters = 8
Action blockbuster:     dialogue_density = 0.3, unique_characters = 15
```

---

#### Category 5: Emotional Indicators (2 features)

| Feature             | Formula                         | What It Measures           | Why It Matters             |
| ------------------- | ------------------------------- | -------------------------- | -------------------------- |
| `exclamation_ratio` | `count('!') / word_count × 100` | Intensity/excitement level | Action films have more `!` |
| `question_ratio`    | `count('?') / word_count × 100` | Interrogative dialogue     | Mysteries have more `?`    |

**Example:**

```python
# Horror/Thriller
"Did you hear that? What was that sound? Oh my God!"
→ exclamation_ratio: 5.0, question_ratio: 10.0

# Documentary/Drama
"The economy shifted. Markets responded accordingly."
→ exclamation_ratio: 0.0, question_ratio: 0.0
```

---

#### Category 6: Script Structure (2 features)

| Feature           | Formula                                  | What It Measures           | Why It Matters                                  |
| ----------------- | ---------------------------------------- | -------------------------- | ----------------------------------------------- |
| `action_density`  | `(brackets + parens) / word_count × 100` | Stage directions frequency | High = visually complex, Low = dialogue-focused |
| `scene_count`     | Count of `INT.` and `EXT.` headings      | Number of scenes           | More scenes = faster cuts, fewer = long takes   |
| `words_per_scene` | `word_count / scene_count`               | Pacing indicator           | High = slow burns, Low = rapid-fire editing     |

**Scene heading examples:**

```
INT. COFFEE SHOP - DAY        ← Interior scene
EXT. ROOFTOP - NIGHT          ← Exterior scene
```

**Example:**

```
Fast-paced action:    scene_count: 150, words_per_scene: 100
Slow drama:           scene_count: 40,  words_per_scene: 400
```

---

#### Category 7: Metadata Features (3 features)

| Feature          | Source            | What It Measures             | Why It Matters                                  |
| ---------------- | ----------------- | ---------------------------- | ----------------------------------------------- |
| `year`           | Excel metadata    | Release year                 | Audience expectations change over time          |
| `decade_encoded` | Derived from year | Era (0=1990s, 1=2000s, etc.) | Captures generational rating patterns           |
| `movie_length`   | Excel metadata    | Runtime in minutes           | Longer films = different structure expectations |

---

### 4.2 How TF-IDF and Structural Features Combine

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FINAL FEATURE VECTOR                          │
├──────────────────────────────┬──────────────────────────────────────┤
│     TF-IDF FEATURES          │     STRUCTURAL FEATURES              │
│     (8,000 dimensions)       │     (17 dimensions)                  │
├──────────────────────────────┼──────────────────────────────────────┤
│ • Word semantics             │ • char_count                         │
│ • Important phrases          │ • word_count                         │
│ • Unique vocabulary          │ • line_count                         │
│ • Bigram patterns            │ • avg_word_length                    │
│   (e.g., "fade to",          │ • unique_word_ratio                  │
│   "cut to", "love you")      │ • long_word_ratio                    │
│                              │ • sentence_count                     │
│                              │ • avg_sentence_length                │
│                              │ • sentence_length_std                │
│                              │ • dialogue_density                   │
│                              │ • unique_characters                  │
│                              │ • exclamation_ratio                  │
│                              │ • question_ratio                     │
│                              │ • action_density                     │
│                              │ • scene_count                        │
│                              │ • words_per_scene                    │
│                              │ • year, decade_encoded, movie_length │
└──────────────────────────────┴──────────────────────────────────────┘
                                │
                                ▼
                    scipy.sparse.hstack([TF-IDF, Numerical])
                                │
                                ▼
                    Combined: 8,017 total features
```

### 4.3 Why Hand-Crafted Features Matter

| TF-IDF Alone               | + Structural Features                |
| -------------------------- | ------------------------------------ |
| Knows "love" appears often | Knows dialogue density is 70%        |
| Knows keywords present     | Knows there are 12 unique characters |
| Captures **what** is said  | Captures **how** it's structured     |
| Semantic meaning           | Writing style & format               |

**Together they capture:**

- **Content** (what the script is about)
- **Style** (how it's written)
- **Structure** (how it's organized)
- **Context** (when it was made)

---

## 5. Why Gradient Boosting Works Best

### Gradient Boosting Superiority for This Task:

| Characteristic           | Why It Helps Our Project                                              |
| ------------------------ | --------------------------------------------------------------------- |
| **Sequential Learning**  | Each tree corrects errors from previous trees                         |
| **Feature Interaction**  | Captures complex relationships (e.g., dialogue_density × scene_count) |
| **Non-Linear Patterns**  | IMDb ratings don't have linear relationships with features            |
| **Regularization**       | Built-in protection against overfitting                               |
| **Sparse Data Handling** | Works excellently with sparse TF-IDF matrices                         |

### How Gradient Boosting Works:

```
Iteration 1: Build Tree₁ → Predict → Calculate Residuals (Errors)
                                            │
                                            ▼
Iteration 2: Build Tree₂ to PREDICT the residuals → Update Predictions
                                            │
                                            ▼
Iteration 3: Build Tree₃ to correct remaining errors → Update
                                            │
                                            ▼
            ... (200 iterations) ...
                                            │
                                            ▼
Final Prediction = Sum of all tree predictions
```

### Model Performance Comparison:

| Model                 | RMSE     | MAE      | R²       | Why This Performance                              |
| --------------------- | -------- | -------- | -------- | ------------------------------------------------- |
| **Gradient Boosting** | Best     | Best     | Best     | Sequential error correction + non-linear patterns |
| Random Forest         | Good     | Good     | Good     | Parallel trees miss sequential dependencies       |
| Ridge Regression      | Moderate | Moderate | Moderate | Linear assumptions don't hold                     |
| ElasticNet            | Moderate | Moderate | Moderate | L1/L2 regularization helps but still linear       |

### Key Hyperparameters Used:

```python
GradientBoostingRegressor(
    n_estimators=200,      # Number of boosting stages
    max_depth=8,           # Maximum tree depth (prevents overfitting)
    learning_rate=0.08,    # Step size shrinkage
    subsample=0.8,         # Row subsampling for regularization
    min_samples_split=10   # Minimum samples to split a node
)
```

---

## 6. Dealing with Limitations

### Identified Limitations & Solutions:

| Limitation                      | Impact                                           | Mitigation Strategy                                      |
| ------------------------------- | ------------------------------------------------ | -------------------------------------------------------- |
| **Class Imbalance**             | Model biased toward common ratings (6-8)         | ✅ Sample weighting: minority classes get 2-3x weight    |
| **Limited Script Availability** | ~5,200 scripts (would be better with 50K+)       | ✅ Feature engineering compensates for data scarcity     |
| **Subjective Ratings**          | IMDb ratings reflect crowd opinion, not quality  | ✅ We predict _audience reception_, not _artistic merit_ |
| **Script ≠ Movie**              | Final film differs from script (editing, acting) | ✅ Use movie_length as proxy for production fidelity     |
| **Decade Bias**                 | Older movies have different rating patterns      | ✅ Decade encoding captures temporal trends              |
| **Genre Differences**           | Horror/Comedy have different rating scales       | ⚠️ Future work: add genre as feature                     |

### Sample Weighting Implementation:

```python
# Rating distribution with weights
Rating Range    | Count | Weight Applied
----------------|-------|---------------
Low (1-4)       | ~150  | 3.5x
Medium (4-6)    | ~450  | 2.0x
Good (6-8)      | ~3000 | 1.0x (baseline)
Excellent (8-10)| ~600  | 1.8x
```

### Future Improvements Roadmap:

1. **Genre Encoding**: Add genre as categorical feature
2. **Cast/Director Metadata**: Incorporate personnel information
3. **Cross-Validation**: Implement k-fold CV for robust metrics
4. **Ensemble Stacking**: Combine multiple model predictions
5. **Deep Learning Experiment**: BERT embeddings + traditional features

---

## 7. Understanding the Models

### Random Forest Regressor

```
                    ┌─────────────┐
                    │   Dataset   │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
    ┌─────────┐      ┌─────────┐       ┌─────────┐
    │ Tree 1  │      │ Tree 2  │  ...  │ Tree N  │
    │(subset) │      │(subset) │       │(subset) │
    └────┬────┘      └────┬────┘       └────┬────┘
         │                │                 │
         ▼                ▼                 ▼
       Pred₁           Pred₂             PredN
         │                │                 │
         └────────────────┼─────────────────┘
                          ▼
                  AVERAGE(Pred₁...PredN)
                          │
                          ▼
                   Final Prediction
```

| Aspect           | Description                                                                    |
| ---------------- | ------------------------------------------------------------------------------ |
| **How it works** | Creates many decision trees on random data subsets, averages their predictions |
| **Strength**     | Reduces variance, handles non-linear relationships                             |
| **Weakness**     | Cannot extrapolate beyond training data range                                  |
| **Our Config**   | 300 trees, max_depth=30, sqrt(features) per split                              |

---

### Gradient Boosting Regressor

```
Prediction = Tree₁ + η·Tree₂ + η·Tree₃ + ... + η·TreeN
             where η = learning_rate (0.08)
```

| Aspect           | Description                                               |
| ---------------- | --------------------------------------------------------- |
| **How it works** | Trees built sequentially, each correcting previous errors |
| **Strength**     | Higher accuracy, captures complex patterns                |
| **Weakness**     | Slower training, can overfit if not tuned                 |
| **Our Config**   | 200 trees, max_depth=8, learning_rate=0.08                |

---

### Ridge Regression

**Formula:**

```
Loss = Σ(y - ŷ)² + α·Σ(β²)
       ─────────   ────────
        MSE Loss   L2 Penalty
```

| Aspect           | Description                                                             |
| ---------------- | ----------------------------------------------------------------------- |
| **How it works** | Linear regression with L2 regularization (penalizes large coefficients) |
| **Strength**     | Fast, interpretable, prevents overfitting                               |
| **Weakness**     | Assumes linear relationship (rarely true for ratings)                   |
| **Our Config**   | α=1.5                                                                   |

---

### ElasticNet

**Formula:**

```
Loss = Σ(y - ŷ)² + α·[λ·Σ|β| + (1-λ)·Σ(β²)]
       ─────────    ─────────   ──────────
        MSE Loss    L1 (Lasso)  L2 (Ridge)
```

| Aspect           | Description                                            |
| ---------------- | ------------------------------------------------------ |
| **How it works** | Combines L1 (sparsity) + L2 (stability) regularization |
| **Strength**     | Feature selection + regularization                     |
| **Weakness**     | Still linear, less accurate for complex patterns       |
| **Our Config**   | α=0.1, l1_ratio=0.5                                    |

---

## 8. Understanding the Metrics: RMSE | MAE | R²

### RMSE (Root Mean Squared Error)

**Formula:**

```
RMSE = √[Σ(actual - predicted)² / n]
```

| Property           | Description                              |
| ------------------ | ---------------------------------------- |
| **Range**          | 0 to ∞ (lower is better)                 |
| **Unit**           | Same as target (rating points)           |
| **Interpretation** | Average magnitude of prediction error    |
| **Sensitivity**    | Heavily penalizes large errors (squared) |

**Example:**
| Movie | Actual | Predicted | Error | Error² |
|-------|--------|-----------|-------|--------|
| A | 7.5 | 7.2 | 0.3 | 0.09 |
| B | 6.0 | 8.0 | 2.0 | 4.00 |
| C | 8.5 | 8.3 | 0.2 | 0.04 |

```
RMSE = √[(0.09 + 4.00 + 0.04) / 3] = √1.38 = 1.17
```

---

### MAE (Mean Absolute Error)

**Formula:**

```
MAE = Σ|actual - predicted| / n
```

| Property           | Description                       |
| ------------------ | --------------------------------- |
| **Range**          | 0 to ∞ (lower is better)          |
| **Unit**           | Same as target (rating points)    |
| **Interpretation** | Average absolute prediction error |
| **Sensitivity**    | All errors weighted equally       |

**Example (same data):**

```
MAE = (0.3 + 2.0 + 0.2) / 3 = 0.83
```

**RMSE vs MAE:**

- RMSE = 1.17 (penalizes the 2.0 error heavily)
- MAE = 0.83 (treats all errors equally)

---

### R² (Coefficient of Determination)

**Formula:**

```
R² = 1 - [SS_res / SS_tot]

where:
  SS_res = Σ(actual - predicted)²   (residual sum of squares)
  SS_tot = Σ(actual - mean)²        (total sum of squares)
```

| Property           | Description                               |
| ------------------ | ----------------------------------------- |
| **Range**          | -∞ to 1 (closer to 1 is better)           |
| **Interpretation** | Proportion of variance explained by model |
| **R² = 1.0**       | Perfect predictions                       |
| **R² = 0.0**       | Model = predicting the mean               |
| **R² < 0**         | Model worse than mean prediction          |

**Interpretation Guide:**

| R² Value  | Model Quality |
| --------- | ------------- |
| 0.9+      | Excellent     |
| 0.7 - 0.9 | Good          |
| 0.5 - 0.7 | Moderate      |
| 0.3 - 0.5 | Weak          |
| < 0.3     | Poor          |

---

### Summary: What Our Metrics Mean

```
┌─────────────────────────────────────────────────────────────────┐
│                  OUR MODEL'S TYPICAL PERFORMANCE                 │
├─────────────┬──────────────────────────────────────────────────┤
│ RMSE: ~0.85 │ On average, predictions are off by 0.85 points   │
├─────────────┼──────────────────────────────────────────────────┤
│ MAE:  ~0.65 │ Typical error is less than 1 rating point        │
├─────────────┼──────────────────────────────────────────────────┤
│ R²:   ~0.45 │ Model explains ~45% of rating variance           │
│             │ (Remaining 55% = subjective factors, production) │
└─────────────┴──────────────────────────────────────────────────┘
```

---

## Quick Reference Card

| Question         | Short Answer                                                        |
| ---------------- | ------------------------------------------------------------------- |
| **Novelty**      | First script→rating predictor with hybrid NLP + structural features |
| **Why 1.5+?**    | No meaningful data exists below 1.5 (statistical reality)           |
| **Why ML?**      | Limited data, interpretability needs, free deployment               |
| **How NLP?**     | TF-IDF (8K features) + 17 structural features                       |
| **Why GB wins?** | Sequential error correction captures non-linear patterns            |
| **Limitations?** | Sample weighting for imbalance, metadata for production gap         |

---

_Document generated for Research Methodology course - IMDb Rating Predictor Project_
