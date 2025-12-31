# Conceptual Guide: Why We Used What We Used

> This document explains the **reasoning** behind every technical choice in the IMDb Rating Predictor project.

---

## 📚 Table of Contents

1. [TF-IDF Vectorization](#1-tf-idf-vectorization)
2. [Hand-Crafted Structural Features](#2-hand-crafted-structural-features)
3. [Train-Test Split (70/30)](#3-train-test-split-7030)
4. [Gradient Boosting](#4-gradient-boosting)
5. [Sample Weighting](#5-sample-weighting)
6. [Evaluation Metrics (RMSE, MAE, R²)](#6-evaluation-metrics-rmse-mae-r)
7. [Why Machine Learning, Not Deep Learning](#7-why-machine-learning-not-deep-learning)

---

## 1. TF-IDF Vectorization

### What is TF-IDF?

**TF-IDF** (Term Frequency-Inverse Document Frequency) is a numerical statistic that reflects how important a word is to a document in a collection.

```
TF-IDF = TF × IDF

TF (Term Frequency) = Number of times word appears in document
                      ─────────────────────────────────────────
                      Total words in document

IDF (Inverse Document Frequency) = log(Total documents / Documents containing word)
```

### Simple Example:

| Word    | Appears in Script | All Scripts Have It? | TF-IDF Score                     |
| ------- | ----------------- | -------------------- | -------------------------------- |
| "the"   | 500 times         | Yes (common)         | **Low** (IDF penalizes)          |
| "heist" | 20 times          | No (rare)            | **High** (unique to this script) |

### Why We Used It:

| Reason                           | Explanation                                                               |
| -------------------------------- | ------------------------------------------------------------------------- |
| **Handles large vocabulary**     | Can process 8,000 unique words efficiently                                |
| **Ignores common words**         | "the", "and", "is" get low scores automatically                           |
| **Highlights distinctive words** | Action scripts get high scores for "explosion", dramas for "relationship" |
| **Sparse representation**        | Memory-efficient for large text data                                      |
| **Works with any text length**   | Unlike BERT (512 token limit)                                             |

### Our Configuration:

```python
TfidfVectorizer(
    max_features=8000,      # Keep top 8000 words
    stop_words='english',   # Remove "the", "and", etc.
    ngram_range=(1, 2),     # Include single words AND two-word phrases
    min_df=3,               # Word must appear in at least 3 scripts
    max_df=0.85,            # Ignore words in >85% of scripts
    sublinear_tf=True       # Use log(TF) to reduce impact of very frequent words
)
```

---

## 2. Hand-Crafted Structural Features

### What Are They?

Numerical features we **manually designed** based on domain knowledge about scripts.

### Why We Used Them:

| Problem with TF-IDF Only   | Hand-Crafted Features Solve It                   |
| -------------------------- | ------------------------------------------------ |
| Knows "love" appears often | But doesn't know it's said by 2 vs 20 characters |
| Captures word semantics    | But misses **writing style**                     |
| Treats all scripts equally | But doesn't know script is from 1990 vs 2020     |

### The 17 Features Explained:

#### Category 1: Basic Statistics

| Feature      | What It Is       | Why It Matters                |
| ------------ | ---------------- | ----------------------------- |
| `char_count` | Total characters | Longer = more complex story   |
| `word_count` | Total words      | Correlates with movie runtime |
| `line_count` | Total lines      | More lines = more dialogue    |

#### Category 2: Vocabulary Complexity

| Feature             | What It Is                 | Why It Matters                       |
| ------------------- | -------------------------- | ------------------------------------ |
| `avg_word_length`   | Average letters per word   | "Said" (4) vs "Instantaneously" (15) |
| `unique_word_ratio` | Unique words / Total words | Higher = richer vocabulary           |
| `long_word_ratio`   | Words ≥8 chars / Total     | Indicates intellectual depth         |

#### Category 3: Sentence Structure

| Feature               | What It Is           | Why It Matters                   |
| --------------------- | -------------------- | -------------------------------- |
| `sentence_count`      | Number of sentences  | Writing density                  |
| `avg_sentence_length` | Words per sentence   | Short = action, Long = drama     |
| `sentence_length_std` | Variation in lengths | Low = monotonous, High = dynamic |

#### Category 4: Dialogue Analysis

| Feature             | What It Is                   | Why It Matters                    |
| ------------------- | ---------------------------- | --------------------------------- |
| `dialogue_density`  | % of lines that are dialogue | Comedy/Drama = high, Action = low |
| `unique_characters` | Number of speaking roles     | Ensemble cast complexity          |

#### Category 5: Emotional Indicators

| Feature             | What It Is   | Why It Matters               |
| ------------------- | ------------ | ---------------------------- |
| `exclamation_ratio` | `!` per word | Action/thriller = high       |
| `question_ratio`    | `?` per word | Mystery/interrogation = high |

#### Category 6: Script Structure

| Feature           | What It Is                 | Why It Matters              |
| ----------------- | -------------------------- | --------------------------- |
| `action_density`  | Stage directions frequency | [GUNSHOT] counts            |
| `scene_count`     | INT./EXT. headings         | More scenes = faster pacing |
| `words_per_scene` | Words / Scenes             | High = slow burns           |

#### Category 7: Metadata

| Feature          | What It Is         | Why It Matters                          |
| ---------------- | ------------------ | --------------------------------------- |
| `year`           | Release year       | 2020 movies rated differently than 1990 |
| `decade_encoded` | Era category       | Captures generational trends            |
| `movie_length`   | Runtime in minutes | Affects story structure                 |

---

## 3. Train-Test Split (70/30)

### What Is It?

Dividing data into two sets:

- **Training Set (70%)**: Model learns from this
- **Test Set (30%)**: Model evaluated on this (never seen during training)

### Why 70/30?

| Split Ratio | Training Data | Test Data          | When to Use                 |
| ----------- | ------------- | ------------------ | --------------------------- |
| 80/20       | More learning | Less validation    | Large datasets (100K+)      |
| **70/30**   | Balanced      | **Robust testing** | Medium datasets (5K-50K)    |
| 60/40       | Less learning | More validation    | When need confident metrics |

### Why We Used It:

| Reason                   | Explanation                                  |
| ------------------------ | -------------------------------------------- |
| **Prevents overfitting** | Model can't just memorize training data      |
| **Realistic evaluation** | Tests on data it's never seen                |
| **Sufficient test size** | 1,559 samples gives statistical significance |
| **Random state = 42**    | Ensures reproducible results                 |

```python
train_test_split(..., test_size=0.30, random_state=42)
```

---

## 4. Gradient Boosting

### What Is It?

An **ensemble learning** method that builds models **sequentially**, where each new model corrects the errors of the previous ones.

### How It Works (Simple Explanation):

```
Step 1: Build Tree₁ → Makes predictions → Calculate errors (residuals)
Step 2: Build Tree₂ → Trained to PREDICT the errors → Adds small correction
Step 3: Build Tree₃ → Corrects remaining errors → Adds another correction
...
Step 200: Final prediction = Tree₁ + 0.08×Tree₂ + 0.08×Tree₃ + ...
```

### Why It Outperformed Other Models:

| Model                | Why It Lost                  | Gradient Boosting Advantage               |
| -------------------- | ---------------------------- | ----------------------------------------- |
| **Random Forest**    | Trees are independent        | GB trees learn from each other's mistakes |
| **Ridge Regression** | Assumes linear relationships | GB captures non-linear patterns           |
| **ElasticNet**       | Also linear                  | Same as Ridge                             |

### Why We Used It:

| Reason                        | Explanation                                     |
| ----------------------------- | ----------------------------------------------- |
| **Handles sparse data**       | TF-IDF matrices are mostly zeros                |
| **Non-linear relationships**  | Rating isn't a linear function of features      |
| **Feature interactions**      | Learns "dialogue_density × scene_count" effects |
| **Built-in regularization**   | `learning_rate=0.08` prevents overfitting       |
| **No feature scaling needed** | Tree-based method doesn't care about scale      |

### Our Configuration:

```python
GradientBoostingRegressor(
    n_estimators=200,      # 200 sequential trees
    max_depth=8,           # Each tree has max 8 levels (prevents overfitting)
    learning_rate=0.08,    # Small steps (shrinks each tree's contribution)
    subsample=0.8,         # Use 80% of data per tree (adds randomness)
    min_samples_split=10   # Need 10 samples to split a node
)
```

---

## 5. Sample Weighting

### What Is It?

Giving **more importance** to underrepresented classes during training.

### The Problem We Solved:

```
Rating Distribution (Imbalanced):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Low (1-4):       371 samples  ████
Medium (4-6):   1947 samples  ████████████████████████
Good (6-8):     1171 samples  ██████████████
Excellent (8-10): 147 samples  ██

Without weights: Model optimizes for Medium ratings (most common)
                 → Poor predictions for Low and Excellent
```

### How Sample Weighting Works:

```
Weight = Max(class_count) / This_class_count

Low (1-4):      1947 / 371 = 5.25x weight
Medium (4-6):   1947 / 1947 = 1.0x weight (baseline)
Good (6-8):     1947 / 1171 = 1.66x weight
Excellent:      1947 / 147 = 13.24x weight
```

### Why We Used It:

| Without Weighting                | With Weighting                      |
| -------------------------------- | ----------------------------------- |
| Model ignores rare ratings       | Model pays attention to all ratings |
| Good on 4-6, bad on 1-4 and 8-10 | Balanced performance across all     |
| RMSE: 0.9652                     | RMSE: 0.9529 (1.3% better)          |

### Impact on Results:

| Metric | Before | After  | Improvement |
| ------ | ------ | ------ | ----------- |
| RMSE   | 0.9652 | 0.9529 | -1.3%       |
| MAE    | 0.7231 | 0.7095 | -1.9%       |
| R²     | 0.5776 | 0.5827 | +0.9%       |

---

## 6. Evaluation Metrics (RMSE, MAE, R²)

### What They Are:

#### RMSE (Root Mean Squared Error)

```
RMSE = √[Average of (Predicted - Actual)²]
```

- **Punishes large errors** more severely
- Our score: **0.9529** → Average error ~1 rating point

#### MAE (Mean Absolute Error)

```
MAE = Average of |Predicted - Actual|
```

- Treats all errors equally
- Our score: **0.7095** → Typical error ~0.7 points

#### R² (R-Squared / Coefficient of Determination)

```
R² = 1 - (Unexplained Variance / Total Variance)
```

- **Proportion of variance explained** by model
- Our score: **0.5827** → Model explains 58% of rating variance

### Why We Used All Three:

| Metric   | What It Tells Us                                           |
| -------- | ---------------------------------------------------------- |
| **RMSE** | How bad are our worst predictions? (penalizes outliers)    |
| **MAE**  | What's the typical prediction error? (average case)        |
| **R²**   | How much of rating variance do we explain? (model quality) |

### Interpreting Our Results:

```
RMSE: 0.9529  → "On average, we're off by ~1 rating point"
MAE:  0.7095  → "Typical prediction error is 0.7 points"
R²:   0.5827  → "We explain 58% of how ratings vary"
                 "The other 42% = subjective factors, production quality, etc."
```

---

## 7. Why Machine Learning, Not Deep Learning

### The Trade-offs:

| Factor               | Machine Learning (Our Choice) | Deep Learning          |
| -------------------- | ----------------------------- | ---------------------- |
| **Dataset Size**     | ✅ Works with 5,000 samples   | ❌ Needs 50,000+       |
| **Training Time**    | ✅ 5 minutes on CPU           | ❌ Hours on GPU        |
| **Hardware**         | ✅ Any laptop                 | ❌ Requires NVIDIA GPU |
| **Model Size**       | ✅ 1.9 MB                     | ❌ 400MB - 1GB         |
| **Interpretability** | ✅ Feature importance visible | ❌ Black box           |
| **Deployment Cost**  | ✅ Free (Render)              | ❌ $50-200/month       |

### Why Deep Learning Would Fail Here:

1. **Overfitting**: 5,000 samples is too small for neural networks (millions of parameters)
2. **Long text problem**: BERT handles max 512 tokens; scripts have 50,000+ words
3. **No interpretability**: Can't explain "why" a script got a certain rating

### When to Switch to Deep Learning:

- Dataset exceeds 50,000 scripts
- Access to GPU infrastructure
- Don't need explainable predictions
- Have budget for cloud GPU hosting

---

## 🎯 Summary: Our Technical Stack

| Component               | Choice                  | Why                                  |
| ----------------------- | ----------------------- | ------------------------------------ |
| **Text Features**       | TF-IDF (8,000 features) | Efficient, handles any length        |
| **Structural Features** | 17 hand-crafted         | Domain knowledge injection           |
| **Split**               | 70/30                   | Balanced learning + reliable testing |
| **Model**               | Gradient Boosting       | Non-linear, handles sparse data      |
| **Class Balancing**     | Sample Weighting        | Fair predictions across all ratings  |
| **Evaluation**          | RMSE + MAE + R²         | Complete picture of performance      |

---

_This document provides the conceptual foundation for understanding every technical decision in the IMDb Rating Predictor project._
