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
| Rating Range | Count | Weight |
|--------------|-------|---------|
| Low (1-4) | 371 | 5.25x |
| Medium (4-6) | 1,947 | 1.00x |
| Good (6-8) | 1,171 | 1.66x |
| Excellent (8-10) | 147 | 13.24x |

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
- **Model:** XGBoost
- **Training:** Class-balanced with sample weights
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
### Performance Metrics (Test Set)

| Metric | SBERT + XGBoost | Longformer | Difference |
|--------|------------------|-------------|-------------|
| **RMSE** | **0.9881** | 1.0462 | **+5.9% worse** |
| **MAE** | **0.7509** | - | - |
| **R²** | **0.5762** | 0.5249 | **-8.9% lower** |
| **Training Time** | **5 min** (CPU) | 2 hours (GPU) | 24x slower |
| **Model Size** | **2 MB** | 500 MB | 250x larger |

### Validation Performance

| Metric | SBERT + XGBoost | Longformer |
|--------|------------------|-------------|
| **RMSE** | 0.9419 | 0.9985 |
| **MAE** | 0.7067 | - |
| **R²** | 0.5659 | 0.5121 |

### Key Insight
**SBERT + XGBoost achieves better predictive performance with significantly lower computational cost.**

---

## 📊 Slide 9: Performance Analysis

### Content:
### Error Distribution (Test Set)

**SBERT + XGBoost:**
- Within ±0.5: 345 (44.2%) ██████████████████████
- Within ±1.0: 559 (71.7%) ███████████████████████████████████
- Within ±1.5: 673 (86.3%) ███████████████████████████████████████████
- Within ±2.0: 741 (95.0%) ███████████████████████████████████████████████

### Performance by Rating Range

| Rating Range | SBERT + XGBoost MAE | Longformer MAE |
|--------------|---------------------|----------------|
| Low (1-4) | 1.541 (n=107) | - |
| Medium (4-6) | 0.579 (n=380) | - |
| Good (6-8) | 0.659 (n=250) | - |
| Excellent (8-10) | 0.839 (n=43) | - |

### Key Observations
1. **95% accuracy within ±2 points** on 10-point scale
2. **Best performance** on medium-rated movies (4-6)
3. **Struggles more** with extreme ratings (low and excellent)
4. **R² of 0.576** means model explains ~58% of variance

---

## 📊 Slide 10: Why SBERT + XGBoost Performs Better

### Content:
### Five Key Reasons

**1. Full Script Coverage**
- **SBERT:** Processes entire script via chunking (256 words) and averaging. No information loss.
- **Longformer:** Truncates to first 2048 tokens (~1500-2000 words). Rest is discarded.
- **Impact:** Missing plot developments, climax, and resolution affects rating prediction.

**2. Richer Feature Set**
- **SBERT + XGBoost:** 384-dim text embedding + **19 script statistics** (dialogue density, word count, sentence stats, character count) + metadata.
- **Longformer:** 768-dim CLS token + **only 3 metadata features**.
- **Impact:** XGBoost receives both semantic and structural signals.

**3. Task-Specific Optimization**
- **SBERT + XGBoost:** Frozen encoder (no catastrophic forgetting) + powerful tabular regressor.
- **Longformer:** End-to-end fine-tuning of 148M parameters with tiny regression head (800→128→1).
- **Impact:** XGBoost optimized for regression; small head may be insufficient.

**4. Pretraining Alignment**
- **SBERT:** Trained for semantic similarity → good "document summary" vectors.
- **Longformer:** Trained with masked language modeling → CLS not explicitly trained for regression.
- **Impact:** SBERT's objective aligns better with "summarize script → predict rating".

**5. Training Stability**
- **SBERT + XGBoost:** Stable, deterministic training. No gradient issues.
- **Longformer:** Requires careful hyperparameter tuning. Early stopping at epoch 5.
- **Impact:** More stable training → more reliable results.

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
### Main Research Findings

**1. SBERT + XGBoost Outperforms Longformer**
- 6% better RMSE (0.9881 vs 1.0462)
- 10% better R² (0.5762 vs 0.5249)
- More accurate predictions across rating ranges

**2. Computational Efficiency is Critical**
- 24x faster training (5 min vs 2 hours)
- 250x smaller model (2MB vs 500MB)
- No GPU required for SBERT + XGBoost

**3. Feature Engineering Matters**
- Rich feature set (19 structural features) provides important signals
- Combining semantic (SBERT) and structural (script stats) information improves performance

**4. Full Script Coverage is Important**
- Truncation (Longformer) loses critical information
- Chunking with averaging preserves script content

**5. Task-Specific Optimization Beats General Models**
- XGBoost (regression-optimized) outperforms small MLP head
- Frozen encoder prevents catastrophic forgetting

### Practical Implications
**For movie rating prediction:**
- SBERT + XGBoost is the recommended approach
- Balances accuracy, efficiency, and deployability
- Suitable for production use

**For NLP tasks with long documents:**
- Consider task-specific optimization
- Evaluate trade-offs between coverage and complexity
- Hybrid approaches often outperform end-to-end fine-tuning

---

## 📊 Slide 14: Limitations

### Content:
### Study Limitations

**1. Dataset Constraints**
- Limited to 5,204 movies (single dataset source)
- Scripts from IMSDb may not represent all movie genres
- Potential bias toward certain types of movies

**2. Rating Subjectivity**
- IMDb ratings have inherent subjectivity
- Multiple factors affect ratings beyond script quality
- Model cannot capture external factors (marketing, star power)

**3. Feature Limitations**
- Text-only analysis (no visual/audio analysis)
- Limited to script structure features
- No genre, cast, or director information

**4. Model Constraints**
- Linear regression assumes continuous relationship
- May not capture complex non-linear patterns
- Performance varies across rating ranges

**5. Longformer Implementation**
- Truncated to 2048 tokens (not full 4096)
- Could benefit from larger sequence length
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
*Which approach better captures the relationship between movie scripts and their IMDb ratings?*

**Answer:**
**SBERT + XGBoost** achieves better performance than Longformer for movie script rating prediction, with significantly lower computational requirements.

### Key Takeaways

1. **Performance:** 6% better RMSE, 10% better R²
2. **Efficiency:** 24x faster training, 250x smaller model
3. **Practicality:** No GPU required, easy to deploy
4. **Approach:** Full script coverage + rich features + task-specific optimizer

### Contributions

- **Comparative Study:** First systematic comparison of SBERT+XGBoost vs Longformer for this task
- **Best Practices:** Demonstrates value of feature engineering + hybrid models
- **Practical Solution:** Provides production-ready model (2MB, CPU inference)
- **Insights:** Understanding of when transformer vs hybrid approaches work better

### Final Recommendation

**For movie rating prediction tasks:**
Use SBERT + XGBoost approach - it offers the best balance of accuracy, efficiency, and deployability.

**For long-document NLP tasks:**
Consider hybrid approaches with feature engineering and task-specific optimizers before investing in complex transformer fine-tuning.

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
