**Model Details:**

- **Text Encoder:** `all-MiniLM-L6-v2` (frozen, no fine-tuning)
- **Features:** SBERT embedding (384-dim) + 19 script statistics + metadata
- **Regressor:** XGBoost (gradient boosting)
- **Training:** Only XGBoost trained (encoder frozen)

---

## 📊 Results Comparison

### Longformer + Metadata Fusion

| Metric            | Validation (Best Epoch 4)          | Test Set   |
| ----------------- | ---------------------------------- | ---------- |
| **RMSE**          | 0.9985                             | **1.0462** |
| **R²**            | 0.5121                             | **0.5249** |
| **Training Time** | ~2 hours (5 epochs, ~25 min/epoch) | -          |
| **Hardware**      | Google Colab GPU (Tesla T4)        | -          |
| **Model Size**    | ~500MB (checkpoint)                | -          |

**Training Details:**

- **Epochs:** 5 (early stopping at epoch 5, best at epoch 4)
- **Batch Size:** 1 (effective batch size: 4 via gradient accumulation)
- **Learning Rate:** 2e-5
- **Optimizer:** AdamW with weight decay 0.01
- **Mixed Precision:** Enabled (FP16)
- **Early Stopping:** Patience = 3 epochs

### SBERT + XGBoost Baseline

| Metric            | Validation            | Test Set   |
| ----------------- | --------------------- | ---------- |
| **RMSE**          | 0.9419                | **0.9881** |
| **MAE**           | 0.7067                | 0.7509     |
| **R²**            | 0.5659                | **0.5762** |
| **Training Time** | ~5 minutes (CPU)      | -          |
| **Hardware**      | CPU (any laptop)      | -          |
| **Model Size**    | ~2MB (XGBoost pickle) | -          |

---

## 🔍 Detailed Analysis

### Performance Gap

| Aspect                   | SBERT + XGBoost | Longformer   | Difference      |
| ------------------------ | --------------- | ------------ | --------------- |
| **Test RMSE**            | 0.9881          | 1.0462       | **+5.9% worse** |
| **Test R²**              | 0.5762          | 0.5249       | **-8.9% lower** |
| **Training Time**        | 5 minutes       | 2 hours      | 24x slower      |
| **Hardware Requirement** | CPU             | GPU required | -               |

### Why SBERT + XGBoost Performs Better

#### 1. **Full Script Coverage vs. Truncation**

- **SBERT:** Processes entire script by chunking (256 words) and averaging embeddings. No information loss.
- **Longformer:** Truncates to first 2048 tokens (~1500-2000 words). Rest of script is discarded.

**Impact:** For scripts longer than 2000 words (most scripts), Longformer only sees the beginning, missing crucial plot developments, climax, and resolution that affect ratings.

#### 2. **Richer Feature Set**

- **SBERT:** 384-dim text embedding + **19 script-derived features** (word count, dialogue density, sentence stats, character count, etc.) + metadata.
- **Longformer:** 768-dim CLS token + **only 3 metadata features** (year, decade, length).

**Impact:** XGBoost receives both semantic (SBERT) and structural (script stats) signals, while Longformer relies primarily on semantic understanding with minimal structural cues.

#### 3. **Task-Specific Optimization**

- **SBERT + XGBoost:** Frozen encoder (no risk of catastrophic forgetting) + powerful tabular regressor optimized for structured data.
- **Longformer:** End-to-end fine-tuning of large transformer (148M parameters) with tiny regression head (800→128→1). Risk of underfitting the head or overfitting the encoder.

**Impact:** XGBoost is specifically designed for regression on mixed feature types, while Longformer's small head may be insufficient to map the 768-dim representation to ratings.

#### 4. **Pretraining Objective Alignment**

- **SBERT:** Trained for semantic similarity (similar texts → similar vectors). Averaging chunk embeddings produces a good "document summary" vector.
- **Longformer:** Pretrained with masked language modeling (MLM) on long documents. CLS token isn't explicitly trained for regression tasks.

**Impact:** SBERT's objective aligns better with "summarize script → predict rating" than Longformer's MLM objective.

#### 5. **Training Stability**

- **SBERT + XGBoost:** Stable, deterministic training. No risk of gradient issues or learning rate sensitivity.
- **Longformer:** Requires careful hyperparameter tuning (learning rate, batch size, gradient accumulation). Early stopping triggered at epoch 5, suggesting potential underfitting or convergence issues.

**Impact:** More stable training leads to more reliable results.

---

## ❓ Questions & Answers

### Q1: What is Longformer?

**A:** Longformer is a transformer model designed to handle **long documents** efficiently. Unlike BERT (limited to ~512 tokens), Longformer uses **local + global attention**:

- **Local attention:** Each token attends to a sliding window (e.g., 512 tokens) → O(n × window_size) complexity
- **Global attention:** Special tokens (like [CLS]) attend to the entire sequence → provides global context

This allows Longformer to process sequences up to **4096 tokens** (or more) while maintaining reasonable memory and compute costs.

**In this project:** We use `allenai/longformer-base-4096` with a 2048-token limit (due to GPU memory constraints), extracting the CLS token (768-dim) as the script representation.

---

### Q2: Why is SBERT + XGBoost performing better than Longformer here?

**A:** Five main reasons:

1. **Full script coverage:** SBERT processes the entire script via chunk-and-average, while Longformer truncates to 2048 tokens, losing information from longer scripts.

2. **Richer features:** SBERT + XGBoost uses 384-dim text embedding + 19 script statistics (dialogue density, word count, etc.) + metadata. Longformer uses only 768-dim CLS + 3 metadata features.

3. **Better regressor:** XGBoost is a powerful gradient boosting model optimized for tabular regression. Longformer uses a small 2-layer MLP (800→128→1) that may be insufficient.

4. **Training efficiency:** SBERT encoder is frozen (no fine-tuning risk), and XGBoost trains quickly and stably. Longformer requires end-to-end fine-tuning of 148M parameters, which is harder to optimize.

5. **Pretraining alignment:** SBERT's similarity-based training aligns better with "document summarization → rating prediction" than Longformer's MLM objective.

**Bottom line:** SBERT + XGBoost combines the best of both worlds: good semantic representation (SBERT) + powerful regression (XGBoost) + full script coverage + rich features.

---

### Q3: Could Longformer be improved?

**A:** Yes, several improvements could help:

1. **Increase sequence length:** Use full 4096 tokens (if GPU memory allows) to capture more of each script.

2. **Add script features:** Include the same 19 script statistics used in SBERT pipeline (dialogue density, word count, etc.) in the fusion layer.

3. **Larger regression head:** Increase hidden dimension (128 → 256 or 512) or add more layers to better map the 768-dim representation to ratings.

4. **More training:** Train for more epochs (10-15) with learning rate scheduling. Current training stopped early at epoch 5.

5. **Different pooling:** Instead of just CLS token, try mean pooling of all token embeddings or attention-weighted pooling.

6. **Multi-task learning:** Train on additional tasks (genre classification, sentiment) alongside rating prediction to improve representation.

7. **Larger model:** Use `longformer-large-4096` (more parameters) if compute allows.

---

### Q4: When would Longformer be preferred over SBERT + XGBoost?

**A:** Longformer would be better when:

1. **Sequential dependencies matter:** If the order of events in a script (beginning → middle → end) is crucial for rating prediction. Longformer's attention mechanism can capture long-range dependencies better than chunk-and-average.

2. **Very long documents:** For scripts longer than what SBERT can efficiently chunk (e.g., >10,000 words), Longformer's efficient attention scales better.

3. **End-to-end learning:** When you want a single model that learns task-specific representations rather than using frozen embeddings.

4. **Transfer learning:** If you plan to fine-tune on multiple related tasks (rating, genre, box office), Longformer's unified architecture is more flexible.

**However**, for this specific task (IMDb rating prediction), SBERT + XGBoost's simplicity, speed, and performance make it the better choice.

---

### Q5: What are the computational requirements?

**A:**

| Aspect                 | SBERT + XGBoost        | Longformer             |
| ---------------------- | ---------------------- | ---------------------- |
| **Training Time**      | ~5 minutes (CPU)       | ~2 hours (GPU)         |
| **Inference Time**     | ~0.1s per script (CPU) | ~0.5s per script (GPU) |
| **Memory (Training)**  | ~2GB RAM               | ~8GB VRAM (GPU)        |
| **Memory (Inference)** | ~500MB RAM             | ~2GB VRAM (GPU)        |
| **Model Size**         | ~2MB                   | ~500MB                 |
| **Hardware**           | Any CPU                | NVIDIA GPU required    |

---

## 🎯 Conclusions

1. **SBERT + XGBoost is the better approach** for this IMDb rating prediction task, achieving **~6% better RMSE** and **~10% better R²** than Longformer.

2. **Key advantages of SBERT + XGBoost:**
   - Processes full scripts (no truncation)
   - Richer feature set (text + statistics)
   - Faster training (5 min vs 2 hours)
   - No GPU required
   - Smaller model size (2MB vs 500MB)

3. **Longformer's limitations in this context:**
   - Truncation loses information from longer scripts
   - Limited feature set (only 3 metadata features)
   - Small regression head may be insufficient
   - Requires GPU and longer training time

4. **Future work:** To improve Longformer, consider:
   - Increasing sequence length to 4096 tokens
   - Adding script-derived features to the fusion layer
   - Larger regression head
   - More training epochs
   - Different pooling strategies

---

## 📁 Files & Resources

- **Notebook:** `longformer_colab_standalone.ipynb` (self-contained Colab notebook)
- **Training Script:** `longformer_trainer.py` (modular training script)
- **Model Checkpoint:** `longformer_imdb_model.pt` (saved to Google Drive)
- **Baseline Results:** `output.md` (SBERT + XGBoost results)

---

## 📚 References

- Longformer Paper: [Beltagy et al. (2020)](https://arxiv.org/abs/2004.05150)
- SBERT Paper: [Reimers & Gurevych (2019)](https://arxiv.org/abs/1908.10084)
- XGBoost: [Chen & Guestrin (2016)](https://arxiv.org/abs/1603.02754)

---

**Document Version:** 1.0  
**Last Updated:** February 2026
