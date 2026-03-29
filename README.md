# IMDb Rating Predictor - SBERT Semantic Embeddings Approach

A machine learning model that predicts IMDb ratings for movie scripts using Sentence-BERT (SBERT) semantic embeddings and classical ML algorithms.

## 🎯 Approach Overview

This branch implements the **SBERT (Sentence-BERT) Semantic Embeddings** approach for IMDb rating prediction. It uses pre-trained SBERT models to convert entire scripts into dense semantic embeddings, then applies classical machine learning algorithms for rating prediction.

### Why SBERT?
- **Semantic Understanding**: Captures meaning rather than surface-level features
- **Pre-trained Models**: Uses models trained on millions of sentence pairs
- **Efficient Embeddings**: 384-dimensional vectors representing script content
- **Fast Inference**: No fine-tuning required; direct embedding + ML prediction
- **Interpretable**: Works with classical ML models (Random Forest, SVM, etc.)

## 📁 Project Structure

```
imdb-predictor/ (sbert branch)
├── config.py                    # Configuration settings
├── preprocessing.py             # Text preprocessing & feature extraction
├── data_loader.py               # Data loading functions
├── trainer.py                   # Model training & evaluation
├── predictor.py                 # Prediction functions
├── imdb_rating_predictor.py     # Main training script
├── example_usage.py             # Usage examples
├── api.py                       # REST API for predictions
├── imdb_model.pkl               # Trained model (generated after training)
├── movie_lengths.xlsx           # Excel file with movie metadata
├── test_model.py                # Model testing utilities
├── test_predictions.py          # Prediction validation
└── scripts/                     # Folder containing script .txt files
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

Key dependencies:
- `sentence-transformers` - SBERT models
- `scikit-learn` - ML algorithms & metrics
- `pandas`, `numpy` - Data processing
- `openpyxl` - Excel file handling

### 2. Training the Model

```bash
python imdb_rating_predictor.py
```

This will:
- Load scripts from the `scripts/` folder
- Generate SBERT embeddings for each script
- Extract semantic features
- Train ML models (ensemble of algorithms)
- Save the best model as `imdb_model.pkl`

### 3. Using the Trained Model

#### Option A: Python Script
```python
from predictor import predict_rating, predict_from_text

# Predict from a file
rating = predict_rating('path/to/script.txt')
print(f"Predicted Rating: {rating}/10")

# Predict from raw text
script_text = "JOHN: Hello!\\nMARY: Hi there!"
rating = predict_from_text(script_text)
print(f"Predicted Rating: {rating}/10")
```

#### Option B: Example Script
```bash
python example_usage.py
```

#### Option C: Interactive Python
```python
from predictor import predict_rating
rating = predict_rating('scripts/your_script.txt')
print(f"Rating: {rating}/10")
```

### 4. Using the API

```bash
python api.py
# Visit http://localhost:5000
```

## 📊 Feature Engineering Pipeline

**SBERT Semantic Embedding Approach:**

1. **Script Input**: Raw screenplay text
2. **Preprocessing**: 
   - Cleaning and normalization
   - Sentence segmentation
3. **SBERT Encoding**: 
   - Convert entire script to SBERT embedding (384 dims)
   - Dimension reduction techniques if needed
4. **Feature Extraction**:
   - SBERT embedding vector itself
   - Statistical features from embeddings
   - Dialogue and narrative analysis
5. **ML Models**:
   - Random Forest
   - Support Vector Regression (SVR)
   - Gradient Boosting
   - Ensemble combination

## 🔧 Configuration

Edit `config.py` to adjust:
- SBERT model selection (`all-MiniLM-L6-v2`, `paraphrase-MiniLM-L6-v2`, etc.)
- Train/test/validation split ratios
- ML model parameters
- Feature scaling options
- File paths for data

## 📈 Performance Metrics

The model evaluates using:
- **R² Score**: Coefficient of determination
- **MAE**: Mean Absolute Error on 1-10 scale
- **RMSE**: Root Mean Squared Error
- **Cross-validation**: K-fold validation for robustness

## 🔍 Key Features

- ✅ **Semantic Understanding**: Deep semantic meaning from SBERT
- ✅ **No Fine-tuning**: Uses pre-trained models directly
- ✅ **Classical ML**: Interpretable models like Random Forest
- ✅ **Fast Training**: Minutes instead of hours
- ✅ **Fast Inference**: Real-time predictions
- ✅ **Production Ready**: Lightweight and deployable

## 📊 What the Model Does

The pipeline analyzes scripts through multiple lens:

**SBERT Embeddings:**
- **Semantic Similarity**: How scripts relate to each other semantically
- **Content Representation**: 384-dimensional feature space
- **Universal Sentence Encoder**: Captures meaning across document

**Classical ML Features:**
- **Text Features**: Word count, vocabulary diversity, readability metrics
- **Dialogue Analysis**: Character count, dialogue density, exchange patterns
- **Emotional Indicators**: Exclamation/question ratios, sentiment markers
- **Script Structure**: Scene distribution, action vs dialogue balance
- **Narrative Elements**: Named entity recognition, topic modeling

## 💡 Tips for Best Results

1. **Model Quality**: Best with scripts similar to training data
2. **Script Format**: Works with standard screenplay formats
3. **File Size**: Scripts should be at least 1KB for reliable embeddings
4. **Encoding**: Handles UTF-8, Latin-1, and CP1252 encodings
5. **Batch Processing**: Use `batch_predict()` for multiple scripts

## 🧪 SBERT Model Options

Different pre-trained models available:

```python
# Balanced speed and quality
"all-MiniLM-L6-v2"  # 384 dims, fast

# Better quality
"paraphrase-MiniLM-L6-v2"  # 384 dims, paraphrase-optimized
"all-mpnet-base-v2"  # 768 dims, high quality

# Custom
Use any model from: https://www.sbert.net/docs/pretrained_models.html
```

## 📚 Research & Documentation

- **CONCEPTUAL_GUIDE.md**: Detailed methodology explanation
- **RESEARCH_QA.md**: Q&A on approach and design decisions
- **PRESENTATION_SPEECH.md**: Project overview and findings

## 🎯 Next Steps

After training, you can:
1. ✅ Use `predict_rating()` to predict ratings for new scripts
2. ✅ Use `batch_predict()` to process multiple scripts at once
3. ✅ Integrate into your own applications via API
4. ✅ Fine-tune by adjusting SBERT model or ML parameters
5. ✅ Combine with other approaches for ensemble predictions

## 📖 Example Use Cases

- **Screenplay Evaluation**: Quick quality assessment
- **Pre-production Analysis**: Early feedback on scripts
- **Batch Processing**: Evaluate multiple scripts efficiently
- **Research**: Analyze patterns in successful scripts
- **A/B Testing**: Compare script variations

## ⚠️ Notes

- The model predicts ratings on a scale of 1-10
- Predictions are based on script content only
- Results may vary for scripts very different from training data
- Always validate predictions with human review
- SBERT embeddings may not capture all domain-specific nuances

## 🔗 References

- [Sentence-BERT Documentation](https://www.sbert.net/)
- [Pre-trained Models](https://www.sbert.net/docs/pretrained_models.html)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Semantic Textual Similarity](https://en.wikipedia.org/wiki/Semantic_similarity)