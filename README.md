# IMDb Rating Predictor

A machine learning model that predicts IMDb ratings for movie scripts based on text analysis and feature extraction.

## 📁 Project Structure

```
imdb-predictor/
├── config.py                    # Configuration settings
├── preprocessing.py             # Text preprocessing & feature extraction
├── data_loader.py               # Data loading functions
├── trainer.py                   # Model training & evaluation
├── predictor.py                 # Prediction functions
├── imdb_rating_predictor.py     # Main training script
├── example_usage.py             # Usage examples
├── imdb_model.pkl               # Trained model (generated after training)
├── isteaq ulab info.xlsx        # Excel file with movie metadata
└── scripts/                     # Folder containing script .txt files
```

## 🚀 Quick Start

### 1. Training the Model

If you haven't trained the model yet:

```bash
python imdb_rating_predictor.py
```

This will:
- Load scripts from the `scripts/` folder
- Extract features from each script
- Train multiple ML models
- Save the best model as `imdb_model.pkl`

### 2. Using the Trained Model

Once you have `imdb_model.pkl`, you can predict ratings for new scripts:

#### Option A: Using Python Script

```python
from predictor import predict_rating, predict_from_text

# Predict from a file
rating = predict_rating('path/to/script.txt')
print(f"Predicted Rating: {rating}/10")

# Predict from raw text
script_text = "JOHN: Hello!\nMARY: Hi there!"
rating = predict_from_text(script_text)
print(f"Predicted Rating: {rating}/10")
```

#### Option B: Using the Example Script

```bash
python example_usage.py
```

Edit `example_usage.py` to uncomment the examples you want to run.

#### Option C: Interactive Python

```python
# Start Python
python

# Import the predictor
>>> from predictor import predict_rating

# Predict a rating
>>> rating = predict_rating('scripts/your_script.txt')
>>> print(f"Rating: {rating}/10")
```

## 📊 What the Model Does

The model analyzes:
- **Text Features**: Word count, vocabulary complexity, sentence structure
- **Dialogue Analysis**: Dialogue density, number of characters
- **Emotional Indicators**: Exclamation/question ratios
- **Script Structure**: Scene count, pacing, action density
- **TF-IDF Features**: Important words and phrases

## 🔧 Configuration

Edit `config.py` to adjust:
- File paths (Excel file, scripts directory)
- Column names in your Excel file
- Model parameters (test size, random state, etc.)

## 📝 Requirements

```
pandas
numpy
scikit-learn
openpyxl
scipy
```

Install with:
```bash
pip install pandas numpy scikit-learn openpyxl scipy
```

## 💡 Tips

1. **Model Quality**: The model works best with scripts similar to the training data
2. **Script Format**: Works with standard screenplay formats
3. **File Size**: Scripts should be at least 1KB to be processed
4. **Encoding**: The loader handles UTF-8, Latin-1, and CP1252 encodings

## 🎯 Next Steps

After training, you can:
1. ✅ Use `predict_rating()` to predict ratings for new scripts
2. ✅ Use `batch_predict()` to process multiple scripts at once
3. ✅ Integrate into your own applications
4. ✅ Fine-tune the model by adjusting parameters in `config.py`

## 📖 Example Use Cases

- **Screenplay Evaluation**: Quickly assess script quality
- **Pre-production Analysis**: Get early feedback on scripts
- **Batch Processing**: Evaluate multiple scripts efficiently
- **Research**: Analyze patterns in successful scripts

## ⚠️ Notes

- The model predicts ratings on a scale of 1-10
- Predictions are based on script content only
- Results may vary for scripts very different from training data
- Always validate predictions with human review





