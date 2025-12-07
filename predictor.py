# ============================================================
# PREDICTION FUNCTIONS
# ============================================================
"""
Prediction module for using trained models.
Use these functions after training to predict ratings for new scripts.
"""

import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from preprocessing import ScriptPreprocessor


def predict_rating(script_path, model_path='imdb_model.pkl'):
    """
    Predict IMDb rating for a new script file.

    Usage:
        rating = predict_rating('path/to/script.txt')
        print(f"Predicted Rating: {rating}/10")
    """
    # Load model package
    with open(model_path, 'rb') as f:
        pkg = pickle.load(f)

    # Read and process script
    with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_text = f.read()

    cleaned_text = ScriptPreprocessor.clean_text(raw_text)
    features = ScriptPreprocessor.extract_features(raw_text)

    # Add default year/decade
    features['year'] = 2020
    features['decade_encoded'] = 0

    # Transform
    X_tfidf = pkg['tfidf'].transform([cleaned_text])
    X_num = pkg['scaler'].transform(pd.DataFrame([features]))
    X = hstack([X_tfidf, csr_matrix(X_num)])

    # Predict
    rating = pkg['model'].predict(X)[0]
    return round(np.clip(rating, 1.0, 10.0), 2)


def predict_from_text(script_text, model_path='imdb_model.pkl'):
    """
    Predict IMDb rating from raw script text.

    Usage:
        script = "JOHN: Hello!\\nMARY: Hi there..."
        rating = predict_from_text(script)
    """
    with open(model_path, 'rb') as f:
        pkg = pickle.load(f)

    cleaned_text = ScriptPreprocessor.clean_text(script_text)
    features = ScriptPreprocessor.extract_features(script_text)
    features['year'] = 2020
    features['decade_encoded'] = 0

    X_tfidf = pkg['tfidf'].transform([cleaned_text])
    X_num = pkg['scaler'].transform(pd.DataFrame([features]))
    X = hstack([X_tfidf, csr_matrix(X_num)])

    rating = pkg['model'].predict(X)[0]
    return round(np.clip(rating, 1.0, 10.0), 2)


def batch_predict(script_paths, model_path='imdb_model.pkl'):
    """
    Predict ratings for multiple scripts.

    Usage:
        paths = ['script1.txt', 'script2.txt', 'script3.txt']
        ratings = batch_predict(paths)
    """
    return [predict_rating(p, model_path) for p in script_paths]

