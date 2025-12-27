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


def predict_rating(script_path, model_path='imdb_model.pkl', year=None, decade_encoded=None, movie_length=None):
    """
    Predict IMDb rating for a new script file.

    Args:
        script_path: Path to the script file
        model_path: Path to the trained model file
        year: Optional year of the movie (default: 2020)
        decade_encoded: Optional encoded decade (default: 0)
        movie_length: Optional movie length in minutes (default: 120)

    Usage:
        rating = predict_rating('path/to/script.txt')
        print(f"Predicted Rating: {rating}/10")
        
        # With actual metadata for more accurate prediction:
        rating = predict_rating('path/to/script.txt', year=2008, decade_encoded=2, movie_length=135)
    """
    # Load model package
    with open(model_path, 'rb') as f:
        pkg = pickle.load(f)

    # Read and process script
    with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_text = f.read()

    cleaned_text = ScriptPreprocessor.clean_text(raw_text)
    features = ScriptPreprocessor.extract_features(raw_text)

    # Add year/decade and movie_length (use provided values or defaults)
    features['year'] = year if year is not None else 2020
    features['decade_encoded'] = decade_encoded if decade_encoded is not None else 0
    features['movie_length'] = movie_length if movie_length is not None else 120

    # Transform - ensure features are in the correct order expected by the scaler
    X_tfidf = pkg['tfidf'].transform([cleaned_text])
    features_df = pd.DataFrame([features])
    # Reorder columns to match what scaler was trained on
    features_df = features_df[pkg['scaler'].feature_names_in_]
    X_num = pkg['scaler'].transform(features_df)
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
    features['movie_length'] = 120  # Default average movie length in minutes

    X_tfidf = pkg['tfidf'].transform([cleaned_text])
    features_df = pd.DataFrame([features])
    # Reorder columns to match what scaler was trained on
    features_df = features_df[pkg['scaler'].feature_names_in_]
    X_num = pkg['scaler'].transform(features_df)
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

