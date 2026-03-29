# ============================================================
# PREDICTION FUNCTIONS
# ============================================================
"""
Prediction module for using trained models.
Use these functions after training to predict ratings for new scripts.

Uses SBERT (Sentence Transformer) embeddings with chunking for long scripts.
"""

import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from preprocessing import ScriptPreprocessor

# Cache for loaded SBERT model (avoid reloading on every prediction)
_sbert_cache = {}


def chunk_text(text, chunk_size, overlap):
    """
    Split long text into overlapping chunks.
    
    Args:
        text: Input text string
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks
    
    Returns:
        List of text chunks
    """
    words = text.split()
    
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    step = chunk_size - overlap
    
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += step
        
        if end >= len(words):
            break
    
    return chunks


def get_sbert_model(model_name):
    """Get or load SBERT model from cache."""
    global _sbert_cache
    if model_name not in _sbert_cache:
        _sbert_cache[model_name] = SentenceTransformer(model_name)
    return _sbert_cache[model_name]


def text_to_sbert_embedding(text, sbert_model, chunk_size, chunk_overlap, embedding_dim):
    """
    Convert text to SBERT embedding with chunking for long texts.
    
    Args:
        text: Input text
        sbert_model: Loaded SentenceTransformer model
        chunk_size: Words per chunk
        chunk_overlap: Overlap between chunks
        embedding_dim: Expected embedding dimension
    
    Returns:
        numpy array of shape (embedding_dim,)
    """
    # Chunk the text
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    if not chunks:
        return np.zeros(embedding_dim)
    
    # Embed all chunks
    chunk_embeddings = sbert_model.encode(chunks, show_progress_bar=False)
    
    # Average chunk embeddings
    if len(chunk_embeddings) > 0:
        return np.mean(chunk_embeddings, axis=0)
    else:
        return np.zeros(embedding_dim)


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

    # Get SBERT model and config from saved package
    sbert_model_name = pkg['sbert_model_name']
    embedding_dim = pkg.get('sbert_embedding_dim', 384)
    chunk_size = pkg.get('chunk_size', 256)
    chunk_overlap = pkg.get('chunk_overlap', 50)
    
    # Load SBERT model (cached)
    sbert_model = get_sbert_model(sbert_model_name)
    
    # Generate SBERT embedding with chunking
    X_sbert = text_to_sbert_embedding(
        cleaned_text, sbert_model, chunk_size, chunk_overlap, embedding_dim
    ).reshape(1, -1)
    
    # Transform numerical features
    features_df = pd.DataFrame([features])
    features_df = features_df[pkg['scaler'].feature_names_in_]
    X_num = pkg['scaler'].transform(features_df)
    
    # Combine features
    X = np.hstack([X_sbert, X_num])

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
    features['movie_length'] = 120

    # Get SBERT config
    sbert_model_name = pkg['sbert_model_name']
    embedding_dim = pkg.get('sbert_embedding_dim', 384)
    chunk_size = pkg.get('chunk_size', 256)
    chunk_overlap = pkg.get('chunk_overlap', 50)
    
    # Load SBERT model (cached)
    sbert_model = get_sbert_model(sbert_model_name)
    
    # Generate SBERT embedding
    X_sbert = text_to_sbert_embedding(
        cleaned_text, sbert_model, chunk_size, chunk_overlap, embedding_dim
    ).reshape(1, -1)
    
    # Transform numerical features
    features_df = pd.DataFrame([features])
    features_df = features_df[pkg['scaler'].feature_names_in_]
    X_num = pkg['scaler'].transform(features_df)
    
    # Combine features
    X = np.hstack([X_sbert, X_num])

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
