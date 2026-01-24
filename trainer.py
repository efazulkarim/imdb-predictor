# ============================================================
# MODEL TRAINING
# ============================================================
"""
Model training and evaluation module.
Handles training multiple models, evaluation, and saving.

Features:
- SBERT (Sentence Transformer) embeddings for semantic text vectorization
- Chunking for long scripts (handles scripts longer than SBERT's token limit)
- Train/Validation/Test split (70%/15%/15%)
- LightGBM & XGBoost models alongside traditional ML models
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sentence_transformers import SentenceTransformer
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from config import (
    TEST_SIZE, VALIDATION_SIZE, RANDOM_STATE,
    SBERT_MODEL_NAME, SBERT_EMBEDDING_DIM, CHUNK_SIZE, CHUNK_OVERLAP
)


def compute_sample_weights(y):
    """
    Compute sample weights based on rating range frequency.
    Underrepresented rating ranges get higher weights.
    
    Rating Ranges:
        Low (1-4), Medium (4-6), Good (6-8), Excellent (8-10)
    """
    weights = np.ones(len(y))
    
    # Define rating ranges
    ranges = [
        (1, 4, 'Low'),
        (4, 6, 'Medium'),
        (6, 8, 'Good'),
        (8, 10, 'Excellent')
    ]
    
    # Count samples in each range
    range_counts = {}
    for low, high, name in ranges:
        mask = (y >= low) & (y < high)
        range_counts[name] = np.sum(mask)
    
    # Compute inverse frequency weights
    total = len(y)
    max_count = max(range_counts.values()) if range_counts.values() else 1
    
    print(f"\n⚖️  Sample Weights (Class Balancing):")
    for low, high, name in ranges:
        mask = (y >= low) & (y < high)
        count = range_counts[name]
        if count > 0:
            # Weight = max_count / count (so minority classes get higher weight)
            weight = max_count / count
            weights[mask] = weight
            print(f"   {name:12s} ({low}-{high}): n={count:4d}, weight={weight:.2f}x")
    
    return weights


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
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
        
        # Stop if we've reached the end
        if end >= len(words):
            break
    
    return chunks


def embed_scripts_sbert(scripts_text, sbert_model, show_progress=True):
    """
    Convert scripts to SBERT embeddings with chunking for long texts.
    
    Args:
        scripts_text: List of script texts
        sbert_model: Loaded SentenceTransformer model
        show_progress: Whether to print progress updates
    
    Returns:
        numpy array of shape (n_scripts, embedding_dim)
    """
    embeddings = []
    total = len(scripts_text)
    
    for i, script in enumerate(scripts_text):
        if show_progress and (i + 1) % 500 == 0:
            print(f"   Embedding progress: {i + 1}/{total} scripts...")
        
        # Chunk the script
        chunks = chunk_text(script)
        
        # Embed all chunks
        chunk_embeddings = sbert_model.encode(chunks, show_progress_bar=False)
        
        # Average chunk embeddings to get single document embedding
        if len(chunk_embeddings) > 0:
            doc_embedding = np.mean(chunk_embeddings, axis=0)
        else:
            doc_embedding = np.zeros(SBERT_EMBEDDING_DIM)
        
        embeddings.append(doc_embedding)
    
    return np.array(embeddings)


def train_and_evaluate(scripts_text, ratings, features_df, movie_names=None, script_files=None):
    """Train multiple models, evaluate, and return the best.
    
    Args:
        scripts_text: List of script texts
        ratings: Array of IMDb ratings
        features_df: DataFrame of extracted features
        movie_names: Optional list of movie names (for saving test set)
        script_files: Optional list of script filenames (for precise test set matching)
    """
    print("\n" + "=" * 70)
    print("  MODEL TRAINING (70% Train / 15% Validation / 15% Test)")
    print("=" * 70)

    # Create indices for tracking which movies go to train/val/test
    indices = np.arange(len(scripts_text))
    
    # Convert scripts to list for easier indexing
    scripts_list = list(scripts_text)
    
    # === Data Split: Train / Temp (Val+Test) ===
    temp_size = TEST_SIZE + VALIDATION_SIZE  # 30% for temp (val + test)
    
    (idx_train, idx_temp, 
     y_train, y_temp) = train_test_split(
        indices, ratings,
        test_size=temp_size,
        random_state=RANDOM_STATE
    )
    
    # === Split Temp into Validation and Test (50/50 of temp = 15% each) ===
    (idx_val, idx_test, 
     y_val, y_test) = train_test_split(
        idx_temp, y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE
    )
    
    # Get corresponding data splits
    X_text_train = [scripts_list[i] for i in idx_train]
    X_text_val = [scripts_list[i] for i in idx_val]
    X_text_test = [scripts_list[i] for i in idx_test]
    
    X_feat_train = features_df.iloc[idx_train]
    X_feat_val = features_df.iloc[idx_val]
    X_feat_test = features_df.iloc[idx_test]
    
    # Save test set information for consistent external testing
    if movie_names is not None:
        test_set_info = {
            'test_indices': idx_test.tolist(),
            'test_movie_names': [movie_names[i] for i in idx_test],
            'test_script_files': [script_files[i] for i in idx_test] if script_files else [],
            'test_ratings': y_test.tolist(),
            'validation_indices': idx_val.tolist(),
            'validation_movie_names': [movie_names[i] for i in idx_val],
            'validation_ratings': y_val.tolist(),
            'train_indices': idx_train.tolist(),
            'total_samples': len(scripts_text),
            'test_size': TEST_SIZE,
            'validation_size': VALIDATION_SIZE,
            'random_state': RANDOM_STATE
        }
        with open('test_set_info.json', 'w', encoding='utf-8') as f:
            json.dump(test_set_info, f, indent=2)
        print(f"\n💾 Test set info saved: 'test_set_info.json'")

    print(f"\n📊 Data Split:")
    print(f"   Training:   {len(X_text_train)} samples ({(1-temp_size)*100:.0f}%)")
    print(f"   Validation: {len(X_text_val)} samples ({VALIDATION_SIZE*100:.0f}%)")
    print(f"   Testing:    {len(X_text_test)} samples ({TEST_SIZE*100:.0f}%)")
    print(f"   Rating range: {ratings.min():.1f} - {ratings.max():.1f}")
    print(f"   Mean rating: {ratings.mean():.2f} (std: {ratings.std():.2f})")

    # === Compute Sample Weights for Training ===
    sample_weights = compute_sample_weights(y_train)

    # === SBERT Embedding ===
    print(f"\n🔤 Loading SBERT model: '{SBERT_MODEL_NAME}'...")
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
    print(f"   Model loaded. Embedding dimension: {SBERT_EMBEDDING_DIM}")
    print(f"   Chunk size: {CHUNK_SIZE} words, Overlap: {CHUNK_OVERLAP} words")
    
    print(f"\n📝 Generating SBERT embeddings for training set...")
    X_sbert_train = embed_scripts_sbert(X_text_train, sbert_model)
    print(f"   Training embeddings shape: {X_sbert_train.shape}")
    
    print(f"\n📝 Generating SBERT embeddings for validation set...")
    X_sbert_val = embed_scripts_sbert(X_text_val, sbert_model, show_progress=False)
    
    print(f"\n📝 Generating SBERT embeddings for test set...")
    X_sbert_test = embed_scripts_sbert(X_text_test, sbert_model, show_progress=False)

    # === Scale Numerical Features ===
    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(X_feat_train)
    X_num_val = scaler.transform(X_feat_val)
    X_num_test = scaler.transform(X_feat_test)

    print(f"   Numerical features: {X_num_train.shape[1]}")

    # === Combine Features (SBERT embeddings + numerical features) ===
    X_train = np.hstack([X_sbert_train, X_num_train])
    X_val = np.hstack([X_sbert_val, X_num_val])
    X_test = np.hstack([X_sbert_test, X_num_test])

    print(f"   Combined features: {X_train.shape[1]} (SBERT: {SBERT_EMBEDDING_DIM} + Numerical: {X_num_train.shape[1]})")

    # === Define Models ===
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=300,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.8,
            min_samples_split=10,
            random_state=RANDOM_STATE
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        ),
        'Ridge Regression': Ridge(
            alpha=1.5,
            random_state=RANDOM_STATE
        ),
        'ElasticNet': ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            random_state=RANDOM_STATE
        )
    }

    # === Train and Evaluate ===
    print("\n" + "-" * 70)
    print("  MODEL EVALUATION RESULTS")
    print("-" * 70)

    results = {}
    best_model_name = None
    best_val_rmse = float('inf')

    for name, model in models.items():
        print(f"\n🔄 Training {name}...")

        # Train with sample weights
        if hasattr(model, 'fit') and 'sample_weight' in model.fit.__code__.co_varnames:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)

        # Predict on validation set
        y_val_pred = model.predict(X_val)
        y_val_pred = np.clip(y_val_pred, 1.0, 10.0)
        
        # Predict on test set
        y_test_pred = model.predict(X_test)
        y_test_pred = np.clip(y_test_pred, 1.0, 10.0)

        # Validation Metrics
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        # Test Metrics
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        results[name] = {
            'model': model,
            'val_predictions': y_val_pred,
            'test_predictions': y_test_pred,
            'Val_RMSE': val_rmse,
            'Val_MAE': val_mae,
            'Val_R2': val_r2,
            'Test_RMSE': test_rmse,
            'Test_MAE': test_mae,
            'Test_R2': test_r2
        }

        print(f"   Validation: RMSE={val_rmse:.4f} | MAE={val_mae:.4f} | R²={val_r2:.4f}")
        print(f"   Test:       RMSE={test_rmse:.4f} | MAE={test_mae:.4f} | R²={test_r2:.4f}")

        # Select best model based on validation RMSE
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_name = name

    # === Best Model Summary ===
    print("\n" + "=" * 70)
    print(f"  🏆 BEST MODEL: {best_model_name}")
    print("=" * 70)
    print(f"   Validation RMSE: {results[best_model_name]['Val_RMSE']:.4f}")
    print(f"   Validation MAE:  {results[best_model_name]['Val_MAE']:.4f}")
    print(f"   Validation R²:   {results[best_model_name]['Val_R2']:.4f}")
    print(f"   ---")
    print(f"   Test RMSE: {results[best_model_name]['Test_RMSE']:.4f}")
    print(f"   Test MAE:  {results[best_model_name]['Test_MAE']:.4f}")
    print(f"   Test R²:   {results[best_model_name]['Test_R2']:.4f}")

    return results, best_model_name, sbert_model, scaler, y_test, y_val


def detailed_analysis(results, best_model_name, y_test):
    """Provide detailed prediction analysis."""
    print("\n" + "=" * 70)
    print("  PREDICTION ANALYSIS (Test Set)")
    print("=" * 70)

    y_pred = results[best_model_name]['test_predictions']

    # Error analysis
    errors = np.abs(y_pred - y_test)

    print("\n📊 Error Distribution:")
    print("-" * 40)
    brackets = [(0.5, '±0.5'), (1.0, '±1.0'), (1.5, '±1.5'), (2.0, '±2.0')]
    for threshold, label in brackets:
        count = np.sum(errors <= threshold)
        pct = count / len(errors) * 100
        bar = '█' * int(pct / 2)
        print(f"   Within {label}: {count:4d} ({pct:5.1f}%) {bar}")

    # Sample predictions
    print("\n📋 Sample Predictions (15 random):")
    print("-" * 50)
    print(f"   {'Actual':>8} | {'Predicted':>10} | {'Error':>8}")
    print("-" * 50)

    np.random.seed(RANDOM_STATE)
    indices = np.random.choice(len(y_test), min(15, len(y_test)), replace=False)
    for i in indices:
        actual = y_test[i]
        pred = y_pred[i]
        error = pred - actual
        print(f"   {actual:8.2f} | {pred:10.2f} | {error:+8.2f}")

    # Rating range performance
    print("\n📈 Performance by Rating Range:")
    print("-" * 50)
    ranges = [(1, 4, 'Low (1-4)'), (4, 6, 'Medium (4-6)'), 
              (6, 8, 'Good (6-8)'), (8, 10, 'Excellent (8-10)')]

    for low, high, label in ranges:
        mask = (y_test >= low) & (y_test < high)
        if np.sum(mask) > 0:
            range_mae = np.mean(np.abs(y_pred[mask] - y_test[mask]))
            count = np.sum(mask)
            print(f"   {label:20s}: MAE = {range_mae:.3f} (n={count})")


def save_model(results, best_model_name, sbert_model_name, scaler, decade_encoder):
    """Save trained model and preprocessors.
    
    Note: We save the SBERT model NAME (not the model itself) to reload at inference time.
    This keeps the pickle file smaller and avoids serialization issues with PyTorch models.
    """
    model_package = {
        'model': results[best_model_name]['model'],
        'sbert_model_name': sbert_model_name,  # Save model name, not the model itself
        'sbert_embedding_dim': SBERT_EMBEDDING_DIM,
        'chunk_size': CHUNK_SIZE,
        'chunk_overlap': CHUNK_OVERLAP,
        'scaler': scaler,
        'decade_encoder': decade_encoder,
        'model_name': best_model_name,
        'metrics': {
            'Val_RMSE': results[best_model_name]['Val_RMSE'],
            'Val_MAE': results[best_model_name]['Val_MAE'],
            'Val_R2': results[best_model_name]['Val_R2'],
            'Test_RMSE': results[best_model_name]['Test_RMSE'],
            'Test_MAE': results[best_model_name]['Test_MAE'],
            'Test_R2': results[best_model_name]['Test_R2']
        }
    }

    with open('imdb_model.pkl', 'wb') as f:
        pickle.dump(model_package, f)

    size_mb = os.path.getsize('imdb_model.pkl') / (1024 * 1024)
    print(f"\n💾 Model saved: 'imdb_model.pkl' ({size_mb:.1f} MB)")
    print(f"   Note: SBERT model '{sbert_model_name}' will be loaded at inference time.")
