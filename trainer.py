# ============================================================
# MODEL TRAINING
# ============================================================
"""
Model training and evaluation module.
Handles training multiple models, evaluation, and saving.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import TEST_SIZE, RANDOM_STATE, MAX_TFIDF_FEATURES


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
    print("  MODEL TRAINING (70% Train / 30% Test)")
    print("=" * 70)

    # Create indices for tracking which movies go to train/test
    indices = np.arange(len(scripts_text))
    
    # === Data Split (with indices) ===
    (X_text_train, X_text_test, 
     y_train, y_test,
     X_feat_train, X_feat_test,
     idx_train, idx_test) = train_test_split(
        scripts_text, ratings, features_df, indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    # Save test set information for consistent external testing
    if movie_names is not None:
        test_set_info = {
            'test_indices': idx_test.tolist(),
            'test_movie_names': [movie_names[i] for i in idx_test],
            'test_script_files': [script_files[i] for i in idx_test] if script_files else [],
            'test_ratings': y_test.tolist(),
            'train_indices': idx_train.tolist(),
            'total_samples': len(scripts_text),
            'test_size': TEST_SIZE,
            'random_state': RANDOM_STATE
        }
        with open('test_set_info.json', 'w', encoding='utf-8') as f:
            json.dump(test_set_info, f, indent=2)
        print(f"\n💾 Test set info saved: 'test_set_info.json'")

    print(f"\n📊 Data Split:")
    print(f"   Training: {len(X_text_train)} samples ({100-TEST_SIZE*100:.0f}%)")
    print(f"   Testing:  {len(X_text_test)} samples ({TEST_SIZE*100:.0f}%)")
    print(f"   Rating range: {ratings.min():.1f} - {ratings.max():.1f}")
    print(f"   Mean rating: {ratings.mean():.2f} (std: {ratings.std():.2f})")

    # === TF-IDF Vectorization ===
    print(f"\n🔤 Creating TF-IDF features (max {MAX_TFIDF_FEATURES} features)...")

    tfidf = TfidfVectorizer(
        max_features=MAX_TFIDF_FEATURES,
        stop_words='english',
        ngram_range=(1, 2),       # Unigrams + Bigrams
        min_df=3,                  # Minimum document frequency
        max_df=0.85,               # Maximum document frequency
        sublinear_tf=True,         # Apply log scaling
        dtype=np.float32
    )

    X_tfidf_train = tfidf.fit_transform(X_text_train)
    X_tfidf_test = tfidf.transform(X_text_test)

    print(f"   TF-IDF shape: {X_tfidf_train.shape}")
    print(f"   Vocabulary size: {len(tfidf.vocabulary_)}")

    # === Scale Numerical Features ===
    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(X_feat_train)
    X_num_test = scaler.transform(X_feat_test)

    print(f"   Numerical features: {X_num_train.shape[1]}")

    # === Combine Features ===
    X_train = hstack([X_tfidf_train, csr_matrix(X_num_train)])
    X_test = hstack([X_tfidf_test, csr_matrix(X_num_test)])

    print(f"   Combined features: {X_train.shape[1]}")

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
    best_rmse = float('inf')

    for name, model in models.items():
        print(f"\n🔄 Training {name}...")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 1.0, 10.0)  # Clip to valid IMDb range

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            'model': model,
            'predictions': y_pred,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

        print(f"   RMSE: {rmse:.4f}  |  MAE: {mae:.4f}  |  R²: {r2:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name

    # === Best Model Summary ===
    print("\n" + "=" * 70)
    print(f"  🏆 BEST MODEL: {best_model_name}")
    print("=" * 70)
    print(f"   RMSE: {results[best_model_name]['RMSE']:.4f} (average prediction error)")
    print(f"   MAE:  {results[best_model_name]['MAE']:.4f}")
    print(f"   R²:   {results[best_model_name]['R2']:.4f}")

    return results, best_model_name, tfidf, scaler, y_test


def detailed_analysis(results, best_model_name, y_test):
    """Provide detailed prediction analysis."""
    print("\n" + "=" * 70)
    print("  PREDICTION ANALYSIS")
    print("=" * 70)

    y_pred = results[best_model_name]['predictions']

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


def save_model(results, best_model_name, tfidf, scaler, decade_encoder):
    """Save trained model and preprocessors."""
    model_package = {
        'model': results[best_model_name]['model'],
        'tfidf': tfidf,
        'scaler': scaler,
        'decade_encoder': decade_encoder,
        'model_name': best_model_name,
        'metrics': {
            'RMSE': results[best_model_name]['RMSE'],
            'MAE': results[best_model_name]['MAE'],
            'R2': results[best_model_name]['R2']
        }
    }

    with open('imdb_model.pkl', 'wb') as f:
        pickle.dump(model_package, f)

    size_mb = os.path.getsize('imdb_model.pkl') / (1024 * 1024)
    print(f"\n💾 Model saved: 'imdb_model.pkl' ({size_mb:.1f} MB)")

