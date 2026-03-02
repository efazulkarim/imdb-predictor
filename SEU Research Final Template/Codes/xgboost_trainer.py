"""
XGBoost Model Training for IMDb Rating Prediction
Complete training pipeline with SBERT embeddings and numerical features.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


def calculate_sample_weights(y):
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
    
    print(f"\\nSample Weights (Class Balancing):")
    for low, high, name in ranges:
        mask = (y >= low) & (y < high)
        count = range_counts[name]
        if count > 0:
            weight = max_count / count
            weights[mask] = weight
            print(f"   {name:12s} ({low}-{high}): n={count:4d}, weight={weight:.2f}x")
    
    return weights


def train_xgboost_model(X_train, y_train, X_val, y_val, sample_weights=None):
    """
    Train XGBoost regressor with early stopping.
    
    Args:
        X_train: Training features (SBERT + numerical)
        y_train: Training ratings
        X_val: Validation features
        y_val: Validation ratings
        sample_weights: Optional sample weights for class balancing
    
    Returns:
        Trained XGBoost model
    """
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    # Fit with or without sample weights
    if sample_weights is not None:
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Returns:
        Dictionary with RMSE, MAE, and R-squared metrics
    """
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 1.0, 10.0)  # Clip to valid rating range
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'predictions': y_pred
    }


def train_pipeline(scripts, ratings, features):
    """
    Complete training pipeline.
    
    Args:
        scripts: List of script texts
        ratings: Array of IMDb ratings
        features: DataFrame of numerical features
    
    Returns:
        Trained model, scaler, and evaluation metrics
    """
    print("=" * 70)
    print("XGBoost Training Pipeline")
    print("=" * 70)
    
    # Split data: 70% train / 15% val / 15% test
    temp_size = 0.30  # 30% for temp (val + test)
    
    indices = np.arange(len(scripts))
    
    # First split: train vs temp
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices, ratings,
        test_size=temp_size,
        random_state=42
    )
    
    # Second split: val vs test (50/50 of temp)
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp,
        test_size=0.5,
        random_state=42
    )
    
    # Get corresponding data
    X_text_train = [scripts[i] for i in idx_train]
    X_text_val = [scripts[i] for i in idx_val]
    X_text_test = [scripts[i] for i in idx_test]
    
    X_feat_train = features.iloc[idx_train]
    X_feat_val = features.iloc[idx_val]
    X_feat_test = features.iloc[idx_test]
    
    print(f"\\nData Split:")
    print(f"   Training:   {len(X_text_train)} samples (70%)")
    print(f"   Validation: {len(X_text_val)} samples (15%)")
    print(f"   Testing:    {len(X_text_test)} samples (15%)")
    
    # Compute sample weights
    sample_weights = calculate_sample_weights(y_train)
    
    # Generate SBERT embeddings (placeholder - requires SBERT model)
    print("\\nGenerating SBERT embeddings...")
    # In practice: X_sbert_train = generate_embeddings(X_text_train)
    # For now, create dummy embeddings
    X_sbert_train = np.random.rand(len(X_text_train), 384)
    X_sbert_val = np.random.rand(len(X_text_val), 384)
    X_sbert_test = np.random.rand(len(X_text_test), 384)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(X_feat_train)
    X_num_val = scaler.transform(X_feat_val)
    X_num_test = scaler.transform(X_feat_test)
    
    # Combine features
    X_train = np.hstack([X_sbert_train, X_num_train])
    X_val = np.hstack([X_sbert_val, X_num_val])
    X_test = np.hstack([X_sbert_test, X_num_test])
    
    print(f"   Combined features: {X_train.shape[1]} (SBERT: 384 + Numerical: {X_num_train.shape[1]})")
    
    # Train XGBoost
    print("\\nTraining XGBoost model...")
    model = train_xgboost_model(X_train, y_train, X_val, y_val, sample_weights)
    
    # Evaluate
    print("\\nEvaluating on validation set...")
    val_metrics = evaluate_model(model, X_val, y_val)
    print(f"   Validation: RMSE={val_metrics['RMSE']:.4f} | MAE={val_metrics['MAE']:.4f} | R2={val_metrics['R2']:.4f}")
    
    print("\\nEvaluating on test set...")
    test_metrics = evaluate_model(model, X_test, y_test)
    print(f"   Test:       RMSE={test_metrics['RMSE']:.4f} | MAE={test_metrics['MAE']:.4f} | R2={test_metrics['R2']:.4f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }


# Example usage
if __name__ == "__main__":
    # This is a placeholder - requires actual data
    print("This script requires:")
    print("  1. Movie scripts (list of strings)")
    print("  2. IMDb ratings (numpy array)")
    print("  3. Numerical features (pandas DataFrame)")
    print("\\nIn production, use trainer.py which includes:")
    print("  - SBERT model loading")
    print("  - Chunking for long scripts")
    print("  - Complete data loading from Excel")
