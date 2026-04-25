# ============================================================
# BASELINE MODELS
# ============================================================
"""
Baseline models for the comparison table.

Each baseline returns predictions on the test set so we can compute
metrics + bootstrap CIs in a unified way (see stats_utils.py).

Baselines included:
    1. predict_mean           - constant predictor (training mean)
    2. ols_metadata           - linear regression on [year, movie_length, decade_encoded]
    3. ols_structural         - linear regression on the 19 structural features
    4. tfidf_xgboost          - TF-IDF + XGBoost (lexical text baseline)

These isolate different signal sources so that the marginal contribution
of SBERT in the main system can be quantified.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor


METADATA_COLS = ['year', 'movie_length', 'decade_encoded']


# ------------------------------------------------------------
# 1. Predict-the-mean
# ------------------------------------------------------------
def predict_mean(y_train, n_test):
    """Constant predictor: training mean for every test sample."""
    mean_val = float(np.mean(y_train))
    return np.full(n_test, mean_val)


# ------------------------------------------------------------
# 2. OLS on metadata only
# ------------------------------------------------------------
def ols_metadata(features_train, features_test, y_train):
    """
    Linear regression on year + movie_length + decade_encoded.

    movie_length can contain NaN; impute with the training median.
    """
    X_train = features_train[METADATA_COLS].copy()
    X_test = features_test[METADATA_COLS].copy()

    median = X_train.median(numeric_only=True)
    X_train = X_train.fillna(median)
    X_test = X_test.fillna(median)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    return np.clip(preds, 1.0, 10.0)


# ------------------------------------------------------------
# 3. OLS on the 19 structural features
# ------------------------------------------------------------
def ols_structural(features_train, features_test, y_train):
    """Linear regression on all numerical features the main system uses."""
    median = features_train.median(numeric_only=True)
    X_train = features_train.fillna(median)
    X_test = features_test.fillna(median)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    return np.clip(preds, 1.0, 10.0)


# ------------------------------------------------------------
# 4. TF-IDF + XGBoost
# ------------------------------------------------------------
def tfidf_xgboost(
    texts_train,
    texts_test,
    y_train,
    max_features=8000,
    sample_weight=None,
    random_state=42,
):
    """
    Classical lexical baseline:
        TF-IDF (1-2 grams, top 8000 features) + XGBoost regressor.

    No metadata, no structural features - pure text-from-bag-of-words.
    Used to isolate the marginal benefit of SBERT semantic embeddings.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(texts_train)
    X_test = vectorizer.transform(texts_test)

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
        tree_method='hist',
    )

    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    preds = model.predict(X_test)
    return np.clip(preds, 1.0, 10.0)


# ------------------------------------------------------------
# Convenience: run all four baselines and return a dict of predictions
# ------------------------------------------------------------
def run_all_baselines(
    texts_train,
    texts_test,
    features_train,
    features_test,
    y_train,
    sample_weight=None,
    random_state=42,
):
    """
    Run every baseline and return a dict {name: y_test_predictions}.

    Args:
        texts_train, texts_test: lists of cleaned script texts
        features_train, features_test: DataFrames with the 19 structural features
        y_train: 1-D array of training ratings
        sample_weight: optional weights for TF-IDF baseline (others ignore)
        random_state: seed for stochastic baselines

    Returns:
        dict mapping baseline name -> np.ndarray of test predictions
    """
    n_test = len(texts_test)
    preds = {}

    print("   Running baseline: predict_mean ...")
    preds['predict_mean'] = predict_mean(y_train, n_test)

    print("   Running baseline: ols_metadata ...")
    preds['ols_metadata'] = ols_metadata(features_train, features_test, y_train)

    print("   Running baseline: ols_structural ...")
    preds['ols_structural'] = ols_structural(features_train, features_test, y_train)

    print("   Running baseline: tfidf_xgboost ...")
    preds['tfidf_xgboost'] = tfidf_xgboost(
        texts_train, texts_test, y_train,
        sample_weight=sample_weight,
        random_state=random_state,
    )

    return preds
