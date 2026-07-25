# ============================================================
# EXPLAINABLE AI (XAI) MODULE USING SHAP
# ============================================================
"""
Provides model interpretability using SHapley Additive exPlanations (SHAP).
Calculates exact game-theoretic feature attributions for predictions made by
the SBERT + XGBoost model.

Outputs:
  - figures/08_shap_summary.png      (Global feature importance & direction)
  - figures/09_shap_dependence.png   (Dependence plot for key features like year/movie_length)
  - figures/10_shap_local.png        (Local waterfall plot for example movie predictions)
  - results/shap_analysis.json       (Numerical summary of SHAP values)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import SBERT_MODEL_NAME, RANDOM_STATE
from data_loader import load_dataset
from experiments import _get_or_build_embeddings_multi, _ensure_dir, RESULTS_DIR


FIG_DIR = 'figures'


def run_shap_analysis(sample_size: int = 200):
    """
    Run SHAP explanation on the SBERT+XGBoost pipeline and save visual/data artifacts.
    """
    _ensure_dir(FIG_DIR)
    _ensure_dir(RESULTS_DIR)

    print("=" * 70)
    print("  EXPLAINABLE AI (SHAP Analysis)")
    print("=" * 70)

    # 1. Load dataset & SBERT embeddings
    (scripts_text, ratings, features_df, movie_names, script_files,
     _, scripts_text_sbert) = load_dataset()

    embeddings = _get_or_build_embeddings_multi(
        scripts_text_sbert, SBERT_MODEL_NAME, poolings=['mean']
    )['mean']

    # Impute missing values in structural features & scale
    from sklearn.preprocessing import StandardScaler
    median = features_df.median(numeric_only=True)
    features_clean = features_df.fillna(median)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_clean)

    # Combined feature matrix: SBERT (384) + Structural (19)
    X = np.hstack([embeddings, features_scaled])
    feature_names = [f'sbert_dim_{i}' for i in range(embeddings.shape[1])] + list(features_clean.columns)

    # Train XGBoost regressor on full dataset or train split
    from xgboost import XGBRegressor
    print(f"\n⚡ Training XGBoost regressor for SHAP evaluation (n={len(X)})...")
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method='hist'
    )
    model.fit(X, ratings)

    # Compute SHAP values
    print(f"📊 Computing SHAP values on {sample_size} representative samples...")
    np.random.seed(RANDOM_STATE)
    sample_idx = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_sample = X[sample_idx]

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        has_shap_pkg = True
    except ImportError:
        print("   [Note] SHAP package not installed. Using XGBoost built-in gain attribution fallback.")
        has_shap_pkg = False
        shap_values = None

    if has_shap_pkg:
        # 1. Global Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, max_display=15, show=False)
        plt.title("SHAP Global Feature Importance & Contribution Direction", fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, '08_shap_summary.png'), dpi=300)
        plt.close()
        print(f"   Saved {FIG_DIR}/08_shap_summary.png")

        # 2. Key Dependence Plot (Year & Movie Length)
        year_idx = feature_names.index('year') if 'year' in feature_names else -1
        length_idx = feature_names.index('movie_length') if 'movie_length' in feature_names else -1

        if year_idx != -1:
            plt.figure(figsize=(8, 5))
            shap.dependence_plot(year_idx, shap_values, X_sample, feature_names=feature_names, show=False)
            plt.title("SHAP Dependence Plot: Release Year Effect on Rating Prediction", fontsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, '09_shap_dependence.png'), dpi=300)
            plt.close()
            print(f"   Saved {FIG_DIR}/09_shap_dependence.png")

        # 3. Local Waterfall / Explanation for a specific sample movie
        plt.figure(figsize=(10, 5))
        sample_movie_name = movie_names[sample_idx[0]] if movie_names else "Sample Movie"
        actual_r = ratings[sample_idx[0]]
        pred_r = model.predict(X_sample[0:1])[0]

        # Summary bar of top 10 contributing features for sample #0
        abs_shap = np.abs(shap_values[0])
        top_k_idx = np.argsort(abs_shap)[-10:]
        top_k_names = [feature_names[i] for i in top_k_idx]
        top_k_vals = [shap_values[0][i] for i in top_k_idx]

        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in top_k_vals]
        plt.barh(top_k_names, top_k_vals, color=colors)
        plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
        plt.xlabel("SHAP Value (Impact on Prediction)", fontsize=10)
        plt.title(f"Local Feature Attributions for '{sample_movie_name}'\nActual: {actual_r:.1f} | Predicted: {pred_r:.2f}", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, '10_shap_local.png'), dpi=300)
        plt.close()
        print(f"   Saved {FIG_DIR}/10_shap_local.png")

        # Export numerical summary to JSON
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:20]
        summary_data = {
            'top_features': [
                {
                    'feature': feature_names[i],
                    'mean_abs_shap': float(mean_abs_shap[i]),
                    'mean_shap': float(np.mean(shap_values[:, i]))
                }
                for i in top_indices
            ]
        }
        with open(os.path.join(RESULTS_DIR, 'shap_analysis.json'), 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"   Saved {RESULTS_DIR}/shap_analysis.json")
    else:
        # Fallback plot if shap package is building/installing
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]
        plt.figure(figsize=(10, 6))
        plt.barh([feature_names[i] for i in top_idx][::-1], importances[top_idx][::-1], color='#3498db')
        plt.xlabel("Feature Importance (Gain)")
        plt.title("XGBoost Feature Importance (XAI Fallback)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, '08_shap_summary.png'), dpi=300)
        plt.close()

    print("\n✅ SHAP XAI Analysis completed successfully!")


if __name__ == '__main__':
    run_shap_analysis()
