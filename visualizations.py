# ============================================================
# TRAINING VISUALIZATIONS
# ============================================================
"""
Generates graphs for model analysis:
- Rating distribution
- Training vs validation loss (for models that support it)
- Error distribution
- Feature importance (for tree-based models)
- Actual vs predicted scatter plot
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files

# Output directory for saved figures
FIGURES_DIR = 'figures'
FIGURE_DPI = 150
FIGURE_FORMAT = 'png'


def _ensure_figures_dir():
    """Create figures directory if it does not exist."""
    os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_rating_distribution(ratings, output_path=None):
    """
    Plot histogram of IMDb rating distribution.
    
    Args:
        ratings: Array of IMDb ratings
        output_path: Optional path to save the figure
    """
    _ensure_figures_dir()
    path = output_path or os.path.join(FIGURES_DIR, '01_rating_distribution.' + FIGURE_FORMAT)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratings, bins=30, color='#4A90D9', edgecolor='white', alpha=0.85)
    ax.axvline(ratings.mean(), color='#E74C3C', linestyle='--', linewidth=2, label=f'Mean: {ratings.mean():.2f}')
    ax.axvline(np.median(ratings), color='#27AE60', linestyle=':', linewidth=2, label=f'Median: {np.median(ratings):.2f}')
    ax.set_xlabel('IMDb Rating', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Rating Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"   📊 Saved: {path}")
    return path


def plot_training_validation_loss(train_loss, val_loss, model_name='Model', output_path=None):
    """
    Plot training vs validation loss (RMSE) across epochs/iterations.
    
    Args:
        train_loss: List/array of training RMSE per iteration
        val_loss: List/array of validation RMSE per iteration
        model_name: Name of the model for the title
        output_path: Optional path to save the figure
    """
    _ensure_figures_dir()
    path = output_path or os.path.join(FIGURES_DIR, '02_training_validation_loss.' + FIGURE_FORMAT)

    epochs = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, color='#4A90D9', linewidth=2, label='Training RMSE')
    ax.plot(epochs, val_loss, color='#E74C3C', linewidth=2, label='Validation RMSE')
    ax.set_xlabel('Iteration / Epoch', fontsize=11)
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title(f'Training vs Validation Loss ({model_name})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"   📊 Saved: {path}")
    return path


def plot_error_distribution(y_actual, y_predicted, output_path=None):
    """
    Plot distribution of prediction errors (absolute error).
    
    Args:
        y_actual: Array of actual ratings
        y_predicted: Array of predicted ratings
        output_path: Optional path to save the figure
    """
    _ensure_figures_dir()
    path = output_path or os.path.join(FIGURES_DIR, '03_error_distribution.' + FIGURE_FORMAT)

    errors = np.abs(np.array(y_predicted) - np.array(y_actual))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=25, color='#9B59B6', edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(errors), color='#E74C3C', linestyle='--', linewidth=2, label=f'Mean MAE: {np.mean(errors):.3f}')
    ax.axvline(np.median(errors), color='#27AE60', linestyle=':', linewidth=2, label=f'Median: {np.median(errors):.3f}')
    ax.set_xlabel('Absolute Error (|Predicted - Actual|)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"   📊 Saved: {path}")
    return path


def plot_feature_importance(model, feature_names, top_n=20, output_path=None):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to display
        output_path: Optional path to save the figure
    """
    if not hasattr(model, 'feature_importances_'):
        print("   ⚠️  Model does not support feature_importances_ (skipping feature importance plot)")
        return None

    _ensure_figures_dir()
    path = output_path or os.path.join(FIGURES_DIR, '04_feature_importance.' + FIGURE_FORMAT)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    names = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' for i in indices]
    values = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
    bars = ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title('Feature Importance (Top {} Features)'.format(top_n), fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"   📊 Saved: {path}")
    return path


def plot_actual_vs_predicted(y_actual, y_predicted, output_path=None):
    """
    Scatter plot of actual vs predicted ratings.
    
    Args:
        y_actual: Array of actual ratings
        y_predicted: Array of predicted ratings
        output_path: Optional path to save the figure
    """
    _ensure_figures_dir()
    path = output_path or os.path.join(FIGURES_DIR, '05_actual_vs_predicted.' + FIGURE_FORMAT)

    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_actual, y_predicted, alpha=0.5, c='#4A90D9', edgecolors='white', s=40)

    # Perfect prediction line
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

    ax.set_xlabel('Actual Rating', fontsize=11)
    ax.set_ylabel('Predicted Rating', fontsize=11)
    ax.set_title('Actual vs Predicted Ratings', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"   📊 Saved: {path}")
    return path


def generate_all_visualizations(
    ratings,
    y_test,
    y_test_pred,
    best_model,
    best_model_name,
    feature_names,
    train_val_history=None,
):
    """
    Generate all visualization graphs.
    
    Args:
        ratings: Full array of ratings (for distribution)
        y_test: Test set actual ratings
        y_test_pred: Test set predicted ratings
        best_model: Trained best model (for feature importance)
        best_model_name: Name of best model
        feature_names: List of feature names
        train_val_history: Optional dict with 'train_rmse' and 'val_rmse' lists
    """
    print("\n" + "=" * 70)
    print("  📈 GENERATING VISUALIZATIONS")
    print("=" * 70)

    plot_rating_distribution(ratings)
    plot_error_distribution(y_test, y_test_pred)
    plot_actual_vs_predicted(y_test, y_test_pred)

    if train_val_history and 'train_rmse' in train_val_history and 'val_rmse' in train_val_history:
        plot_training_validation_loss(
            train_val_history['train_rmse'],
            train_val_history['val_rmse'],
            model_name=best_model_name,
        )
    else:
        print("   ⚠️  Training/validation loss not available (model does not track eval history)")

    plot_feature_importance(best_model, feature_names)

    print(f"\n   ✅ All figures saved to '{FIGURES_DIR}/'")
