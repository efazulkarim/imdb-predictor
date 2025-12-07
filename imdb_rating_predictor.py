# IMDb RATING PREDICTOR - OPTIMIZED FOR MOVIE SCRIPTS
# Version: 2.0 (Production Ready)
# Dataset: 5000 Movie Scripts + Excel Metadata
# Split: 70% Training / 30% Testing
# ============================================================

import os
import warnings
warnings.filterwarnings('ignore')

# Import from modules
from config import EXCEL_FILE, SCRIPTS_DIR
from data_loader import load_dataset
from trainer import train_and_evaluate, detailed_analysis, save_model

# Also export prediction functions for easy access
from predictor import predict_rating, predict_from_text, batch_predict

# Make prediction functions available at package level
__all__ = ['predict_rating', 'predict_from_text', 'batch_predict']


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 70)
    print("  🎬 IMDb RATING PREDICTOR - TRAINING PIPELINE")
    print("  📁 Dataset: Movie Scripts + Excel Metadata")
    print("=" * 70)

    # Validate paths
    if not os.path.exists(EXCEL_FILE):
        print(f"\n❌ ERROR: Excel file not found: '{EXCEL_FILE}'")
        return

    if not os.path.exists(SCRIPTS_DIR):
        print(f"\n❌ ERROR: Scripts folder not found: '{SCRIPTS_DIR}'")
        return

    # Load data
    scripts_text, ratings, features_df, movie_names, decade_encoder = load_dataset()

    if len(scripts_text) < 100:
        print("\n⚠️  WARNING: Less than 100 scripts loaded!")
        print("   Results may be unreliable. Check file paths.")
        response = input("   Continue? (y/n): ")
        if response.lower() != 'y':
            return

    # Train models
    results, best_model_name, tfidf, scaler, y_test = train_and_evaluate(
        scripts_text, ratings, features_df
    )

    # Detailed analysis
    detailed_analysis(results, best_model_name, y_test)

    # Save model
    save_model(results, best_model_name, tfidf, scaler, decade_encoder)

    # Usage instructions
    print("\n" + "=" * 70)
    print("  📖 HOW TO USE THE TRAINED MODEL")
    print("=" * 70)
    print("""
    # In Python:
    from predictor import predict_rating, predict_from_text

    # Predict from file
    rating = predict_rating('new_script.txt')
    print(f"Predicted: {rating}/10")

    # Predict from text
    script = "JOHN: Hello!\\nMARY: Hi there!"
    rating = predict_from_text(script)
    """)

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
