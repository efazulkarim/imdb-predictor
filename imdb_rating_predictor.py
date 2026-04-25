# IMDb RATING PREDICTOR - OPTIMIZED FOR MOVIE SCRIPTS
# Version: 4.0 (SBERT + LightGBM/XGBoost)
# Dataset: 5000 Movie Scripts + Excel Metadata
# Split: 70% Training / 15% Validation / 15% Test
# ============================================================

import os
import warnings
warnings.filterwarnings('ignore')

# Import from modules
import config
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
    # Get configuration values safely
    EXCEL_FILE = getattr(config, 'EXCEL_FILE', 'isteaq ulab info.xlsx')
    SCRIPTS_DIR = getattr(config, 'SCRIPTS_DIR', 'scripts/')
    
    # Check for multiple Excel files or single file
    excel_files_to_check = []
    try:
        # Get EXCEL_FILES safely (may not be defined)
        EXCEL_FILES = getattr(config, 'EXCEL_FILES', None)
        
        # Normalize EXCEL_FILES to always be a list
        if EXCEL_FILES:
            if isinstance(EXCEL_FILES, str):
                # If it's a string, convert to list
                excel_files_to_check = [EXCEL_FILES]
            elif isinstance(EXCEL_FILES, list) and len(EXCEL_FILES) > 0:
                # If it's a non-empty list, use it
                excel_files_to_check = EXCEL_FILES
            else:
                # Empty list or other type, fall back to single file
                excel_files_to_check = [EXCEL_FILE]
        else:
            # EXCEL_FILES is None/False, fall back to single file
            excel_files_to_check = [EXCEL_FILE]
    except Exception:
        # Any error, fall back to single file
        excel_files_to_check = [EXCEL_FILE]
    
    # Check if at least one Excel file exists
    found_files = [f for f in excel_files_to_check if os.path.exists(f)]
    if not found_files:
        print(f"\n❌ ERROR: No Excel files found!")
        print(f"   Checked: {excel_files_to_check}")
        print(f"\n   💡 Tip: Update EXCEL_FILE or EXCEL_FILES in config.py")
        return
    
    if not os.path.exists(SCRIPTS_DIR):
        print(f"\n❌ ERROR: Scripts folder not found: '{SCRIPTS_DIR}'")
        return

    # Load data
    scripts_text, ratings, features_df, movie_names, script_files, decade_encoder, scripts_text_sbert = load_dataset()

    if len(scripts_text) < 100:
        print("\n⚠️  WARNING: Less than 100 scripts loaded!")
        print("   Results may be unreliable. Check file paths.")
        response = input("   Continue? (y/n): ")
        if response.lower() != 'y':
            return

    # Train models (pass movie_names and script_files to save test set info)
    results, best_model_name, sbert_model, scaler, y_test, y_val = train_and_evaluate(
        scripts_text, ratings, features_df, movie_names, script_files,
        scripts_text_sbert=scripts_text_sbert,
    )

    # Detailed analysis
    detailed_analysis(results, best_model_name, y_test)

    # Save model (pass SBERT model name for loading at inference)
    from config import SBERT_MODEL_NAME
    save_model(results, best_model_name, SBERT_MODEL_NAME, scaler, decade_encoder)

    # Usage instructions
    print("\n" + "=" * 70)
    print("  📖 HOW TO USE THE TRAINED MODEL")
    print("=" * 70)
    print("""
    # In Python:
    from predictor import predict_rating; print(f'Predicted Rating: {predict_rating(\"scripts/file10074.txt\")}/10')

    # Predict from text
    script = "JOHN: Hello!\\nMARY: Hi there!"
    rating = predict_from_text(script)
    """)

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
