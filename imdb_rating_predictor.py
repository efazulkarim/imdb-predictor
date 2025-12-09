# IMDb RATING PREDICTOR - OPTIMIZED FOR MOVIE SCRIPTS
# Version: 2.0 (Production Ready)
# Dataset: 5000 Movie Scripts + Excel Metadata
# Split: 70% Training / 30% Testing
# ============================================================

import os
import sys
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
    scripts_text, ratings, features_df, movie_names, decade_encoder = load_dataset()

    # Show data loading summary
    print(f"\n✅ Data Loading Complete!")
    print(f"   Successfully loaded: {len(scripts_text)} scripts")
    print(f"   Rating range: {ratings.min():.2f} - {ratings.max():.2f}")
    print(f"   Mean rating: {ratings.mean():.2f}")

    # Check if user wants to skip training (just test data loading)
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['--test', '--data-only', '-t']:
        print("\n✅ Data loading test complete! (Training skipped)")
        return

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
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['--help', '-h', '/?']:
        print("""
IMDb Rating Predictor - Usage:

  python imdb_rating_predictor.py          # Full pipeline: Load data + Train model
  python imdb_rating_predictor.py --test    # Test data loading only (skip training)
  python imdb_rating_predictor.py --help    # Show this help message

Or simply double-click: run.bat (Windows)
""")
        sys.exit(0)
    
    main()
