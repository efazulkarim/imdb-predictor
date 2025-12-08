# ============================================================
# EXAMPLE: How to Use Your Trained IMDb Rating Predictor
# ============================================================
"""
This script demonstrates how to use your trained model to predict
IMDb ratings for new movie scripts.
"""

from predictor import predict_rating, predict_from_text, batch_predict


# ============================================================
# Example 1: Predict from a script file
# ============================================================
def example_predict_from_file():
    """Predict rating from a script file."""
    print("=" * 70)
    print("Example 1: Predict from Script File")
    print("=" * 70)
    
    # Replace with path to your script file
    script_path = 'scripts/your_script.txt'
    
    try:
        rating = predict_rating(script_path)
        print(f"\n📊 Predicted IMDb Rating: {rating}/10")
        print(f"   Script: {script_path}")
    except FileNotFoundError:
        print(f"\n❌ Script file not found: {script_path}")
        print("   Please update the script_path variable with a valid file.")
    except Exception as e:
        print(f"\n❌ Error: {e}")


# ============================================================
# Example 2: Predict from raw text
# ============================================================
def example_predict_from_text():
    """Predict rating from raw script text."""
    print("\n" + "=" * 70)
    print("Example 2: Predict from Raw Text")
    print("=" * 70)
    
    # Example script text
    sample_script = """
    JOHN: I can't believe this is happening.
    MARY: We need to find a way out of here.
    JOHN: There's no time. They're coming.
    [GUNSHOT]
    MARY: What was that?
    JOHN: We need to move. Now!
    """
    
    rating = predict_from_text(sample_script)
    print(f"\n📊 Predicted IMDb Rating: {rating}/10")
    print(f"\nSample Script:")
    print(sample_script)


# ============================================================
# Example 3: Batch prediction for multiple scripts
# ============================================================
def example_batch_predict():
    """Predict ratings for multiple scripts at once."""
    print("\n" + "=" * 70)
    print("Example 3: Batch Prediction")
    print("=" * 70)
    
    # List of script file paths
    script_paths = [
        'scripts/script1.txt',
        'scripts/script2.txt',
        'scripts/script3.txt',
    ]
    
    try:
        ratings = batch_predict(script_paths)
        
        print(f"\n📊 Batch Prediction Results:")
        print("-" * 50)
        for path, rating in zip(script_paths, ratings):
            print(f"   {path:30s} → {rating:5.2f}/10")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Please update script_paths with valid file paths.")


# ============================================================
# Example 4: Analyze a specific script
# ============================================================
def analyze_script(script_path):
    """Analyze a script and provide detailed prediction."""
    print("\n" + "=" * 70)
    print(f"Analyzing: {script_path}")
    print("=" * 70)
    
    try:
        rating = predict_rating(script_path)
        
        # Rating interpretation
        if rating >= 8.0:
            quality = "Excellent"
            emoji = "⭐⭐⭐⭐⭐"
        elif rating >= 6.5:
            quality = "Good"
            emoji = "⭐⭐⭐⭐"
        elif rating >= 5.0:
            quality = "Average"
            emoji = "⭐⭐⭐"
        elif rating >= 3.5:
            quality = "Below Average"
            emoji = "⭐⭐"
        else:
            quality = "Poor"
            emoji = "⭐"
        
        print(f"\n📊 Prediction Results:")
        print(f"   Rating: {rating}/10 {emoji}")
        print(f"   Quality: {quality}")
        print(f"\n💡 Interpretation:")
        if rating >= 8.0:
            print("   This script shows strong potential for critical acclaim.")
        elif rating >= 6.5:
            print("   This script has good commercial potential.")
        elif rating >= 5.0:
            print("   This script may need some improvements.")
        else:
            print("   This script likely needs significant revision.")
            
    except FileNotFoundError:
        print(f"\n❌ Script file not found: {script_path}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  🎬 IMDb Rating Predictor - Usage Examples")
    print("=" * 70)
    
    # Run examples
    # Uncomment the examples you want to try:
    
    # example_predict_from_file()
    # example_predict_from_text()
    # example_batch_predict()
    
    # Or analyze a specific script:
    # analyze_script('scripts/your_script.txt')
    
    print("\n" + "=" * 70)
    print("  📖 Quick Usage Guide")
    print("=" * 70)
    print("""
    # In Python or Jupyter Notebook:
    
    from predictor import predict_rating, predict_from_text
    
    # Predict from file
    rating = predict_rating('path/to/script.txt')
    print(f"Rating: {rating}/10")
    
    # Predict from text
    script = "JOHN: Hello!\\nMARY: Hi there!"
    rating = predict_from_text(script)
    
    # Batch predict
    from predictor import batch_predict
    ratings = batch_predict(['script1.txt', 'script2.txt'])
    """)
    
    print("\n✅ Examples loaded! Uncomment the functions above to run them.")


