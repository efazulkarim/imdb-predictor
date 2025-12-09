# ============================================================
# Quick Test Script - Verify Your Model Works
# ============================================================
"""
Simple script to test if your trained model is working correctly.
"""

import os
from predictor import predict_rating, predict_from_text


def test_model():
    """Test the trained model with a sample script."""
    print("=" * 70)
    print("  🧪 Testing IMDb Rating Predictor Model")
    print("=" * 70)
    
    # Check if model exists
    if not os.path.exists('imdb_model.pkl'):
        print("\n❌ ERROR: Model file 'imdb_model.pkl' not found!")
        print("   Please train the model first by running:")
        print("   python imdb_rating_predictor.py")
        return
    
    print("\n✅ Model file found: imdb_model.pkl")
    
    # Test 1: Predict from text
    print("\n" + "-" * 70)
    print("Test 1: Predicting from sample text...")
    print("-" * 70)
    
    sample_script = """
    INT. DARK ROOM - NIGHT
    
    JOHN (whispering)
    We need to get out of here. Now.
    
    MARY
    But how? They're everywhere.
    
    JOHN
    Trust me. I have a plan.
    
    [GUNSHOT in the distance]
    
    MARY
    What was that?
    
    JOHN
    We're running out of time.
    """
    
    try:
        rating = predict_from_text(sample_script)
        print(f"✅ Prediction successful!")
        print(f"   Sample script rating: {rating}/10")
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return
    
    # Test 2: Check if scripts folder exists and has files
    print("\n" + "-" * 70)
    print("Test 2: Checking for script files...")
    print("-" * 70)
    
    scripts_dir = 'scripts/'
    if os.path.exists(scripts_dir):
        txt_files = [f for f in os.listdir(scripts_dir) if f.endswith('.txt')]
        if txt_files:
            print(f"✅ Found {len(txt_files)} script files in '{scripts_dir}'")
            print(f"   Example file: {txt_files[0]}")
            
            # Test 3: Predict from an actual file
            print("\n" + "-" * 70)
            print(f"Test 3: Predicting from actual file...")
            print("-" * 70)
            
            test_file = os.path.join(scripts_dir, txt_files[0])
            try:
                rating = predict_rating(test_file)
                print(f"✅ Prediction successful!")
                print(f"   File: {txt_files[0]}")
                print(f"   Predicted rating: {rating}/10")
            except Exception as e:
                print(f"⚠️  Could not predict from file: {e}")
        else:
            print(f"⚠️  No .txt files found in '{scripts_dir}'")
    else:
        print(f"⚠️  Scripts folder '{scripts_dir}' not found")
    
    # Summary
    print("\n" + "=" * 70)
    print("  ✅ Model Test Complete!")
    print("=" * 70)
    print("\n📖 Next Steps:")
    print("   1. Use predict_rating('path/to/script.txt') for file predictions")
    print("   2. Use predict_from_text(script_text) for text predictions")
    print("   3. See example_usage.py for more examples")
    print("   4. Check README.md for detailed documentation")


if __name__ == "__main__":
    test_model()




