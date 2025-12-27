# Large-scale test: Compare predictions with ACTUAL ratings
# Uses the SAME test set as training evaluation for consistency
# Now also uses actual metadata (year, decade, movie_length) for accurate predictions
import json
import os
import pandas as pd
import numpy as np
from predictor import predict_rating
from sklearn.preprocessing import LabelEncoder

print("Loading data...")

# Check if test_set_info.json exists (created during training)
TEST_SET_FILE = 'test_set_info.json'

if os.path.exists(TEST_SET_FILE):
    print(f"[OK] Found saved test set info: {TEST_SET_FILE}")
    with open(TEST_SET_FILE, 'r', encoding='utf-8') as f:
        test_set_info = json.load(f)
    
    # Check if script files are saved (new format)
    if 'test_script_files' in test_set_info and test_set_info['test_script_files']:
        # Use script filenames directly (most reliable)
        test_script_files = test_set_info['test_script_files']
        test_movie_names = test_set_info['test_movie_names']
        test_ratings = test_set_info['test_ratings']
        
        print(f"   Test set size: {len(test_script_files)} movies")
        print(f"   Random state used: {test_set_info['random_state']}")
        print(f"   Test split: {test_set_info['test_size']*100:.0f}%")
        print(f"   Using script filenames for precise matching")
        
        USE_SCRIPT_FILES = True
        USE_SAVED_TEST_SET = True
    else:
        print(f"   [WARNING] Old format - run training again to get script filenames")
        USE_SCRIPT_FILES = False
        USE_SAVED_TEST_SET = False
else:
    print(f"[WARNING] No saved test set found: {TEST_SET_FILE}")
    print("   Run 'python imdb_rating_predictor.py' first to generate it.")
    USE_SAVED_TEST_SET = False
    USE_SCRIPT_FILES = False

if not USE_SAVED_TEST_SET:
    print("Cannot proceed without test set info. Exiting.")
    exit(1)

# Load Excel to get metadata (year, decade, movie_length) for each script
print("\nLoading metadata from Excel...")
df = pd.read_excel('movie_lengths.xlsx')

# Create mappings from script filename to metadata
script_to_year = {}
script_to_decade = {}
script_to_length = {}
script_to_name = {}

for _, row in df.iterrows():
    txt_file = str(row['.txt Files']).strip() if pd.notna(row['.txt Files']) else None
    if txt_file:
        if not txt_file.endswith('.txt'):
            txt_file = txt_file + '.txt'
        
        # Year
        year = int(row['Year']) if pd.notna(row.get('Year')) else 2000
        script_to_year[txt_file] = year
        
        # Decade (as string for encoding)
        decade = str(row.get('Decade', '2000s')) if pd.notna(row.get('Decade')) else '2000s'
        script_to_decade[txt_file] = decade
        
        # Movie length
        try:
            length = float(row.get('Movie length')) if pd.notna(row.get('Movie length')) else 120
            if length <= 0:
                length = 120
        except (ValueError, TypeError):
            length = 120
        script_to_length[txt_file] = length
        
        # Movie name
        name = str(row['Movie name']).strip() if pd.notna(row.get('Movie name')) else 'Unknown'
        script_to_name[txt_file] = name

# Fit decade encoder on all decades to match training
all_decades = list(script_to_decade.values())
decade_encoder = LabelEncoder()
decade_encoder.fit(all_decades)

# Create decade encoding mapping
script_to_decade_encoded = {}
for script, decade in script_to_decade.items():
    try:
        script_to_decade_encoded[script] = int(decade_encoder.transform([decade])[0])
    except ValueError:
        script_to_decade_encoded[script] = 0  # Default if unknown decade

print(f"   Loaded metadata for {len(script_to_year)} scripts")

# Verify scripts exist
valid_count = sum(1 for s in test_script_files if os.path.exists(f'scripts/{s}'))
print(f"\nVerified {valid_count} out of {len(test_script_files)} test scripts exist")

print(f"\nTesting {len(test_script_files)} scripts... This may take a few minutes.\n")
print("=" * 70)
print("  LARGE-SCALE PREDICTION TEST")
print("  (Using SAME test set AND metadata as training evaluation)")
print("=" * 70)

# Process all scripts
predictions = []
actuals = []
errors = []
movie_names_tested = []
skipped = 0

for i, script in enumerate(test_script_files):
    if (i + 1) % 100 == 0:
        print(f"  Progress: {i + 1}/{len(test_script_files)} scripts processed...")
    
    path = f'scripts/{script}'
    
    # Skip if file doesn't exist
    if not os.path.exists(path):
        skipped += 1
        continue
    
    # Get actual metadata for this script
    year = script_to_year.get(script, 2020)
    decade_encoded = script_to_decade_encoded.get(script, 0)
    movie_length = script_to_length.get(script, 120)
        
    try:
        # Pass actual metadata to predictor
        predicted = predict_rating(
            path, 
            year=year, 
            decade_encoded=decade_encoded, 
            movie_length=movie_length
        )
        actual = test_ratings[i]
        error = abs(predicted - actual)
        
        predictions.append(predicted)
        actuals.append(actual)
        errors.append(error)
        movie_names_tested.append(test_movie_names[i] if i < len(test_movie_names) else 'Unknown')
    except Exception as e:
        print(f"  Error with {script}: {e}")
        skipped += 1

print(f"\n  Completed: {len(predictions)} successful predictions")
if skipped > 0:
    print(f"  Skipped: {skipped} scripts (missing or errors)")

# Calculate metrics
mae = np.mean(errors)
rmse = np.sqrt(np.mean(np.array(errors)**2))
r2 = 1 - (np.sum((np.array(actuals) - np.array(predictions))**2) / 
          np.sum((np.array(actuals) - np.mean(actuals))**2))

print("\n" + "=" * 70)
print("  RESULTS SUMMARY")
print("=" * 70)

print(f"\n  Total Scripts Tested: {len(predictions)}")
print(f"\n  METRICS:")
print(f"    Mean Absolute Error (MAE): {mae:.4f}")
print(f"    Root Mean Square Error (RMSE): {rmse:.4f}")
print(f"    R-squared (R2): {r2:.4f}")

# Error distribution
print(f"\n  ERROR DISTRIBUTION:")
brackets = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
for threshold in brackets:
    count = sum(1 for e in errors if e <= threshold)
    pct = count / len(errors) * 100
    bar = '#' * int(pct / 2)
    print(f"    Within +/-{threshold}: {count:5d} ({pct:5.1f}%) {bar}")

# Performance by rating range
print(f"\n  PERFORMANCE BY RATING RANGE:")
ranges = [(1, 4, 'Low (1-4)'), (4, 6, 'Medium (4-6)'), 
          (6, 8, 'Good (6-8)'), (8, 10, 'Excellent (8-10)')]

for low, high, label in ranges:
    mask = [(a >= low and a < high) for a in actuals]
    range_errors = [e for e, m in zip(errors, mask) if m]
    if range_errors:
        range_mae = np.mean(range_errors)
        count = len(range_errors)
        print(f"    {label:20s}: MAE = {range_mae:.3f} (n={count})")

# Show some examples
print(f"\n  SAMPLE PREDICTIONS (first 20):")
print("-" * 70)
print(f"    {'Movie':<30} | {'Actual':>8} | {'Predicted':>10} | {'Error':>8}")
print("-" * 70)
for i in range(min(20, len(predictions))):
    name = movie_names_tested[i][:28] if i < len(movie_names_tested) else 'Unknown'
    print(f"    {name:<30} | {actuals[i]:>8.2f} | {predictions[i]:>10.2f} | {errors[i]:>+8.2f}")

print("\n" + "=" * 70)
print("  TEST COMPLETE")
print("  (Results should now MATCH training evaluation)")
print("=" * 70)
