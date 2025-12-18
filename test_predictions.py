# Large-scale test: Compare predictions with ACTUAL ratings
# Testing with 1300 random script files
from predictor import predict_rating
import random
import os
import pandas as pd
import numpy as np
from collections import defaultdict

print("Loading data...")

# Load Excel with actual ratings
df = pd.read_excel('movie_lengths.xlsx')

# Create mapping: .txt filename -> actual rating
script_to_rating = {}
script_to_name = {}
for _, row in df.iterrows():
    txt_file = str(row['.txt Files']).strip() if pd.notna(row['.txt Files']) else None
    rating = row['IMDb Rating'] if pd.notna(row['IMDb Rating']) else None
    name = row['Movie name'] if pd.notna(row['Movie name']) else 'Unknown'
    
    if txt_file and rating:
        if not txt_file.endswith('.txt'):
            txt_file = txt_file + '.txt'
        script_to_rating[txt_file] = float(rating)
        script_to_name[txt_file] = name

print(f"Found {len(script_to_rating)} scripts with known ratings")

# Get scripts that exist AND have known ratings
scripts_dir = os.listdir('scripts')
valid_scripts = [s for s in scripts_dir if s in script_to_rating]
print(f"Scripts in folder with ratings: {len(valid_scripts)}")

# Sample 1300 random scripts
NUM_SAMPLES = 1300
samples = random.sample(valid_scripts, min(NUM_SAMPLES, len(valid_scripts)))

print(f"\nTesting {len(samples)} scripts... This may take a few minutes.\n")
print("=" * 70)
print("  LARGE-SCALE PREDICTION TEST")
print("=" * 70)

# Process all scripts
predictions = []
actuals = []
errors = []

for i, script in enumerate(samples):
    if (i + 1) % 100 == 0:
        print(f"  Progress: {i + 1}/{len(samples)} scripts processed...")
    
    path = f'scripts/{script}'
    try:
        predicted = predict_rating(path)
        actual = script_to_rating[script]
        error = abs(predicted - actual)
        
        predictions.append(predicted)
        actuals.append(actual)
        errors.append(error)
    except Exception as e:
        print(f"  Error with {script}: {e}")

print(f"\n  Completed: {len(predictions)} successful predictions")

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
for i in range(min(20, len(samples))):
    name = script_to_name.get(samples[i], samples[i])[:28]
    print(f"    {name:<30} | {actuals[i]:>8.2f} | {predictions[i]:>10.2f} | {errors[i]:>+8.2f}")

print("\n" + "=" * 70)
print("  TEST COMPLETE")
print("=" * 70)
