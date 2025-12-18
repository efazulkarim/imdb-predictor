# Compare predictions with ACTUAL ratings from Excel
from predictor import predict_rating
import random
import os
import pandas as pd

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
        # Handle different formats of txt file names
        if not txt_file.endswith('.txt'):
            txt_file = txt_file + '.txt'
        script_to_rating[txt_file] = float(rating)
        script_to_name[txt_file] = name

print(f"Found {len(script_to_rating)} scripts with known ratings\n")

# Get scripts that exist AND have known ratings
scripts_dir = os.listdir('scripts')
valid_scripts = [s for s in scripts_dir if s in script_to_rating]

print(f"Scripts in folder with ratings: {len(valid_scripts)}\n")

# Sample 15 random scripts (like training evaluation)
samples = random.sample(valid_scripts, min(15, len(valid_scripts)))

print("=" * 80)
print("  PREDICTION vs ACTUAL - 15 Random Scripts")
print("=" * 80)
print(f"{'Movie Name':<30} | {'Actual':>8} | {'Predicted':>10} | {'Error':>8}")
print("-" * 80)

errors = []
for script in samples:
    path = f'scripts/{script}'
    predicted = predict_rating(path)
    actual = script_to_rating[script]
    error = predicted - actual
    errors.append(abs(error))
    name = script_to_name.get(script, script)[:28]
    print(f"{name:<30} | {actual:>8.2f} | {predicted:>10.2f} | {error:>+8.2f}")

print("-" * 80)
print(f"\n📊 ERROR METRICS:")
print(f"   Mean Absolute Error (MAE): {sum(errors)/len(errors):.3f}")
print(f"   Max Error: {max(errors):.3f}")
print(f"   Min Error: {min(errors):.3f}")

# Error distribution
within_05 = sum(1 for e in errors if e <= 0.5)
within_10 = sum(1 for e in errors if e <= 1.0)
within_15 = sum(1 for e in errors if e <= 1.5)
print(f"\n📈 ERROR DISTRIBUTION:")
print(f"   Within ±0.5: {within_05}/{len(errors)} ({within_05/len(errors)*100:.1f}%)")
print(f"   Within ±1.0: {within_10}/{len(errors)} ({within_10/len(errors)*100:.1f}%)")
print(f"   Within ±1.5: {within_15}/{len(errors)} ({within_15/len(errors)*100:.1f}%)")
