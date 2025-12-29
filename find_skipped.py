# Script to find skipped files
import os
import pandas as pd

df = pd.read_excel('movie_lengths.xlsx')

# Remove duplicates like the loader does
df = df.drop_duplicates(subset=['.txt Files'], keep='first')
df = df.drop_duplicates(subset=['Movie name'], keep='first')

print("=" * 60)
print("FINDING SKIPPED FILES")
print("=" * 60)

invalid_rating = []
too_short = []
missing = []

for idx, row in df.iterrows():
    rating = row.get('IMDb Rating')
    script_file = str(row['.txt Files']).strip() if pd.notna(row['.txt Files']) else ''
    movie_name = row.get('Movie name', 'Unknown')
    
    # Check invalid rating
    try:
        r = float(rating)
        if r < 0 or r > 10:
            invalid_rating.append((movie_name, script_file, rating))
            continue
    except:
        invalid_rating.append((movie_name, script_file, rating))
        continue
    
    # Build path
    if not script_file.endswith('.txt'):
        script_file += '.txt'
    path = f'scripts/{script_file}'
    
    if not os.path.exists(path):
        missing.append((movie_name, script_file))
        continue
    
    size = os.path.getsize(path)
    if size < 1000:
        too_short.append((movie_name, script_file, size))

print(f"\n=== INVALID RATING ({len(invalid_rating)}) ===")
for item in invalid_rating:
    print(f"  {item[0]}: rating={item[2]}")

print(f"\n=== TOO SHORT <1KB ({len(too_short)}) ===")
for item in too_short:
    print(f"  {item[0]}: {item[1]} ({item[2]} bytes)")

print(f"\n=== MISSING FILES ({len(missing)}) ===")
for item in missing[:10]:
    print(f"  {item[0]}: {item[1]}")

print(f"\nTotal skipped: {len(invalid_rating) + len(too_short) + len(missing)}")
