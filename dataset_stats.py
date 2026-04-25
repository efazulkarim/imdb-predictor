"""
One-shot script: pull descriptive statistics from the dataset for the
paper's Dataset section. Reads movie_lengths.xlsx + the scripts/ folder.

Outputs to stdout (and writes a JSON snapshot for reproducibility).
"""

import os
import json
import numpy as np
import pandas as pd

EXCEL = 'movie_lengths.xlsx'
SCRIPTS_DIR = 'scripts/'

df = pd.read_excel(EXCEL)
df.columns = df.columns.str.strip()
print(f"Excel rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

ratings = pd.to_numeric(df['IMDb Rating'], errors='coerce').dropna()
years = pd.to_numeric(df['Year'], errors='coerce').dropna()
lengths = pd.to_numeric(df['Movie length'], errors='coerce').dropna()

print(f"\n--- Ratings (n={len(ratings)})")
print(f"  range:  {ratings.min():.1f} – {ratings.max():.1f}")
print(f"  mean:   {ratings.mean():.3f}")
print(f"  median: {ratings.median():.3f}")
print(f"  std:    {ratings.std():.3f}")
print(f"  IQR:    {ratings.quantile(0.25):.2f} – {ratings.quantile(0.75):.2f}")

# Bucket counts
buckets = [(1, 4, 'Low'), (4, 6, 'Medium'), (6, 8, 'Good'), (8, 10, 'Excellent')]
print(f"\n--- Bucketed rating counts")
for lo, hi, name in buckets:
    n = ((ratings >= lo) & (ratings < hi)).sum()
    print(f"  {name:10s} [{lo}, {hi}): {n:5d}  ({n / len(ratings) * 100:5.2f}%)")

# Histogram-style check for bimodality
print(f"\n--- Rating histogram (bin width 0.5)")
for lo in np.arange(1, 10, 0.5):
    n = ((ratings >= lo) & (ratings < lo + 0.5)).sum()
    bar = '#' * int(n / 30)
    print(f"  [{lo:.1f}, {lo + 0.5:.1f}): {n:4d}  {bar}")

print(f"\n--- Years (n={len(years)})")
print(f"  range: {int(years.min())} – {int(years.max())}")
print(f"  mean:  {years.mean():.1f}")
print(f"  median: {int(years.median())}")
print(f"\nYear distribution by decade:")
decades = (years // 10 * 10).astype(int).value_counts().sort_index()
for dec, n in decades.items():
    print(f"  {dec}s: {n:5d}")

print(f"\n--- Movie length, minutes (n={len(lengths)})")
print(f"  range: {lengths.min():.0f} – {lengths.max():.0f}")
print(f"  mean:  {lengths.mean():.1f}")
print(f"  median: {int(lengths.median())}")

# Script availability
script_files = set(os.listdir(SCRIPTS_DIR))
print(f"\n--- Script files on disk: {len(script_files)}")

# Sample script lengths (chars)
sample = list(script_files)[:200]
sizes = []
for f in sample:
    p = os.path.join(SCRIPTS_DIR, f)
    if os.path.isfile(p):
        sizes.append(os.path.getsize(p))
sizes = np.array(sizes)
print(f"  sample (n={len(sizes)}) byte size: "
      f"min={sizes.min()}, median={int(np.median(sizes))}, max={sizes.max()}")

# Cross-tab: rating vs decade (mean rating)
df_clean = df.copy()
df_clean['rating_num'] = pd.to_numeric(df_clean['IMDb Rating'], errors='coerce')
df_clean['year_num'] = pd.to_numeric(df_clean['Year'], errors='coerce')
df_clean['decade'] = (df_clean['year_num'] // 10 * 10).astype('Int64')
print(f"\n--- Mean rating by decade")
gb = df_clean.dropna(subset=['rating_num', 'decade']).groupby('decade')['rating_num']
for dec, mean in gb.mean().items():
    n = gb.size().loc[dec]
    print(f"  {int(dec)}s (n={n:5d}): mean={mean:.3f}")

# Save snapshot
snapshot = {
    'n_records': int(len(df)),
    'rating': {
        'n': int(len(ratings)),
        'min': float(ratings.min()),
        'max': float(ratings.max()),
        'mean': float(ratings.mean()),
        'median': float(ratings.median()),
        'std': float(ratings.std()),
        'q25': float(ratings.quantile(0.25)),
        'q75': float(ratings.quantile(0.75)),
        'buckets': {name: int(((ratings >= lo) & (ratings < hi)).sum())
                    for lo, hi, name in buckets},
    },
    'year': {
        'min': int(years.min()),
        'max': int(years.max()),
        'mean': float(years.mean()),
        'median': float(years.median()),
        'by_decade': {f'{int(d)}s': int(n) for d, n in decades.items()},
    },
    'movie_length_min': {
        'min': float(lengths.min()),
        'max': float(lengths.max()),
        'mean': float(lengths.mean()),
        'median': float(lengths.median()),
    },
    'scripts_on_disk': len(script_files),
    'mean_rating_by_decade': {f'{int(d)}s': float(m)
                              for d, m in gb.mean().items()},
}
with open('results/dataset_snapshot.json', 'w') as f:
    json.dump(snapshot, f, indent=2)
print("\nSaved results/dataset_snapshot.json")
