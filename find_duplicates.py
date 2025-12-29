# Script to find duplicate movies
import pandas as pd

df = pd.read_excel('movie_lengths.xlsx')

print("=" * 70)
print("FINDING DUPLICATE MOVIES")
print("=" * 70)

# Find duplicates by movie name
movie_name_counts = df['Movie name'].value_counts()
duplicates = movie_name_counts[movie_name_counts > 1]

print(f"\n=== DUPLICATE MOVIE NAMES ({len(duplicates)} unique names, {duplicates.sum() - len(duplicates)} extra rows) ===\n")

for movie_name, count in duplicates.items():
    rows = df[df['Movie name'] == movie_name]
    print(f"'{movie_name}' appears {count} times:")
    for idx, row in rows.iterrows():
        script = row.get('.txt Files', 'N/A')
        rating = row.get('IMDb Rating', 'N/A')
        year = row.get('Year', 'N/A')
        print(f"    Row {idx}: {script}, Rating={rating}, Year={year}")
    print()

print(f"\nTotal duplicate entries (extras to remove): {duplicates.sum() - len(duplicates)}")
