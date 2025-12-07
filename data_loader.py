# ============================================================
# DATA LOADING
# ============================================================
"""
Data loading module.
Handles loading Excel files and script text files.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import (
    EXCEL_FILE, SCRIPTS_DIR, SCRIPT_COL, RATING_COL,
    YEAR_COL, DECADE_COL, MOVIE_NAME_COL
)
from preprocessing import ScriptPreprocessor


def load_dataset():
    """Load Excel and script files, extract all features."""
    print("=" * 70)
    print("  LOADING DATASET")
    print("=" * 70)

    # Load Excel
    df = pd.read_excel(EXCEL_FILE)
    print(f"\n📊 Excel loaded: {len(df)} records")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Note: Scripts not found will be skipped automatically")

    # Data containers
    scripts_text = []
    script_features = []
    ratings = []
    years = []
    decades = []
    movie_names = []

    # Track loading stats
    loaded = 0
    skipped = {'missing': 0, 'short': 0, 'invalid_rating': 0, 'error': 0}

    print(f"\n📁 Loading scripts from: {SCRIPTS_DIR}")
    print("   Processing", end="", flush=True)

    for idx, row in df.iterrows():
        # Progress indicator
        if (idx + 1) % 500 == 0:
            print(".", end="", flush=True)

        # Skip if rating is missing or invalid
        try:
            rating = float(row[RATING_COL])
            if rating < 0 or rating > 10 or pd.isna(rating):
                skipped['invalid_rating'] += 1
                continue
        except:
            skipped['invalid_rating'] += 1
            continue

        # Get script filename
        script_file = str(row[SCRIPT_COL]).strip()

        # Try multiple filename patterns
        possible_paths = [
            os.path.join(SCRIPTS_DIR, script_file),
            os.path.join(SCRIPTS_DIR, script_file + '.txt'),
            os.path.join(SCRIPTS_DIR, script_file.replace(' ', '-') + '.txt'),
            os.path.join(SCRIPTS_DIR, script_file.replace(' ', '-')),
            os.path.join(SCRIPTS_DIR, script_file.replace(' ', '_') + '.txt'),
        ]

        # Extract number and try more patterns
        num_match = re.search(r'(\d+)', script_file)
        if num_match:
            num = num_match.group(1)
            possible_paths.extend([
                os.path.join(SCRIPTS_DIR, f'file-{num}.txt'),
                os.path.join(SCRIPTS_DIR, f'file_{num}.txt'),
                os.path.join(SCRIPTS_DIR, f'file {num}.txt'),
            ])

        # Find existing file
        filepath = None
        for p in possible_paths:
            if os.path.exists(p):
                filepath = p
                break

        if not filepath:
            skipped['missing'] += 1
            continue

        # Read script
        try:
            raw_text = None
            for enc in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(filepath, 'r', encoding=enc) as f:
                        raw_text = f.read()
                    break
                except UnicodeDecodeError:
                    continue

            if not raw_text or len(raw_text) < 1000:
                skipped['short'] += 1
                continue

            # Preprocess and extract features
            cleaned_text = ScriptPreprocessor.clean_text(raw_text)
            features = ScriptPreprocessor.extract_features(raw_text)

            # Store data
            scripts_text.append(cleaned_text)
            script_features.append(features)
            ratings.append(rating)
            years.append(int(row[YEAR_COL]) if pd.notna(row.get(YEAR_COL)) else 2000)
            decades.append(str(row.get(DECADE_COL, '2000s')))
            movie_names.append(str(row.get(MOVIE_NAME_COL, f'Movie_{idx}')))
            loaded += 1

        except Exception as e:
            skipped['error'] += 1
            continue

    print(" Done!")

    # Summary
    print(f"\n📈 Loading Summary:")
    print(f"   ✅ Successfully loaded: {loaded} scripts")
    print(f"   ❌ Skipped: {sum(skipped.values())} total")
    print(f"      - Missing files: {skipped['missing']}")
    print(f"      - Too short (<1KB): {skipped['short']}")
    print(f"      - Invalid rating: {skipped['invalid_rating']}")
    print(f"      - Read errors: {skipped['error']}")

    # Convert to arrays/dataframes
    features_df = pd.DataFrame(script_features)
    features_df['year'] = years

    # Encode decades
    le = LabelEncoder()
    features_df['decade_encoded'] = le.fit_transform(decades)

    return scripts_text, np.array(ratings), features_df, movie_names, le

