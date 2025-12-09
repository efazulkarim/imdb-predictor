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
    """Load Excel and script files, extract all features.
    
    Supports loading from multiple Excel files. If EXCEL_FILES is defined
    and contains file paths, it will load from all of them. Otherwise,
    it falls back to EXCEL_FILE. Duplicate scripts (same filename) are
    automatically handled - only the first occurrence is processed.
    """
    print("=" * 70)
    print("  LOADING DATASET")
    print("=" * 70)

    # Determine which Excel files to load
    excel_files_to_load = []
    
    # Check if EXCEL_FILES is defined and has content
    try:
        # Get EXCEL_FILES safely (may not be defined)
        import config
        EXCEL_FILES = getattr(config, 'EXCEL_FILES', None)
        
        # Normalize EXCEL_FILES to always be a list
        if EXCEL_FILES:
            if isinstance(EXCEL_FILES, str):
                # If it's a string, convert to list
                excel_files_to_load = [EXCEL_FILES]
            elif isinstance(EXCEL_FILES, list) and len(EXCEL_FILES) > 0:
                # If it's a non-empty list, use it
                excel_files_to_load = EXCEL_FILES
            else:
                # Empty list or other type, fall back to single file
                excel_files_to_load = [EXCEL_FILE]
        else:
            # EXCEL_FILES is None/False, fall back to single file
            excel_files_to_load = [EXCEL_FILE]
        
        if len(excel_files_to_load) > 1:
            print(f"\n📊 Loading from {len(excel_files_to_load)} Excel file(s):")
        else:
            print(f"\n📊 Loading from single Excel file:")
    except Exception:
        # Any error, fall back to single file
        excel_files_to_load = [EXCEL_FILE]
        print(f"\n📊 Loading from single Excel file:")

    # Load and combine Excel files
    dataframes = []
    for excel_file in excel_files_to_load:
        if not os.path.exists(excel_file):
            print(f"   ⚠️  Warning: File not found: {excel_file}")
            continue
        
        try:
            df_temp = pd.read_excel(excel_file)
            
            # Normalize column names BEFORE combining
            # This ensures columns with same name (but different whitespace) are treated as the same
            column_renames = {}
            
            # 1. Normalize all whitespace (leading, trailing, and normalize internal spaces)
            for col in df_temp.columns:
                col_normalized = ' '.join(col.split())  # Normalize all whitespace to single spaces and strip
                if col != col_normalized:
                    column_renames[col] = col_normalized
            
            # Apply whitespace normalization first
            if column_renames:
                df_temp = df_temp.rename(columns=column_renames)
                column_renames = {}  # Reset for next step
            
            # 2. Fix typos/variations (after whitespace normalization)
            if 'Decaade' in df_temp.columns:
                column_renames['Decaade'] = 'Decade'
            
            # Apply typo fixes
            if column_renames:
                df_temp = df_temp.rename(columns=column_renames)
            
            # Final strip to ensure no trailing/leading spaces remain
            df_temp.columns = df_temp.columns.str.strip()
            
            # Show rating range for this file
            if RATING_COL in df_temp.columns:
                valid_ratings = pd.to_numeric(df_temp[RATING_COL], errors='coerce').dropna()
                if len(valid_ratings) > 0:
                    min_rating = valid_ratings.min()
                    max_rating = valid_ratings.max()
                    print(f"   ✅ {excel_file}: {len(df_temp)} records (Ratings: {min_rating:.1f}-{max_rating:.1f})")
                else:
                    print(f"   ✅ {excel_file}: {len(df_temp)} records (no valid ratings)")
            else:
                print(f"   ✅ {excel_file}: {len(df_temp)} records")
            
            dataframes.append(df_temp)
        except Exception as e:
            print(f"   ❌ Error loading {excel_file}: {e}")
            continue

    if not dataframes:
        raise FileNotFoundError("No valid Excel files found to load!")

    # Combine all dataframes
    # Column names are already normalized in the loading loop above
    # This should ensure columns with same name (but different whitespace) are treated as the same
    df = pd.concat(dataframes, ignore_index=True, sort=False)
    
    # Final check: ensure no duplicate column names
    # This should not happen after normalization, but check just in case
    if len(df.columns) != len(set(df.columns)):
        # Find and report duplicate column names
        seen = set()
        duplicates = []
        for col in df.columns:
            if col in seen:
                duplicates.append(col)
            else:
                seen.add(col)
        
        if duplicates:
            print(f"   ⚠️  Warning: Found duplicate column names: {', '.join(duplicates)}")
            # Drop duplicate columns (keep first occurrence)
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
            print(f"   🔧 Removed duplicate columns, keeping first occurrence")
    
    # Ensure all expected columns exist (add missing ones with NaN if needed)
    expected_columns = [
        MOVIE_NAME_COL,
        YEAR_COL,
        RATING_COL,
        'IMDb ID',  # This might not be in config, so hardcode it
        SCRIPT_COL,
        'Collected By',  # This might not be in config, so hardcode it
        DECADE_COL
    ]
    
    # Add any missing expected columns (fill with NaN)
    for col in expected_columns:
        if col not in df.columns:
            df[col] = None
            print(f"   ⚠️  Warning: Column '{col}' not found in Excel files, added as empty")
    
    # Remove duplicates carefully
    # Check for duplicates based on Movie Name OR Script filename
    initial_count = len(df)
    
    # Check for duplicates by script filename (most reliable - same script = same movie)
    script_duplicates_before = df.duplicated(subset=[SCRIPT_COL], keep='first').sum()
    
    # Check for duplicates by movie name (same movie name might appear in both sheets)
    movie_duplicates_before = df.duplicated(subset=[MOVIE_NAME_COL], keep='first').sum()
    
    # Remove duplicates: First remove by script filename (most reliable identifier)
    # This handles cases where the same script file appears in both sheets
    df = df.drop_duplicates(subset=[SCRIPT_COL], keep='first')
    
    # Then remove by movie name (handles cases where same movie has different script files)
    # This catches any remaining duplicates where movie name matches but script file differs
    movie_duplicates_after = df.duplicated(subset=[MOVIE_NAME_COL], keep='first').sum()
    if movie_duplicates_after > 0:
        df = df.drop_duplicates(subset=[MOVIE_NAME_COL], keep='first')
    
    total_duplicates_removed = initial_count - len(df)
    
    if total_duplicates_removed > 0:
        print(f"\n📝 Duplicate Removal Summary:")
        print(f"   - Removed {script_duplicates_before} duplicate(s) by script filename")
        if movie_duplicates_after > 0:
            print(f"   - Removed {movie_duplicates_after} additional duplicate(s) by movie name")
        print(f"   - Total duplicates removed: {total_duplicates_removed}")
        print(f"   - Kept first occurrence of each duplicate")
    else:
        print(f"\n✅ No duplicates found - all {len(df)} records are unique")
    
    print(f"\n📊 Combined dataset: {len(df)} unique records")
    
    # Display columns in a readable format
    all_columns = list(df.columns)
    print(f"   Columns ({len(all_columns)} total): {', '.join(all_columns)}")
    
    # Verify expected columns are present
    expected_columns_check = [
        MOVIE_NAME_COL,
        YEAR_COL,
        RATING_COL,
        'IMDb ID',
        SCRIPT_COL,
        'Collected By',
        DECADE_COL
    ]
    missing_columns = [col for col in expected_columns_check if col not in all_columns]
    if missing_columns:
        print(f"   ⚠️  Missing expected columns: {', '.join(missing_columns)}")
    else:
        print(f"   ✅ All expected columns present")
    
    # Show combined rating range
    if RATING_COL in df.columns:
        valid_ratings = pd.to_numeric(df[RATING_COL], errors='coerce').dropna()
        if len(valid_ratings) > 0:
            print(f"   Rating range: {valid_ratings.min():.1f} - {valid_ratings.max():.1f} (mean: {valid_ratings.mean():.2f})")
    
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
            # Handle double extensions (e.g., "file 3000.txt.txt")
            os.path.join(SCRIPTS_DIR, script_file + '.txt.txt'),
        ]

        # Extract number and try more patterns
        num_match = re.search(r'(\d+)', script_file)
        if num_match:
            num = num_match.group(1)
            possible_paths.extend([
                os.path.join(SCRIPTS_DIR, f'file-{num}.txt'),
                os.path.join(SCRIPTS_DIR, f'file_{num}.txt'),
                os.path.join(SCRIPTS_DIR, f'file {num}.txt'),
                os.path.join(SCRIPTS_DIR, f'file {num}.txt.txt'),  # Handle double extension
                os.path.join(SCRIPTS_DIR, f'file{num}.txt'),  # No space
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


# ============================================================
# TEST/MAIN BLOCK (for testing the data loader)
# ============================================================
if __name__ == "__main__":
    print("Testing data loader...")
    try:
        scripts_text, ratings, features_df, movie_names, decade_encoder = load_dataset()
        print(f"\n✅ Successfully loaded {len(scripts_text)} scripts!")
        print(f"   Rating range: {ratings.min():.2f} - {ratings.max():.2f}")
        print(f"   Mean rating: {ratings.mean():.2f}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()