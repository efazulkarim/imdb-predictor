"""
Corpus integrity audit (read-only).

Reproduces the exact record ordering used by data_loader.load_dataset(),
then layers three integrity checks on top of it WITHOUT changing the
loader itself:

    1. Exact-duplicate detection via SHA-256 over the raw script bytes.
    2. Title-family assignment (normalized title key) for grouped splits.
    3. Release-year validity screening.

Writes results/corpus_audit.json + results/corpus_audit_index.npz so the
downstream leakage-controlled evaluation can reuse the cached SBERT
embeddings by positional index.

Usage:
    python audit_corpus.py
"""

import os
import re
import json
import hashlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from config import (
    SCRIPTS_DIR, SCRIPT_COL, RATING_COL, YEAR_COL,
    DECADE_COL, MOVIE_NAME_COL, MOVIE_LENGTH_COL,
)

EXCEL = 'movie_lengths.xlsx'
RESULTS_DIR = 'results'

# Plausible release-year window for a feature film with an IMDb entry.
YEAR_MIN, YEAR_MAX = 1900, 2025


def resolve_path(script_file):
    """Mirror of the filename-resolution ladder in data_loader.load_dataset()."""
    candidates = [
        os.path.join(SCRIPTS_DIR, script_file),
        os.path.join(SCRIPTS_DIR, script_file + '.txt'),
        os.path.join(SCRIPTS_DIR, script_file.replace(' ', '-') + '.txt'),
        os.path.join(SCRIPTS_DIR, script_file.replace(' ', '-')),
        os.path.join(SCRIPTS_DIR, script_file.replace(' ', '_') + '.txt'),
        os.path.join(SCRIPTS_DIR, script_file + '.txt.txt'),
    ]
    num_match = re.search(r'(\d+)', script_file)
    if num_match:
        num = num_match.group(1)
        candidates.extend([
            os.path.join(SCRIPTS_DIR, f'file-{num}.txt'),
            os.path.join(SCRIPTS_DIR, f'file_{num}.txt'),
            os.path.join(SCRIPTS_DIR, f'file {num}.txt'),
            os.path.join(SCRIPTS_DIR, f'file {num}.txt.txt'),
            os.path.join(SCRIPTS_DIR, f'file{num}.txt'),
        ])
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def read_raw(filepath):
    """Same decoding ladder as the loader; returns (text, raw_bytes)."""
    with open(filepath, 'rb') as f:
        raw_bytes = f.read()
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            return raw_bytes.decode(enc), raw_bytes
        except UnicodeDecodeError:
            continue
    return None, raw_bytes


_ROMAN = {
    'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5,
    'vi': 6, 'vii': 7, 'viii': 8, 'ix': 9, 'x': 10,
}
_SEQUEL_TAIL = re.compile(
    r'\s+(part\s+)?(\d+|i{1,3}|iv|vi{0,3}|ix|x)$'
)
_ARTICLE = re.compile(r'^(the|a|an)\s+')


def title_family(title):
    """
    Normalized grouping key for a title.

    Two records land in the same family when they share a canonical title
    stem after (a) case folding, (b) punctuation stripping, (c) leading
    article removal, and (d) removal of a trailing sequel/part marker.
    "Toy Story", "Toy Story 2" and "Toy Story 3" therefore form one group,
    which prevents a franchise's installments from straddling a fold
    boundary and leaking franchise-level style and vocabulary.
    """
    t = str(title).lower().strip()
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    t = _ARTICLE.sub('', t)
    prev = None
    while prev != t:
        prev = t
        t = _SEQUEL_TAIL.sub('', t).strip()
    return t or str(title).lower().strip()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_excel(EXCEL)
    df.columns = df.columns.str.strip()
    n_excel = len(df)
    print(f"Excel records: {n_excel}")

    dup_filenames = int(df.duplicated(subset=[SCRIPT_COL], keep='first').sum())
    df = df.drop_duplicates(subset=[SCRIPT_COL], keep='first')
    print(f"Duplicate script filenames removed: {dup_filenames}")

    rows = []          # one entry per successfully loaded record (loader order)
    skipped = Counter()

    for excel_idx, row in df.iterrows():
        try:
            rating = float(row[RATING_COL])
            if rating < 0 or rating > 10 or pd.isna(rating):
                skipped['invalid_rating'] += 1
                continue
        except Exception:
            skipped['invalid_rating'] += 1
            continue

        script_file = str(row[SCRIPT_COL]).strip()
        filepath = resolve_path(script_file)
        if not filepath:
            skipped['missing'] += 1
            continue

        text, raw_bytes = read_raw(filepath)
        if not text or len(text) < 1000:
            skipped['short'] += 1
            continue

        try:
            year = int(row[YEAR_COL]) if pd.notna(row.get(YEAR_COL)) else 2000
        except Exception:
            year = 2000

        rows.append({
            'load_idx': len(rows),
            'excel_idx': int(excel_idx),
            'script_file': os.path.basename(filepath),
            'movie_name': str(row.get(MOVIE_NAME_COL, '')),
            'family': title_family(row.get(MOVIE_NAME_COL, '')),
            'year': year,
            'rating': rating,
            'length': (float(row[MOVIE_LENGTH_COL])
                       if pd.notna(row.get(MOVIE_LENGTH_COL)) else None),
            'decade': str(row.get(DECADE_COL, '2000s')),
            'sha256': hashlib.sha256(raw_bytes).hexdigest(),
            'n_bytes': len(raw_bytes),
        })

    n_loaded = len(rows)
    print(f"Loaded records (matches loader): {n_loaded}  skipped={dict(skipped)}")

    # ---- 1. Exact-duplicate detection over SHA-256 digests -------------
    by_hash = defaultdict(list)
    for r in rows:
        by_hash[r['sha256']].append(r)

    dup_groups = {h: rs for h, rs in by_hash.items() if len(rs) > 1}
    n_dup_groups = len(dup_groups)
    n_dup_records = sum(len(rs) for rs in dup_groups.values())
    n_dup_dropped = n_dup_records - n_dup_groups
    print(f"SHA-256 unique digests: {len(by_hash)}")
    print(f"Duplicate groups: {n_dup_groups} "
          f"covering {n_dup_records} records -> drop {n_dup_dropped}")

    group_size_hist = Counter(len(rs) for rs in dup_groups.values())
    print(f"Duplicate group sizes: {dict(sorted(group_size_hist.items()))}")

    # Rating disagreement inside a duplicate group = label noise that a
    # random split would silently convert into an unbeatable test signal.
    disagreeing = 0
    rating_spread = []
    for rs in dup_groups.values():
        ratings = [r['rating'] for r in rs]
        spread = max(ratings) - min(ratings)
        rating_spread.append(spread)
        if spread > 0:
            disagreeing += 1
    print(f"Duplicate groups with disagreeing ratings: {disagreeing}")
    if rating_spread:
        print(f"  max rating spread within a group: {max(rating_spread):.1f}")

    # Keep the first record of every digest (loader order).
    keep_dedup = []
    seen_hash = set()
    for r in rows:
        if r['sha256'] in seen_hash:
            continue
        seen_hash.add(r['sha256'])
        keep_dedup.append(r)
    print(f"After exact dedup: {len(keep_dedup)}")

    # ---- 2. Year validity screening -----------------------------------
    invalid_year = [r for r in keep_dedup
                    if not (YEAR_MIN <= r['year'] <= YEAR_MAX)]
    print(f"Records with out-of-range year: {len(invalid_year)}")
    for r in invalid_year:
        print(f"   {r['movie_name']!r} year={r['year']} file={r['script_file']}")

    final = [r for r in keep_dedup if YEAR_MIN <= r['year'] <= YEAR_MAX]
    print(f"FINAL analysis corpus: {len(final)}")

    # ---- 3. Title-family statistics -----------------------------------
    fam_counts = Counter(r['family'] for r in final)
    multi_fam = {f: c for f, c in fam_counts.items() if c > 1}
    print(f"Title families: {len(fam_counts)} "
          f"({len(multi_fam)} with >1 member, "
          f"largest={max(fam_counts.values())})")
    top_fams = sorted(multi_fam.items(), key=lambda kv: -kv[1])[:10]
    print(f"  largest families: {top_fams}")

    # Near-duplicate risk that exact hashing cannot catch: same family,
    # same year -> almost certainly the same film under two file names.
    same_fam_year = Counter((r['family'], r['year']) for r in final)
    n_fam_year_collisions = sum(c - 1 for c in same_fam_year.values() if c > 1)
    print(f"Same-family + same-year record pairs beyond the first: "
          f"{n_fam_year_collisions}")

    # ---- 4. Chronological holdout sizes -------------------------------
    n_hist = sum(1 for r in final if r['year'] <= 2019)
    n_recent = sum(1 for r in final if 2020 <= r['year'] <= 2025)
    print(f"Chronological split: train<=2019 n={n_hist}, "
          f"test 2020-2025 n={n_recent}")

    ratings_all = np.array([r['rating'] for r in final])
    ratings_hist = np.array([r['rating'] for r in final if r['year'] <= 2019])
    ratings_recent = np.array([r['rating'] for r in final
                               if 2020 <= r['year'] <= 2025])
    print(f"  mean rating <=2019: {ratings_hist.mean():.4f} "
          f"(sd {ratings_hist.std(ddof=1):.4f})")
    print(f"  mean rating 2020-25: {ratings_recent.mean():.4f} "
          f"(sd {ratings_recent.std(ddof=1):.4f})")

    # ---- Save artifacts ------------------------------------------------
    audit = {
        'excel_records': n_excel,
        'duplicate_filenames_removed': dup_filenames,
        'loaded_records': n_loaded,
        'skipped': dict(skipped),
        'sha256': {
            'unique_digests': len(by_hash),
            'duplicate_groups': n_dup_groups,
            'records_in_duplicate_groups': n_dup_records,
            'records_dropped': n_dup_dropped,
            'group_size_histogram': {str(k): v for k, v
                                     in sorted(group_size_hist.items())},
            'groups_with_disagreeing_ratings': disagreeing,
            'max_rating_spread_in_group': (float(max(rating_spread))
                                           if rating_spread else 0.0),
            'n_after_dedup': len(keep_dedup),
        },
        'year_screen': {
            'min_valid': YEAR_MIN,
            'max_valid': YEAR_MAX,
            'dropped': len(invalid_year),
            'dropped_records': [
                {'movie_name': r['movie_name'], 'year': r['year'],
                 'script_file': r['script_file']} for r in invalid_year
            ],
        },
        'final_n': len(final),
        'families': {
            'n_families': len(fam_counts),
            'n_multi_member': len(multi_fam),
            'largest': max(fam_counts.values()),
            'top': [{'family': f, 'n': c} for f, c in top_fams],
            'same_family_same_year_extra_records': n_fam_year_collisions,
        },
        'chronological': {
            'train_max_year': 2019,
            'n_train_pool': n_hist,
            'n_holdout': n_recent,
            'mean_rating_train_pool': float(ratings_hist.mean()),
            'sd_rating_train_pool': float(ratings_hist.std(ddof=1)),
            'mean_rating_holdout': float(ratings_recent.mean()),
            'sd_rating_holdout': float(ratings_recent.std(ddof=1)),
        },
        'rating_all': {
            'mean': float(ratings_all.mean()),
            'sd': float(ratings_all.std(ddof=1)),
            'min': float(ratings_all.min()),
            'max': float(ratings_all.max()),
        },
    }
    with open(os.path.join(RESULTS_DIR, 'corpus_audit.json'), 'w') as f:
        json.dump(audit, f, indent=2)
    print("\nSaved results/corpus_audit.json")

    np.savez(
        os.path.join(RESULTS_DIR, 'corpus_audit_index.npz'),
        load_idx=np.array([r['load_idx'] for r in final], dtype=np.int64),
        excel_idx=np.array([r['excel_idx'] for r in final], dtype=np.int64),
        year=np.array([r['year'] for r in final], dtype=np.int64),
        rating=np.array([r['rating'] for r in final], dtype=np.float64),
        family=np.array([r['family'] for r in final], dtype=object),
        script_file=np.array([r['script_file'] for r in final], dtype=object),
        movie_name=np.array([r['movie_name'] for r in final], dtype=object),
        sha256=np.array([r['sha256'] for r in final], dtype=object),
        n_bytes=np.array([r['n_bytes'] for r in final], dtype=np.int64),
    )
    print("Saved results/corpus_audit_index.npz")


if __name__ == '__main__':
    main()
