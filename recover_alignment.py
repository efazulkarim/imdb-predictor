"""
Recover the positional mapping load_idx -> Excel row (read-only).

The cached SBERT matrix in results/ is stored in loader order, but the
loader silently dropped 9 records whose script text fell below the 1 KB
threshold, so position i in the matrix is NOT Excel row i. This script
reconstructs the offset function from the split bookkeeping already
committed in test_set_info.json:

    offset(i) = excel_idx(i) - i,  non-decreasing, 0 -> 9

780 test anchors give (load_idx, script_file) pairs that fix the offset
exactly at 780 positions; 779 validation anchors give (load_idx, title,
rating) pairs used to disambiguate the remaining gaps.

Writes results/alignment.npz with the recovered excel_idx per load_idx
and a boolean mask flagging any record whose alignment stays ambiguous.

Usage:
    python recover_alignment.py
"""

import json
import itertools

import numpy as np
import pandas as pd

EXCEL = 'movie_lengths.xlsx'
N_LOAD = 5195


def norm_title(s):
    return ' '.join(str(s).lower().split())


def main():
    df = pd.read_excel(EXCEL)
    df.columns = df.columns.str.strip()
    df = df.reset_index(drop=True)
    n_excel = len(df)

    fmap = {}
    for i, v in enumerate(df['.txt Files']):
        fmap.setdefault(str(v).strip().lower(), []).append(i)

    info = json.load(open('test_set_info.json'))

    # --- hard anchors from the test split (filename is unique) ----------
    anchors = {}
    for li, sf in zip(info['test_indices'], info['test_script_files']):
        key = sf.strip().lower()
        cands = fmap.get(key) or fmap.get(key[:-4])
        anchors[int(li)] = cands[0]
    anchor_items = sorted(anchors.items())
    assert all(b[1] - b[0] >= a[1] - a[0]
               for a, b in zip(anchor_items, anchor_items[1:])), \
        "offset must be non-decreasing"

    n_skips = anchor_items[-1][1] - anchor_items[-1][0]
    print(f"Anchors: {len(anchor_items)}, total skips implied: {n_skips}")

    # --- soft anchors from the validation split (title + rating) --------
    val = {}
    for li, name, rating in zip(info['validation_indices'],
                                info['validation_movie_names'],
                                info['validation_ratings']):
        val[int(li)] = (norm_title(name), float(rating))
    print(f"Validation soft anchors: {len(val)}")

    excel_titles = [norm_title(t) for t in df['Movie name']]
    excel_ratings = pd.to_numeric(df['IMDb Rating'], errors='coerce').values

    def soft_ok(load_idx, excel_idx):
        """Does the validation record at load_idx match this Excel row?"""
        if load_idx not in val:
            return True
        t, r = val[load_idx]
        if excel_titles[excel_idx] != t:
            return False
        return abs(excel_ratings[excel_idx] - r) < 1e-9

    # --- enumerate gaps between consecutive hard anchors -----------------
    # A gap that raises the offset by k hides k skipped Excel rows strictly
    # between the two anchored Excel rows.
    gaps = []
    for (l1, e1), (l2, e2) in zip(anchor_items, anchor_items[1:]):
        k = (e2 - l2) - (e1 - l1)
        if k > 0:
            gaps.append({'l1': l1, 'e1': e1, 'l2': l2, 'e2': e2, 'k': k})
    # leading / trailing regions carry no offset change, so nothing to solve
    print(f"Gaps to resolve: {len(gaps)} "
          f"(total skips {sum(g['k'] for g in gaps)})")

    skip_rows = []          # confirmed skipped Excel rows
    ambiguous_loads = set()  # load indices whose Excel row stays uncertain

    for g in gaps:
        # Candidate skipped Excel rows lie strictly inside (e1, e2).
        cand = list(range(g['e1'] + 1, g['e2']))
        loads = list(range(g['l1'] + 1, g['l2']))
        solutions = []
        for combo in itertools.combinations(cand, g['k']):
            # Build the implied load->excel map inside the gap.
            skipset = set(combo)
            ok = True
            mapping = {}
            e = g['e1'] + 1
            for li in loads:
                while e in skipset:
                    e += 1
                mapping[li] = e
                e += 1
            if e != g['e2']:
                continue
            for li, ei in mapping.items():
                if not soft_ok(li, ei):
                    ok = False
                    break
            if ok:
                solutions.append((combo, mapping))

        print(f"  gap load {g['l1']}->{g['l2']} (k={g['k']}, "
              f"{len(cand)} candidate rows): {len(solutions)} consistent solution(s)")
        if len(solutions) == 1:
            skip_rows.extend(solutions[0][0])
        else:
            # Keep the intersection: load indices on which every surviving
            # solution agrees are still trustworthy.
            agree = {}
            for li in loads:
                vals = {sol[1][li] for sol in solutions}
                if len(vals) == 1:
                    agree[li] = vals.pop()
                else:
                    ambiguous_loads.add(li)
            if solutions:
                skip_rows.extend(solutions[0][0])  # provisional
            print(f"     -> {len(ambiguous_loads)} ambiguous load indices so far")

    skip_rows = sorted(set(skip_rows))
    print(f"\nRecovered skipped Excel rows ({len(skip_rows)}): {skip_rows}")
    for r in skip_rows:
        print(f"   row {r}: {df['Movie name'].iloc[r]!r} "
              f"file={df['.txt Files'].iloc[r]} "
              f"year={df['Year'].iloc[r]} rating={df['IMDb Rating'].iloc[r]}")

    # --- build the full mapping ------------------------------------------
    skipset = set(skip_rows)
    excel_idx = []
    e = 0
    for li in range(N_LOAD):
        while e in skipset:
            e += 1
        excel_idx.append(e)
        e += 1
    excel_idx = np.array(excel_idx, dtype=np.int64)
    assert e == n_excel, f"consumed {e} of {n_excel} Excel rows"

    # --- verify against every anchor --------------------------------------
    bad_hard = [(li, ei, excel_idx[li]) for li, ei in anchor_items
                if excel_idx[li] != ei]
    print(f"\nHard-anchor mismatches: {len(bad_hard)}")
    bad_soft = [li for li in val if not soft_ok(li, excel_idx[li])]
    print(f"Soft-anchor mismatches: {len(bad_soft)}")
    if bad_soft[:5]:
        for li in bad_soft[:5]:
            print(f"   load {li}: expected {val[li]}, "
                  f"got ({excel_titles[excel_idx[li]]!r}, "
                  f"{excel_ratings[excel_idx[li]]})")

    ambiguous = np.zeros(N_LOAD, dtype=bool)
    for li in ambiguous_loads:
        ambiguous[li] = True
    print(f"Ambiguous load indices: {int(ambiguous.sum())}")

    np.savez('results/alignment.npz',
             excel_idx=excel_idx,
             ambiguous=ambiguous,
             skipped_excel_rows=np.array(skip_rows, dtype=np.int64))
    print("\nSaved results/alignment.npz")


if __name__ == '__main__':
    main()
