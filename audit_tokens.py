"""
Token-budget and structural-feature audit over the screenplay files that
are present in scripts/ (read-only).

Two questions:

  1. The pipeline chunks on WORDS (CHUNK_SIZE=256, CHUNK_OVERLAP=50) but
     all-MiniLM-L6-v2 truncates on WORDPIECE TOKENS at max_seq_length=256,
     of which [CLS] and [SEP] consume two, leaving 254 content positions.
     How much text is silently dropped by that mismatch?

  2. What do the 16 structural statistics produced by
     ScriptPreprocessor.extract_features() actually look like, and how do
     they correlate with the IMDb rating?

Writes results/token_audit.json.

Usage:
    python audit_tokens.py
"""

import os
import json
import hashlib

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from config import CHUNK_SIZE, CHUNK_OVERLAP, SBERT_MODEL_NAME
from preprocessing import ScriptPreprocessor
from trainer import chunk_text

EXCEL = 'movie_lengths.xlsx'
SCRIPTS_DIR = 'scripts/'
RESULTS_DIR = 'results'

STRUCTURAL_16 = [
    'char_count', 'word_count', 'line_count', 'avg_word_length',
    'unique_word_ratio', 'long_word_ratio', 'sentence_count',
    'avg_sentence_length', 'sentence_length_std', 'dialogue_density',
    'unique_characters', 'exclamation_ratio', 'question_ratio',
    'action_density', 'scene_count', 'words_per_scene',
]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_excel(EXCEL)
    df.columns = df.columns.str.strip()
    by_file = {str(v).strip().lower(): i for i, v in enumerate(df['.txt Files'])}

    files = sorted(f for f in os.listdir(SCRIPTS_DIR)
                   if os.path.isfile(os.path.join(SCRIPTS_DIR, f)))
    print(f"Script files available locally: {len(files)}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f'sentence-transformers/{SBERT_MODEL_NAME}')
    max_seq = 256
    budget = max_seq - 2          # [CLS] ... [SEP]
    print(f"Tokenizer: {SBERT_MODEL_NAME}, max_seq_length={max_seq}, "
          f"content budget={budget}")

    digests = {}
    rows = []
    chunk_tok_lens = []
    per_script = []

    for fn in files:
        path = os.path.join(SCRIPTS_DIR, fn)
        with open(path, 'rb') as f:
            raw_bytes = f.read()
        text = None
        for enc in ['utf-8', 'latin-1', 'cp1252']:
            try:
                text = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None or len(text) < 1000:
            continue

        digests.setdefault(hashlib.sha256(raw_bytes).hexdigest(), []).append(fn)

        feats = ScriptPreprocessor.extract_features(text)
        sbert_text = ScriptPreprocessor.clean_text_for_sbert(text)
        chunks = chunk_text(sbert_text, CHUNK_SIZE, CHUNK_OVERLAP)

        lens = [len(tok.encode(c, add_special_tokens=False)) for c in chunks]
        lens = np.array(lens)
        chunk_tok_lens.append(lens)

        over = lens > budget
        kept = np.minimum(lens, budget).sum()
        total = lens.sum()

        per_script.append({
            'file': fn,
            'n_chunks': int(len(lens)),
            'tokens_total': int(total),
            'tokens_kept': int(kept),
            'frac_truncated_chunks': float(over.mean()),
            'frac_tokens_dropped': float(1 - kept / total) if total else 0.0,
        })

        idx = by_file.get(fn.strip().lower())
        rec = dict(feats)
        rec['file'] = fn
        rec['rating'] = (float(df['IMDb Rating'].iloc[idx])
                         if idx is not None else np.nan)
        rec['year'] = (float(df['Year'].iloc[idx])
                       if idx is not None else np.nan)
        rec['n_chunks'] = len(lens)
        rec['frac_tokens_dropped'] = per_script[-1]['frac_tokens_dropped']
        rows.append(rec)

    fdf = pd.DataFrame(rows)
    print(f"Scripts analysed: {len(fdf)}")

    # ---- 1. exact-duplicate rate on the available subset ---------------
    dup_groups = {h: fs for h, fs in digests.items() if len(fs) > 1}
    print(f"SHA-256 duplicate groups among available files: {len(dup_groups)} "
          f"({sum(len(v) for v in dup_groups.values())} records)")

    # ---- 2. truncation ---------------------------------------------------
    all_lens = np.concatenate(chunk_tok_lens)
    ps = pd.DataFrame(per_script)
    trunc = {
        'n_scripts': int(len(ps)),
        'n_chunks_total': int(len(all_lens)),
        'chunk_token_len': {
            'mean': float(all_lens.mean()),
            'median': float(np.median(all_lens)),
            'p90': float(np.percentile(all_lens, 90)),
            'p99': float(np.percentile(all_lens, 99)),
            'max': int(all_lens.max()),
        },
        'content_budget': budget,
        'frac_chunks_over_budget': float((all_lens > budget).mean()),
        'frac_tokens_dropped_corpuswide': float(
            1 - np.minimum(all_lens, budget).sum() / all_lens.sum()),
        'per_script_frac_tokens_dropped': {
            'mean': float(ps['frac_tokens_dropped'].mean()),
            'median': float(ps['frac_tokens_dropped'].median()),
            'max': float(ps['frac_tokens_dropped'].max()),
        },
        'chunks_per_script': {
            'mean': float(ps['n_chunks'].mean()),
            'median': float(ps['n_chunks'].median()),
            'min': int(ps['n_chunks'].min()),
            'max': int(ps['n_chunks'].max()),
        },
    }
    print(f"Chunks over the {budget}-token budget: "
          f"{trunc['frac_chunks_over_budget']*100:.2f}%")
    print(f"Corpus-wide tokens silently dropped: "
          f"{trunc['frac_tokens_dropped_corpuswide']*100:.2f}%")
    print(f"Chunk token length: mean={trunc['chunk_token_len']['mean']:.1f}, "
          f"p99={trunc['chunk_token_len']['p99']:.0f}, "
          f"max={trunc['chunk_token_len']['max']}")

    # ---- 3. structural feature behaviour ---------------------------------
    valid = fdf.dropna(subset=['rating'])
    print(f"\nStructural features vs rating (n={len(valid)}):")
    feat_stats = {}
    for c in STRUCTURAL_16:
        x = valid[c].values.astype(float)
        y = valid['rating'].values.astype(float)
        rho, p = scipy_stats.spearmanr(x, y)
        r, pr = scipy_stats.pearsonr(x, y)
        feat_stats[c] = {
            'mean': float(np.mean(x)),
            'sd': float(np.std(x, ddof=1)),
            'median': float(np.median(x)),
            'p05': float(np.percentile(x, 5)),
            'p95': float(np.percentile(x, 95)),
            'spearman_rho': float(rho),
            'spearman_p': float(p),
            'pearson_r': float(r),
            'pearson_p': float(pr),
        }
        print(f"  {c:22s} mean={np.mean(x):12.3f} sd={np.std(x, ddof=1):12.3f} "
              f"rho={rho:+.3f} (p={p:.3g})")

    out = {
        'scripts_available': len(files),
        'scripts_analysed': int(len(fdf)),
        'chunking': {'chunk_size_words': CHUNK_SIZE,
                     'chunk_overlap_words': CHUNK_OVERLAP},
        'sha256_duplicate_groups_local': len(dup_groups),
        'truncation': trunc,
        'structural_features': feat_stats,
        'rating_summary_local': {
            'n': int(len(valid)),
            'mean': float(valid['rating'].mean()),
            'sd': float(valid['rating'].std(ddof=1)),
            'min': float(valid['rating'].min()),
            'max': float(valid['rating'].max()),
        },
    }
    with open(os.path.join(RESULTS_DIR, 'token_audit.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/token_audit.json")


if __name__ == '__main__':
    main()
