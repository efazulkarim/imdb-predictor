# ============================================================
# ALTERNATIVE EMBEDDING MODULE
# ============================================================
"""
Provides feature extraction for alternative word and sentence embedding methods:
  1. GloVe (Global Vectors for Word Representation - 300d)
  2. Word2Vec (Continuous Bag-of-Words / Skip-Gram - 300d)
  3. SBERT MPNet (all-mpnet-base-v2 - 768d)

Used for comparative evaluation against the primary Sentence-BERT (all-MiniLM-L6-v2) baseline.
"""

import os
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Union

# Word Tokenizer for classical embeddings (GloVe / Word2Vec)
def _tokenize_script(text: str) -> List[str]:
    """Tokenize script into lowercase words, stripping non-alphanumeric chars."""
    text = text.lower()
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'\(.*?\)', ' ', text)
    words = re.findall(r'\b[a-z]{2,}\b', text)
    return words


# ------------------------------------------------------------
# 1. Word2Vec Embedding Generator
# ------------------------------------------------------------
def generate_word2vec_embeddings(
    scripts_text: List[str],
    vector_size: int = 300,
    window: int = 5,
    min_count: int = 2,
    seed: int = 42,
) -> np.ndarray:
    """
    Train a Word2Vec model directly on the script corpus and compute
    mean word-embedding vectors per script.

    Args:
        scripts_text: List of script texts
        vector_size: Embedding dimension (default: 300)
        window: Context window size
        min_count: Minimum word frequency count
        seed: Random seed

    Returns:
        np.ndarray of shape (n_scripts, vector_size)
    """
    try:
        from gensim.models import Word2Vec
    except ImportError:
        raise ImportError("gensim is required for Word2Vec embeddings. Install via `pip install gensim`.")

    print(f"\n🧠 Training Word2Vec model on {len(scripts_text)} scripts (dim={vector_size})...")
    tokenized_corpus = [_tokenize_script(doc) for doc in scripts_text]

    w2v_model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        seed=seed,
    )

    embeddings = []
    for tokens in tokenized_corpus:
        vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if vectors:
            doc_vec = np.mean(vectors, axis=0)
        else:
            doc_vec = np.zeros(vector_size)
        embeddings.append(doc_vec)

    out = np.array(embeddings)
    print(f"   Word2Vec embeddings generated: shape {out.shape}")
    return out


# ------------------------------------------------------------
# 2. GloVe Embedding Generator
# ------------------------------------------------------------
def generate_glove_embeddings(
    scripts_text: List[str],
    glove_path: str = None,
    vector_size: int = 300,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute document embeddings using pre-trained GloVe vectors (or corpus-trained SVD fallback).

    If glove_path is provided and exists, reads pre-trained vectors.
    Otherwise, uses GloVe-style co-occurrence matrix factorization (TruncatedSVD on PPMI)
    to generate 300-d semantic word vectors from the corpus.

    Args:
        scripts_text: List of script texts
        glove_path: Path to glove.6B.300d.txt (optional)
        vector_size: Vector dimension
        seed: Random seed

    Returns:
        np.ndarray of shape (n_scripts, vector_size)
    """
    tokenized_corpus = [_tokenize_script(doc) for doc in scripts_text]
    vocab_dict = {}

    if glove_path and os.path.exists(glove_path):
        print(f"\n📖 Loading pre-trained GloVe vectors from '{glove_path}'...")
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ')
                word = parts[0]
                vec = np.asarray(parts[1:], dtype='float32')
                if len(vec) == vector_size:
                    vocab_dict[word] = vec
    else:
        print(f"\n📖 Building GloVe representation (co-occurrence SVD, dim={vector_size})...")
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import TruncatedSVD

        vectorizer = CountVectorizer(max_features=10000, min_df=3, stop_words='english')
        X_counts = vectorizer.fit_transform(scripts_text)
        
        # Word-word co-occurrence
        X_cooc = (X_counts.T * X_counts).astype(np.float32)
        X_cooc.setdiag(0)

        # Truncated SVD (GloVe approximation)
        svd = TruncatedSVD(n_components=vector_size, random_state=seed)
        word_vecs = svd.fit_transform(X_cooc)
        
        vocab_words = vectorizer.get_feature_names_out()
        for idx, word in enumerate(vocab_words):
            vocab_dict[word] = word_vecs[idx]

    embeddings = []
    for tokens in tokenized_corpus:
        vectors = [vocab_dict[word] for word in tokens if word in vocab_dict]
        if vectors:
            doc_vec = np.mean(vectors, axis=0)
        else:
            doc_vec = np.zeros(vector_size)
        embeddings.append(doc_vec)

    out = np.array(embeddings)
    print(f"   GloVe embeddings generated: shape {out.shape}")
    return out


# ------------------------------------------------------------
# 3. SBERT MPNet (all-mpnet-base-v2 - 768d)
# ------------------------------------------------------------
def get_mpnet_model_name() -> str:
    """Return the name of the state-of-the-art SBERT model."""
    return 'all-mpnet-base-v2'
