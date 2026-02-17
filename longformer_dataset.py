# ============================================================
# MOVIE SCRIPT DATASET (PyTorch)
# ============================================================
"""
Custom Dataset that tokenizes scripts on the fly and returns
metadata alongside the text tokens.

On-the-fly tokenization keeps memory usage low -- important
when working with ~5 000 full-length movie scripts.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from longformer_config import MAX_LENGTH


class MovieScriptDataset(Dataset):
    """
    Yields tokenized script + metadata + rating for each sample.

    Parameters
    ----------
    texts      : list of raw script strings
    ratings    : 1-D numpy array of float ratings
    metadata   : 2-D numpy array (n_samples, 3) -- already scaled
    tokenizer  : HuggingFace tokenizer (LongformerTokenizerFast)
    max_length : max token count (default from config)
    """

    def __init__(
        self,
        texts: list[str],
        ratings: np.ndarray,
        metadata: np.ndarray,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = MAX_LENGTH,
    ):
        self.texts = texts
        self.ratings = ratings.astype(np.float32)
        self.metadata = metadata.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),        # (max_length,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (max_length,)
            "metadata": torch.tensor(self.metadata[idx]),          # (3,)
            "rating": torch.tensor(self.ratings[idx]),             # scalar
        }
