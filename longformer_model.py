# ============================================================
# LONGFORMER + METADATA FUSION REGRESSOR
# ============================================================
"""
Multi-input regression model:
  Text branch  : Longformer encoder -> CLS embedding (768-dim)
  Meta branch  : Linear(3, 32) + ReLU
  Fusion       : Concatenate(800) -> Linear(128) -> ReLU -> Dropout -> Linear(1)
  Output       : single float rating per sample
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LongformerModel

from longformer_config import (
    MODEL_NAME,
    DROPOUT,
    METADATA_INPUT_DIM,
    METADATA_EMBED_DIM,
    FUSION_HIDDEN_DIM,
)


class LongformerRegressor(nn.Module):
    """Longformer encoder fused with structured metadata for rating prediction."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        dropout: float = DROPOUT,
        metadata_input_dim: int = METADATA_INPUT_DIM,
        metadata_embed_dim: int = METADATA_EMBED_DIM,
        fusion_hidden_dim: int = FUSION_HIDDEN_DIM,
    ):
        super().__init__()

        # --- Text branch ---
        self.longformer = LongformerModel.from_pretrained(model_name)
        hidden_size = self.longformer.config.hidden_size  # 768

        # --- Metadata branch ---
        self.metadata_fc = nn.Linear(metadata_input_dim, metadata_embed_dim)

        # --- Fusion head ---
        fused_dim = hidden_size + metadata_embed_dim  # 768 + 32 = 800
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids      : (B, seq_len)
        attention_mask  : (B, seq_len)
        metadata       : (B, metadata_input_dim)  -- already scaled

        Returns
        -------
        predictions : (B,) float tensor
        """
        # Longformer encoding -> CLS token
        longformer_output = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_embedding = longformer_output.last_hidden_state[:, 0, :]  # (B, 768)

        # Metadata embedding
        meta_embedding = F.relu(self.metadata_fc(metadata))  # (B, 32)

        # Concatenate and predict
        combined = torch.cat([text_embedding, meta_embedding], dim=1)  # (B, 800)
        prediction = self.regressor(combined).squeeze(-1)  # (B,)
        return prediction
