from __future__ import annotations

import torch
import torch.nn as nn

from src.models.transformer_classification import PositionalEncoding


class TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        sequence_length: int,
        patch_size: int = 8,
        d_model: int = 32,
        nhead: int = 2,
        num_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if sequence_length % patch_size != 0:
            raise ValueError("sequence_length must be divisible by patch_size.")

        self.input_size = input_size
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.patch_dim = patch_size * input_size
        self.num_patches = sequence_length // patch_size

        self.patch_projection = nn.Linear(self.patch_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=self.num_patches + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.reconstruction_head = nn.Linear(d_model, self.patch_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, num_features = inputs.shape
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, got {sequence_length}."
            )
        if num_features != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {num_features}.")

        patches = inputs.reshape(batch_size, self.num_patches, self.patch_dim)
        hidden = self.patch_projection(patches)
        hidden = self.positional_encoding(hidden)
        hidden = self.encoder(hidden)
        hidden = self.norm(hidden)
        reconstructed = self.reconstruction_head(self.dropout(hidden))
        return reconstructed.reshape(batch_size, sequence_length, num_features)
