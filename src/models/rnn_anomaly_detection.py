from __future__ import annotations

import torch
import torch.nn as nn


class RNNAutoencoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        latent_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.decoder = nn.RNN(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.output_projection = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(inputs)
        latent = self.to_latent(hidden[-1])
        repeated = latent.unsqueeze(1).expand(-1, inputs.size(1), -1)
        decoded, _ = self.decoder(repeated)
        return self.output_projection(self.dropout(decoded))
