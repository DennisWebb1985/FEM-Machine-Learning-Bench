from __future__ import annotations

import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        output_size = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Linear(output_size, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, hidden = self.rnn(inputs)
        if self.rnn.bidirectional:
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            final_hidden = hidden[-1]
        return self.classifier(self.dropout(final_hidden))
