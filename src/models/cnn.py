from __future__ import annotations

import torch
import torch.nn as nn


class VanillaCNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        conv_channels: list[int] | None = None,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        channels = conv_channels or [32, 64, 128]
        if len(channels) < 2:
            raise ValueError("conv_channels must contain at least two entries.")

        blocks: list[nn.Module] = []
        in_channels = input_channels
        for out_channels in channels:
            blocks.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.pool(features)
        return self.classifier(pooled)
