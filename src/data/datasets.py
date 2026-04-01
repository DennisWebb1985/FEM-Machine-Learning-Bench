from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SequenceAugmentor:
    feature_indices: list[int] | None = None
    gaussian_noise_std: float = 0.0
    scale_std: float = 0.0
    feature_dropout_prob: float = 0.0
    time_mask_prob: float = 0.0
    mask_fill_value: float = 0.0

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        augmented = features.clone()
        if augmented.ndim != 2:
            return augmented

        indices = self.feature_indices or list(range(augmented.shape[-1]))
        if not indices:
            return augmented

        selected = augmented[:, indices]

        if self.gaussian_noise_std > 0.0:
            selected = selected + torch.randn_like(selected) * self.gaussian_noise_std

        if self.scale_std > 0.0:
            scale = 1.0 + torch.randn(selected.shape[-1], device=selected.device) * self.scale_std
            selected = selected * scale.unsqueeze(0)

        if self.feature_dropout_prob > 0.0:
            feature_mask = torch.rand(selected.shape[-1], device=selected.device) < self.feature_dropout_prob
            if feature_mask.any():
                selected[:, feature_mask] = self.mask_fill_value

        if self.time_mask_prob > 0.0:
            time_mask = torch.rand(selected.shape[0], device=selected.device) < self.time_mask_prob
            if time_mask.any():
                selected[time_mask] = self.mask_fill_value

        augmented[:, indices] = selected
        return augmented


class SequenceClassificationDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
        augmentor: SequenceAugmentor | None = None,
    ) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.long)
        self.metadata = metadata or [{} for _ in range(len(self.targets))]
        self.augmentor = augmentor

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features[index]
        if self.augmentor is not None:
            features = self.augmentor(features)
        return features, self.targets[index]

    def feature_array(self) -> np.ndarray:
        return self.features.numpy()

    def target_array(self) -> np.ndarray:
        return self.targets.numpy()


class SequenceRegressionDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
        augmentor: SequenceAugmentor | None = None,
    ) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.metadata = metadata or [{} for _ in range(len(self.targets))]
        self.augmentor = augmentor

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features[index]
        if self.augmentor is not None:
            features = self.augmentor(features)
        return features, self.targets[index]

    def input_array(self) -> np.ndarray:
        return self.features.numpy()

    def target_array(self) -> np.ndarray:
        return self.targets.numpy()
