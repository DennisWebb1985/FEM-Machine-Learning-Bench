from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def compute_class_weights(targets: np.ndarray, num_classes: int) -> torch.Tensor:
    targets = np.asarray(targets)
    class_counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    class_counts[class_counts == 0.0] = 1.0
    weights = targets.size / (num_classes * class_counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_class_weights_with_mode(
    targets: np.ndarray,
    num_classes: int,
    mode: str = "inverse",
) -> torch.Tensor | None:
    if mode == "none":
        return None

    base_weights = compute_class_weights(targets=targets, num_classes=num_classes)
    if mode == "inverse":
        return base_weights
    if mode == "sqrt_inverse":
        return torch.sqrt(base_weights)
    raise ValueError(f"Unsupported class_weight_mode: {mode}")


def build_cross_entropy_loss(
    targets: np.ndarray,
    num_classes: int,
    device: torch.device,
    label_smoothing: float = 0.0,
    class_weight_mode: str = "inverse",
) -> nn.Module:
    weights = compute_class_weights_with_mode(
        targets,
        num_classes=num_classes,
        mode=class_weight_mode,
    )
    if weights is not None:
        weights = weights.to(device)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
