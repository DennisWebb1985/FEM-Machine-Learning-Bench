from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(history: dict[str, list[float]], output_path: str | Path) -> None:
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    if not list(epochs):
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()

    axes[1].plot(epochs, history["train_macro_f1"], label="Train Macro-F1")
    axes[1].plot(epochs, history["val_macro_f1"], label="Val Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_title("Validation Performance")
    axes[1].legend()

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > threshold else "black"
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color=color)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(
    history: dict[str, list[float]],
    output_path: str | Path,
    title: str = "Loss Curve",
) -> None:
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    if not list(epochs):
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.plot(epochs, history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_anomaly_score_trend(
    file_scores: np.ndarray | Sequence[float],
    file_labels: Sequence[str],
    output_path: str | Path,
    title: str = "File-Level Anomaly Score",
) -> None:
    scores = np.asarray(file_scores, dtype=float)
    if len(scores) == 0:
        return

    x = np.arange(len(scores))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, scores, marker="o", linewidth=1.5)
    ax.set_xlabel("File Order")
    ax.set_ylabel("Mean Window Score")
    ax.set_title(title)

    tick_step = max(1, len(file_labels) // 12)
    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels([file_labels[index] for index in x[::tick_step]], rotation=45, ha="right")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_score_histogram(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    output_path: str | Path,
    title: str = "Window Score Distribution",
    log1p_transform: bool = False,
    clip_quantile: float | None = None,
) -> None:
    train_array = np.asarray(train_scores, dtype=float)
    val_array = np.asarray(val_scores, dtype=float)
    test_array = np.asarray(test_scores, dtype=float)

    if clip_quantile is not None:
        combined = np.concatenate([train_array, val_array, test_array])
        if len(combined) > 0:
            upper = float(np.quantile(combined, clip_quantile))
            train_array = np.clip(train_array, a_min=None, a_max=upper)
            val_array = np.clip(val_array, a_min=None, a_max=upper)
            test_array = np.clip(test_array, a_min=None, a_max=upper)

    if log1p_transform:
        train_array = np.log1p(train_array)
        val_array = np.log1p(val_array)
        test_array = np.log1p(test_array)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = 40
    ax.hist(train_array, bins=bins, alpha=0.5, label="Train", density=True)
    ax.hist(val_array, bins=bins, alpha=0.5, label="Val", density=True)
    ax.hist(test_array, bins=bins, alpha=0.5, label="Test", density=True)
    if log1p_transform:
        ax.set_xlabel("log1p(Window Score)")
    elif clip_quantile is not None:
        ax.set_xlabel(f"Window Score (clipped at q={clip_quantile:.3f})")
    else:
        ax.set_xlabel("Window Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_channel_error_bar(
    channel_names: Sequence[str],
    channel_errors: Sequence[float],
    output_path: str | Path,
    title: str = "Mean Channel Error",
) -> None:
    if not channel_names:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(channel_names))
    ax.bar(x, channel_errors)
    ax.set_xticks(x)
    ax.set_xticklabels(channel_names, rotation=45, ha="right")
    ax.set_ylabel("Mean Error")
    ax.set_title(title)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sequence_comparison(
    target_sequence: np.ndarray,
    predicted_sequence: np.ndarray,
    feature_names: Sequence[str],
    output_path: str | Path,
    title: str = "Sequence Reconstruction",
    max_features: int = 6,
    feature_selection: str = "largest_error",
) -> None:
    if target_sequence.size == 0 or predicted_sequence.size == 0:
        return

    per_feature_error = ((target_sequence - predicted_sequence) ** 2).mean(axis=0)
    per_feature_variance = np.var(target_sequence, axis=0)

    if feature_selection == "largest_error":
        top_indices = np.argsort(per_feature_error)[-max_features:]
    elif feature_selection == "lowest_error_high_variance":
        candidate_count = max(max_features, len(per_feature_error) // 2)
        candidate_indices = np.argsort(per_feature_error)[:candidate_count]
        ranked_candidates = candidate_indices[np.argsort(per_feature_variance[candidate_indices])]
        top_indices = ranked_candidates[-max_features:]
    else:
        raise ValueError(f"Unsupported feature_selection: {feature_selection}")

    top_indices = np.sort(top_indices)

    fig, axes = plt.subplots(len(top_indices), 1, figsize=(10, 2.2 * len(top_indices)), sharex=True)
    if len(top_indices) == 1:
        axes = [axes]

    time_axis = np.arange(target_sequence.shape[0])
    for axis, feature_index in zip(axes, top_indices):
        axis.plot(time_axis, target_sequence[:, feature_index], label="Target", linewidth=1.5)
        axis.plot(time_axis, predicted_sequence[:, feature_index], label="Prediction", linewidth=1.2)
        axis.set_ylabel(feature_names[feature_index])
        axis.legend(loc="upper right")

    axes[-1].set_xlabel("Step")
    fig.suptitle(title)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
