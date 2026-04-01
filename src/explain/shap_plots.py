from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_global_importance(
    feature_names: Sequence[str],
    importances: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    order = np.argsort(importances)[::-1]
    ordered_names = [feature_names[idx] for idx in order]
    ordered_values = importances[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(ordered_names)), ordered_values)
    ax.set_xticks(range(len(ordered_names)))
    ax.set_xticklabels(ordered_names, rotation=45, ha="right")
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title(title)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_time_feature_heatmap(
    values: np.ndarray,
    feature_names: Sequence[str],
    output_path: str | Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(values, aspect="auto", cmap="viridis")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Time Step")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_title(title)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _prepare_image_for_display(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return np.clip(array, 0.0, 1.0)


def _reduce_shap_map(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        signed_map = array.mean(axis=0)
        abs_map = np.abs(array).mean(axis=0)
        return signed_map, abs_map
    if array.ndim == 3 and array.shape[-1] in (1, 3):
        signed_map = array.mean(axis=-1)
        abs_map = np.abs(array).mean(axis=-1)
        return signed_map, abs_map
    if array.ndim == 2:
        return array, np.abs(array)
    raise ValueError(f"Unexpected SHAP image shape: {array.shape}")


def plot_image_shap_overlay(
    image: np.ndarray,
    shap_values: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    display_image = _prepare_image_for_display(image)
    signed_map, abs_map = _reduce_shap_map(shap_values)

    signed_scale = float(np.quantile(np.abs(signed_map), 0.995)) if signed_map.size else 0.0
    abs_scale = float(np.quantile(abs_map, 0.995)) if abs_map.size else 0.0
    signed_scale = signed_scale if signed_scale > 0 else 1e-6
    abs_scale = abs_scale if abs_scale > 0 else 1e-6

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    cmap = "gray" if display_image.ndim == 2 else None

    axes[0].imshow(display_image, cmap=cmap)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(display_image, cmap=cmap)
    signed_image = axes[1].imshow(
        signed_map,
        cmap="coolwarm",
        alpha=0.55,
        vmin=-signed_scale,
        vmax=signed_scale,
    )
    axes[1].set_title("Signed SHAP")
    axes[1].axis("off")
    fig.colorbar(signed_image, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(display_image, cmap=cmap)
    abs_image = axes[2].imshow(abs_map, cmap="inferno", alpha=0.65, vmin=0.0, vmax=abs_scale)
    axes[2].set_title("Mean |SHAP|")
    axes[2].axis("off")
    fig.colorbar(abs_image, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_image_shap_summary(
    mean_image: np.ndarray,
    importance_map: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    display_image = _prepare_image_for_display(mean_image)
    importance = np.asarray(importance_map, dtype=np.float32)
    scale = float(np.quantile(importance, 0.995)) if importance.size else 0.0
    scale = scale if scale > 0 else 1e-6

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    cmap = "gray" if display_image.ndim == 2 else None

    axes[0].imshow(display_image, cmap=cmap)
    axes[0].set_title("Mean Input")
    axes[0].axis("off")

    axes[1].imshow(display_image, cmap=cmap)
    image = axes[1].imshow(importance, cmap="magma", alpha=0.7, vmin=0.0, vmax=scale)
    axes[1].set_title("Mean |SHAP|")
    axes[1].axis("off")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
