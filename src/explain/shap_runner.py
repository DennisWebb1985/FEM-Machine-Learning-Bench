from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.explain.shap_plots import (
    plot_global_importance,
    plot_image_shap_overlay,
    plot_image_shap_summary,
    plot_time_feature_heatmap,
)
from src.utils.io import write_dataframe

try:
    import shap
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    shap = None


def _normalize_shap_values(shap_values: Any, num_classes: int) -> list[np.ndarray]:
    if isinstance(shap_values, list):
        return [np.asarray(values) for values in shap_values]

    array_values = np.asarray(shap_values)
    if array_values.ndim >= 2 and array_values.shape[-1] == num_classes:
        return [array_values[..., class_idx] for class_idx in range(num_classes)]
    if array_values.ndim >= 1:
        return [array_values]

    raise ValueError(f"Unexpected SHAP output shape: {array_values.shape}")


def _collect_image_tensors(dataset: Any, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    images: list[torch.Tensor] = []
    targets: list[int] = []

    for index in range(min(len(dataset), max_samples)):
        image, target = dataset[index]
        images.append(image.detach().cpu())
        targets.append(int(target))

    if not images:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

    stacked = torch.stack(images).numpy()
    return stacked, np.asarray(targets, dtype=np.int64)


def _unnormalize_images(
    images: np.ndarray,
    normalize_mean: list[float] | None,
    normalize_std: list[float] | None,
) -> np.ndarray:
    if normalize_mean is None or normalize_std is None:
        return np.clip(images, 0.0, 1.0)

    mean = np.asarray(normalize_mean, dtype=np.float32).reshape(1, -1, 1, 1)
    std = np.asarray(normalize_std, dtype=np.float32).reshape(1, -1, 1, 1)
    return np.clip(images * std + mean, 0.0, 1.0)


def run_shap_analysis(
    model: torch.nn.Module,
    background_features: np.ndarray,
    explain_features: np.ndarray,
    feature_names: list[str],
    output_dir: str | Path,
    class_names: list[str],
    device: torch.device,
    prefix: str,
    max_background: int = 32,
    max_samples: int = 24,
) -> dict[str, Any]:
    if shap is None:
        raise ModuleNotFoundError("The 'shap' package is required for SHAP analysis.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(background_features) == 0 or len(explain_features) == 0:
        return {}

    # Run gradient-based SHAP on CPU for broader backend compatibility.
    analysis_device = torch.device("cpu")
    model_cpu = copy.deepcopy(model).to(analysis_device)
    background = torch.as_tensor(background_features[:max_background], dtype=torch.float32, device=analysis_device)
    explain = torch.as_tensor(explain_features[:max_samples], dtype=torch.float32, device=analysis_device)

    model_cpu.eval()
    with torch.no_grad():
        num_classes = int(model_cpu(background[:1]).shape[-1])

    explainer = shap.GradientExplainer(model_cpu, background)
    shap_values = explainer.shap_values(explain)
    shap_arrays = _normalize_shap_values(shap_values, num_classes=num_classes)

    overall_rows: list[pd.DataFrame] = []
    for class_idx, class_values in enumerate(shap_arrays):
        abs_values = np.abs(class_values)
        feature_importance = abs_values.mean(axis=(0, 1))
        mean_time_feature = abs_values.mean(axis=0)
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
        safe_name = class_name.lower().replace(" ", "_")

        frame = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": feature_importance,
                "class_name": class_name,
            }
        ).sort_values("mean_abs_shap", ascending=False)
        write_dataframe(frame, output_dir / f"{prefix}_shap_importance_{safe_name}.csv", index=False)
        overall_rows.append(frame)

        plot_global_importance(
            feature_names=feature_names,
            importances=feature_importance,
            output_path=output_dir / f"{prefix}_shap_global_{safe_name}.png",
            title=f"{prefix} SHAP Feature Importance - {class_name}",
        )
        plot_time_feature_heatmap(
            values=mean_time_feature,
            feature_names=feature_names,
            output_path=output_dir / f"{prefix}_shap_heatmap_{safe_name}.png",
            title=f"{prefix} Mean |SHAP| Heatmap - {class_name}",
        )
        np.save(output_dir / f"{prefix}_raw_shap_{safe_name}.npy", class_values)

    overall_frame = pd.concat(overall_rows, ignore_index=True)
    write_dataframe(overall_frame, output_dir / f"{prefix}_shap_importance_all_classes.csv", index=False)
    return {
        "analysis_device": str(analysis_device),
        "num_background": int(background.shape[0]),
        "num_explained": int(explain.shape[0]),
        "num_classes": num_classes,
    }


def run_image_shap_analysis(
    model: torch.nn.Module,
    background_dataset: Any,
    explain_dataset: Any,
    output_dir: str | Path,
    class_names: list[str],
    prefix: str,
    normalize_mean: list[float] | None = None,
    normalize_std: list[float] | None = None,
    explain_metadata: list[dict[str, Any]] | None = None,
    max_background: int = 16,
    max_samples: int = 8,
) -> dict[str, Any]:
    if shap is None:
        raise ModuleNotFoundError("The 'shap' package is required for SHAP analysis.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    background_images, _ = _collect_image_tensors(background_dataset, max_background)
    explain_images, explain_targets = _collect_image_tensors(explain_dataset, max_samples)
    if len(background_images) == 0 or len(explain_images) == 0:
        return {}

    model_cpu = copy.deepcopy(model).to(torch.device("cpu"))
    model_cpu.eval()

    background = torch.as_tensor(background_images, dtype=torch.float32)
    explain = torch.as_tensor(explain_images, dtype=torch.float32)

    with torch.no_grad():
        logits = model_cpu(explain)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        num_classes = int(logits.shape[-1])

    explainer = shap.GradientExplainer(model_cpu, background)
    shap_values = explainer.shap_values(explain)
    shap_arrays = _normalize_shap_values(shap_values, num_classes=num_classes)

    display_images = _unnormalize_images(explain_images, normalize_mean, normalize_std)
    mean_image = display_images.mean(axis=0)

    sample_rows: list[dict[str, Any]] = []
    safe_metadata = explain_metadata[: len(explain_images)] if explain_metadata is not None else [{}] * len(explain_images)

    for class_idx, class_values in enumerate(shap_arrays):
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
        safe_name = class_name.lower().replace(" ", "_")
        np.save(output_dir / f"{prefix}_raw_shap_{safe_name}.npy", class_values)

        mean_abs_map = np.abs(class_values).mean(axis=(0, 1))
        plot_image_shap_summary(
            mean_image=mean_image,
            importance_map=mean_abs_map,
            output_path=output_dir / f"{prefix}_shap_summary_{safe_name}.png",
            title=f"{prefix} Mean |SHAP| - {class_name}",
        )

    for sample_index, metadata in enumerate(safe_metadata):
        predicted_class = int(predictions[sample_index])
        true_class = int(explain_targets[sample_index])
        predicted_name = class_names[predicted_class]
        true_name = class_names[true_class]
        shap_for_prediction = shap_arrays[predicted_class][sample_index]
        source_name = Path(str(metadata.get("path", f"sample_{sample_index:02d}"))).stem
        output_name = f"{prefix}_shap_sample_{sample_index:02d}_{source_name}.png"

        plot_image_shap_overlay(
            image=display_images[sample_index],
            shap_values=shap_for_prediction,
            output_path=output_dir / output_name,
            title=(
                f"{prefix} sample {sample_index} | true={true_name} | "
                f"pred={predicted_name} ({probabilities[sample_index, predicted_class]:.3f})"
            ),
        )

        row = {
            **metadata,
            "y_true": true_class,
            "y_true_name": true_name,
            "y_pred": predicted_class,
            "y_pred_name": predicted_name,
            "explained_class": predicted_name,
            "shap_image": output_name,
        }
        for class_idx, class_name in enumerate(class_names):
            safe_name = class_name.lower().replace(" ", "_")
            row[f"prob_{safe_name}"] = float(probabilities[sample_index, class_idx])
        sample_rows.append(row)

    write_dataframe(pd.DataFrame(sample_rows), output_dir / f"{prefix}_shap_samples.csv", index=False)
    return {
        "analysis_device": "cpu",
        "num_background": int(background.shape[0]),
        "num_explained": int(explain.shape[0]),
        "num_classes": int(num_classes),
        "sample_outputs": [row["shap_image"] for row in sample_rows],
    }
