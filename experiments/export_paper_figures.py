from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.runtime import prepare_environment

REPO_ROOT = prepare_environment(ROOT)

from src.data.anomaly_detection_preprocess import (
    build_anomaly_reconstruction_windows,
    prepare_anomaly_detection_data,
)
from src.data.classification_labels import class_names_from_labels
from src.data.datasets import SequenceRegressionDataset
from src.models.lstm_anomaly_detection_autoencoder import LSTMAutoencoder
from src.models.rnn_anomaly_detection import RNNAutoencoder
from src.models.transformer_anomaly_detection import TransformerAutoencoder
from src.training.anomaly_engine import evaluate_sequence_model, representative_window_index
from src.utils.config import load_config

OUTPUT_ROOT = REPO_ROOT / "outputs" / "paper_figures"

MLP_RUN = REPO_ROOT / "outputs" / "mlp"
CNN_RUN = REPO_ROOT / "outputs" / "cnn"
BUILDING_RUN_SPECS = {
    "RNN": {
        "config": REPO_ROOT / "experiments/configs/rnn_classification.yaml",
        "required_metadata": ["auto_window_selection.json"],
    },
    "LSTM": {
        "config": REPO_ROOT / "experiments/configs/lstm_classification.yaml",
        "required_metadata": ["auto_window_selection.json"],
    },
    "Transformer": {
        "config": REPO_ROOT / "experiments/configs/transformer_classification.yaml",
        "required_metadata": ["auto_window_selection.json", "auto_hparam_selection.json"],
    },
}
ANOMALY_RUN_SPECS = {
    "RNN": {
        "config": REPO_ROOT / "experiments/configs/rnn_anomaly_detection.yaml",
        "required_metadata": ["auto_window_selection.json"],
    },
    "LSTM": {
        "config": REPO_ROOT / "experiments/configs/lstm_anomaly_detection_autoencoder.yaml",
        "required_metadata": ["auto_window_selection.json"],
    },
    "Transformer": {
        "config": REPO_ROOT / "experiments/configs/transformer_anomaly_detection.yaml",
        "required_metadata": ["auto_window_selection.json"],
    },
}
ANOMALY_MODEL_KEYS = {
    "RNN": "rnn",
    "LSTM": "lstm_autoencoder",
    "Transformer": "transformer",
}
BUILDING_RUNS: dict[str, Path] = {}
ANOMALY_RUNS: dict[str, Path] = {}
SOURCE_SELECTION_NOTES: list[str] = []
OPTIONAL_EXPORT_NOTES: list[str] = []

MODEL_COLORS = {
    "RNN": "#3B6EA8",
    "LSTM": "#D17C2F",
    "Transformer": "#4E8B6A",
}
CLASS_COLORS = {
    "No crack": "#3B6EA8",
    "Crack": "#B24C44",
}
BUILDING_CLASS_LABELS = class_names_from_labels([0, 1, 2])
CONFUSION_CMAP = LinearSegmentedColormap.from_list(
    "paper_blues",
    ["#F7FBFF", "#D8E7F3", "#88AEC7", "#2F6D91"],
)
IMPORTANCE_CMAP = LinearSegmentedColormap.from_list(
    "paper_magma",
    ["#FFF8ED", "#F6C290", "#D66D5A", "#8B3F62", "#2A1D39"],
)


def configure_matplotlib() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 400,
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "stix",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#4C4C4C",
            "axes.labelcolor": "#222222",
            "text.color": "#161616",
            "axes.titleweight": "semibold",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "grid.color": "#DADADA",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.55,
            "axes.grid": True,
            "axes.axisbelow": True,
        }
    )


def ensure_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def export_optional(section_name: str, exporter: Callable[[], list[Path]]) -> list[Path]:
    try:
        return exporter()
    except FileNotFoundError as exc:
        OPTIONAL_EXPORT_NOTES.append(f"Skipped {section_name}: {exc}")
        return []


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def style_axis(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis is None:
        ax.grid(False)
    elif grid_axis == "both":
        ax.grid(True)
    else:
        ax.grid(True, axis=grid_axis)


def load_json(path: Path) -> dict:
    return json.loads(ensure_exists(path).read_text(encoding="utf-8"))


def resolve_source_runs() -> tuple[dict[str, Path], dict[str, Path], list[str]]:
    building_runs = {
        label: resolve_classification_run(label, spec["config"], spec["required_metadata"])
        for label, spec in BUILDING_RUN_SPECS.items()
    }
    anomaly_runs = {
        label: resolve_anomaly_run(label, spec["config"], spec["required_metadata"])
        for label, spec in ANOMALY_RUN_SPECS.items()
    }

    notes = [
        summarize_run_selection("Classification", building_runs),
        summarize_run_selection("Anomaly-detection", anomaly_runs),
    ]
    return building_runs, anomaly_runs, notes


def resolve_classification_run(label: str, config_path: Path, required_metadata: list[str]) -> Path:
    shap_prefix = f"{label.lower()}_classification"
    required_files = [
        "metrics.json",
        "predictions.csv",
        "resolved_config.yaml",
        f"{shap_prefix}_shap_importance_all_classes.csv",
    ]
    return resolve_run_from_config(
        label=label,
        config_path=config_path,
        required_files=required_files,
        required_metadata=required_metadata,
        task_name="classification",
    )


def resolve_anomaly_run(label: str, config_path: Path, required_metadata: list[str]) -> Path:
    required_files = [
        "best_model.pt",
        "metrics.json",
        "file_scores.csv",
        "channel_error.csv",
        "resolved_config.yaml",
        "example_low_score.png",
        "example_high_score.png",
    ]
    return resolve_run_from_config(
        label=label,
        config_path=config_path,
        required_files=required_files,
        required_metadata=required_metadata,
        task_name="anomaly_detection",
    )


def resolve_run_from_config(
    *,
    label: str,
    config_path: Path,
    required_files: list[str],
    required_metadata: list[str],
    task_name: str,
) -> Path:
    configured_run_dir = configured_output_dir(config_path)
    missing_files = [file_name for file_name in required_files if not (configured_run_dir / file_name).exists()]
    missing_metadata = [file_name for file_name in required_metadata if not (configured_run_dir / file_name).exists()]
    if not missing_files and not missing_metadata:
        return configured_run_dir

    missing = ", ".join([*missing_files, *missing_metadata])
    raise FileNotFoundError(
        f"Could not resolve the required final auto-selected run for {task_name}/{label}. "
        f"Checked configured output '{configured_run_dir}'. Missing: {missing}."
    )


def configured_output_dir(config_path: Path) -> Path:
    config = load_config(config_path)
    return REPO_ROOT / config["paths"]["output_dir"]

def summarize_run_selection(task_label: str, runs: dict[str, Path]) -> str:
    entries = []
    window_lengths = {}
    source_labels = {}

    for label, run_dir in runs.items():
        resolved_config = load_config(run_dir / "resolved_config.yaml")
        window_length = int(resolved_config["window"]["length"])
        window_lengths[label] = window_length
        source_labels[label] = describe_run_source(run_dir)
        entries.append(
            f"{label} -> {run_dir.relative_to(REPO_ROOT)} ({source_labels[label]}, window {window_length})"
        )

    if len(set(window_lengths.values())) == 1:
        shared_window = next(iter(window_lengths.values()))
        return (
            f"{task_label} source selection: "
            + "; ".join(entries)
            + f". Shared window length = {shared_window}."
        )

    per_model_windows = ", ".join(f"{label}={window}" for label, window in window_lengths.items())
    return (
        f"{task_label} source selection: "
        + "; ".join(entries)
        + f". Model-specific window lengths: {per_model_windows}."
    )


def describe_run_source(run_dir: Path) -> str:
    auto_window_path = run_dir / "auto_window_selection.json"
    auto_window = load_json(auto_window_path)
    metric = auto_window.get("selection_metric", "selection metric")
    window_length = auto_window.get("best_window_length")

    auto_hparam_path = run_dir / "auto_hparam_selection.json"
    if auto_hparam_path.exists():
        auto_hparams = load_json(auto_hparam_path)
        profile = auto_hparams.get("best_profile", "selected profile")
        return (
            f"auto-window final run selected by {metric} ({window_length}); "
            f"auto-hparams profile={profile}"
        )

    return f"auto-window final run selected by {metric} ({window_length})"


def compute_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, labels: list[int]) -> np.ndarray:
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    index = {label: idx for idx, label in enumerate(labels)}
    for truth, pred in zip(y_true, y_pred):
        matrix[index[int(truth)], index[int(pred)]] += 1
    return matrix


def draw_confusion_matrix(
    ax: plt.Axes,
    matrix: np.ndarray,
    class_labels: list[str],
    title: str,
) -> matplotlib.image.AxesImage:
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)

    image = ax.imshow(normalized, cmap=CONFUSION_CMAP, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=25, ha="right")
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.grid(False)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            proportion = normalized[row, col] * 100.0
            text_color = "white" if normalized[row, col] >= 0.55 else "#1B1B1B"
            ax.text(
                col,
                row,
                f"{matrix[row, col]}\n{proportion:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    return image


def read_rgb_image(path: str | Path) -> np.ndarray:
    image = mpimg.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    return image


def format_file_timestamp_label(file_name: str) -> str:
    stem = Path(file_name).stem
    try:
        timestamp = pd.to_datetime(stem, format="%Y_%m_%dT%H_%M_%S")
        return timestamp.strftime("%m-%d\n%H:%M")
    except (ValueError, TypeError):
        return stem


def build_anomaly_model(model_key: str, config: dict[str, Any], input_size: int) -> torch.nn.Module:
    if model_key == "rnn":
        return RNNAutoencoder(
            input_size=input_size,
            hidden_size=config["model"]["hidden_size"],
            latent_size=config["model"]["latent_size"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
        )
    if model_key == "lstm_autoencoder":
        return LSTMAutoencoder(
            input_size=input_size,
            hidden_size=config["model"]["hidden_size"],
            latent_size=config["model"]["latent_size"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
        )
    if model_key == "transformer":
        return TransformerAutoencoder(
            input_size=input_size,
            sequence_length=config["window"]["length"],
            patch_size=config["model"]["patch_size"],
            d_model=config["model"]["d_model"],
            nhead=config["model"]["nhead"],
            num_layers=config["model"]["num_layers"],
            dim_feedforward=config["model"]["dim_feedforward"],
            dropout=config["model"]["dropout"],
        )
    raise ValueError(f"Unsupported anomaly model key: {model_key}")


def anomaly_window_key(metadata_row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        metadata_row["file_name"],
        int(metadata_row["start_index"]),
        int(metadata_row["end_index"]),
        metadata_row["start_ts"],
        metadata_row["end_ts"],
    )


def load_anomaly_reconstruction_bundle(model_key: str, run_dir: Path) -> dict[str, Any]:
    config = load_config(run_dir / "resolved_config.yaml")
    prepared = prepare_anomaly_detection_data(
        data_dir=REPO_ROOT / config["paths"]["data_dir"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        drop_channels=config["data"]["drop_channels"],
        missing_sentinel=config["data"]["missing_sentinel"],
        extreme_value_threshold=config["data"]["extreme_value_threshold"],
        downsample_factor=config["data"]["downsample_factor"],
    )
    test_x, test_y, test_meta = build_anomaly_reconstruction_windows(
        prepared["test"],
        prepared["feature_cols"],
        window_length=config["window"]["length"],
        stride=config["window"]["stride"],
        drop_nan_windows=config["window"]["drop_nan_windows"],
    )

    test_ds = SequenceRegressionDataset(test_x, test_y, test_meta)
    test_loader = DataLoader(
        test_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 0),
    )

    device = torch.device("cpu")
    model = build_anomaly_model(model_key=model_key, config=config, input_size=test_x.shape[-1]).to(device)
    state_dict = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    criterion = torch.nn.MSELoss()

    test_eval = evaluate_sequence_model(model, test_loader, criterion, device, collect_outputs=True)
    low_index = representative_window_index(test_eval["window_scores"], quantile=0.10)
    high_index = representative_window_index(test_eval["window_scores"], quantile=0.95)
    if low_index is None or high_index is None:
        raise ValueError(f"Unable to determine representative windows for {run_dir}")

    window_index_by_key = {anomaly_window_key(row): idx for idx, row in enumerate(test_meta)}
    return {
        "feature_names": prepared["feature_cols"],
        "metadata": test_meta,
        "window_scores": test_eval["window_scores"],
        "targets": test_eval["targets"],
        "predictions": test_eval["predictions"],
        "window_index_by_key": window_index_by_key,
        "low_target": test_eval["targets"][low_index],
        "low_prediction": test_eval["predictions"][low_index],
        "high_target": test_eval["targets"][high_index],
        "high_prediction": test_eval["predictions"][high_index],
    }


def select_shared_feature_indices(
    bundles: dict[str, dict[str, Any]],
    row_key: str,
    max_features: int = 6,
) -> list[int]:
    per_model_error = []
    per_model_variance = []
    for bundle in bundles.values():
        target = bundle[f"{row_key}_target"]
        prediction = bundle[f"{row_key}_prediction"]
        per_model_error.append(((target - prediction) ** 2).mean(axis=0))
        per_model_variance.append(np.var(target, axis=0))

    mean_error = np.mean(np.stack(per_model_error, axis=0), axis=0)
    mean_variance = np.mean(np.stack(per_model_variance, axis=0), axis=0)

    if row_key == "low":
        candidate_count = max(max_features, len(mean_error) // 2)
        candidate_indices = np.argsort(mean_error)[:candidate_count]
        ranked_candidates = candidate_indices[np.argsort(mean_variance[candidate_indices])]
        selected = ranked_candidates[-max_features:]
    elif row_key == "high":
        selected = np.argsort(mean_error)[-max_features:]
    else:
        raise ValueError(f"Unsupported row key: {row_key}")

    return np.sort(selected).tolist()


def compute_shared_feature_limits(
    bundles: dict[str, dict[str, Any]],
    row_key: str,
    feature_indices: list[int],
) -> dict[int, tuple[float, float]]:
    limits: dict[int, tuple[float, float]] = {}
    for feature_index in feature_indices:
        values = []
        for bundle in bundles.values():
            values.append(bundle[f"{row_key}_target"][:, feature_index])
            values.append(bundle[f"{row_key}_prediction"][:, feature_index])
        combined = np.concatenate(values)
        lower = float(combined.min())
        upper = float(combined.max())
        if np.isclose(lower, upper):
            padding = 0.1 if np.isclose(lower, 0.0) else abs(lower) * 0.1
        else:
            padding = (upper - lower) * 0.08
        limits[feature_index] = (lower - padding, upper + padding)
    return limits


def select_shared_window_keys(
    bundles: dict[str, dict[str, Any]],
    low_quantile: float = 0.10,
    high_quantile: float = 0.95,
) -> dict[str, tuple[Any, ...]]:
    ordered_keys = [anomaly_window_key(row) for row in next(iter(bundles.values()))["metadata"]]
    key_sets = [{anomaly_window_key(row) for row in bundle["metadata"]} for bundle in bundles.values()]
    common_keys = [key for key in ordered_keys if all(key in key_set for key_set in key_sets)]
    if not common_keys:
        raise ValueError("No shared anomaly-detection reconstruction windows were found across models.")

    percentile_rows = []
    for bundle in bundles.values():
        scores = np.asarray(
            [bundle["window_scores"][bundle["window_index_by_key"][key]] for key in common_keys],
            dtype=float,
        )
        percentile_rows.append(pd.Series(scores).rank(method="average", pct=True).to_numpy())

    mean_percentiles = np.mean(np.stack(percentile_rows, axis=0), axis=0)
    low_idx = int(np.argmin(np.abs(mean_percentiles - low_quantile)))
    high_idx = int(np.argmin(np.abs(mean_percentiles - high_quantile)))
    if high_idx == low_idx and len(common_keys) > 1:
        ranked = np.argsort(np.abs(mean_percentiles - high_quantile))
        high_idx = int(next(idx for idx in ranked if int(idx) != low_idx))

    return {
        "low": common_keys[low_idx],
        "high": common_keys[high_idx],
    }


def select_shared_feature_indices_for_window(
    bundles: dict[str, dict[str, Any]],
    window_key: tuple[Any, ...],
    row_key: str,
    max_features: int = 6,
) -> list[int]:
    per_model_error = []
    reference_bundle = next(iter(bundles.values()))
    reference_index = reference_bundle["window_index_by_key"][window_key]
    reference_target = reference_bundle["targets"][reference_index]
    reference_variance = np.var(reference_target, axis=0)

    for bundle in bundles.values():
        window_index = bundle["window_index_by_key"][window_key]
        target = bundle["targets"][window_index]
        prediction = bundle["predictions"][window_index]
        per_model_error.append(((target - prediction) ** 2).mean(axis=0))

    mean_error = np.mean(np.stack(per_model_error, axis=0), axis=0)

    if row_key == "low":
        candidate_count = max(max_features, len(mean_error) // 2)
        candidate_indices = np.argsort(mean_error)[:candidate_count]
        ranked_candidates = candidate_indices[np.argsort(reference_variance[candidate_indices])]
        selected = ranked_candidates[-max_features:]
    elif row_key == "high":
        selected = np.argsort(mean_error)[-max_features:]
    else:
        raise ValueError(f"Unsupported row key: {row_key}")

    return np.sort(selected).tolist()


def compute_shared_feature_limits_for_window(
    bundles: dict[str, dict[str, Any]],
    window_key: tuple[Any, ...],
    feature_indices: list[int],
) -> dict[int, tuple[float, float]]:
    limits: dict[int, tuple[float, float]] = {}
    for feature_index in feature_indices:
        values = []
        for bundle in bundles.values():
            window_index = bundle["window_index_by_key"][window_key]
            values.append(bundle["targets"][window_index][:, feature_index])
            values.append(bundle["predictions"][window_index][:, feature_index])
        combined = np.concatenate(values)
        lower = float(combined.min())
        upper = float(combined.max())
        if np.isclose(lower, upper):
            padding = 0.1 if np.isclose(lower, 0.0) else abs(lower) * 0.1
        else:
            padding = (upper - lower) * 0.08
        limits[feature_index] = (lower - padding, upper + padding)
    return limits


def export_building_model_performance() -> list[Path]:
    rows: list[dict[str, float | str]] = []
    for model_name, run_dir in BUILDING_RUNS.items():
        metrics = load_json(run_dir / "metrics.json")["test_metrics"]
        rows.append(
            {
                "model": model_name,
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "ovr_auroc": float(metrics["ovr_auroc"]),
            }
        )

    frame = pd.DataFrame(rows)
    metrics_to_plot = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("balanced_accuracy", "Balanced Acc."),
        ("ovr_auroc", "AUROC"),
    ]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(10.8, 3.3), sharey=True)
    for ax, (metric_key, metric_label) in zip(axes, metrics_to_plot):
        bars = ax.bar(
            frame["model"],
            frame[metric_key],
            color=[MODEL_COLORS[model] for model in frame["model"]],
            width=0.64,
        )
        ax.set_title(metric_label)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("")
        style_axis(ax, grid_axis="y")
        for bar, value in zip(bars, frame[metric_key]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.02, 0.98),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axes[0].set_ylabel("Score")
    fig.suptitle("Classification Models: Sequence Model Comparison", y=1.02)
    fig.tight_layout()
    return save_figure(fig, OUTPUT_ROOT / "classification_comparison" / "classification_model_performance")


def export_building_confusion_matrices() -> list[Path]:
    fig, axes = plt.subplots(1, len(BUILDING_RUNS), figsize=(11.6, 3.9))
    image = None

    for idx, (ax, (model_name, run_dir)) in enumerate(zip(axes, BUILDING_RUNS.items())):
        predictions = pd.read_csv(ensure_exists(run_dir / "predictions.csv"))
        matrix = compute_confusion_matrix(predictions["y_true"], predictions["y_pred"], labels=[0, 1, 2])
        image = draw_confusion_matrix(ax, matrix, BUILDING_CLASS_LABELS, model_name)
        if idx > 0:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False)

    if image is not None:
        fig.subplots_adjust(left=0.08, right=0.88, top=0.84, bottom=0.18, wspace=0.28)
        colorbar = fig.colorbar(image, cax=fig.add_axes([0.90, 0.22, 0.016, 0.56]))
        colorbar.set_label("Row-normalized proportion")

    fig.suptitle("Classification Models: Normalized Confusion Matrices", y=1.03)
    return save_figure(fig, OUTPUT_ROOT / "classification_comparison" / "classification_confusion_matrices")


def export_building_feature_importance() -> list[Path]:
    rows = {}
    for model_name, run_dir in BUILDING_RUNS.items():
        csv_path = run_dir / f"{model_name.lower()}_classification_shap_importance_all_classes.csv"
        frame = pd.read_csv(ensure_exists(csv_path))
        rows[model_name] = frame.groupby("feature")["mean_abs_shap"].mean()

    data = pd.DataFrame(rows).T.fillna(0.0)
    ordered_features = data.mean(axis=0).sort_values(ascending=False).index.tolist()
    data = data[ordered_features]
    normalized = data.div(data.max(axis=1).replace(0.0, 1.0), axis=0)

    fig, ax = plt.subplots(figsize=(10.2, 3.6))
    image = ax.imshow(normalized.to_numpy(), cmap=CONFUSION_CMAP, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(ordered_features)))
    ax.set_xticklabels(ordered_features, rotation=25, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index.tolist())
    ax.set_xlabel("Sensor feature")
    ax.set_title("Classification Models: Relative SHAP Importance by Model")
    ax.grid(False)

    for row_idx, model_name in enumerate(data.index):
        for col_idx, feature_name in enumerate(ordered_features):
            raw_value = data.loc[model_name, feature_name]
            text_color = "white" if normalized.loc[model_name, feature_name] >= 0.55 else "#1B1B1B"
            ax.text(col_idx, row_idx, f"{raw_value:.3f}", ha="center", va="center", fontsize=8, color=text_color)

    fig.subplots_adjust(left=0.10, right=0.90, top=0.84, bottom=0.26)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.025)
    colorbar.set_label("Relative importance within model", labelpad=12)
    return save_figure(
        fig,
        OUTPUT_ROOT / "classification_comparison" / "classification_feature_importance_heatmap",
    )


def export_mlp_parity_grid() -> list[Path]:
    predictions = pd.read_csv(ensure_exists(MLP_RUN / "predictions.csv"))
    metrics = pd.read_csv(ensure_exists(MLP_RUN / "test_metrics.csv")).set_index("output")
    targets = ["Mr_t", "Mt_t", "Mr_c", "Mt_c"]

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2))
    hexbins = []

    for ax, target_name in zip(axes.flat, targets):
        actual = predictions[f"actual_{target_name}"].to_numpy()
        predicted = predictions[f"pred_{target_name}"].to_numpy()
        lower = float(min(actual.min(), predicted.min()))
        upper = float(max(actual.max(), predicted.max()))
        padding = 0.06 * (upper - lower) if upper > lower else 1.0

        hexbin = ax.hexbin(
            actual,
            predicted,
            gridsize=28,
            mincnt=1,
            cmap="Blues",
            linewidths=0.0,
        )
        hexbins.append(hexbin)
        ax.plot(
            [lower - padding, upper + padding],
            [lower - padding, upper + padding],
            linestyle="--",
            linewidth=1.4,
            color="#B24C44",
        )
        ax.set_xlim(lower - padding, upper + padding)
        ax.set_ylim(lower - padding, upper + padding)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(target_name)
        ax.set_xlabel(f"Actual {target_name}")
        ax.set_ylabel(f"Predicted {target_name}")
        style_axis(ax, grid_axis="both")

        metric_row = metrics.loc[target_name]
        ax.text(
            0.04,
            0.96,
            f"R2 = {metric_row['R2']:.3f}\nRMSE = {metric_row['RMSE']:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#D8D8D8", "boxstyle": "round,pad=0.25"},
        )

    fig.subplots_adjust(left=0.08, right=0.86, top=0.90, bottom=0.08, wspace=0.24, hspace=0.28)
    colorbar = fig.colorbar(hexbins[-1], cax=fig.add_axes([0.88, 0.20, 0.02, 0.58]))
    colorbar.set_label("Sample count")
    fig.suptitle("FEM Surrogate MLP: Parity Plot Grid", y=0.99)
    return save_figure(fig, OUTPUT_ROOT / "mlp" / "mlp_parity_grid")


def export_mlp_feature_importance() -> list[Path]:
    targets = ["Mr_t", "Mt_t", "Mr_c", "Mt_c"]
    columns = {}
    for target_name in targets:
        frame = pd.read_csv(ensure_exists(MLP_RUN / f"shap_importance_{target_name}.csv"))
        columns[target_name] = frame.set_index("feature")["mean_abs_shap"]

    data = pd.DataFrame(columns).fillna(0.0)
    ordered_features = data.mean(axis=1).sort_values(ascending=False).index.tolist()
    data = data.loc[ordered_features, targets]

    fig_height = max(3.4, 0.42 * len(ordered_features) + 1.4)
    fig, ax = plt.subplots(figsize=(6.6, fig_height))
    image = ax.imshow(data.to_numpy(), cmap=IMPORTANCE_CMAP, aspect="auto")
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets)
    ax.set_yticks(range(len(ordered_features)))
    ax.set_yticklabels(ordered_features)
    ax.set_xlabel("Target response")
    ax.set_title("FEM Surrogate MLP: Mean |SHAP| by Input Feature")
    ax.grid(False)

    max_value = float(data.to_numpy().max()) if len(data) else 0.0
    threshold = 0.55 * max_value if max_value > 0 else 0.0
    for row_idx, feature_name in enumerate(ordered_features):
        for col_idx, target_name in enumerate(targets):
            value = float(data.loc[feature_name, target_name])
            text_color = "white" if value >= threshold else "#1B1B1B"
            ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=8, color=text_color)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Mean |SHAP|")
    fig.tight_layout()
    return save_figure(fig, OUTPUT_ROOT / "mlp" / "mlp_shap_importance_heatmap")


def export_cnn_performance_overview() -> list[Path]:
    predictions = pd.read_csv(ensure_exists(CNN_RUN / "predictions.csv"))
    metrics = load_json(CNN_RUN / "metrics.json")["test_metrics"]
    matrix = compute_confusion_matrix(predictions["y_true"], predictions["y_pred"], labels=[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), gridspec_kw={"width_ratios": [1.0, 1.2]})
    image = draw_confusion_matrix(axes[0], matrix, ["No crack", "Crack"], "Normalized confusion matrix")
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label("Row-normalized proportion")

    bins = np.linspace(0.0, 1.0, 40)
    no_crack = predictions.loc[predictions["y_true"] == 0, "prob_positive"]
    crack = predictions.loc[predictions["y_true"] == 1, "prob_positive"]
    axes[1].hist(
        no_crack,
        bins=bins,
        density=True,
        alpha=0.55,
        color=CLASS_COLORS["No crack"],
        label="True no crack",
    )
    axes[1].hist(
        crack,
        bins=bins,
        density=True,
        alpha=0.55,
        color=CLASS_COLORS["Crack"],
        label="True crack",
    )
    axes[1].axvline(0.5, linestyle="--", linewidth=1.2, color="#4C4C4C")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel("Predicted probability of crack")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Positive-class probability distribution")
    style_axis(axes[1], grid_axis="y")
    axes[1].legend(loc="upper left", ncol=2)
    axes[1].text(
        0.98,
        0.80,
        f"Accuracy = {metrics['accuracy']:.4f}\nMacro-F1 = {metrics['macro_f1']:.4f}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#D8D8D8", "boxstyle": "round,pad=0.25"},
    )

    fig.suptitle("Concrete Crack CNN: Performance Overview", y=1.02)
    fig.tight_layout()
    return save_figure(fig, OUTPUT_ROOT / "cnn" / "cnn_performance_overview")


def export_cnn_shap_summary() -> list[Path]:
    samples = pd.read_csv(ensure_exists(CNN_RUN / "cnn_shap_samples.csv"))
    class_specs = [
        ("Negative", "No crack", CNN_RUN / "cnn_raw_shap_negative.npy"),
        ("Positive", "Crack", CNN_RUN / "cnn_raw_shap_positive.npy"),
    ]

    mean_inputs: dict[str, np.ndarray] = {}
    importance_maps: dict[str, np.ndarray] = {}
    scales: list[float] = []

    for explained_name, display_name, shap_path in class_specs:
        subset = samples.loc[samples["explained_class"] == explained_name]
        images = np.stack([read_rgb_image(path) for path in subset["path"]], axis=0)
        shap_values = np.load(ensure_exists(shap_path))
        mean_inputs[display_name] = images.mean(axis=0)
        importance = np.abs(shap_values).mean(axis=(0, 1))
        importance_maps[display_name] = importance
        scales.append(float(np.quantile(importance, 0.995)))

    vmax = max(scales) if scales else 1.0
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.5))

    last_image = None
    for row_idx, (_, display_name, _) in enumerate(class_specs):
        input_ax = axes[row_idx, 0]
        overlay_ax = axes[row_idx, 1]

        input_ax.imshow(mean_inputs[display_name])
        input_ax.set_title(f"{display_name}: mean input")
        input_ax.axis("off")

        overlay_ax.imshow(mean_inputs[display_name])
        last_image = overlay_ax.imshow(
            importance_maps[display_name],
            cmap=IMPORTANCE_CMAP,
            alpha=0.72,
            vmin=0.0,
            vmax=vmax,
        )
        overlay_ax.set_title(f"{display_name}: mean |SHAP| overlay")
        overlay_ax.axis("off")

    if last_image is not None:
        fig.subplots_adjust(left=0.05, right=0.88, top=0.90, bottom=0.06, wspace=0.08, hspace=0.12)
        colorbar = fig.colorbar(last_image, cax=fig.add_axes([0.90, 0.25, 0.016, 0.42]))
        colorbar.set_label("Mean |SHAP|")

    fig.suptitle("Concrete Crack CNN: Classwise Saliency Summary", y=0.98)
    return save_figure(fig, OUTPUT_ROOT / "cnn" / "cnn_shap_summary")


def export_anomaly_model_summary() -> list[Path]:
    rows: list[dict[str, float | str]] = []
    for label, run_dir in ANOMALY_RUNS.items():
        metrics = load_json(run_dir / "metrics.json")
        rows.append(
            {
                "model": label,
                "best_val_loss": float(metrics["best_val_loss"]),
                "test_median_score": float(metrics["score_summary"]["test"]["median_score"]),
                "test_q95_score": float(metrics["score_summary"]["test"]["q95"]),
                "test_exceedance_q95": float(metrics["threshold_summary"]["test_exceedance_q95"]),
            }
        )

    frame = pd.DataFrame(rows)
    metrics_to_plot = [
        ("best_val_loss", "Best val. loss"),
        ("test_median_score", "Test median score"),
        ("test_q95_score", "Test q95 score"),
        ("test_exceedance_q95", "Test exceedance @ q95"),
    ]
    color_lookup = {label: MODEL_COLORS[label] for label in ANOMALY_RUNS}

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(11.3, 3.4))
    for ax, (metric_key, metric_label) in zip(axes, metrics_to_plot):
        values = frame[metric_key].to_numpy()
        bars = ax.bar(
            frame["model"],
            values,
            color=[color_lookup[model] for model in frame["model"]],
            width=0.64,
        )
        ax.set_title(metric_label)
        ax.set_xlabel("")
        style_axis(ax, grid_axis="y")
        ax.tick_params(axis="x", rotation=15)
        ax.margins(y=0.15)
        for bar, value in zip(bars, values):
            label_text = f"{value:.3f}" if value < 10 else f"{value:.1f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.03 if value > 0 else 0.01,
                label_text,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axes[0].set_ylabel("Value")
    fig.suptitle("Anomaly Detection Models: Reconstruction and Detection Summary", y=1.02)
    fig.tight_layout()
    return save_figure(
        fig,
        OUTPUT_ROOT / "anomaly_detection_comparison" / "anomaly_detection_model_summary",
    )


def export_anomaly_mean_score_trends() -> list[Path]:
    model_frames: dict[str, pd.DataFrame] = {}
    ordered_files: list[str] | None = None

    for label, run_dir in ANOMALY_RUNS.items():
        frame = pd.read_csv(ensure_exists(run_dir / "file_scores.csv"))
        test_frame = (
            frame.loc[frame["split"] == "test", ["file_order", "file_name", "mean_score"]]
            .sort_values("file_order")
            .reset_index(drop=True)
        )
        model_frames[label] = test_frame

        current_files = test_frame["file_name"].tolist()
        if ordered_files is None:
            ordered_files = current_files
        elif current_files != ordered_files:
            raise ValueError("Test file ordering is inconsistent across anomaly-detection runs.")

    if ordered_files is None:
        raise ValueError("No anomaly-detection test files were found.")

    mean_score_frame = pd.DataFrame(
        {
            label: frame.set_index("file_name")["mean_score"].reindex(ordered_files)
            for label, frame in model_frames.items()
        }
    )
    x = np.arange(len(ordered_files))
    tick_labels = [format_file_timestamp_label(name) for name in ordered_files]
    global_min = float(mean_score_frame.to_numpy().min())
    global_max = float(mean_score_frame.to_numpy().max())

    fig, axes = plt.subplots(len(ANOMALY_RUNS), 1, figsize=(11.2, 8.2), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for ax, (label, _) in zip(axes, ANOMALY_RUNS.items()):
        values = mean_score_frame[label].to_numpy(dtype=float)
        max_idx = int(np.argmax(values))
        median_value = float(np.median(values))
        top_file = ordered_files[max_idx]

        ax.plot(
            x,
            values,
            color=MODEL_COLORS[label],
            marker="o",
            markersize=5.5,
            linewidth=1.9,
        )
        ax.scatter(
            [x[max_idx]],
            [values[max_idx]],
            marker="*",
            s=130,
            color="#B24C44",
            edgecolors="white",
            linewidths=0.7,
            zorder=4,
        )
        ax.axhline(
            median_value,
            color="#7A7A7A",
            linestyle="--",
            linewidth=1.0,
            alpha=0.85,
        )
        ax.set_yscale("log")
        ax.set_ylim(global_min / 1.4, global_max * 1.35)
        ax.set_title(label, loc="left", color=MODEL_COLORS[label], fontsize=11)
        style_axis(ax, grid_axis="y")
        ax.grid(True, which="both", axis="y", alpha=0.4)
        ax.text(
            0.015,
            0.90,
            f"Median = {median_value:.3f}\nMax = {values[max_idx]:.1f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#D8D8D8", "boxstyle": "round,pad=0.25"},
        )
        ax.annotate(
            format_file_timestamp_label(top_file).replace("\n", " "),
            xy=(x[max_idx], values[max_idx]),
            xytext=(10, -14),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=7.5,
            color="#4A2A2A",
            bbox={"facecolor": "white", "edgecolor": "#D8D8D8", "boxstyle": "round,pad=0.18"},
            arrowprops={"arrowstyle": "-", "color": "#7A7A7A", "linewidth": 0.8},
        )

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(tick_labels)
    axes[-1].set_xlabel("Test file timestamp")
    for tick in axes[-1].get_xticklabels():
        tick.set_fontsize(8)

    fig.text(0.015, 0.5, "Mean anomaly score (log scale)", va="center", rotation=90, fontsize=10)
    fig.suptitle("Anomaly Detection Models: Test-file Mean Score Trends", y=0.995)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.93, bottom=0.11, hspace=0.26)
    return save_figure(
        fig,
        OUTPUT_ROOT / "anomaly_detection_comparison" / "anomaly_detection_test_file_mean_score_trends",
    )


def export_anomaly_test_file_heatmap() -> list[Path]:
    rows = {}
    for label, run_dir in ANOMALY_RUNS.items():
        frame = pd.read_csv(ensure_exists(run_dir / "file_scores.csv"))
        test_frame = frame.loc[frame["split"] == "test", ["file_name", "exceedance_rate"]]
        rows[label] = test_frame.set_index("file_name")["exceedance_rate"]

    data = pd.DataFrame(rows).T.fillna(0.0)
    ordered_files = data.mean(axis=0).sort_values(ascending=False).index.tolist()
    data = data[ordered_files]

    fig, ax = plt.subplots(figsize=(10.8, 3.1))
    image = ax.imshow(data.to_numpy(), cmap=IMPORTANCE_CMAP, aspect="auto")
    short_names = [name.replace(".csv", "") for name in ordered_files]
    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index.tolist())
    ax.set_xlabel("Test file")
    ax.set_title("Anomaly Detection Models: Test-file Exceedance Rate Heatmap")
    ax.grid(False)

    max_value = float(data.to_numpy().max()) if len(data) else 0.0
    threshold = 0.55 * max_value if max_value > 0 else 0.0
    for row_idx, model_name in enumerate(data.index):
        for col_idx, file_name in enumerate(ordered_files):
            value = float(data.loc[model_name, file_name])
            text_color = "white" if value >= threshold else "#1B1B1B"
            ax.text(col_idx, row_idx, f"{value * 100:.1f}%", ha="center", va="center", fontsize=8, color=text_color)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("Exceedance rate")
    fig.tight_layout()
    return save_figure(
        fig,
        OUTPUT_ROOT / "anomaly_detection_comparison" / "anomaly_detection_test_file_exceedance_heatmap",
    )


def export_anomaly_channel_error_heatmap() -> list[Path]:
    rows = {}
    for label, run_dir in ANOMALY_RUNS.items():
        frame = pd.read_csv(ensure_exists(run_dir / "channel_error.csv"))
        rows[label] = frame.set_index("feature")["mean_channel_error"]

    raw = pd.DataFrame(rows).T.fillna(0.0)
    log_data = np.log10(raw + 1e-6)
    ordered_channels = log_data.mean(axis=0).sort_values(ascending=False).index.tolist()[:10]
    raw = raw[ordered_channels]
    log_data = log_data[ordered_channels]

    fig, ax = plt.subplots(figsize=(8.6, 3.1))
    image = ax.imshow(log_data.to_numpy(), cmap=CONFUSION_CMAP, aspect="auto")
    ax.set_xticks(range(len(ordered_channels)))
    ax.set_xticklabels(ordered_channels, rotation=25, ha="right")
    ax.set_yticks(range(len(log_data.index)))
    ax.set_yticklabels(log_data.index.tolist())
    ax.set_xlabel("Sensor channel")
    ax.set_title("Anomaly Detection Models: Channel Reconstruction Error (log10 scale)")
    ax.grid(False)

    midpoint = float(np.nanmedian(log_data.to_numpy())) if len(log_data) else 0.0
    for row_idx, model_name in enumerate(log_data.index):
        for col_idx, channel_name in enumerate(ordered_channels):
            raw_value = float(raw.loc[model_name, channel_name])
            text_color = "white" if log_data.loc[model_name, channel_name] >= midpoint else "#1B1B1B"
            ax.text(
                col_idx,
                row_idx,
                f"{raw_value:.2e}" if raw_value >= 100 else f"{raw_value:.3f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=text_color,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("log10(mean channel error)")
    fig.tight_layout()
    return save_figure(
        fig,
        OUTPUT_ROOT / "anomaly_detection_comparison" / "anomaly_detection_channel_error_heatmap",
    )


def export_anomaly_reconstruction_examples() -> list[Path]:
    fig, axes = plt.subplots(2, len(ANOMALY_RUNS), figsize=(12.0, 8.6))
    row_specs = [
        ("example_low_score.png", "Low-score window / matched channels"),
        ("example_high_score.png", "High-score window / error channels"),
    ]

    for col_idx, (label, run_dir) in enumerate(ANOMALY_RUNS.items()):
        for row_idx, (image_name, row_title) in enumerate(row_specs):
            ax = axes[row_idx, col_idx]
            ax.imshow(read_rgb_image(ensure_exists(run_dir / image_name)))
            if row_idx == 0:
                ax.set_title(label)
            if col_idx == 0:
                ax.set_ylabel(row_title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.suptitle("Anomaly Detection Models: Representative Windows", y=0.995)
    fig.tight_layout()
    return save_figure(
        fig,
        OUTPUT_ROOT / "anomaly_detection_comparison" / "anomaly_detection_reconstruction_examples",
    )


def export_anomaly_reconstruction_examples_shared_channels() -> list[Path]:
    bundles = {
        label: load_anomaly_reconstruction_bundle(ANOMALY_MODEL_KEYS[label], run_dir)
        for label, run_dir in ANOMALY_RUNS.items()
    }
    feature_names = next(iter(bundles.values()))["feature_names"]
    row_specs = [
        ("low", "Low-score window / shared matched channels"),
        ("high", "High-score window / shared error channels"),
    ]
    shared_indices = {
        row_key: select_shared_feature_indices(bundles, row_key=row_key)
        for row_key, _ in row_specs
    }
    feature_limits = {
        row_key: compute_shared_feature_limits(bundles, row_key=row_key, feature_indices=shared_indices[row_key])
        for row_key, _ in row_specs
    }

    fig = plt.figure(figsize=(12.6, 15.2))
    outer = fig.add_gridspec(2, len(ANOMALY_RUNS), hspace=0.18, wspace=0.22)

    for col_idx, (label, bundle) in enumerate(bundles.items()):
        for row_idx, (row_key, _) in enumerate(row_specs):
            feature_indices = shared_indices[row_key]
            cell = outer[row_idx, col_idx].subgridspec(len(feature_indices), 1, hspace=0.08)
            time_axis = np.arange(bundle[f"{row_key}_target"].shape[0])

            for plot_idx, feature_index in enumerate(feature_indices):
                ax = fig.add_subplot(cell[plot_idx, 0])
                target = bundle[f"{row_key}_target"][:, feature_index]
                prediction = bundle[f"{row_key}_prediction"][:, feature_index]

                ax.plot(time_axis, target, label="Target", linewidth=1.4, color="#6BA3D9")
                ax.plot(time_axis, prediction, label="Prediction", linewidth=1.1, color="#F0A35E")
                ax.set_ylabel(feature_names[feature_index], fontsize=7.5)
                ax.set_ylim(*feature_limits[row_key][feature_index])
                ax.tick_params(axis="both", labelsize=7)
                style_axis(ax, grid_axis="y")

                if plot_idx == 0:
                    ax.set_title(label)
                    ax.legend(loc="upper right", fontsize=6.5)
                if plot_idx < len(feature_indices) - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("Step", fontsize=8)

    fig.text(0.018, 0.74, row_specs[0][1], rotation=90, va="center", ha="center", fontsize=10)
    fig.text(0.018, 0.29, row_specs[1][1], rotation=90, va="center", ha="center", fontsize=10)
    fig.suptitle("Anomaly Detection Models: Representative Windows (Shared Channels)", y=0.995)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.955, bottom=0.055)
    return save_figure(
        fig,
        OUTPUT_ROOT / "anomaly_detection_comparison" / "anomaly_detection_reconstruction_examples_shared_channels",
    )


def export_anomaly_reconstruction_examples_shared_window_shared_channels() -> list[Path]:
    bundles = {
        label: load_anomaly_reconstruction_bundle(ANOMALY_MODEL_KEYS[label], run_dir)
        for label, run_dir in ANOMALY_RUNS.items()
    }
    feature_names = next(iter(bundles.values()))["feature_names"]
    row_specs = [
        ("low", "Shared low-score window / shared channels"),
        ("high", "Shared high-score window / shared channels"),
    ]
    try:
        shared_window_keys = select_shared_window_keys(bundles)
    except ValueError as exc:
        print(f"[skip] {exc}")
        return []
    shared_indices = {
        row_key: select_shared_feature_indices_for_window(
            bundles,
            window_key=shared_window_keys[row_key],
            row_key=row_key,
        )
        for row_key, _ in row_specs
    }
    feature_limits = {
        row_key: compute_shared_feature_limits_for_window(
            bundles,
            window_key=shared_window_keys[row_key],
            feature_indices=shared_indices[row_key],
        )
        for row_key, _ in row_specs
    }

    fig = plt.figure(figsize=(12.6, 15.2))
    outer = fig.add_gridspec(2, len(ANOMALY_RUNS), hspace=0.18, wspace=0.22)

    for col_idx, (label, bundle) in enumerate(bundles.items()):
        for row_idx, (row_key, _) in enumerate(row_specs):
            window_key = shared_window_keys[row_key]
            window_index = bundle["window_index_by_key"][window_key]
            feature_indices = shared_indices[row_key]
            cell = outer[row_idx, col_idx].subgridspec(len(feature_indices), 1, hspace=0.08)
            time_axis = np.arange(bundle["targets"][window_index].shape[0])

            for plot_idx, feature_index in enumerate(feature_indices):
                ax = fig.add_subplot(cell[plot_idx, 0])
                target = bundle["targets"][window_index][:, feature_index]
                prediction = bundle["predictions"][window_index][:, feature_index]

                ax.plot(time_axis, target, label="Target", linewidth=1.4, color="#6BA3D9")
                ax.plot(time_axis, prediction, label="Prediction", linewidth=1.1, color="#F0A35E")
                ax.set_ylabel(feature_names[feature_index], fontsize=7.5)
                ax.set_ylim(*feature_limits[row_key][feature_index])
                ax.tick_params(axis="both", labelsize=7)
                style_axis(ax, grid_axis="y")

                if plot_idx == 0:
                    ax.set_title(label)
                    ax.legend(loc="upper right", fontsize=6.5)
                if plot_idx < len(feature_indices) - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("Step", fontsize=8)

    fig.text(0.018, 0.74, row_specs[0][1], rotation=90, va="center", ha="center", fontsize=10)
    fig.text(0.018, 0.29, row_specs[1][1], rotation=90, va="center", ha="center", fontsize=10)
    fig.suptitle("Anomaly Detection Models: Same Window and Same Channels", y=0.995)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.955, bottom=0.055)
    return save_figure(
        fig,
        OUTPUT_ROOT
        / "anomaly_detection_comparison"
        / "anomaly_detection_reconstruction_examples_same_window_shared_channels",
    )


def write_manifest(generated_paths: list[Path]) -> Path:
    lines = [
        "# Paper Figures",
        "",
        "This directory stores publication-oriented figures exported from the structured result files.",
        "",
        "Selected source runs:",
        f"- MLP: `{MLP_RUN.relative_to(REPO_ROOT)}`",
        f"- CNN: `{CNN_RUN.relative_to(REPO_ROOT)}`",
        f"- RNN: `{BUILDING_RUNS['RNN'].relative_to(REPO_ROOT)}`",
        f"- LSTM: `{BUILDING_RUNS['LSTM'].relative_to(REPO_ROOT)}`",
        f"- Transformer: `{BUILDING_RUNS['Transformer'].relative_to(REPO_ROOT)}`",
        f"- RNN anomaly detection: `{ANOMALY_RUNS['RNN'].relative_to(REPO_ROOT)}`",
        f"- LSTM anomaly detection: `{ANOMALY_RUNS['LSTM'].relative_to(REPO_ROOT)}`",
        f"- Transformer anomaly detection: `{ANOMALY_RUNS['Transformer'].relative_to(REPO_ROOT)}`",
        "",
        "Notes:",
    ]
    lines.extend(f"- {note}" for note in SOURCE_SELECTION_NOTES)
    lines.extend(f"- {note}" for note in OPTIONAL_EXPORT_NOTES)
    lines.extend(
        [
        "- Every figure is exported as both PNG and PDF.",
        "",
        "Generated files:",
        ]
    )
    lines.extend(f"- `{path.relative_to(OUTPUT_ROOT)}`" for path in sorted(generated_paths))
    manifest_path = OUTPUT_ROOT / "README.md"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    global BUILDING_RUNS, ANOMALY_RUNS, SOURCE_SELECTION_NOTES, OPTIONAL_EXPORT_NOTES

    configure_matplotlib()
    BUILDING_RUNS, ANOMALY_RUNS, SOURCE_SELECTION_NOTES = resolve_source_runs()
    OPTIONAL_EXPORT_NOTES = []

    generated: list[Path] = []
    generated.extend(export_building_model_performance())
    generated.extend(export_building_confusion_matrices())
    generated.extend(export_building_feature_importance())
    generated.extend(export_optional("MLP parity grid", export_mlp_parity_grid))
    generated.extend(export_optional("MLP feature importance", export_mlp_feature_importance))
    generated.extend(export_optional("CNN performance overview", export_cnn_performance_overview))
    generated.extend(export_optional("CNN SHAP summary", export_cnn_shap_summary))
    generated.extend(export_anomaly_model_summary())
    generated.extend(export_anomaly_mean_score_trends())
    generated.extend(export_anomaly_test_file_heatmap())
    generated.extend(export_anomaly_channel_error_heatmap())
    generated.extend(export_anomaly_reconstruction_examples())
    generated.extend(export_anomaly_reconstruction_examples_shared_channels())
    generated.extend(export_anomaly_reconstruction_examples_shared_window_shared_channels())
    manifest_path = write_manifest(generated)

    print("Generated paper figures:")
    for path in sorted(generated):
        print(path.relative_to(REPO_ROOT))
    print(manifest_path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
