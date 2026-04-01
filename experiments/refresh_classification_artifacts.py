from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.runtime import prepare_environment

REPO_ROOT = prepare_environment(ROOT)

from src.data.classification_labels import (
    LEGACY_TO_CURRENT_CLASS_NAMES,
    class_name_for_label,
    class_names_from_labels,
    class_slug_for_label,
)
from src.explain.shap_plots import plot_global_importance, plot_time_feature_heatmap
from src.training.metrics import classification_report_frame
from src.utils.config import load_config
from src.utils.io import write_dataframe, write_json
from src.utils.plotting import plot_confusion_matrix

LABELS = [0, 1, 2]
CLASS_NAMES = class_names_from_labels(LABELS)
OUTPUT_ROOT = REPO_ROOT / "outputs"
CONFIG_PATHS = [
    REPO_ROOT / "experiments/configs/rnn_classification.yaml",
    REPO_ROOT / "experiments/configs/lstm_classification.yaml",
    REPO_ROOT / "experiments/configs/transformer_classification.yaml",
]
LEGACY_SLUGS = {label: f"condition_{label}" for label in LABELS}


def replace_legacy_class_names(text: str) -> str:
    updated = text
    for legacy_name, current_name in LEGACY_TO_CURRENT_CLASS_NAMES.items():
        updated = updated.replace(legacy_name, current_name)
    return updated


def replace_legacy_names_recursive(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            replace_legacy_class_names(key) if isinstance(key, str) else key: replace_legacy_names_recursive(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [replace_legacy_names_recursive(item) for item in value]
    if isinstance(value, str):
        return replace_legacy_class_names(value)
    return value


def iter_run_dirs() -> list[Path]:
    run_dirs: list[Path] = []
    for config_path in CONFIG_PATHS:
        config = load_config(config_path)
        run_dir = REPO_ROOT / config["paths"]["output_dir"]
        if run_dir.is_dir() and (run_dir / "predictions.csv").exists():
            run_dirs.append(run_dir)
    return run_dirs


def refresh_predictions(run_dir: Path) -> pd.DataFrame:
    predictions = pd.read_csv(run_dir / "predictions.csv")
    predictions["y_true_name"] = predictions["y_true"].map(class_name_for_label)
    predictions["y_pred_name"] = predictions["y_pred"].map(class_name_for_label)
    write_dataframe(predictions, run_dir / "predictions.csv", index=False)
    return predictions


def compute_test_metrics(predictions: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    y_true = predictions["y_true"].to_numpy()
    y_pred = predictions["y_pred"].to_numpy()
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }
    return matrix, report, metrics


def refresh_metrics_json(run_dir: Path, metrics: dict[str, Any]) -> None:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        payload = {}

    payload = replace_legacy_names_recursive(payload)
    payload.setdefault("test_metrics", {})
    payload["test_metrics"].update(metrics)
    write_json(payload, metrics_path)


def refresh_classification_report(run_dir: Path, report: dict[str, Any]) -> None:
    frame = classification_report_frame(report)
    write_dataframe(frame, run_dir / "classification_report.csv", index=False)


def refresh_confusion_matrix(run_dir: Path, matrix: np.ndarray) -> None:
    plot_confusion_matrix(
        matrix=matrix,
        class_names=CLASS_NAMES,
        output_path=run_dir / "confusion_matrix.png",
        title="Confusion Matrix",
    )


def rename_legacy_shap_files(run_dir: Path) -> None:
    for path in sorted(run_dir.iterdir()):
        if not path.is_file():
            continue
        updated_name = path.name
        for label, legacy_slug in LEGACY_SLUGS.items():
            updated_name = updated_name.replace(legacy_slug, class_slug_for_label(label))
        if updated_name == path.name:
            continue
        target = path.with_name(updated_name)
        if target.exists():
            if target.is_file():
                target.unlink()
        path.rename(target)


def refresh_text_outputs(run_dir: Path) -> None:
    for path in sorted(run_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".yaml", ".yml", ".md", ".txt"}:
            continue
        if path.name in {"predictions.csv", "classification_report.csv", "metrics.json"}:
            continue
        original = path.read_text(encoding="utf-8")
        updated = replace_legacy_class_names(original)
        if updated != original:
            path.write_text(updated, encoding="utf-8")


def refresh_shap_plots(run_dir: Path) -> None:
    summary_candidates = sorted(run_dir.glob("*_shap_importance_all_classes.csv"))
    if not summary_candidates:
        return

    for summary_path in summary_candidates:
        prefix = summary_path.name.removesuffix("_shap_importance_all_classes.csv")
        for label in LABELS:
            class_name = class_name_for_label(label)
            slug = class_slug_for_label(label)
            csv_path = run_dir / f"{prefix}_shap_importance_{slug}.csv"
            npy_path = run_dir / f"{prefix}_raw_shap_{slug}.npy"
            if csv_path.exists():
                frame = pd.read_csv(csv_path)
                plot_global_importance(
                    feature_names=frame["feature"].tolist(),
                    importances=frame["mean_abs_shap"].to_numpy(),
                    output_path=run_dir / f"{prefix}_shap_global_{slug}.png",
                    title=f"{prefix} SHAP Feature Importance - {class_name}",
                )
            if npy_path.exists():
                values = np.load(npy_path)
                mean_time_feature = np.abs(values).mean(axis=0)
                feature_names = (
                    pd.read_csv(csv_path)["feature"].tolist() if csv_path.exists() else [f"feature_{idx}" for idx in range(values.shape[-1])]
                )
                plot_time_feature_heatmap(
                    values=mean_time_feature,
                    feature_names=feature_names,
                    output_path=run_dir / f"{prefix}_shap_heatmap_{slug}.png",
                    title=f"{prefix} Mean |SHAP| Heatmap - {class_name}",
                )


def regenerate_paper_figures() -> None:
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "experiments" / "export_paper_figures.py")],
        cwd=REPO_ROOT,
        check=True,
    )


def main() -> None:
    refreshed = []
    for run_dir in iter_run_dirs():
        predictions = refresh_predictions(run_dir)
        matrix, report, metrics = compute_test_metrics(predictions)
        rename_legacy_shap_files(run_dir)
        refresh_metrics_json(run_dir, metrics)
        refresh_classification_report(run_dir, report)
        refresh_confusion_matrix(run_dir, matrix)
        refresh_text_outputs(run_dir)
        refresh_shap_plots(run_dir)
        refreshed.append(run_dir.relative_to(REPO_ROOT))

    regenerate_paper_figures()

    print("Refreshed classification outputs:")
    for path in refreshed:
        print(path)


if __name__ == "__main__":
    main()
