from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.runtime import prepare_environment

PROJECT_ROOT = prepare_environment(ROOT)

import pandas as pd
import torch

from src.data.classification_labels import class_slug_for_label
from src.utils.config import dump_config, load_config
from src.utils.io import ensure_dir, write_dataframe, write_json


MODEL_SPECS: dict[str, dict[str, str]] = {
    "rnn": {
        "task": "classification",
        "runner_model": "rnn",
        "base_config": "experiments/configs/rnn_classification.yaml",
        "experiment_name": "rnn_classification",
        "output_prefix": "rnn_classification",
    },
    "lstm": {
        "task": "classification",
        "runner_model": "lstm",
        "base_config": "experiments/configs/lstm_classification.yaml",
        "experiment_name": "lstm_classification",
        "output_prefix": "lstm_classification",
    },
    "transformer": {
        "task": "classification",
        "runner_model": "transformer",
        "base_config": "experiments/configs/transformer_classification.yaml",
        "experiment_name": "transformer_classification",
        "output_prefix": "transformer_classification",
    },
}

SWEEP_ROOT = PROJECT_ROOT / "outputs" / "window_sweeps"


def parse_lengths(raw: str) -> list[int]:
    lengths = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not lengths:
        raise ValueError("At least one window length is required.")
    if any(length <= 0 for length in lengths):
        raise ValueError("Window lengths must be positive integers.")
    return sorted(set(lengths))


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_sweep_config(model_key: str, window_length: int, generated_config_dir: Path) -> tuple[dict[str, Any], Path, Path]:
    spec = MODEL_SPECS[model_key]
    config = load_config(PROJECT_ROOT / spec["base_config"])

    experiment_name = f"{spec['experiment_name']}_sweep_win{window_length:02d}"
    output_dir = SWEEP_ROOT / spec["output_prefix"] / f"win_{window_length:02d}"
    config["experiment"]["name"] = experiment_name
    config["paths"]["output_dir"] = str(output_dir.relative_to(PROJECT_ROOT))
    config["window"]["length"] = window_length

    config_path = generated_config_dir / f"{spec['output_prefix']}_win_{window_length:02d}.yaml"
    dump_config(config, config_path)
    return config, config_path, output_dir


def run_experiment(
    model_key: str,
    config_path: Path,
    device: str,
    skip_shap: bool,
) -> None:
    spec = MODEL_SPECS[model_key]
    command = [
        sys.executable,
        "-m",
        "fem_ml_bench",
        "run",
        "--task",
        spec["task"],
        "--model",
        spec["runner_model"],
        "--config",
        str(config_path),
        "--device",
        device,
        "--fixed-window",
        "--fixed-hparams",
    ]
    if skip_shap:
        command.append("--skip-shap")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def collect_row(model_key: str, window_length: int, output_dir: Path) -> dict[str, Any]:
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(output_dir / "predictions.csv")
    class_counts = predictions["y_true"].value_counts().sort_index().to_dict()

    row: dict[str, Any] = {
        "model": model_key,
        "window_length": window_length,
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "best_epoch": int(metrics["best_epoch"]),
        "test_count": int(len(predictions)),
    }

    for prefix, metric_block_name in (("val", "best_val_metrics"), ("test", "test_metrics")):
        metric_block = metrics[metric_block_name]
        row[f"{prefix}_accuracy"] = float(metric_block["accuracy"])
        row[f"{prefix}_macro_f1"] = float(metric_block["macro_f1"])
        row[f"{prefix}_balanced_accuracy"] = float(metric_block["balanced_accuracy"])
        row[f"{prefix}_ovr_auroc"] = float(metric_block["ovr_auroc"])

    for class_index in range(3):
        row[f"test_{class_slug_for_label(class_index)}_count"] = int(class_counts.get(class_index, 0))

    return row


def build_best_by_model(summary: pd.DataFrame) -> pd.DataFrame:
    order = ["model", "val_macro_f1", "val_balanced_accuracy", "val_accuracy", "window_length"]
    ranked = summary.sort_values(order, ascending=[True, False, False, False, True]).reset_index(drop=True)
    return ranked.groupby("model", as_index=False).first()


def build_shared_window_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    aggregated = (
        summary.groupby("window_length", as_index=False)
        .agg(
            mean_val_accuracy=("val_accuracy", "mean"),
            mean_val_macro_f1=("val_macro_f1", "mean"),
            mean_val_balanced_accuracy=("val_balanced_accuracy", "mean"),
            mean_val_ovr_auroc=("val_ovr_auroc", "mean"),
            mean_test_accuracy=("test_accuracy", "mean"),
            mean_test_macro_f1=("test_macro_f1", "mean"),
            mean_test_balanced_accuracy=("test_balanced_accuracy", "mean"),
            mean_test_ovr_auroc=("test_ovr_auroc", "mean"),
        )
        .sort_values(
            ["mean_val_macro_f1", "mean_val_balanced_accuracy", "mean_val_accuracy", "window_length"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="2,3,4,6,8,12,16,24")
    parser.add_argument("--models", default="rnn,lstm,transformer")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-shap", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    lengths = parse_lengths(args.lengths)
    model_keys = [part.strip().lower() for part in args.models.split(",") if part.strip()]
    unknown = sorted(set(model_keys) - set(MODEL_SPECS))
    if unknown:
        raise ValueError(f"Unsupported model keys: {unknown}")

    device = select_device(args.device)
    skip_shap = not args.include_shap

    generated_config_dir = ensure_dir(SWEEP_ROOT / "generated_configs")
    write_json(
        {
            "lengths": lengths,
            "models": model_keys,
            "device": device,
            "skip_shap": skip_shap,
            "force": bool(args.force),
        },
        SWEEP_ROOT / "run_metadata.json",
    )

    rows: list[dict[str, Any]] = []
    for model_key in model_keys:
        for window_length in lengths:
            _, config_path, output_dir = build_sweep_config(model_key, window_length, generated_config_dir)
            metrics_path = output_dir / "metrics.json"
            if args.force or not metrics_path.exists():
                print(f"[run] model={model_key} window={window_length} device={device}")
                run_experiment(
                    model_key=model_key,
                    config_path=config_path,
                    device=device,
                    skip_shap=skip_shap,
                )
            else:
                print(f"[skip] model={model_key} window={window_length} existing={metrics_path.relative_to(PROJECT_ROOT)}")

            rows.append(collect_row(model_key=model_key, window_length=window_length, output_dir=output_dir))

    summary = pd.DataFrame(rows).sort_values(["model", "window_length"]).reset_index(drop=True)
    best_by_model = build_best_by_model(summary)
    shared_window_ranking = build_shared_window_ranking(summary)

    write_dataframe(summary, SWEEP_ROOT / "summary.csv", index=False)
    write_dataframe(best_by_model, SWEEP_ROOT / "best_by_model_val_macro_f1.csv", index=False)
    write_dataframe(shared_window_ranking, SWEEP_ROOT / "shared_window_ranking.csv", index=False)

    print("\nBest window by model (validation macro-F1):")
    print(best_by_model[["model", "window_length", "val_macro_f1", "test_macro_f1"]].to_string(index=False))
    print("\nShared window ranking (mean validation macro-F1):")
    print(
        shared_window_ranking[
            ["window_length", "mean_val_macro_f1", "mean_test_macro_f1"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
