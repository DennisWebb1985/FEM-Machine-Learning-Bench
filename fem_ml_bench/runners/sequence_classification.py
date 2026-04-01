from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.building_preprocess import prepare_building_data
from src.data.classification_labels import class_name_for_label, class_slug_for_name
from src.data.datasets import SequenceClassificationDataset
from src.data.windowing import build_sliding_windows
from src.explain.shap_runner import run_shap_analysis
from src.training.engine import fit_model
from src.training.losses import build_cross_entropy_loss
from src.training.metrics import classification_report_frame
from src.utils.config import merge_overrides
from src.utils.io import write_dataframe, write_json
from src.utils.plotting import plot_confusion_matrix, plot_learning_curve
from src.utils.seed import set_seed

from .auto_tuning import TuningProfile, parse_tuning_profiles, run_auto_tuning_search
from .auto_window import parse_window_lengths, run_auto_window_search
from .common import default_train_device, load_runner_config, prepare_output_dir, resolve_device, resolve_repo_path


ModelBuilder = Callable[[dict[str, Any], int, int], torch.nn.Module]
OptimizerBuilder = Callable[[torch.nn.Module, dict[str, Any]], torch.optim.Optimizer]

DEFAULT_CLASSIFICATION_WINDOW_LENGTHS = (2, 3, 4, 6, 8, 12, 16, 24)


@dataclass(frozen=True)
class SequenceClassificationSpec:
    default_config: Path
    model_key: str
    display_name: str
    shap_prefix: str
    model_builder: ModelBuilder
    optimizer_builder: OptimizerBuilder
    scheduler_patience: int = 4
    tuning_profiles: tuple[TuningProfile, ...] = ()


def _make_building_windows(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    time_col: str,
    window_length: int,
    stride: int,
    label_mode: str,
):
    return build_sliding_windows(
        features=frame[feature_cols].to_numpy(),
        labels=frame[target_col].to_numpy(),
        window_length=window_length,
        stride=stride,
        label_mode=label_mode,
        metadata={time_col: frame[time_col].astype(str).to_numpy()},
    )


def _run_single_sequence_classification(
    spec: SequenceClassificationSpec,
    config: dict[str, Any],
    *,
    device_name: str,
    skip_shap: bool,
) -> dict[str, Any]:
    output_dir = prepare_output_dir(config)
    set_seed(int(config["training"].get("seed", 42)))

    prepared = prepare_building_data(
        csv_path=resolve_repo_path(config["paths"]["data"]),
        feature_cols=config["data"]["feature_cols"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
    )

    train_x, train_y, train_meta = _make_building_windows(
        prepared["train"],
        prepared["feature_cols"],
        prepared["target_col"],
        prepared["time_col"],
        config["window"]["length"],
        config["window"]["stride"],
        config["window"]["label_mode"],
    )
    val_x, val_y, val_meta = _make_building_windows(
        prepared["val"],
        prepared["feature_cols"],
        prepared["target_col"],
        prepared["time_col"],
        config["window"]["length"],
        config["window"]["stride"],
        config["window"]["label_mode"],
    )
    test_x, test_y, test_meta = _make_building_windows(
        prepared["test"],
        prepared["feature_cols"],
        prepared["target_col"],
        prepared["time_col"],
        config["window"]["length"],
        config["window"]["stride"],
        config["window"]["label_mode"],
    )

    train_ds = SequenceClassificationDataset(train_x, train_y, train_meta)
    val_ds = SequenceClassificationDataset(val_x, val_y, val_meta)
    test_ds = SequenceClassificationDataset(test_x, test_y, test_meta)

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = resolve_device(device_name)
    model = spec.model_builder(config, int(train_x.shape[-1]), len(prepared["class_names"])).to(device)

    criterion = build_cross_entropy_loss(
        train_y,
        len(prepared["class_names"]),
        device=device,
        label_smoothing=float(config["training"].get("label_smoothing", 0.0)),
        class_weight_mode=str(config["training"].get("class_weight_mode", "inverse")),
    )
    optimizer = spec.optimizer_builder(model, config)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=spec.scheduler_patience,
    )

    results = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        class_names=prepared["class_names"],
        num_epochs=int(config["training"]["epochs"]),
        patience=int(config["training"]["patience"]),
        grad_clip=config["training"].get("grad_clip"),
        scheduler=scheduler,
        monitor_metric=str(config["training"].get("monitor_metric", "val_macro_f1")),
        monitor_mode=str(config["training"].get("monitor_mode", "max")),
        monitor_min_delta=float(config["training"].get("monitor_min_delta", 0.0)),
    )

    torch.save(results["model"].state_dict(), output_dir / "best_model.pt")
    write_json(results["history"], output_dir / "history.json")
    plot_learning_curve(results["history"], output_dir / "learning_curve.png")
    plot_confusion_matrix(
        matrix=results["test_metrics"]["confusion_matrix"],
        class_names=prepared["class_names"],
        output_path=output_dir / "confusion_matrix.png",
        title=f"Building {spec.display_name} Confusion Matrix",
    )

    metrics_to_save = {
        "best_epoch": results["best_epoch"],
        "best_val_metrics": results["best_val_metrics"],
        "test_metrics": results["test_metrics"],
        "monitor_metric": results["monitor_metric"],
        "monitor_mode": results["monitor_mode"],
    }
    write_json(metrics_to_save, output_dir / "metrics.json")

    report_df = classification_report_frame(results["test_metrics"]["classification_report"])
    write_dataframe(report_df, output_dir / "classification_report.csv")

    predictions_df = pd.DataFrame(test_meta)
    predictions_df["y_true"] = results["test_outputs"]["targets"]
    predictions_df["y_pred"] = results["test_outputs"]["predictions"]
    predictions_df["y_true_name"] = predictions_df["y_true"].map(class_name_for_label)
    predictions_df["y_pred_name"] = predictions_df["y_pred"].map(class_name_for_label)
    for class_index, class_name in enumerate(prepared["class_names"]):
        predictions_df[f"prob_{class_slug_for_name(class_name)}"] = results["test_outputs"]["probabilities"][
            :,
            class_index,
        ]
    write_dataframe(predictions_df, output_dir / "predictions.csv", index=False)

    if config.get("shap", {}).get("enabled", False) and not skip_shap:
        try:
            shap_summary = run_shap_analysis(
                model=results["model"],
                background_features=train_ds.feature_array(),
                explain_features=test_ds.feature_array(),
                feature_names=prepared["feature_cols"],
                output_dir=output_dir,
                class_names=prepared["class_names"],
                device=device,
                prefix=spec.shap_prefix,
                max_background=int(config["shap"]["background_size"]),
                max_samples=int(config["shap"]["num_explain_samples"]),
            )
            write_json(shap_summary, output_dir / "shap_summary.json")
        except Exception as exc:  # pragma: no cover - diagnostic path
            write_json({"shap_error": str(exc)}, output_dir / "shap_error.json")

    return {
        "output_dir": output_dir,
        "metrics": metrics_to_save,
    }


def _classification_summary_row(metrics: dict[str, Any]) -> dict[str, Any]:
    best_val = metrics["best_val_metrics"]
    test_metrics = metrics["test_metrics"]
    return {
        "best_epoch": int(metrics["best_epoch"]),
        "val_accuracy": float(best_val["accuracy"]),
        "val_macro_f1": float(best_val["macro_f1"]),
        "val_balanced_accuracy": float(best_val["balanced_accuracy"]),
        "val_ovr_auroc": float(best_val["ovr_auroc"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        "test_ovr_auroc": float(test_metrics["ovr_auroc"]),
    }


def _load_classification_summary_row(output_dir: Path, _: int) -> dict[str, Any]:
    with (output_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return _classification_summary_row(metrics)


def _apply_classification_window(config: dict[str, Any], window_length: int) -> None:
    config.setdefault("window", {})
    config["window"]["length"] = window_length


def run_sequence_classification(spec: SequenceClassificationSpec, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(spec.default_config))
    parser.add_argument("--device", default=default_train_device())
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-shap", action="store_true")

    parser.set_defaults(auto_window=True)
    parser.add_argument(
        "--auto-window",
        dest="auto_window",
        action="store_true",
        help="Search candidate window lengths before the final training run. Enabled by default.",
    )
    parser.add_argument(
        "--fixed-window",
        dest="auto_window",
        action="store_false",
        help="Use the config window length without automatic window search.",
    )
    parser.add_argument(
        "--window-lengths",
        help="Comma-separated candidate window lengths for automatic window search.",
    )
    parser.add_argument(
        "--auto-window-force",
        action="store_true",
        help="Re-run window-search candidates even if cached metrics already exist.",
    )

    parser.set_defaults(auto_tune=bool(spec.tuning_profiles))
    parser.add_argument(
        "--auto-tune",
        dest="auto_tune",
        action="store_true",
        help="Search supported hyperparameter profiles before the final training run.",
    )
    parser.add_argument(
        "--fixed-hparams",
        dest="auto_tune",
        action="store_false",
        help="Use config hyperparameters without automatic profile search.",
    )
    parser.add_argument(
        "--tuning-profiles",
        help="Comma-separated hyperparameter profile names for automatic tuning.",
    )
    parser.add_argument(
        "--auto-tune-force",
        action="store_true",
        help="Re-run hyperparameter candidates even if cached metrics already exist.",
    )

    args = parser.parse_args(argv)
    config = load_runner_config(args.config, epochs=args.epochs)
    final_config = deepcopy(config)
    project_root = resolve_repo_path(".")

    if args.auto_window:
        window_lengths = parse_window_lengths(args.window_lengths, DEFAULT_CLASSIFICATION_WINDOW_LENGTHS)
        window_selection = run_auto_window_search(
            task_name="classification",
            model_key=spec.model_key,
            base_config=final_config,
            window_lengths=window_lengths,
            selection_metric="val_macro_f1",
            sort_columns=["val_macro_f1", "val_balanced_accuracy", "val_accuracy", "window_length"],
            ascending=[False, False, False, True],
            adjust_window=_apply_classification_window,
            run_candidate=lambda candidate_config: _classification_summary_row(
                _run_single_sequence_classification(
                    spec,
                    candidate_config,
                    device_name=args.device,
                    skip_shap=True,
                )["metrics"]
            ),
            load_candidate_row=_load_classification_summary_row,
            force=bool(args.auto_window_force),
        )
        _apply_classification_window(final_config, window_selection.best_window_length)
    else:
        window_lengths = [int(final_config["window"]["length"])]
        window_selection = None

    if args.auto_tune:
        if not spec.tuning_profiles:
            raise ValueError(f"Automatic hyperparameter search is not configured for classification/{spec.model_key}.")
        tuning_profiles = parse_tuning_profiles(args.tuning_profiles, spec.tuning_profiles)
        tuning_selection = run_auto_tuning_search(
            task_name="classification",
            model_key=spec.model_key,
            base_config=final_config,
            profiles=tuning_profiles,
            selection_metric="val_macro_f1",
            sort_columns=["val_macro_f1", "val_balanced_accuracy", "val_accuracy", "tuning_profile"],
            ascending=[False, False, False, True],
            run_candidate=lambda candidate_config: _classification_summary_row(
                _run_single_sequence_classification(
                    spec,
                    candidate_config,
                    device_name=args.device,
                    skip_shap=True,
                )["metrics"]
            ),
            load_candidate_row=_load_classification_summary_row,
            force=bool(args.auto_tune_force),
            context_label=f"window_{int(final_config['window']['length']):02d}",
        )
        final_config = merge_overrides(final_config, tuning_selection.best_profile.overrides)
    else:
        tuning_profiles = ()
        tuning_selection = None

    final_result = _run_single_sequence_classification(
        spec,
        final_config,
        device_name=args.device,
        skip_shap=bool(args.skip_shap),
    )

    if window_selection is not None:
        write_json(
            {
                "best_window_length": window_selection.best_window_length,
                "candidate_lengths": window_lengths,
                "selection_metric": "val_macro_f1",
                "search_root": str(window_selection.search_root.relative_to(project_root)),
                "summary_path": str(window_selection.summary_path.relative_to(project_root)),
                "ranked_summary_path": str(window_selection.ranked_summary_path.relative_to(project_root)),
                "best_candidate": window_selection.best_row,
            },
            final_result["output_dir"] / "auto_window_selection.json",
        )

    if tuning_selection is not None:
        write_json(
            {
                "best_profile": tuning_selection.best_profile.name,
                "candidate_profiles": [profile.name for profile in tuning_profiles],
                "selection_metric": "val_macro_f1",
                "search_root": str(tuning_selection.search_root.relative_to(project_root)),
                "summary_path": str(tuning_selection.summary_path.relative_to(project_root)),
                "ranked_summary_path": str(tuning_selection.ranked_summary_path.relative_to(project_root)),
                "best_candidate": tuning_selection.best_row,
                "applied_overrides": tuning_selection.best_profile.overrides,
            },
            final_result["output_dir"] / "auto_hparam_selection.json",
        )

    return 0
