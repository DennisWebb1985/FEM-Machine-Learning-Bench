from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.anomaly_detection_preprocess import (
    build_anomaly_reconstruction_windows,
    prepare_anomaly_detection_data,
)
from src.data.datasets import SequenceRegressionDataset
from src.training.anomaly_engine import (
    aggregate_file_scores,
    evaluate_sequence_model,
    fit_sequence_model,
    representative_window_index,
    summarize_score_distribution,
    window_scores_frame,
)
from src.utils.io import write_dataframe, write_json
from src.utils.plotting import (
    plot_anomaly_score_trend,
    plot_channel_error_bar,
    plot_loss_curve,
    plot_score_histogram,
    plot_sequence_comparison,
)
from src.utils.seed import set_seed

from .auto_window import parse_window_lengths, run_auto_window_search
from .common import default_train_device, load_runner_config, prepare_output_dir, resolve_device, resolve_repo_path


ModelBuilder = Callable[[dict[str, Any], int, int], torch.nn.Module]
OptimizerBuilder = Callable[[torch.nn.Module, dict[str, Any]], torch.optim.Optimizer]

DEFAULT_ANOMALY_WINDOW_LENGTHS = (64, 128, 256, 384, 512)


@dataclass(frozen=True)
class SequenceAnomalySpec:
    default_config: Path
    model_key: str
    example_name: str
    title_root: str
    model_builder: ModelBuilder
    optimizer_builder: OptimizerBuilder
    scheduler_patience: int = 3
    qualifier: str | None = None


def _make_threshold_summary(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    quantiles: list[float],
) -> dict[str, float]:
    summary: dict[str, float] = {}
    for quantile in quantiles:
        label = f"q{int(quantile * 100):02d}"
        threshold = float(np.quantile(train_scores, quantile))
        summary[f"threshold_{label}"] = threshold
        summary[f"val_exceedance_{label}"] = float(np.mean(val_scores > threshold))
        summary[f"test_exceedance_{label}"] = float(np.mean(test_scores > threshold))
    return summary


def _format_title(base: str, qualifier: str | None = None, extra: str | None = None) -> str:
    if qualifier and extra:
        return f"{base} ({qualifier}, {extra})"
    if qualifier:
        return f"{base} ({qualifier})"
    if extra:
        return f"{base} ({extra})"
    return base


def _run_single_sequence_anomaly_detection(
    spec: SequenceAnomalySpec,
    config: dict[str, Any],
    *,
    device_name: str,
) -> dict[str, Any]:
    output_dir = prepare_output_dir(config)
    set_seed(int(config["training"].get("seed", 42)))

    prepared = prepare_anomaly_detection_data(
        data_dir=resolve_repo_path(config["paths"]["data_dir"]),
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        drop_channels=config["data"]["drop_channels"],
        missing_sentinel=config["data"]["missing_sentinel"],
        extreme_value_threshold=config["data"]["extreme_value_threshold"],
        downsample_factor=config["data"]["downsample_factor"],
    )

    train_x, train_y, train_meta = build_anomaly_reconstruction_windows(
        prepared["train"],
        prepared["feature_cols"],
        window_length=config["window"]["length"],
        stride=config["window"]["stride"],
        drop_nan_windows=config["window"]["drop_nan_windows"],
    )
    val_x, val_y, val_meta = build_anomaly_reconstruction_windows(
        prepared["val"],
        prepared["feature_cols"],
        window_length=config["window"]["length"],
        stride=config["window"]["stride"],
        drop_nan_windows=config["window"]["drop_nan_windows"],
    )
    test_x, test_y, test_meta = build_anomaly_reconstruction_windows(
        prepared["test"],
        prepared["feature_cols"],
        window_length=config["window"]["length"],
        stride=config["window"]["stride"],
        drop_nan_windows=config["window"]["drop_nan_windows"],
    )

    train_ds = SequenceRegressionDataset(train_x, train_y, train_meta)
    val_ds = SequenceRegressionDataset(val_x, val_y, val_meta)
    test_ds = SequenceRegressionDataset(test_x, test_y, test_meta)

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = resolve_device(device_name)
    model = spec.model_builder(config, int(train_x.shape[-1]), int(config["window"]["length"])).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = spec.optimizer_builder(model, config)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=spec.scheduler_patience,
    )

    fit_results = fit_sequence_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=int(config["training"]["epochs"]),
        patience=int(config["training"]["patience"]),
        grad_clip=config["training"].get("grad_clip"),
        scheduler=scheduler,
        min_delta=float(config["training"].get("min_delta", 0.0)),
    )

    train_eval = evaluate_sequence_model(fit_results["model"], train_loader, criterion, device, False)
    val_eval = evaluate_sequence_model(fit_results["model"], val_loader, criterion, device, False)
    test_eval = evaluate_sequence_model(fit_results["model"], test_loader, criterion, device, True)

    threshold_quantiles = config["evaluation"]["threshold_quantiles"]
    threshold_summary = _make_threshold_summary(
        train_eval["window_scores"],
        val_eval["window_scores"],
        test_eval["window_scores"],
        threshold_quantiles,
    )
    main_threshold = threshold_summary[f"threshold_q{int(threshold_quantiles[0] * 100):02d}"]

    train_window_df = window_scores_frame(train_meta, train_eval["window_scores"])
    val_window_df = window_scores_frame(val_meta, val_eval["window_scores"])
    test_window_df = window_scores_frame(test_meta, test_eval["window_scores"])
    window_scores_df = pd.concat([train_window_df, val_window_df, test_window_df], ignore_index=True)

    file_scores_df = pd.concat(
        [
            aggregate_file_scores(train_meta, train_eval["window_scores"], main_threshold),
            aggregate_file_scores(val_meta, val_eval["window_scores"], main_threshold),
            aggregate_file_scores(test_meta, test_eval["window_scores"], main_threshold),
        ],
        ignore_index=True,
    )

    channel_error = test_eval["channel_scores"].mean(axis=0)
    channel_error_df = pd.DataFrame(
        {"feature": prepared["feature_cols"], "mean_channel_error": channel_error}
    ).sort_values("mean_channel_error", ascending=False)

    metrics = {
        "best_epoch": fit_results["best_epoch"],
        "best_val_loss": fit_results["best_val_loss"],
        "train_loss": train_eval["loss"],
        "val_loss": val_eval["loss"],
        "test_loss": test_eval["loss"],
        "score_summary": summarize_score_distribution(
            train_eval["window_scores"],
            val_eval["window_scores"],
            test_eval["window_scores"],
            quantiles=threshold_quantiles,
        ),
        "threshold_summary": threshold_summary,
        "num_train_windows": int(len(train_x)),
        "num_val_windows": int(len(val_x)),
        "num_test_windows": int(len(test_x)),
        "num_features": int(train_x.shape[-1]),
    }

    torch.save(fit_results["model"].state_dict(), output_dir / "best_model.pt")
    write_json(metrics, output_dir / "metrics.json")
    write_json(fit_results["history"], output_dir / "history.json")
    write_dataframe(window_scores_df, output_dir / "window_scores.csv", index=False)
    write_dataframe(file_scores_df, output_dir / "file_scores.csv", index=False)
    write_dataframe(channel_error_df, output_dir / "channel_error.csv", index=False)

    plot_loss_curve(
        fit_results["history"],
        output_dir / "learning_curve.png",
        title=_format_title(f"{spec.title_root} Loss", spec.qualifier),
    )
    plot_score_histogram(
        train_eval["window_scores"],
        val_eval["window_scores"],
        test_eval["window_scores"],
        output_dir / "score_histogram.png",
        title=_format_title(f"{spec.title_root} Window Score Distribution", spec.qualifier),
    )
    plot_score_histogram(
        train_eval["window_scores"],
        val_eval["window_scores"],
        test_eval["window_scores"],
        output_dir / "score_histogram_log.png",
        title=_format_title(f"{spec.title_root} Window Score Distribution", spec.qualifier, "log1p"),
        log1p_transform=True,
    )
    plot_score_histogram(
        train_eval["window_scores"],
        val_eval["window_scores"],
        test_eval["window_scores"],
        output_dir / "score_histogram_clipped.png",
        title=_format_title(f"{spec.title_root} Window Score Distribution", spec.qualifier, "clipped"),
        clip_quantile=0.995,
    )

    test_file_scores = file_scores_df[file_scores_df["split"] == "test"].sort_values("file_order").reset_index(drop=True)
    plot_anomaly_score_trend(
        file_scores=test_file_scores["mean_score"].to_numpy(),
        file_labels=test_file_scores["file_name"].tolist(),
        output_path=output_dir / "score_trend.png",
        title=_format_title(f"{spec.title_root} Test File Scores", spec.qualifier),
    )
    plot_channel_error_bar(
        channel_names=channel_error_df["feature"].tolist(),
        channel_errors=channel_error_df["mean_channel_error"].tolist(),
        output_path=output_dir / "channel_error.png",
        title=_format_title(f"{spec.title_root} Mean Channel Error", spec.qualifier),
    )

    if len(test_eval["window_scores"]) > 0:
        low_index = representative_window_index(test_eval["window_scores"], quantile=0.10)
        high_index = representative_window_index(test_eval["window_scores"], quantile=0.95)
        if low_index is not None:
            plot_sequence_comparison(
                target_sequence=test_eval["targets"][low_index],
                predicted_sequence=test_eval["predictions"][low_index],
                feature_names=prepared["feature_cols"],
                output_path=output_dir / "example_low_score.png",
                title=f"{spec.example_name} Low-Score Window (q10)",
                feature_selection="lowest_error_high_variance",
            )
        if high_index is not None:
            plot_sequence_comparison(
                target_sequence=test_eval["targets"][high_index],
                predicted_sequence=test_eval["predictions"][high_index],
                feature_names=prepared["feature_cols"],
                output_path=output_dir / "example_high_score.png",
                title=f"{spec.example_name} High-Score Window (q95)",
                feature_selection="largest_error",
            )

    return {
        "output_dir": output_dir,
        "metrics": metrics,
    }


def _anomaly_summary_row(metrics: dict[str, Any]) -> dict[str, Any]:
    threshold_summary = metrics["threshold_summary"]
    return {
        "best_epoch": int(metrics["best_epoch"]),
        "best_val_loss": float(metrics["best_val_loss"]),
        "train_loss": float(metrics["train_loss"]),
        "val_loss": float(metrics["val_loss"]),
        "test_loss": float(metrics["test_loss"]),
        "num_train_windows": int(metrics["num_train_windows"]),
        "num_val_windows": int(metrics["num_val_windows"]),
        "num_test_windows": int(metrics["num_test_windows"]),
        "threshold_q95": float(threshold_summary.get("threshold_q95", 0.0)),
        "val_exceedance_q95": float(threshold_summary.get("val_exceedance_q95", 0.0)),
        "test_exceedance_q95": float(threshold_summary.get("test_exceedance_q95", 0.0)),
    }


def _load_anomaly_summary_row(output_dir: Path, _: int) -> dict[str, Any]:
    with (output_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return _anomaly_summary_row(metrics)


def _apply_anomaly_window(config: dict[str, Any], window_length: int) -> None:
    config.setdefault("window", {})
    config["window"]["length"] = window_length
    config["window"]["stride"] = max(1, window_length // 2)

    patch_size = config.get("model", {}).get("patch_size")
    if patch_size is not None and window_length % int(patch_size) != 0:
        raise ValueError(
            f"Window length {window_length} must be divisible by transformer patch_size {patch_size}."
        )


def run_sequence_anomaly_detection(spec: SequenceAnomalySpec, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(spec.default_config))
    parser.add_argument("--device", default=default_train_device())
    parser.add_argument("--epochs", type=int, default=None)

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
        help="Re-run candidate window searches even if cached metrics already exist.",
    )
    args = parser.parse_args(argv)

    config = load_runner_config(args.config, epochs=args.epochs)

    if args.auto_window:
        window_lengths = parse_window_lengths(args.window_lengths, DEFAULT_ANOMALY_WINDOW_LENGTHS)
        selection = run_auto_window_search(
            task_name="anomaly_detection",
            model_key=spec.model_key,
            base_config=config,
            window_lengths=window_lengths,
            selection_metric="best_val_loss",
            sort_columns=["best_val_loss", "val_loss", "test_loss", "window_length"],
            ascending=[True, True, True, True],
            adjust_window=_apply_anomaly_window,
            run_candidate=lambda candidate_config: _anomaly_summary_row(
                _run_single_sequence_anomaly_detection(
                    spec,
                    candidate_config,
                    device_name=args.device,
                )["metrics"]
            ),
            load_candidate_row=_load_anomaly_summary_row,
            force=bool(args.auto_window_force),
        )

        final_config = deepcopy(config)
        _apply_anomaly_window(final_config, selection.best_window_length)
        final_result = _run_single_sequence_anomaly_detection(
            spec,
            final_config,
            device_name=args.device,
        )
        write_json(
            {
                "best_window_length": selection.best_window_length,
                "candidate_lengths": window_lengths,
                "selection_metric": "best_val_loss",
                "search_root": str(selection.search_root.relative_to(resolve_repo_path("."))),
                "summary_path": str(selection.summary_path.relative_to(resolve_repo_path("."))),
                "ranked_summary_path": str(selection.ranked_summary_path.relative_to(resolve_repo_path("."))),
                "best_candidate": selection.best_row,
            },
            final_result["output_dir"] / "auto_window_selection.json",
        )
        return 0

    _run_single_sequence_anomaly_detection(
        spec,
        config,
        device_name=args.device,
    )
    return 0
