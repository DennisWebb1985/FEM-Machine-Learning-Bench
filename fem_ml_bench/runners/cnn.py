from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.image_datasets import (
    ImageClassificationDataset,
    _subset_per_class,
    discover_image_samples,
    split_image_samples,
)
from src.explain.shap_runner import run_image_shap_analysis
from src.models.cnn import VanillaCNN
from src.training.engine import fit_model
from src.training.losses import build_cross_entropy_loss
from src.training.metrics import classification_report_frame
from src.utils.io import write_dataframe, write_json
from src.utils.plotting import plot_confusion_matrix, plot_learning_curve
from src.utils.seed import set_seed

from .common import load_runner_config, prepare_output_dir, resolve_device, resolve_repo_path


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "experiments/configs/cnn.yaml"


def _override_normalization(config: dict, grayscale: bool) -> None:
    channels = 1 if grayscale else 3
    if len(config["data"]["normalize_mean"]) != channels:
        config["data"]["normalize_mean"] = [0.5] * channels
    if len(config["data"]["normalize_std"]) != channels:
        config["data"]["normalize_std"] = [0.5] * channels


def _count_by_class(samples, class_names: list[str]) -> dict[str, int]:
    counts = {class_name: 0 for class_name in class_names}
    for sample in samples:
        counts[sample.class_name] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args(argv)

    config = load_runner_config(args.config, epochs=args.epochs)
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.limit_per_class is not None:
        config["data"]["max_samples_per_class"] = args.limit_per_class

    _override_normalization(config, grayscale=bool(config["data"]["grayscale"]))

    output_dir = prepare_output_dir(config)
    set_seed(int(config["training"].get("seed", 42)))

    samples = discover_image_samples(
        root_dir=resolve_repo_path(config["paths"]["data_dir"]),
        class_names=config["data"]["class_names"],
        file_extensions=config["data"]["file_extensions"],
    )
    samples = _subset_per_class(
        samples,
        max_samples_per_class=config["data"]["max_samples_per_class"],
        seed=int(config["training"].get("seed", 42)),
    )
    splits = split_image_samples(
        samples=samples,
        train_ratio=float(config["data"]["train_ratio"]),
        val_ratio=float(config["data"]["val_ratio"]),
        seed=int(config["training"].get("seed", 42)),
    )

    train_ds = ImageClassificationDataset(
        samples=splits["train"],
        image_size=int(config["data"]["image_size"]),
        grayscale=bool(config["data"]["grayscale"]),
        normalize_mean=config["data"]["normalize_mean"],
        normalize_std=config["data"]["normalize_std"],
        horizontal_flip_prob=float(config["augmentation"]["horizontal_flip_prob"]),
    )
    val_ds = ImageClassificationDataset(
        samples=splits["val"],
        image_size=int(config["data"]["image_size"]),
        grayscale=bool(config["data"]["grayscale"]),
        normalize_mean=config["data"]["normalize_mean"],
        normalize_std=config["data"]["normalize_std"],
        horizontal_flip_prob=0.0,
    )
    test_ds = ImageClassificationDataset(
        samples=splits["test"],
        image_size=int(config["data"]["image_size"]),
        grayscale=bool(config["data"]["grayscale"]),
        normalize_mean=config["data"]["normalize_mean"],
        normalize_std=config["data"]["normalize_std"],
        horizontal_flip_prob=0.0,
    )

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = resolve_device(args.device, allow_auto=True)
    input_channels = 1 if config["data"]["grayscale"] else 3
    model = VanillaCNN(
        input_channels=input_channels,
        num_classes=len(config["data"]["class_names"]),
        conv_channels=list(config["model"]["conv_channels"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)

    criterion = build_cross_entropy_loss(
        train_ds.target_array(),
        len(config["data"]["class_names"]),
        device=device,
        label_smoothing=float(config["training"].get("label_smoothing", 0.0)),
        class_weight_mode=str(config["training"].get("class_weight_mode", "inverse")),
    )
    optimizer = Adam(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    results = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        class_names=list(config["data"]["class_names"]),
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
        class_names=config["data"]["class_names"],
        output_path=output_dir / "confusion_matrix.png",
        title="CNN Confusion Matrix",
    )

    split_summary = {
        split_name: {
            "num_samples": len(split_samples),
            "class_counts": _count_by_class(split_samples, list(config["data"]["class_names"])),
        }
        for split_name, split_samples in splits.items()
    }
    metrics_to_save = {
        "device": str(device),
        "best_epoch": results["best_epoch"],
        "best_val_metrics": results["best_val_metrics"],
        "split_summary": split_summary,
        "monitor_metric": results["monitor_metric"],
        "monitor_mode": results["monitor_mode"],
        "test_metrics": {
            key: value
            for key, value in results["test_metrics"].items()
            if key not in {"confusion_matrix", "classification_report"}
        },
    }
    write_json(metrics_to_save, output_dir / "metrics.json")

    report_df = classification_report_frame(results["test_metrics"]["classification_report"])
    write_dataframe(report_df, output_dir / "classification_report.csv")

    probabilities = results["test_outputs"]["probabilities"]
    predictions_df = pd.DataFrame(test_ds.metadata_frame())
    predictions_df["y_true"] = results["test_outputs"]["targets"]
    predictions_df["y_pred"] = results["test_outputs"]["predictions"]
    for class_index, class_name in enumerate(config["data"]["class_names"]):
        predictions_df[f"prob_{class_name.lower()}"] = probabilities[:, class_index]
    write_dataframe(predictions_df, output_dir / "predictions.csv", index=False)

    if config.get("shap", {}).get("enabled", False) and not args.skip_shap:
        shap_train_ds = ImageClassificationDataset(
            samples=splits["train"],
            image_size=int(config["data"]["image_size"]),
            grayscale=bool(config["data"]["grayscale"]),
            normalize_mean=config["data"]["normalize_mean"],
            normalize_std=config["data"]["normalize_std"],
            horizontal_flip_prob=0.0,
        )
        shap_test_ds = ImageClassificationDataset(
            samples=splits["test"],
            image_size=int(config["data"]["image_size"]),
            grayscale=bool(config["data"]["grayscale"]),
            normalize_mean=config["data"]["normalize_mean"],
            normalize_std=config["data"]["normalize_std"],
            horizontal_flip_prob=0.0,
        )
        try:
            shap_summary = run_image_shap_analysis(
                model=results["model"],
                background_dataset=shap_train_ds,
                explain_dataset=shap_test_ds,
                output_dir=output_dir,
                class_names=list(config["data"]["class_names"]),
                prefix="cnn",
                normalize_mean=config["data"]["normalize_mean"],
                normalize_std=config["data"]["normalize_std"],
                explain_metadata=shap_test_ds.metadata_frame(),
                max_background=int(config["shap"]["background_size"]),
                max_samples=int(config["shap"]["num_explain_samples"]),
            )
            write_json(shap_summary, output_dir / "shap_summary.json")
        except Exception as exc:  # pragma: no cover - diagnostic path
            write_json({"shap_error": str(exc)}, output_dir / "shap_error.json")

    return 0
