from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.training.callbacks import EarlyStopping


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
    collect_outputs: bool = False,
) -> dict[str, Any]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_examples = 0
    all_scores: list[np.ndarray] = []
    all_channel_scores: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(is_training):
            predictions = model(features)
            loss = criterion(predictions, targets)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        squared_error = (predictions - targets).pow(2)
        window_scores = squared_error.mean(dim=(1, 2)).detach().cpu().numpy()
        channel_scores = squared_error.mean(dim=1).detach().cpu().numpy()

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        all_scores.append(window_scores)
        all_channel_scores.append(channel_scores)

        if collect_outputs:
            all_predictions.append(predictions.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    outputs: dict[str, Any] = {
        "loss": total_loss / max(total_examples, 1),
        "window_scores": np.concatenate(all_scores) if all_scores else np.empty((0,), dtype=np.float32),
        "channel_scores": (
            np.concatenate(all_channel_scores, axis=0)
            if all_channel_scores
            else np.empty((0, 0), dtype=np.float32)
        ),
    }
    if collect_outputs:
        outputs["predictions"] = (
            np.concatenate(all_predictions, axis=0)
            if all_predictions
            else np.empty((0,), dtype=np.float32)
        )
        outputs["targets"] = (
            np.concatenate(all_targets, axis=0)
            if all_targets
            else np.empty((0,), dtype=np.float32)
        )
    return outputs


def fit_sequence_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    patience: int,
    grad_clip: float | None = None,
    scheduler: Any | None = None,
    min_delta: float = 0.0,
) -> dict[str, Any]:
    history = {"train_loss": [], "val_loss": []}
    early_stopping = EarlyStopping(patience=patience, mode="min", min_delta=min_delta)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_outputs = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            grad_clip=grad_clip,
            collect_outputs=False,
        )
        val_outputs = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            collect_outputs=False,
        )

        history["train_loss"].append(train_outputs["loss"])
        history["val_loss"].append(val_outputs["loss"])

        if scheduler is not None:
            scheduler.step(val_outputs["loss"])

        improved, should_stop = early_stopping.step(val_outputs["loss"])
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_loss = val_outputs["loss"]

        if should_stop:
            break

    model.load_state_dict(best_state)
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


def evaluate_sequence_model(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    collect_outputs: bool = False,
) -> dict[str, Any]:
    return _run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        collect_outputs=collect_outputs,
    )


def representative_window_index(window_scores: np.ndarray, quantile: float) -> int | None:
    scores = np.asarray(window_scores, dtype=np.float32)
    if len(scores) == 0:
        return None
    quantile = float(np.clip(quantile, 0.0, 1.0))
    target_score = float(np.quantile(scores, quantile))
    return int(np.argmin(np.abs(scores - target_score)))


def window_scores_frame(
    metadata: list[dict[str, Any]],
    window_scores: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(metadata)
    frame["window_score"] = window_scores
    return frame


def aggregate_file_scores(
    metadata: list[dict[str, Any]],
    window_scores: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    score_frame = window_scores_frame(metadata=metadata, window_scores=window_scores)
    if score_frame.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "file_name",
                "file_order",
                "num_windows",
                "mean_score",
                "median_score",
                "p95_score",
                "max_score",
                "exceedance_rate",
            ]
        )

    aggregated = (
        score_frame.groupby(["split", "file_name", "file_order"], as_index=False)
        .agg(
            num_windows=("window_score", "size"),
            mean_score=("window_score", "mean"),
            median_score=("window_score", "median"),
            p95_score=("window_score", lambda values: float(np.quantile(values, 0.95))),
            max_score=("window_score", "max"),
        )
        .sort_values(["split", "file_order", "file_name"])
        .reset_index(drop=True)
    )

    exceedance = (
        score_frame.assign(exceeds=score_frame["window_score"] > threshold)
        .groupby(["split", "file_name", "file_order"], as_index=False)["exceeds"]
        .mean()
        .rename(columns={"exceeds": "exceedance_rate"})
    )
    return aggregated.merge(exceedance, on=["split", "file_name", "file_order"], how="left")


def summarize_score_distribution(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    quantiles: list[float] | None = None,
) -> dict[str, Any]:
    quantiles = quantiles or [0.95, 0.99]
    summary: dict[str, Any] = {}

    for split_name, scores in {
        "train": train_scores,
        "val": val_scores,
        "test": test_scores,
    }.items():
        split_summary: dict[str, Any] = {
            "num_windows": int(len(scores)),
            "mean_score": float(np.mean(scores)) if len(scores) else float("nan"),
            "median_score": float(np.median(scores)) if len(scores) else float("nan"),
            "max_score": float(np.max(scores)) if len(scores) else float("nan"),
        }
        for quantile in quantiles:
            label = f"q{int(quantile * 100):02d}"
            split_summary[label] = float(np.quantile(scores, quantile)) if len(scores) else float("nan")
        summary[split_name] = split_summary

    return summary
