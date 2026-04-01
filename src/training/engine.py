from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.training.callbacks import EarlyStopping
from src.training.metrics import compute_classification_metrics


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
) -> dict[str, Any]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_examples = 0
    all_targets: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(is_training):
            logits = model(features)
            loss = criterion(logits, targets)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

        all_targets.append(targets.detach().cpu().numpy())
        all_preds.append(predictions.detach().cpu().numpy())
        all_probabilities.append(probabilities.detach().cpu().numpy())

    targets_np = np.concatenate(all_targets)
    preds_np = np.concatenate(all_preds)
    probabilities_np = np.concatenate(all_probabilities)

    return {
        "loss": total_loss / max(total_examples, 1),
        "targets": targets_np,
        "predictions": preds_np,
        "probabilities": probabilities_np,
    }


def fit_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_names: list[str],
    num_epochs: int,
    patience: int,
    grad_clip: float | None = None,
    scheduler: Any | None = None,
    monitor_metric: str = "val_macro_f1",
    monitor_mode: str = "max",
    monitor_min_delta: float = 0.0,
) -> dict[str, Any]:
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
    }
    early_stopping = EarlyStopping(
        patience=patience,
        mode=monitor_mode,
        min_delta=monitor_min_delta,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_metrics: dict[str, Any] | None = None

    for epoch in range(1, num_epochs + 1):
        train_outputs = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            grad_clip=grad_clip,
        )
        val_outputs = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        train_metrics = compute_classification_metrics(
            y_true=train_outputs["targets"],
            y_pred=train_outputs["predictions"],
            probabilities=train_outputs["probabilities"],
            class_names=class_names,
        )
        val_metrics = compute_classification_metrics(
            y_true=val_outputs["targets"],
            y_pred=val_outputs["predictions"],
            probabilities=val_outputs["probabilities"],
            class_names=class_names,
        )

        history["train_loss"].append(train_outputs["loss"])
        history["val_loss"].append(val_outputs["loss"])
        history["train_macro_f1"].append(train_metrics["macro_f1"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])

        if scheduler is not None:
            scheduler.step(val_outputs["loss"])

        monitor_value_map = {
            "val_loss": val_outputs["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        }
        if monitor_metric not in monitor_value_map:
            raise ValueError(f"Unsupported monitor_metric: {monitor_metric}")

        monitored_value = monitor_value_map[monitor_metric]
        improved, should_stop = early_stopping.step(monitored_value)
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_metrics = val_metrics

        if should_stop:
            break

    model.load_state_dict(best_state)

    test_outputs = _run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
    )
    test_metrics = compute_classification_metrics(
        y_true=test_outputs["targets"],
        y_pred=test_outputs["predictions"],
        probabilities=test_outputs["probabilities"],
        class_names=class_names,
    )

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "monitor_metric": monitor_metric,
        "monitor_mode": monitor_mode,
        "test_metrics": test_metrics,
        "test_outputs": test_outputs,
    }
