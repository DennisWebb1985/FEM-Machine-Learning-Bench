from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None,
    class_names: list[str],
) -> dict[str, Any]:
    labels = list(range(len(class_names)))
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }
    if probabilities is not None:
        try:
            metrics["ovr_auroc"] = float(
                roc_auc_score(y_true, probabilities, multi_class="ovr", labels=labels)
            )
        except ValueError:
            metrics["ovr_auroc"] = None
    else:
        metrics["ovr_auroc"] = None
    return metrics


def classification_report_frame(report_dict: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(report_dict).transpose().rename_axis("class_name").reset_index()
