from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


def _resolve_window_label(window_labels: np.ndarray, label_mode: str) -> int:
    if label_mode == "last":
        return int(window_labels[-1])
    if label_mode == "majority":
        return Counter(window_labels.tolist()).most_common(1)[0][0]
    raise ValueError(f"Unsupported label_mode: {label_mode}")


def build_sliding_windows(
    features: np.ndarray,
    labels: np.ndarray,
    window_length: int,
    stride: int = 1,
    label_mode: str = "last",
    metadata: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    feature_array = np.asarray(features, dtype=np.float32)
    label_array = np.asarray(labels)
    metadata = metadata or {}

    if feature_array.ndim != 2:
        raise ValueError("features must be a 2D array of shape (num_rows, num_features).")
    if len(feature_array) != len(label_array):
        raise ValueError("features and labels must have matching lengths.")
    if window_length <= 0:
        raise ValueError("window_length must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    if len(feature_array) < window_length:
        empty_x = np.empty((0, window_length, feature_array.shape[1]), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.int64)
        return empty_x, empty_y, []

    meta_arrays = {key: np.asarray(value) for key, value in metadata.items()}
    windows: list[np.ndarray] = []
    targets: list[int] = []
    meta_rows: list[dict[str, Any]] = []

    for start in range(0, len(feature_array) - window_length + 1, stride):
        end = start + window_length
        windows.append(feature_array[start:end])
        targets.append(_resolve_window_label(label_array[start:end], label_mode))

        row_meta: dict[str, Any] = {
            "start_index": start,
            "end_index": end - 1,
        }
        for key, value in meta_arrays.items():
            row_meta[key] = value[end - 1]
        meta_rows.append(row_meta)

    return np.stack(windows), np.asarray(targets, dtype=np.int64), meta_rows


def build_group_windows(
    frame: pd.DataFrame,
    group_cols: list[str],
    feature_cols: list[str],
    target_col: str,
    time_col: str,
    window_length: int,
    stride: int = 1,
    label_mode: str = "last",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    all_windows: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_metadata: list[dict[str, Any]] = []

    for group_key, group_frame in frame.groupby(group_cols, sort=False):
        group_frame = group_frame.sort_values(time_col).reset_index(drop=True)
        group_metadata = {time_col: group_frame[time_col].to_numpy()}
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        for key_name, key_value in zip(group_cols, group_key):
            group_metadata[key_name] = np.asarray([key_value] * len(group_frame))

        windows, targets, metadata = build_sliding_windows(
            features=group_frame[feature_cols].to_numpy(dtype=np.float32),
            labels=group_frame[target_col].to_numpy(),
            window_length=window_length,
            stride=stride,
            label_mode=label_mode,
            metadata=group_metadata,
        )

        if len(windows) == 0:
            continue

        all_windows.append(windows)
        all_targets.append(targets)
        all_metadata.extend(metadata)

    if not all_windows:
        empty_x = np.empty((0, window_length, len(feature_cols)), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.int64)
        return empty_x, empty_y, []

    return (
        np.concatenate(all_windows, axis=0),
        np.concatenate(all_targets, axis=0),
        all_metadata,
    )
