from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


TIME_COLUMN = "ts"
DROP_METADATA_COLUMNS = ["Unnamed: 0", "id", "fileid", "event"]
DEFAULT_DROP_CHANNELS = ["ch_12", "ch_23", "ch_24", "ch_26"]


def _resolve_anomaly_detection_data_dir(data_dir: str | Path) -> Path:
    path = Path(data_dir)
    if path.exists():
        return path
    raise FileNotFoundError(f"Anomaly-detection data directory not found: {path}")


def _list_anomaly_detection_files(data_dir: str | Path) -> list[Path]:
    resolved_dir = _resolve_anomaly_detection_data_dir(data_dir)
    return sorted(path for path in resolved_dir.glob("*.csv") if path.name != ".DS_Store")


def _resolve_feature_columns(columns: list[str], drop_channels: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if column.startswith("ch_") and column not in set(drop_channels)
    ]


def _downsample_frame(frame: pd.DataFrame, feature_cols: list[str], factor: int) -> pd.DataFrame:
    if factor <= 1:
        return frame.reset_index(drop=True)

    groups = np.arange(len(frame)) // factor
    aggregated = frame.groupby(groups, sort=False).agg(
        {TIME_COLUMN: "last", **{column: "mean" for column in feature_cols}}
    )
    return aggregated.reset_index(drop=True)


def _clean_anomaly_detection_frame(
    csv_path: Path,
    drop_channels: list[str],
    missing_sentinel: float,
    extreme_value_threshold: float,
    downsample_factor: int,
) -> tuple[pd.DataFrame, list[str]]:
    missing_sentinel = float(missing_sentinel)
    extreme_value_threshold = float(extreme_value_threshold)
    downsample_factor = int(downsample_factor)

    frame = pd.read_csv(csv_path)
    frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN], format="ISO8601")

    feature_cols = _resolve_feature_columns(frame.columns.tolist(), drop_channels)
    keep_cols = [TIME_COLUMN] + feature_cols
    cleaned = frame[keep_cols].copy()

    cleaned.loc[:, feature_cols] = cleaned[feature_cols].replace(missing_sentinel, np.nan)
    extreme_mask = cleaned[feature_cols].abs() > extreme_value_threshold
    cleaned.loc[:, feature_cols] = cleaned[feature_cols].mask(extreme_mask, np.nan)

    downsampled = _downsample_frame(cleaned, feature_cols=feature_cols, factor=downsample_factor)
    return downsampled, feature_cols


def _split_file_paths(
    file_paths: list[Path],
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[Path], list[Path], list[Path]]:
    num_files = len(file_paths)
    train_end = max(1, int(num_files * train_ratio))
    val_end = max(train_end + 1, int(num_files * (train_ratio + val_ratio)))
    if val_end >= num_files:
        val_end = num_files - 1

    return (
        file_paths[:train_end],
        file_paths[train_end:val_end],
        file_paths[val_end:],
    )


def _fit_scaler(records: list[dict[str, Any]], feature_cols: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    fitted = False

    for record in records:
        values = record["frame"][feature_cols].dropna().to_numpy(dtype=np.float64)
        if len(values) == 0:
            continue
        scaler.partial_fit(values)
        fitted = True

    if not fitted:
        raise ValueError("Could not fit scaler because no complete train rows were found.")
    return scaler


def _apply_scaler(
    frame: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler,
) -> pd.DataFrame:
    scaled = frame.copy()
    means = pd.Series(scaler.mean_, index=feature_cols)
    scales = pd.Series(np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_), index=feature_cols)
    scaled.loc[:, feature_cols] = (scaled[feature_cols] - means) / scales
    return scaled


def _load_split_records(
    file_paths: list[Path],
    split_name: str,
    drop_channels: list[str],
    missing_sentinel: float,
    extreme_value_threshold: float,
    downsample_factor: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    feature_cols: list[str] | None = None

    for file_path in file_paths:
        frame, current_feature_cols = _clean_anomaly_detection_frame(
            csv_path=file_path,
            drop_channels=drop_channels,
            missing_sentinel=missing_sentinel,
            extreme_value_threshold=extreme_value_threshold,
            downsample_factor=downsample_factor,
        )
        if feature_cols is None:
            feature_cols = current_feature_cols
        records.append(
            {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "split": split_name,
                "frame": frame,
            }
        )

    if feature_cols is None:
        feature_cols = []
    return records, feature_cols


def prepare_anomaly_detection_data(
    data_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    drop_channels: list[str] | None = None,
    missing_sentinel: float = -1000000.0,
    extreme_value_threshold: float = 1.0e12,
    downsample_factor: int = 5,
) -> dict[str, Any]:
    file_paths = _list_anomaly_detection_files(data_dir)
    if not file_paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}.")

    drop_channels = drop_channels or DEFAULT_DROP_CHANNELS
    train_paths, val_paths, test_paths = _split_file_paths(
        file_paths=file_paths,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    train_records, feature_cols = _load_split_records(
        train_paths,
        split_name="train",
        drop_channels=drop_channels,
        missing_sentinel=missing_sentinel,
        extreme_value_threshold=extreme_value_threshold,
        downsample_factor=downsample_factor,
    )
    val_records, _ = _load_split_records(
        val_paths,
        split_name="val",
        drop_channels=drop_channels,
        missing_sentinel=missing_sentinel,
        extreme_value_threshold=extreme_value_threshold,
        downsample_factor=downsample_factor,
    )
    test_records, _ = _load_split_records(
        test_paths,
        split_name="test",
        drop_channels=drop_channels,
        missing_sentinel=missing_sentinel,
        extreme_value_threshold=extreme_value_threshold,
        downsample_factor=downsample_factor,
    )

    scaler = _fit_scaler(train_records, feature_cols=feature_cols)
    for records in (train_records, val_records, test_records):
        for record in records:
            record["frame"] = _apply_scaler(record["frame"], feature_cols=feature_cols, scaler=scaler)

    return {
        "train": train_records,
        "val": val_records,
        "test": test_records,
        "feature_cols": feature_cols,
        "time_col": TIME_COLUMN,
        "scaler": scaler,
        "drop_channels": drop_channels,
        "all_file_names": [path.name for path in file_paths],
    }


def build_anomaly_reconstruction_windows(
    records: list[dict[str, Any]],
    feature_cols: list[str],
    window_length: int,
    stride: int,
    drop_nan_windows: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    windows: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []

    for file_order, record in enumerate(records):
        frame = record["frame"].reset_index(drop=True)
        values = frame[feature_cols].to_numpy(dtype=np.float32)
        timestamps = frame[TIME_COLUMN].astype(str).to_numpy()

        if len(values) < window_length:
            continue

        for start in range(0, len(values) - window_length + 1, stride):
            end = start + window_length
            window = values[start:end]
            if drop_nan_windows and np.isnan(window).any():
                continue

            windows.append(window)
            metadata.append(
                {
                    "split": record["split"],
                    "file_name": record["file_name"],
                    "file_order": file_order,
                    "start_index": start,
                    "end_index": end - 1,
                    "start_ts": timestamps[start],
                    "end_ts": timestamps[end - 1],
                }
            )

    if not windows:
        empty_x = np.empty((0, window_length, len(feature_cols)), dtype=np.float32)
        return empty_x, empty_x.copy(), []

    stacked = np.stack(windows).astype(np.float32)
    return stacked, stacked.copy(), metadata


def build_anomaly_forecast_windows(
    records: list[dict[str, Any]],
    feature_cols: list[str],
    input_length: int,
    horizon: int,
    stride: int,
    drop_nan_windows: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    total_length = input_length + horizon

    for file_order, record in enumerate(records):
        frame = record["frame"].reset_index(drop=True)
        values = frame[feature_cols].to_numpy(dtype=np.float32)
        timestamps = frame[TIME_COLUMN].astype(str).to_numpy()

        if len(values) < total_length:
            continue

        for start in range(0, len(values) - total_length + 1, stride):
            split_index = start + input_length
            end = start + total_length
            input_window = values[start:split_index]
            target_window = values[split_index:end]
            if drop_nan_windows and (np.isnan(input_window).any() or np.isnan(target_window).any()):
                continue

            inputs.append(input_window)
            targets.append(target_window)
            metadata.append(
                {
                    "split": record["split"],
                    "file_name": record["file_name"],
                    "file_order": file_order,
                    "input_start_index": start,
                    "input_end_index": split_index - 1,
                    "target_start_index": split_index,
                    "target_end_index": end - 1,
                    "input_start_ts": timestamps[start],
                    "input_end_ts": timestamps[split_index - 1],
                    "target_start_ts": timestamps[split_index],
                    "target_end_ts": timestamps[end - 1],
                }
            )

    if not inputs:
        empty_x = np.empty((0, input_length, len(feature_cols)), dtype=np.float32)
        empty_y = np.empty((0, horizon, len(feature_cols)), dtype=np.float32)
        return empty_x, empty_y, []

    return (
        np.stack(inputs).astype(np.float32),
        np.stack(targets).astype(np.float32),
        metadata,
    )
