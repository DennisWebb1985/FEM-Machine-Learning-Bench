from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.classification_labels import class_names_from_labels


TIMESTAMP_COLUMN = "Timestamp"
TARGET_COLUMN = "Damage State Label"
LEGACY_TARGET_COLUMNS = ("Condition Label",)
LEGACY_DATA_FILENAMES = {
    "building_health_monitoring_dataset.csv",
}
DEFAULT_FEATURE_COLUMNS = [
    "Accel_X (m/s^2)",
    "Accel_Y (m/s^2)",
    "Accel_Z (m/s^2)",
    "Strain (με)",
    "Temp (°C)",
]


def _fill_missing_features(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    filled = frame.copy()
    filled[feature_cols] = filled[feature_cols].interpolate(
        method="linear",
        limit_direction="both",
    )
    filled[feature_cols] = filled[feature_cols].ffill().bfill()
    return filled


def _resolve_classification_csv_path(csv_path: str | Path) -> Path:
    path = Path(csv_path)
    if path.exists():
        return path

    if path.name in LEGACY_DATA_FILENAMES:
        classification_path = path.with_name("classification.csv")
        if classification_path.exists():
            return classification_path

    raise FileNotFoundError(f"Classification data file not found: {path}")


def _resolve_target_column(frame: pd.DataFrame) -> str:
    if TARGET_COLUMN in frame.columns:
        return TARGET_COLUMN
    for legacy_column in LEGACY_TARGET_COLUMNS:
        if legacy_column in frame.columns:
            return legacy_column
    raise KeyError(f"Target column not found. Expected one of: {[TARGET_COLUMN, *LEGACY_TARGET_COLUMNS]}")


def prepare_building_data(
    csv_path: str | Path,
    feature_cols: list[str] | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, Any]:
    feature_cols = feature_cols or DEFAULT_FEATURE_COLUMNS
    resolved_csv_path = _resolve_classification_csv_path(csv_path)
    frame = pd.read_csv(resolved_csv_path, parse_dates=[TIMESTAMP_COLUMN]).sort_values(TIMESTAMP_COLUMN)
    frame = frame.reset_index(drop=True)
    target_column = _resolve_target_column(frame)
    if target_column != TARGET_COLUMN:
        frame = frame.rename(columns={target_column: TARGET_COLUMN})

    num_rows = len(frame)
    train_end = int(num_rows * train_ratio)
    val_end = int(num_rows * (train_ratio + val_ratio))

    train_frame = frame.iloc[:train_end].copy()
    val_frame = frame.iloc[train_end:val_end].copy()
    test_frame = frame.iloc[val_end:].copy()

    train_frame = _fill_missing_features(train_frame, feature_cols)
    val_frame = _fill_missing_features(val_frame, feature_cols)
    test_frame = _fill_missing_features(test_frame, feature_cols)

    scaler = StandardScaler()
    train_frame.loc[:, feature_cols] = scaler.fit_transform(train_frame[feature_cols])
    val_frame.loc[:, feature_cols] = scaler.transform(val_frame[feature_cols])
    test_frame.loc[:, feature_cols] = scaler.transform(test_frame[feature_cols])

    labels = sorted(int(label) for label in frame[TARGET_COLUMN].unique().tolist())
    class_names = class_names_from_labels(labels)

    return {
        "train": train_frame.reset_index(drop=True),
        "val": val_frame.reset_index(drop=True),
        "test": test_frame.reset_index(drop=True),
        "feature_cols": feature_cols,
        "target_col": TARGET_COLUMN,
        "time_col": TIMESTAMP_COLUMN,
        "class_names": class_names,
        "scaler": scaler,
    }
