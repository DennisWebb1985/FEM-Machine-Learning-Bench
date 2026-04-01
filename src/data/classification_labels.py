from __future__ import annotations

import re
from collections.abc import Iterable


CLASS_NAME_BY_LABEL = {
    0: "Healthy",
    1: "Minor Damage",
    2: "Severe Damage",
}
LEGACY_CLASS_NAME_BY_LABEL = {
    label: f"Condition {label}" for label in CLASS_NAME_BY_LABEL
}
CLASS_LABEL_BY_NAME = {name: label for label, name in CLASS_NAME_BY_LABEL.items()}
LEGACY_TO_CURRENT_CLASS_NAMES = {
    legacy_name: CLASS_NAME_BY_LABEL[label]
    for label, legacy_name in LEGACY_CLASS_NAME_BY_LABEL.items()
}


def class_name_for_label(label: int) -> str:
    int_label = int(label)
    if int_label not in CLASS_NAME_BY_LABEL:
        raise ValueError(f"Unsupported classification label: {int_label}")
    return CLASS_NAME_BY_LABEL[int_label]


def class_slug_for_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def class_slug_for_label(label: int) -> str:
    return class_slug_for_name(class_name_for_label(label))


def class_names_from_labels(labels: Iterable[int]) -> list[str]:
    ordered_labels = [int(label) for label in labels]
    unknown = sorted({label for label in ordered_labels if label not in CLASS_NAME_BY_LABEL})
    if unknown:
        raise ValueError(f"Unsupported classification labels: {unknown}")
    return [CLASS_NAME_BY_LABEL[label] for label in ordered_labels]

