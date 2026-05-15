from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _normalize_label_value(label: Any):
    if isinstance(label, np.generic):
        return label.item()
    return label


def _label_key(label: Any) -> str:
    value = _normalize_label_value(label)
    return f"{type(value).__name__}:{value}"


def _json_safe_label(label: Any):
    value = _normalize_label_value(label)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def build_label_classes(labels) -> list:
    unique_labels = list(pd.unique(np.asarray(labels)))

    try:
        numeric_values = pd.to_numeric(pd.Series(unique_labels), errors="raise").to_numpy()
        order = np.argsort(numeric_values, kind="stable")
        return [unique_labels[i] for i in order]
    except (TypeError, ValueError):
        return sorted(unique_labels, key=lambda value: str(value))


def encode_labels(labels, class_labels: list | None = None) -> tuple[np.ndarray, list]:
    if class_labels is None:
        class_labels = build_label_classes(labels)

    class_to_index = {_label_key(label): idx for idx, label in enumerate(class_labels)}
    encoded = []
    for label in labels:
        key = _label_key(label)
        if key not in class_to_index:
            raise ValueError(f"Unknown label '{label}' found while applying label encoding.")
        encoded.append(class_to_index[key])

    return np.asarray(encoded, dtype=np.int64), class_labels


def save_label_mapping(path: str | Path, class_labels: list) -> Path:
    path = Path(path)
    payload = {
        "classes": [
            {"encoded": idx, "label": _json_safe_label(label)}
            for idx, label in enumerate(class_labels)
        ]
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
