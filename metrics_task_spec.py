from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)


FOUR_CLASS_LABELS = ["None", "Mild", "Moderate", "Severe+"]
BINARY_LABELS = ["None", "Drought"]
SEVERITY_LABELS = ["Mild", "Moderate", "Severe+"]


def _as_1d_int(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D label array, got shape {arr.shape}")
    return arr


def _float(value: Any) -> float:
    return float(value)


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def per_class_metrics(y_true: Any, y_pred: Any, labels: list[str]) -> dict[str, dict[str, float | int]]:
    y_true = _as_1d_int(y_true)
    y_pred = _as_1d_int(y_pred)
    class_ids = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_ids,
        zero_division=0,
    )
    return {
        label: {
            "precision": _float(precision[i]),
            "recall": _float(recall[i]),
            "f1": _float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }


def multiclass_metrics(y_true: Any, y_pred: Any, labels: list[str] | None = None) -> dict[str, Any]:
    """Return the full metric set required by the AlfaFlow task spec."""
    if labels is None:
        labels = FOUR_CLASS_LABELS
    y_true = _as_1d_int(y_true)
    y_pred = _as_1d_int(y_pred)
    class_ids = list(range(len(labels)))
    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    out: dict[str, Any] = {
        "Accuracy": _float(accuracy_score(y_true, y_pred)),
        "Macro F1": _float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "MAE": _float(mean_absolute_error(y_true, y_pred)),
        "RMSE": _float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_metrics(y_true, y_pred, labels),
    }
    for label in ("Moderate", "Severe+"):
        if label in out["per_class"]:
            out[f"{label} Recall"] = out["per_class"][label]["recall"]
            out[f"{label} Precision"] = out["per_class"][label]["precision"]
    if "Severe+" in labels:
        severe_idx = labels.index("Severe+")
        severe_total = int(cm[severe_idx, :].sum())
        severe_fn = severe_total - int(cm[severe_idx, severe_idx])
        out["Severe+ False Negative Rate"] = _safe_rate(severe_fn, severe_total)
    return out


def binary_detection_metrics(y_true_binary: Any, y_pred_binary: Any) -> dict[str, Any]:
    """Metrics for Task 2 stage 1: None vs any drought."""
    y_true = _as_1d_int(y_true_binary)
    y_pred = _as_1d_int(y_pred_binary)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    return {
        "Accuracy": _float(accuracy_score(y_true, y_pred)),
        "Precision": _float(precision[1]),
        "Recall": _float(recall[1]),
        "F1": _float(f1[1]),
        "Macro F1": _float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Drought Recall": _float(recall[1]),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            BINARY_LABELS[i]: {
                "precision": _float(precision[i]),
                "recall": _float(recall[i]),
                "f1": _float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(2)
        },
    }


def task2_critical_errors(y_true_4class: Any, y_pred_4class: Any) -> dict[str, float | int]:
    """Named Task 2 operational error categories for final 4-class outputs."""
    y_true = _as_1d_int(y_true_4class)
    y_pred = _as_1d_int(y_pred_4class)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    severe_total = int((y_true == 3).sum())
    moderate_total = int((y_true == 2).sum())
    mild_total = int((y_true == 1).sum())
    none_total = int((y_true == 0).sum())

    severe_as_none = int(((y_true == 3) & (y_pred == 0)).sum())
    moderate_as_mild = int(((y_true == 2) & (y_pred == 1)).sum())
    mild_as_severe = int(((y_true == 1) & (y_pred == 3)).sum())
    none_as_drought = int(((y_true == 0) & (y_pred > 0)).sum())

    return {
        "Severe+ predicted as None": severe_as_none,
        "Severe+ predicted as None rate": _safe_rate(severe_as_none, severe_total),
        "Moderate predicted as Mild": moderate_as_mild,
        "Moderate predicted as Mild rate": _safe_rate(moderate_as_mild, moderate_total),
        "Mild predicted as Severe+": mild_as_severe,
        "Mild predicted as Severe+ rate": _safe_rate(mild_as_severe, mild_total),
        "None predicted as Drought": none_as_drought,
        "None predicted as Drought rate": _safe_rate(none_as_drought, none_total),
    }


def summarize_runs(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, dict[str, float]]:
    """Mean/std/min/max summary for repeated-seed result rows."""
    summary: dict[str, dict[str, float]] = {}
    for name in metric_names:
        vals = np.array([row[name] for row in rows if name in row], dtype=np.float64)
        if len(vals) == 0:
            continue
        summary[name] = {
            "mean": _float(vals.mean()),
            "std": _float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": _float(vals.min()),
            "max": _float(vals.max()),
        }
    return summary
