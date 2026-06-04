"""Shared metric helpers for thesis evaluation scripts."""

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_multiclass_metrics(y_true, y_pred, num_classes: int):
    labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def compute_binary_metrics(y_true, y_pred):
    labels = [0, 1]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
