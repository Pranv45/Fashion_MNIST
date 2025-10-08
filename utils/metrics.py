# utils/metrics.py
"""
Evaluation metrics for classification tasks.
"""

import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.
    """
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    """
    Compute confusion matrix for multi-class classification.
    """
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max()) + 1)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm
