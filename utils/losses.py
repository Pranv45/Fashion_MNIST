# utils/losses.py
"""
Loss functions and helpers for feedforward neural networks.
"""

import numpy as np

def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities for each row of the input matrix.
    Stable: subtract per-row max and use keepdims for safe broadcasting.
    """
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def to_one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer labels into one-hot encoded matrix.
    Example: [1, 2] -> [[0,1,0], [0,0,1]]
    """
    y_onehot = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    y_onehot[np.arange(y.shape[0]), y] = 1.0
    return y_onehot

# -------- Cross-Entropy -------- #

def cross_entropy_loss(probs: np.ndarray, y_onehot: np.ndarray, eps: float = 1e-12) -> float:
    """
    Average cross-entropy loss given predicted probabilities and one-hot labels.
    """
    probs = np.clip(probs, eps, 1 - eps)
    return -np.mean(np.sum(y_onehot * np.log(probs), axis=1))

def cross_entropy_grad(probs: np.ndarray, y_onehot: np.ndarray) -> np.ndarray:
    """
    Gradient of softmax-cross-entropy w.r.t. logits (not averaged).
    For softmax+CE, dL/dz = probs - y_onehot.
    """
    return (probs - y_onehot)

# -------- Mean Squared Error -------- #

def mse_loss(preds: np.ndarray, y_onehot: np.ndarray) -> float:
    """
    Mean Squared Error between predictions and one-hot labels.
    """
    return 0.5 * np.mean((preds - y_onehot) ** 2)

def mse_grad(preds: np.ndarray, y_onehot: np.ndarray) -> np.ndarray:
    """
    Gradient of MSE w.r.t. predictions (unnormalized; chain via softmax if needed).
    """
    return (preds - y_onehot)
