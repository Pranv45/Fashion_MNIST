"""
This module handles loading and batching of the Fashion-MNIST dataset.
We'll write:
1. load_data() - loads, normalizes, flattens, splits
2. DataLoader - creates mini-batches for training
"""

import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split


def load_data(dataset='fashion_mnist', val_frac=0.1, flatten=True, normalize=True, random_seed=42):
    """
    Loads and preprocesses the Fashion-MNIST dataset.

    Parameters:
    -----------
    dataset : str
        Which dataset to load ('fashion_mnist' or 'mnist')
    val_frac : float
        Fraction of training data to use for validation
    flatten : bool
        Whether to flatten 28x28 images into 784-d vectors
    normalize : bool
        Whether to scale pixel values to [0,1]
    random_seed : int
        Random seed for reproducible validation split

    Returns:
    --------
    X_train, y_train, X_val, y_val, X_test, y_test : np.ndarrays
    """
    # Load dataset
    if dataset == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        from tensorflow.keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Convert to float32 and normalize to [0,1]
    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test  = X_test.astype(np.float32) / 255.0

    # Flatten 28x28 → 784
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test  = X_test.reshape(X_test.shape[0], -1)

    # Split off validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_frac, random_state=random_seed, stratify=y_train)

    return X_train, y_train, X_val, y_val, X_test, y_test


class DataLoader:
    """
    Lightweight DataLoader to create mini-batches.
    """

    def __init__(self, X, y, batch_size=64, shuffle=True, seed=None):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)  # random generator for reproducibility

    def __iter__(self):
        # Generate indices from 0 to len(X)-1
        indices = np.arange(self.X.shape[0])
        if self.shuffle:
            self.rng.shuffle(indices)
        # Yield mini-batches
        for start in range(0, len(indices), self.batch_size):
            end = start + self.batch_size
            batch_idx = indices[start:end]
            yield self.X[batch_idx], self.y[batch_idx]

    def __len__(self):
        # Returns number of batches
        return int(np.ceil(self.X.shape[0] / self.batch_size))
