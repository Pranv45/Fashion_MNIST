# models/initializers.py
"""
Weight initialization methods for feedforward neural networks.
"""

import numpy as np

def random_init(n_in, n_out, scale=0.01):
    """
    Initialize weights randomly from a normal distribution.

    Parameters
    ----------
    n_in : int
        Number of input neurons to the layer
    n_out : int
        Number of output neurons from the layer
    scale : float, optional
        Standard deviation for the random values (default 0.01)

    Returns
    -------
    W : np.ndarray of shape (n_in, n_out)
        Randomly initialized weights
    """
    return np.random.randn(n_in, n_out) * scale


def xavier_init(n_in, n_out):
    """
    Xavier (Glorot Uniform) initialization.
    This keeps the variance of activations constant across layers.

    Parameters
    ----------
    n_in : int
        Number of input neurons
    n_out : int
        Number of output neurons

    Returns
    -------
    W : np.ndarray of shape (n_in, n_out)
        Initialized weights from uniform[-limit, limit]
    """
    limit = np.sqrt(6.0 / (n_in + n_out))
    return np.random.uniform(-limit, limit, size=(n_in, n_out))
