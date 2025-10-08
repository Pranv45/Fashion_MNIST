# models/activations.py
"""
Activation functions and their derivatives for feedforward networks.
"""

import numpy as np

# -------------------------------
# Activation Functions
# -------------------------------

def identity(x):
    """Linear activation: f(x) = x"""
    return x

def sigmoid(x):
    """Sigmoid activation: f(x) = 1 / (1 + e^-x)"""
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """Tanh activation"""
    return np.tanh(x)

def relu(x):
    """ReLU activation: f(x) = max(0, x)"""
    return np.maximum(0, x)


# -------------------------------
# Derivatives (for backprop)
# -------------------------------

def identity_deriv(x):
    """Derivative of identity: f'(x) = 1"""
    return np.ones_like(x)

def sigmoid_deriv(output):
    """Derivative of sigmoid wrt output: f'(x) = y * (1 - y)"""
    return output * (1 - output)

def tanh_deriv(output):
    """Derivative of tanh wrt output: f'(x) = 1 - y^2"""
    return 1 - output ** 2

def relu_deriv(x):
    """Derivative of ReLU: f'(x) = 1 if x>0 else 0"""
    return (x > 0).astype(float)


# -------------------------------
# Helper: Get activation by name
# -------------------------------

def get_activation(name):
    """
    Returns activation and derivative functions.

    Returns:
    --------
    act_func : callable
        Function for forward pass
    deriv_func : callable
        Function for derivative computation
    deriv_input_type : str
        'a' if derivative takes activation output, 'z' if it takes pre-activation
    """
    name = name.lower()

    if name in ['identity', 'linear']:
        return identity, identity_deriv, 'z'

    elif name == 'sigmoid':
        return sigmoid, sigmoid_deriv, 'a'  # derivative depends on output

    elif name == 'tanh':
        return tanh, tanh_deriv, 'a'  # derivative depends on output

    elif name in ['relu', 'ReLU']:
        return relu, relu_deriv, 'z'  # derivative depends on pre-activation

    else:
        raise ValueError(f"Unknown activation function: {name}")
