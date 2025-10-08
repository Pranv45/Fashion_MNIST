"""
utils/optimizer_utils.py

Utility to select and initialize optimizers dynamically from config or CLI.
"""

from models.optimizers import SGD, Momentum, NAG, RMSProp, Adam, Nadam


def get_optimizer(name: str, lr: float, **kwargs):
    """
    Returns an optimizer instance by name.

    Parameters
    ----------
    name : str
        Name of the optimizer. One of:
        'sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam'
    lr : float
        Learning rate.
    kwargs : dict
        Additional optimizer parameters (momentum, beta1, beta2, etc.)

    Returns
    -------
    Optimizer
        Instantiated optimizer object.
    """
    name = name.lower()
    if name == "sgd":
        return SGD(lr)
    elif name == "momentum":
        return Momentum(lr, **{k: kwargs.get(k, 0.9) for k in ["momentum"]})
    elif name == "nesterov":
        return NAG(lr, **{k: kwargs.get(k, 0.9) for k in ["momentum"]})
    elif name == "rmsprop":
        return RMSProp(lr, **{k: kwargs.get(k, v) for k, v in {"beta": 0.9, "eps": 1e-8}.items()})
    elif name == "adam":
        return Adam(lr, **{k: kwargs.get(k, v) for k, v in {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8}.items()})
    elif name == "nadam":
        return Nadam(lr, **{k: kwargs.get(k, v) for k, v in {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8}.items()})
    else:
        raise ValueError(f"Unknown optimizer: {name}")
