"""
models/optimizers.py

Implementation of gradient-based optimizers for NumPy-based neural networks.
All optimizers share a unified interface:
    optimizer = OptimizerClass(lr, **kwargs)
    optimizer.update(params, grads)
"""

import numpy as np


# ----------------------------------------------------------------
# Base Class
# ----------------------------------------------------------------
class Optimizer:
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def update(self, params, grads):
        raise NotImplementedError("Subclasses must implement update() method.")


# ----------------------------------------------------------------
# 1. Stochastic Gradient Descent (SGD)
# ----------------------------------------------------------------
class SGD(Optimizer):
    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


# ----------------------------------------------------------------
# 2. Momentum
# ----------------------------------------------------------------
class Momentum(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.v = None  # velocity term

    def update(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]

        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            params[i] += self.v[i]


# ----------------------------------------------------------------
# 3. Nesterov Accelerated Gradient (NAG)
# ----------------------------------------------------------------
class NAG(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]

        for i in range(len(params)):
            prev_v = self.v[i].copy()
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            params[i] += -self.momentum * prev_v + (1 + self.momentum) * self.v[i]


# ----------------------------------------------------------------
# 4. RMSProp
# ----------------------------------------------------------------
class RMSProp(Optimizer):
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):
        super().__init__(lr)
        self.beta = beta
        self.eps = eps
        self.s = None  # running average of squared gradients

    def update(self, params, grads):
        if self.s is None:
            self.s = [np.zeros_like(p) for p in params]

        for i in range(len(params)):
            self.s[i] = self.beta * self.s[i] + (1 - self.beta) * (grads[i] ** 2)
            params[i] -= self.lr * grads[i] / (np.sqrt(self.s[i]) + self.eps)


# ----------------------------------------------------------------
# 5. Adam
# ----------------------------------------------------------------
class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0  # timestep

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ----------------------------------------------------------------
# 6. Nadam (Adam + Nesterov momentum)
# ----------------------------------------------------------------
class Nadam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1
        mu_t = self.beta1 * (1 - 0.5 * (0.96 ** (self.t / 250)))
        mu_t_next = self.beta1 * (1 - 0.5 * (0.96 ** ((self.t + 1) / 250)))

        for i in range(len(params)):
            self.m[i] = mu_t * self.m[i] + (1 - mu_t) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)

            m_hat = self.m[i] / (1 - np.prod([(1 - mu_t_next)]))
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= self.lr * (mu_t_next * m_hat + (1 - mu_t) * grads[i] / (1 - mu_t_next)) / (np.sqrt(v_hat) + self.eps)
