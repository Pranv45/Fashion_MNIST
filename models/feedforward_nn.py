# models/feedforward_nn.py
"""
Flexible FeedForward Neural Network (NumPy).
"""

from typing import List, Union, Tuple, Dict, Any
import numpy as np

from models.activations import get_activation
from models.initializers import xavier_init, random_init
from utils.losses import softmax, to_one_hot, cross_entropy_grad, mse_grad

class FeedForwardNN:
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: List[int],
        output_dim: int,
        activation: Union[str, List[str]] = "relu",
        weight_init: str = "xavier",
        weight_decay: float = 0.0,
        seed: int | None = None,
        dtype = np.float32,
    ):
        self.dtype = dtype
        self.rng = np.random.default_rng(seed)
        self.layer_sizes = [int(input_dim)] + [int(h) for h in hidden_sizes] + [int(output_dim)]
        self.n_layers = len(self.layer_sizes) - 1
        if self.n_layers < 1:
            raise ValueError("Network must have at least one layer (input->output)")

        n_hidden = max(0, self.n_layers - 1)
        if isinstance(activation, list):
            if len(activation) != n_hidden:
                raise ValueError(f"If activation is a list it must have length {n_hidden}")
            activations_list = activation
        else:
            activations_list = [activation] * n_hidden

        self.activations: List[Tuple] = []
        for act_name in activations_list:
            act_fwd, act_deriv, deriv_arg = get_activation(act_name)
            self.activations.append((act_fwd, act_deriv, deriv_arg))

        self.weight_decay = float(weight_decay)
        self.weight_init = weight_init.lower()
        self._init_weights(self.weight_init)

        self.caches: List[Dict[str, np.ndarray]] = []

    def _init_weights(self, scheme: str):
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(self.n_layers):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            if scheme in ("xavier", "glorot", "glorot_uniform"):
                W = xavier_init(n_in, n_out)
            elif scheme in ("random", "normal"):
                W = random_init(n_in, n_out, scale=0.01)
            else:
                raise ValueError(f"Unknown weight_init scheme: {scheme}")
            b = np.zeros((n_out,), dtype=self.dtype)
            self.weights.append(W.astype(self.dtype))
            self.biases.append(b.astype(self.dtype))

    def forward(self, X: np.ndarray, return_logits: bool = False) -> np.ndarray:
        a = X.astype(self.dtype)
        caches: List[Dict[str, np.ndarray]] = []
        caches.append({'a': a.copy()})
        for i in range(self.n_layers):
            W = self.weights[i]
            b = self.biases[i]
            z = np.dot(a, W)
            z = z + b.reshape(1, -1)
            if i == self.n_layers - 1:
                logits = z
                probs = softmax(logits)
                caches.append({'z': logits, 'a': probs})
                a = probs
            else:
                act_fwd, _, _ = self.activations[i]
                a = act_fwd(z)
                caches.append({'z': z, 'a': a})
        self.caches = caches
        return (logits if return_logits else a)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X, return_logits=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def compute_gradients(self, X_batch: np.ndarray, y_batch: np.ndarray,
                          loss: str = "cross_entropy", n_classes: int = 10) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        batch_size = X_batch.shape[0]
        probs = self.forward(X_batch)  # (batch, output_dim)
        y_oh = to_one_hot(y_batch, n_classes)

        # Compute scalar loss (stored for monitoring)
        if loss == "cross_entropy":
            self.current_loss = -np.mean(np.sum(y_oh * np.log(np.clip(probs, 1e-12, 1.0)), axis=1))
            # For softmax+CE, gradient w.r.t logits: dZ = probs - y
            dZ = cross_entropy_grad(probs, y_oh)
        elif loss in ("mse", "mean_squared_error"):
            self.current_loss = 0.5 * np.mean((probs - y_oh) ** 2)
            # Chain MSE gradient through softmax Jacobian:
            # dL/dp = g; dL/dz = p * (g - <g,p>)
            g = mse_grad(probs, y_oh)
            gp = np.sum(g * probs, axis=1, keepdims=True)
            dZ = probs * (g - gp)
        else:
            raise ValueError("Unsupported loss. Use 'cross_entropy' or 'mean_squared_error'.")

        dWs: List[np.ndarray] = [None] * self.n_layers
        dbs: List[np.ndarray] = [None] * self.n_layers

        for i in reversed(range(self.n_layers)):
            a_prev = self.caches[i]['a']  # (batch, n_in)
            dW = np.dot(a_prev.T, dZ)     # (n_in, n_out)
            db = np.sum(dZ, axis=0)       # (n_out,)

            # L2 weight decay term (added to gradient)
            if self.weight_decay and self.weight_decay != 0.0:
                dW = dW + (self.weight_decay * self.weights[i])

            # Average over batch once (no pre-division inside *_grad)
            dW = dW / batch_size
            db = db / batch_size

            dWs[i] = dW
            dbs[i] = db

            if i > 0:
                dA_prev = np.dot(dZ, self.weights[i].T)
                _, act_deriv, deriv_arg = self.activations[i - 1]
                prev_cache = self.caches[i]  # z/a for layer (i-1)
                if deriv_arg == 'a':
                    deriv_vals = act_deriv(prev_cache['a'])
                else:
                    deriv_vals = act_deriv(prev_cache['z'])
                dZ = dA_prev * deriv_vals

        return dWs, dbs

    def get_params(self) -> Dict[str, Any]:
        return {'weights': [w.copy() for w in self.weights],
                'biases': [b.copy() for b in self.biases]}

    def set_params(self, params: Dict[str, Any]):
        ws = params.get('weights')
        bs = params.get('biases')
        if ws is None or bs is None or len(ws) != self.n_layers or len(bs) != self.n_layers:
            raise ValueError("Invalid params structure for set_params.")
        self.weights = [w.astype(self.dtype).copy() for w in ws]
        self.biases = [b.astype(self.dtype).copy() for b in bs]

    def save(self, path: str):
        to_save = {}
        for i, W in enumerate(self.weights):
            to_save[f'w{i}'] = W
        for i, b in enumerate(self.biases):
            to_save[f'b{i}'] = b
        np.savez(path, **to_save)

    def load(self, path: str):
        npz = np.load(path, allow_pickle=True)
        ws = []
        bs = []
        for i in range(self.n_layers):
            ws.append(npz[f'w{i}'].astype(self.dtype))
        for i in range(self.n_layers):
            bs.append(npz[f'b{i}'].astype(self.dtype))
        self.weights = ws
        self.biases = bs

    def __repr__(self):
        desc = f"FeedForwardNN(layers={self.layer_sizes}, weight_init={self.weight_init}, weight_decay={self.weight_decay})"
        return desc
