from __future__ import annotations

import numpy as np


class MLPRegressorNumpy:
    """Small fully-connected regressor with ReLU activations (NumPy only)."""

    def __init__(self, in_dim: int, hidden_dims: list[int], seed: int = 42):
        rng = np.random.default_rng(seed)
        dims = [in_dim] + hidden_dims + [1]
        self.weights = []
        self.biases = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            scale = np.sqrt(2.0 / d_in)
            self.weights.append(rng.normal(0.0, scale, size=(d_in, d_out)))
            self.biases.append(np.zeros((1, d_out)))
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    @staticmethod
    def _relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(np.float64)

    def _forward(self, x: np.ndarray):
        a = x
        pre_acts = []
        acts = [x]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            pre_acts.append(z)
            if i < len(self.weights) - 1:
                a = self._relu(z)
            else:
                a = z
            acts.append(a)
        return a, pre_acts, acts

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.x_mean is None or self.x_std is None:
            raise RuntimeError("Model is not fitted.")
        x_n = (x - self.x_mean) / self.x_std
        y_n, _, _ = self._forward(x_n)
        y = y_n * self.y_std + self.y_mean
        return y.ravel()

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        lr: float,
        batch_size: int,
    ) -> list[float]:
        y = y.reshape(-1, 1)
        self.x_mean = np.mean(x, axis=0, keepdims=True)
        self.x_std = np.std(x, axis=0, keepdims=True) + 1e-8
        self.y_mean = np.mean(y, axis=0, keepdims=True)
        self.y_std = np.std(y, axis=0, keepdims=True) + 1e-8

        x = (x - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std
        n = x.shape[0]
        history = []
        for _ in range(epochs):
            idx = np.random.permutation(n)
            x_shuf, y_shuf = x[idx], y[idx]
            epoch_loss = 0.0
            for s in range(0, n, batch_size):
                xb = x_shuf[s : s + batch_size]
                yb = y_shuf[s : s + batch_size]

                pred, pre_acts, acts = self._forward(xb)
                diff = pred - yb
                loss = np.mean(diff**2)
                epoch_loss += loss * xb.shape[0]

                grad = 2.0 * diff / xb.shape[0]
                grad_ws = [None] * len(self.weights)
                grad_bs = [None] * len(self.biases)

                for l in reversed(range(len(self.weights))):
                    grad_ws[l] = acts[l].T @ grad
                    grad_bs[l] = np.sum(grad, axis=0, keepdims=True)
                    if l > 0:
                        grad = (grad @ self.weights[l].T) * self._relu_grad(pre_acts[l - 1])

                for l in range(len(self.weights)):
                    np.clip(grad_ws[l], -5.0, 5.0, out=grad_ws[l])
                    np.clip(grad_bs[l], -5.0, 5.0, out=grad_bs[l])
                    self.weights[l] -= lr * grad_ws[l]
                    self.biases[l] -= lr * grad_bs[l]

            history.append(epoch_loss / n)
        return history
