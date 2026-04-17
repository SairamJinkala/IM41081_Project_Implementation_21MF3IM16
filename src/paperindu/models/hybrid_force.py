from __future__ import annotations

import numpy as np

from paperindu.models.nn import MLPRegressorNumpy


def build_sequence_matrix(obs: np.ndarray, seq_len: int) -> np.ndarray:
    n, f = obs.shape
    if n < seq_len:
        raise ValueError("Not enough samples for the chosen sequence length.")
    out = np.zeros((n - seq_len + 1, seq_len * f), dtype=np.float64)
    for i in range(seq_len - 1, n):
        w = obs[i - seq_len + 1 : i + 1]
        out[i - seq_len + 1] = w.reshape(-1)
    return out


class HybridForceResidualModel:
    """Eq. (5): residual learner over physical force model using observation signals."""

    def __init__(self, seq_len: int = 32, seed: int = 42):
        self.seq_len = seq_len
        self.seed = seed
        self.net: MLPRegressorNumpy | None = None

    def fit(
        self,
        obs: np.ndarray,
        force_phy: np.ndarray,
        force_sensor: np.ndarray,
        epochs: int = 250,
        lr: float = 1e-3,
        batch_size: int = 128,
    ) -> list[float]:
        x = build_sequence_matrix(obs, self.seq_len)
        y = force_sensor[self.seq_len - 1 :] - force_phy[self.seq_len - 1 :]

        self.net = MLPRegressorNumpy(x.shape[1], [96, 96], seed=self.seed)
        return self.net.fit(x, y, epochs=epochs, lr=lr, batch_size=batch_size)

    def predict_residual(self, obs: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Model is not trained.")
        x = build_sequence_matrix(obs, self.seq_len)
        pred = self.net.predict(x)
        head = np.repeat(pred[0], self.seq_len - 1)
        return np.concatenate([head, pred])

    def predict_force(self, obs: np.ndarray, force_phy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta_f = self.predict_residual(obs)
        return force_phy + delta_f, delta_f
