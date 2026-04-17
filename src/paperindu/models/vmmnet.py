from __future__ import annotations

import numpy as np

from paperindu.models.nn import MLPRegressorNumpy


class VMMNetLikeModel:
    """Eq. (9): hybrid quality residual network with temporal + spatial branches."""

    def __init__(self, seq_len: int = 24, seed: int = 42):
        self.seq_len = seq_len
        self.seed = seed
        self.net: MLPRegressorNumpy | None = None

    def _temporal_features(self, temporal_inputs: np.ndarray) -> np.ndarray:
        n, f = temporal_inputs.shape
        out = np.zeros((n, f * 3), dtype=np.float64)
        for i in range(n):
            s = max(0, i - self.seq_len + 1)
            w = temporal_inputs[s : i + 1]
            out[i, :f] = w[-1]
            out[i, f : 2 * f] = np.mean(w, axis=0)
            out[i, 2 * f :] = np.std(w, axis=0)
        return out

    @staticmethod
    def _spatial_features(xy: np.ndarray) -> np.ndarray:
        x = xy[:, 0]
        y = xy[:, 1]
        return np.column_stack([x, y, x * x, y * y, x * y])

    def build_features(self, temporal_inputs: np.ndarray, xy: np.ndarray) -> np.ndarray:
        t_feat = self._temporal_features(temporal_inputs)
        s_feat = self._spatial_features(xy)
        return np.concatenate([t_feat, s_feat], axis=1)

    def fit(
        self,
        temporal_inputs: np.ndarray,
        xy: np.ndarray,
        y: np.ndarray,
        epochs: int = 300,
        lr: float = 8e-4,
        batch_size: int = 128,
    ) -> list[float]:
        x = self.build_features(temporal_inputs, xy)
        self.net = MLPRegressorNumpy(x.shape[1], [128, 128, 64], seed=self.seed)
        return self.net.fit(x, y, epochs=epochs, lr=lr, batch_size=batch_size)

    def predict(self, temporal_inputs: np.ndarray, xy: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Model is not trained.")
        x = self.build_features(temporal_inputs, xy)
        return self.net.predict(x)
