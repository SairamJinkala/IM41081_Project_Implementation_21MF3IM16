from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StiffnessModel:
    uncertainty_ratio: float = 0.08

    def predict(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = 3500.0 + 170.0 * np.sin(0.07 * x) + 110.0 * np.cos(0.11 * y)
        k = np.clip(k, 3000.0, None)
        delta_k = self.uncertainty_ratio * k
        return k, delta_k
