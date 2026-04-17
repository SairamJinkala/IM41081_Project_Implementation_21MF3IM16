from __future__ import annotations

import numpy as np


class VMMEProcessModel:
    """Eq. (6)-(7): simplified edge-ready virtual metrology model."""

    def __init__(self, cutter_radius: float = 1.8, local_window: int = 24):
        self.cutter_radius = cutter_radius
        self.local_window = local_window

    def estimate_quality(
        self,
        trace_xyz: np.ndarray,
        delta_x: np.ndarray,
        quality_xyz: np.ndarray,
        quality_trace_index: np.ndarray,
    ) -> np.ndarray:
        q = np.zeros(quality_xyz.shape[0], dtype=np.float64)
        r2 = self.cutter_radius * self.cutter_radius

        for i in range(quality_xyz.shape[0]):
            qx, qy = quality_xyz[i, 0], quality_xyz[i, 1]
            center = quality_trace_index[i]
            s = max(0, center - self.local_window)
            e = min(trace_xyz.shape[0], center + self.local_window + 1)

            local_xy = trace_xyz[s:e, :2]
            dx = local_xy[:, 0] - qx
            dy = local_xy[:, 1] - qy
            d2 = dx * dx + dy * dy
            engaged = d2 <= r2

            if np.any(engaged):
                weights = 1.0 - (d2[engaged] / r2)
                q[i] = np.sum(weights * delta_x[s:e][engaged]) / (np.sum(weights) + 1e-12)
            else:
                q[i] = delta_x[center]
        return q
