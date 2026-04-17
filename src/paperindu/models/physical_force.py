from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PhysicalForceModel:
    """Eq. (4): Physics/domain-knowledge force model approximation."""

    def predict(
        self,
        x_pos: np.ndarray,
        z_velocity: np.ndarray,
        z_acceleration: np.ndarray,
        z_current: np.ndarray,
    ) -> np.ndarray:
        k_enc = 190.0 + 18.0 * np.sin(0.09 * x_pos)
        f_fric = 0.58 * np.tanh(2.8 * z_velocity) + 0.10 * z_velocity
        f_acc = 0.45 * z_acceleration
        f_sp = 0.22 * np.sin(0.15 * x_pos)
        drive_dev = 0.012 * z_current
        return k_enc * (drive_dev - f_fric - f_acc - f_sp)
