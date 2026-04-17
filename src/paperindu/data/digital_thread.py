from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from paperindu.config import PipelineConfig


@dataclass
class DigitalThreadData:
    trace_xyz: np.ndarray
    quality_xyz: np.ndarray
    quality_trace_index: np.ndarray
    z_current: np.ndarray
    spindle_current: np.ndarray
    z_velocity: np.ndarray
    z_acceleration: np.ndarray
    force_sensor: np.ndarray
    stiffness_true: np.ndarray
    quality_sensor: np.ndarray

    def split_quality_train_test(self) -> Tuple[np.ndarray, np.ndarray]:
        """80/20 split similar to paper setup with sectioned quality traces."""
        n = self.quality_xyz.shape[0]
        cut_90 = int(0.9 * n)
        section_size = max(cut_90 // 7, 1)
        sec7_start = 6 * section_size
        sec7_end = min(7 * section_size, cut_90)

        all_idx = np.arange(n)
        section7_idx = np.arange(sec7_start, sec7_end)
        tail10_idx = np.arange(cut_90, n)
        test_idx = np.unique(np.concatenate([section7_idx, tail10_idx]))
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False

        train_idx = all_idx[train_mask]
        return train_idx, test_idx


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x.copy()
    kernel = np.ones(k) / k
    pad = k // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(x_pad, kernel, mode="valid")[: x.shape[0]]


def generate_synthetic_digital_thread(cfg: PipelineConfig) -> DigitalThreadData:
    """Create a synthetic but structured dataset that follows the paper's model chain."""
    rng = np.random.default_rng(cfg.random_seed)
    n = cfg.num_trace_points
    m = cfg.num_quality_points

    t = np.linspace(0.0, 1.0, n)
    x = 120.0 * t
    y = 24.0 * np.sin(2.3 * np.pi * t) + 8.0 * np.sin(11.0 * np.pi * t)
    z = np.zeros_like(t)
    trace_xyz = np.column_stack([x, y, z])

    z_velocity = np.gradient(y)
    z_acceleration = np.gradient(z_velocity)

    z_current = 4.5 + 1.3 * np.sin(16 * np.pi * t) + 0.4 * np.sign(np.sin(4 * np.pi * t))
    z_current += 0.12 * rng.normal(size=n)
    spindle_current = 8.0 + 0.7 * np.cos(13 * np.pi * t + 0.3) + 0.08 * rng.normal(size=n)

    k_enc = 190.0 + 18.0 * np.sin(0.09 * x)
    f_fric = 0.58 * np.tanh(2.8 * z_velocity) + 0.10 * z_velocity
    f_acc = 0.45 * z_acceleration
    f_sp = 0.22 * np.sin(0.15 * x)
    drive_dev = 0.012 * z_current
    force_phy = k_enc * (drive_dev - f_fric - f_acc - f_sp)

    dyn_term = 0.8 * _moving_average(z_current, 7) + 0.2 * _moving_average(spindle_current, 11)
    nonlinear_term = 0.14 * np.sin(0.25 * x) * np.cos(0.31 * y)
    force_residual_true = 2.2 * dyn_term + 8.0 * nonlinear_term + 0.4 * rng.normal(size=n)
    force_sensor = force_phy + force_residual_true

    stiffness_true = 3500.0 + 180.0 * np.sin(0.07 * x) + 120.0 * np.cos(0.11 * y)
    stiffness_true = np.clip(stiffness_true, 3000.0, None)

    delta_x_true = force_sensor / stiffness_true

    quality_trace_index = np.linspace(0, n - 1, m).astype(int)
    qx = x[quality_trace_index]
    qy = y[quality_trace_index]
    qz = np.zeros(m)
    quality_xyz = np.column_stack([qx, qy, qz])

    base_quality = delta_x_true[quality_trace_index]
    residual_quality = (
        0.006 * np.sin(0.6 * qx)
        + 0.008 * np.cos(0.35 * qy)
        + 0.0025 * _moving_average(delta_x_true, 15)[quality_trace_index]
    )
    quality_sensor = base_quality + residual_quality + 0.0015 * rng.normal(size=m)

    return DigitalThreadData(
        trace_xyz=trace_xyz,
        quality_xyz=quality_xyz,
        quality_trace_index=quality_trace_index,
        z_current=z_current,
        spindle_current=spindle_current,
        z_velocity=z_velocity,
        z_acceleration=z_acceleration,
        force_sensor=force_sensor,
        stiffness_true=stiffness_true,
        quality_sensor=quality_sensor,
    )
