from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def _load_csv(path: Path) -> tuple[np.ndarray, list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data, header


def render_dashboard(result_dir: str | Path) -> Path:
    result_dir = Path(result_dir)
    quality, q_cols = _load_csv(result_dir / "quality_predictions.csv")
    trace, t_cols = _load_csv(result_dir / "trace_signals.csv")

    qi = {k: i for i, k in enumerate(q_cols)}
    ti = {k: i for i, k in enumerate(t_cols)}

    q_sensor = quality[:, qi["quality_sensor"]]
    q_pred = quality[:, qi["q_vmmnet"]]
    q_low = quality[:, qi["q_lower"]]
    q_up = quality[:, qi["q_upper"]]
    qx = quality[:, qi["x"]]
    qy = quality[:, qi["y"]]

    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    idx = np.arange(q_sensor.shape[0])
    ax1.plot(idx, q_sensor * 1e3, label="CMM/Real", linewidth=1.0)
    ax1.plot(idx, q_pred * 1e3, label="Digital Twin", linewidth=1.0)
    ax1.fill_between(idx, q_low * 1e3, q_up * 1e3, alpha=0.2, label="Uncertainty band")
    ax1.set_title("Quality Profile (um)")
    ax1.set_xlabel("Quality path index")
    ax1.set_ylabel("Deviation (um)")
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(2, 2, 2)
    sc = ax2.scatter(qx, qy, c=(q_pred - q_sensor) * 1e3, s=8, cmap="coolwarm")
    ax2.set_title("Residual Field: Twin - Real (um)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    fig.colorbar(sc, ax=ax2)

    ax3 = fig.add_subplot(2, 2, 3)
    tx = trace[:, ti["x"]]
    ty = trace[:, ti["y"]]
    ax3.plot(tx, ty, color="gray", linewidth=0.8, label="Machining path")
    ax3.plot(qx, qy, color="tab:green", linewidth=0.8, alpha=0.8, label="Virtual metrology path")
    ax3.set_title("Digital Thread Geometry")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.legend(loc="best")

    ax4 = fig.add_subplot(2, 2, 4)
    force_h = trace[:, ti["force_hybrid"]]
    stiffness = trace[:, ti["stiffness"]]
    ax4.plot(force_h, label="Hybrid force")
    ax4_t = ax4.twinx()
    ax4_t.plot(stiffness, color="tab:orange", alpha=0.8, label="Stiffness")
    ax4.set_title("Twin States over Time")
    ax4.set_xlabel("Trace index")
    ax4.set_ylabel("Force")
    ax4_t.set_ylabel("Stiffness")

    fig.suptitle("Hybrid Learning Digital Twin Simulation Dashboard", fontsize=14)
    fig.tight_layout()

    out = result_dir / "digital_twin_dashboard.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def render_animation(result_dir: str | Path, frame_stride: int = 20) -> Path:
    result_dir = Path(result_dir)
    quality, q_cols = _load_csv(result_dir / "quality_predictions.csv")
    qi = {k: i for i, k in enumerate(q_cols)}

    qx = quality[:, qi["x"]]
    qy = quality[:, qi["y"]]
    qs = quality[:, qi["quality_sensor"]] * 1e3
    qp = quality[:, qi["q_vmmnet"]] * 1e3
    ql = quality[:, qi["q_lower"]] * 1e3
    qu = quality[:, qi["q_upper"]] * 1e3

    frames = np.arange(1, len(qx), frame_stride)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(qx, qy, color="lightgray", linewidth=0.8)
    tool_pt, = ax1.plot([], [], "ro", markersize=4)
    ax1.set_title("Virtual Tool / Feature Tracking")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    ax2.set_title("Online Quality Estimation")
    ax2.set_xlabel("Quality index")
    ax2.set_ylabel("Deviation (um)")
    ax2.set_xlim(0, len(qx))
    ymin = min(qs.min(), ql.min()) - 1
    ymax = max(qs.max(), qu.max()) + 1
    ax2.set_ylim(ymin, ymax)

    line_real, = ax2.plot([], [], label="Real", linewidth=1.2)
    line_twin, = ax2.plot([], [], label="Twin", linewidth=1.2)
    ax2.legend(loc="best")

    fill_holder = [None]

    def init():
        tool_pt.set_data([], [])
        line_real.set_data([], [])
        line_twin.set_data([], [])
        return tool_pt, line_real, line_twin

    def update(frame):
        i = int(frame)
        tool_pt.set_data([qx[i]], [qy[i]])
        idx = np.arange(i)
        line_real.set_data(idx, qs[:i])
        line_twin.set_data(idx, qp[:i])

        if fill_holder[0] is not None:
            fill_holder[0].remove()
        fill_holder[0] = ax2.fill_between(idx, ql[:i], qu[:i], alpha=0.2, color="tab:blue")
        return tool_pt, line_real, line_twin

    anim = FuncAnimation(fig, update, init_func=init, frames=frames, interval=120, blit=False)
    out = result_dir / "digital_twin_simulation.gif"
    anim.save(out, writer=PillowWriter(fps=8))
    plt.close(fig)
    return out
