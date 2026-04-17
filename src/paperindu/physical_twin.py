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


def _tool_cylinder(xc: float, yc: float, radius: float, z_bottom: float, z_top: float):
    theta = np.linspace(0, 2 * np.pi, 30)
    z = np.linspace(z_bottom, z_top, 2)
    tt, zz = np.meshgrid(theta, z)
    xx = xc + radius * np.cos(tt)
    yy = yc + radius * np.sin(tt)
    return xx, yy, zz


def _simulate_material_removal(
    x: np.ndarray,
    y: np.ndarray,
    force: np.ndarray,
    stiffness: np.ndarray,
    z_current: np.ndarray,
    tool_radius: float,
    grid_nx: int,
    grid_ny: int,
    n_frames: int,
    seed: int,
):
    rng = np.random.default_rng(seed)

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    margin = 3.0
    gx = np.linspace(x_min - margin, x_max + margin, grid_nx)
    gy = np.linspace(y_min - margin, y_max + margin, grid_ny)
    X, Y = np.meshgrid(gx, gy)

                                                                                  
    stock = 0.65 + 0.05 * np.sin(0.08 * X) + 0.035 * np.cos(0.12 * Y) + 0.01 * rng.normal(size=X.shape)

    digital = stock.copy()
    physical = stock.copy()

    n = x.shape[0]
    frame_steps = np.linspace(0, n - 1, n_frames).astype(int)
    frame_steps_set = set(frame_steps.tolist())

    frames = []

    zc_mean = float(np.mean(z_current))
    for i in range(n):
        xt = float(x[i])
        yt = float(y[i])

                                                               
        defl_pred = 0.22 * (force[i] / max(stiffness[i], 1e-6))

                                                                                    
        transient = (
            0.0035 * np.sin(0.1 * xt)
            + 0.0025 * np.cos(0.07 * yt)
            + 0.0008 * (z_current[i] - zc_mean)
        )
        defl_true = defl_pred + transient

                                                                                               
        mask = (X - xt) ** 2 + (Y - yt) ** 2 <= tool_radius**2
        digital[mask] = np.minimum(digital[mask], defl_pred)
        physical[mask] = np.minimum(physical[mask], defl_true)

        if i in frame_steps_set:
            local_mask = mask if np.any(mask) else np.ones_like(mask, dtype=bool)
            z_tool_phys = float(np.min(physical[local_mask]) + 0.35)
            z_tool_dig = float(np.min(digital[local_mask]) + 0.35)
            frames.append(
                {
                    "i": i,
                    "xt": xt,
                    "yt": yt,
                    "z_tool_phys": z_tool_phys,
                    "z_tool_dig": z_tool_dig,
                    "physical": physical.copy(),
                    "digital": digital.copy(),
                }
            )

    return X, Y, frames


def render_physical_twin_simulation(
    result_dir: str | Path,
    n_frames: int = 80,
    grid_nx: int = 75,
    grid_ny: int = 45,
    tool_radius: float = 1.8,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Render a physical twin simulation: machine/workpiece/tool + digital counterpart."""
    result_dir = Path(result_dir)
    trace, cols = _load_csv(result_dir / "trace_signals.csv")
    col = {k: i for i, k in enumerate(cols)}

    x = trace[:, col["x"]]
    y = trace[:, col["y"]]
    force = trace[:, col["force_hybrid"]]
    stiffness = trace[:, col["stiffness"]]
    z_current = trace[:, col["z_current"]]

    X, Y, frames = _simulate_material_removal(
        x=x,
        y=y,
        force=force,
        stiffness=stiffness,
        z_current=z_current,
        tool_radius=tool_radius,
        grid_nx=grid_nx,
        grid_ny=grid_ny,
        n_frames=n_frames,
        seed=seed,
    )

    z_min = min(float(np.min(f["physical"])) for f in frames) - 0.1
    z_max = max(float(np.max(f["physical"])) for f in frames) + 0.2

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.8])
    ax_phys = fig.add_subplot(gs[0, 0], projection="3d")
    ax_dig = fig.add_subplot(gs[0, 1], projection="3d")
    ax_res = fig.add_subplot(gs[1, :])

    def draw_frame(frame_idx: int):
        fr = frames[frame_idx]
        ax_phys.cla()
        ax_dig.cla()
        ax_res.cla()

        phys = fr["physical"]
        dig = fr["digital"]

        ax_phys.plot_surface(X, Y, phys, cmap="terrain", linewidth=0, antialiased=False, alpha=0.95)
        cx, cy, cz = _tool_cylinder(fr["xt"], fr["yt"], tool_radius, fr["z_tool_phys"] - 0.28, fr["z_tool_phys"])
        ax_phys.plot_surface(cx, cy, cz, color="silver", alpha=0.9, linewidth=0)

        ax_dig.plot_surface(X, Y, dig, cmap="viridis", linewidth=0, antialiased=False, alpha=0.95)
        dx, dy, dz = _tool_cylinder(fr["xt"], fr["yt"], tool_radius, fr["z_tool_dig"] - 0.28, fr["z_tool_dig"])
        ax_dig.plot_surface(dx, dy, dz, color="lightgray", alpha=0.9, linewidth=0)

        for ax in (ax_phys, ax_dig):
            ax.set_xlim(float(np.min(X)), float(np.max(X)))
            ax.set_ylim(float(np.min(Y)), float(np.max(Y)))
            ax.set_zlim(z_min, z_max)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Surface Z")
            ax.view_init(elev=28, azim=-60)

        ax_phys.set_title("Physical Machine Twin (as-manufactured surface)")
        ax_dig.set_title("Digital Process Twin (as-predicted surface)")

        residual_um = (phys - dig) * 1e3
        im = ax_res.imshow(
            residual_um,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            extent=[float(np.min(X)), float(np.max(X)), float(np.min(Y)), float(np.max(Y))],
        )
        ax_res.scatter([fr["xt"]], [fr["yt"]], c="k", s=12, label="Current cutter position")
        ax_res.set_title(f"Residual Field (Physical - Digital) in um, step {fr['i']}")
        ax_res.set_xlabel("X")
        ax_res.set_ylabel("Y")
        ax_res.legend(loc="upper right")
        fig.colorbar(im, ax=ax_res, fraction=0.02, pad=0.01)

    def init():
        draw_frame(0)
        return []

    def update(k):
        draw_frame(k)
        return []

    anim = FuncAnimation(fig, update, init_func=init, frames=len(frames), interval=140, blit=False)

    gif_path = result_dir / "physical_digital_twin_simulation.gif"
    anim.save(gif_path, writer=PillowWriter(fps=7))

    draw_frame(len(frames) - 1)
    png_path = result_dir / "physical_digital_twin_final.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    return gif_path, png_path
