from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paperindu.config import PipelineConfig
from paperindu.data.digital_thread import generate_synthetic_digital_thread
from paperindu.models.hybrid_force import HybridForceResidualModel
from paperindu.models.physical_force import PhysicalForceModel
from paperindu.models.process_vmme import VMMEProcessModel
from paperindu.models.stiffness import StiffnessModel
from paperindu.models.vmmnet import VMMNetLikeModel


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run_pipeline(cfg: PipelineConfig, output_dir: str | Path) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = generate_synthetic_digital_thread(cfg)

    x = data.trace_xyz[:, 0]
    y = data.trace_xyz[:, 1]

                                       
    phy_model = PhysicalForceModel()
    force_phy = phy_model.predict(x, data.z_velocity, data.z_acceleration, data.z_current)

                                   
    obs_force = np.column_stack([
        data.z_current,
        data.spindle_current,
        data.z_velocity,
        data.z_acceleration,
    ])
    force_res_model = HybridForceResidualModel(
        seq_len=cfg.sequence_length_force,
        seed=cfg.random_seed,
    )
    force_loss = force_res_model.fit(
        obs=obs_force,
        force_phy=force_phy,
        force_sensor=data.force_sensor,
        epochs=cfg.force_epochs,
        lr=cfg.learning_rate_force,
        batch_size=cfg.batch_size,
    )
    force_hybrid, delta_f = force_res_model.predict_force(obs_force, force_phy)

                                                   
    stiffness_model = StiffnessModel(cfg.stiffness_uncertainty_ratio)
    stiffness_est, delta_k = stiffness_model.predict(x, y)
    delta_x_phy = force_phy / stiffness_est
    delta_x_hybrid = force_hybrid / stiffness_est

    vmme = VMMEProcessModel(cutter_radius=cfg.cutter_radius, local_window=cfg.vmme_window)
    q_phy = vmme.estimate_quality(
        data.trace_xyz, delta_x_phy, data.quality_xyz, data.quality_trace_index
    )
    q_hybrid = vmme.estimate_quality(
        data.trace_xyz, delta_x_hybrid, data.quality_xyz, data.quality_trace_index
    )

                                                                                
    f_upper = force_hybrid + np.abs(delta_f)
    f_lower = force_hybrid - np.abs(delta_f)
    k_upper = stiffness_est + delta_k
    k_lower = np.maximum(stiffness_est - delta_k, 1e-6)

    dx_upper = f_upper / k_lower
    dx_lower = f_lower / k_upper
    q_upper = vmme.estimate_quality(
        data.trace_xyz, dx_upper, data.quality_xyz, data.quality_trace_index
    )
    q_lower = vmme.estimate_quality(
        data.trace_xyz, dx_lower, data.quality_xyz, data.quality_trace_index
    )

                                                                     
    q_idx = data.quality_trace_index
    temporal_quality_features = np.column_stack([
        data.trace_xyz[q_idx, 0],
        data.trace_xyz[q_idx, 1],
        data.trace_xyz[q_idx, 2],
        force_hybrid[q_idx],
        np.zeros_like(q_idx, dtype=np.float64),
        np.zeros_like(q_idx, dtype=np.float64),
        data.z_current[q_idx],
        data.z_velocity[q_idx],
        data.spindle_current[q_idx],
    ])

    target_residual = data.quality_sensor - q_hybrid
    train_idx, test_idx = data.split_quality_train_test()

    vmmnet = VMMNetLikeModel(seq_len=cfg.sequence_length_quality, seed=cfg.random_seed)
    q_loss = vmmnet.fit(
        temporal_inputs=temporal_quality_features[train_idx],
        xy=data.quality_xyz[train_idx, :2],
        y=target_residual[train_idx],
        epochs=cfg.quality_epochs,
        lr=cfg.learning_rate_quality,
        batch_size=cfg.batch_size,
    )

    pred_residual_all = np.zeros_like(target_residual)
    pred_residual_all[train_idx] = vmmnet.predict(
        temporal_quality_features[train_idx], data.quality_xyz[train_idx, :2]
    )
    pred_residual_all[test_idx] = vmmnet.predict(
        temporal_quality_features[test_idx], data.quality_xyz[test_idx, :2]
    )

    q_vmmnet = q_hybrid + pred_residual_all

    metrics = {
        "rmse_domain_knowledge": rmse(q_phy, data.quality_sensor),
        "rmse_hybrid_model_chain": rmse(q_hybrid, data.quality_sensor),
        "rmse_hybrid_plus_vmmnet": rmse(q_vmmnet, data.quality_sensor),
        "rmse_hybrid_test_only": rmse(q_vmmnet[test_idx], data.quality_sensor[test_idx]),
        "force_train_final_loss": float(force_loss[-1]),
        "quality_train_final_loss": float(q_loss[-1]),
    }

    np.savetxt(out_dir / "quality_predictions.csv", np.column_stack([
        data.quality_xyz[:, 0],
        data.quality_xyz[:, 1],
        data.quality_sensor,
        q_phy,
        q_hybrid,
        q_vmmnet,
        q_lower,
        q_upper,
    ]), delimiter=",", header="x,y,quality_sensor,q_phy,q_hybrid,q_vmmnet,q_lower,q_upper", comments="")

    np.savetxt(
        out_dir / "trace_signals.csv",
        np.column_stack(
            [
                data.trace_xyz[:, 0],
                data.trace_xyz[:, 1],
                data.z_current,
                data.spindle_current,
                force_phy,
                force_hybrid,
                delta_f,
                stiffness_est,
                delta_k,
            ]
        ),
        delimiter=",",
        header="x,y,z_current,spindle_current,force_phy,force_hybrid,delta_f,stiffness,delta_k",
        comments="",
    )

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
