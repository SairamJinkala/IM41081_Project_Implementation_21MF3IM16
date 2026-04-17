from dataclasses import dataclass


@dataclass
class PipelineConfig:
    num_trace_points: int = 6000
    num_quality_points: int = 1800
    sequence_length_force: int = 32
    sequence_length_quality: int = 24
    cutter_radius: float = 1.8
    vmme_window: int = 24
    force_epochs: int = 250
    quality_epochs: int = 300
    learning_rate_force: float = 1e-3
    learning_rate_quality: float = 8e-4
    batch_size: int = 128
    stiffness_uncertainty_ratio: float = 0.08
    random_seed: int = 42
