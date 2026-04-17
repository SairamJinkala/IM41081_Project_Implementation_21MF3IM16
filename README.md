# Hybrid Learning Digital Twin (Paper Implementation)

This project implements the core modeling chain described in:

- Huang et al., "Hybrid learning-based digital twin for manufacturing process: Modeling framework and implementation" (Robotics and Computer-Integrated Manufacturing, 82, 2023, 102545)
- DOI: `10.1016/j.rcim.2023.102545`
- ScienceDirect page: https://www.sciencedirect.com/science/article/pii/S0736584523000212


## Outputs

- `results/metrics.json`: RMSE and training losses.
- `results/quality_predictions.csv`: measured and predicted quality traces with uncertainty bounds.
- `results/trace_signals.csv`: trace-level process signals and modeled force/stiffness states.
- `results/digital_twin_dashboard.png`: visual state of the digital twin.
- `results/digital_twin_simulation.gif`: simulated online twin behavior over the machining path.
- `results/physical_digital_twin_simulation.gif`: physical + digital machining twin animation.
- `results/physical_digital_twin_final.png`: final surface state comparison snapshot.

