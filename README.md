# Hybrid Learning Digital Twin (Paper Implementation)

This project implements the core modeling chain described in:

- Huang et al., "Hybrid learning-based digital twin for manufacturing process: Modeling framework and implementation" (Robotics and Computer-Integrated Manufacturing, 82, 2023, 102545)
- DOI: `10.1016/j.rcim.2023.102545`
- ScienceDirect page: https://www.sciencedirect.com/science/article/pii/S0736584523000212

Implementation is created in a **new folder outside `Downloads`**:
`/home/sairam/PaperInduImplementation`

## What is implemented

- Feature-based digital-thread style synthetic data generation across process and quality domains.
- Physics/domain-knowledge force model (paper Eq. 4).
- Hybrid force uncertainty learner (paper Eq. 5) using sequence-aware residual modeling.
- Process model (VMME-style) for quality estimation with local tool engagement (paper Eq. 6-7).
- Uncertainty propagation to upper/lower quality bands.
- VMMNet-like residual learner to fuse process model + quality feedback (paper Eq. 9).
- Evaluation metrics and exported predictions.

## Project layout

- `src/paperindu/data/digital_thread.py`: digital thread data creation and split strategy.
- `src/paperindu/models/physical_force.py`: domain force model.
- `src/paperindu/models/hybrid_force.py`: hybrid force residual model.
- `src/paperindu/models/process_vmme.py`: VMME-style process model.
- `src/paperindu/models/vmmnet.py`: residual quality model.
- `src/paperindu/pipeline.py`: end-to-end training + evaluation pipeline.
- `scripts/run_demo.py`: runnable entry point.

## Run

```bash
cd /home/sairam/PaperInduImplementation
python3 scripts/run_demo.py --output-dir /home/sairam/PaperInduImplementation/results
```

Run with digital twin visualization (dashboard + animation):

```bash
cd /home/sairam/PaperInduImplementation
python3 scripts/run_twin_simulation.py --output-dir /home/sairam/PaperInduImplementation/results
```

Run with physical machining twin simulation (workpiece + moving cutter + material removal):

```bash
cd /home/sairam/PaperInduImplementation
python3 scripts/run_physical_twin.py --output-dir /home/sairam/PaperInduImplementation/results
```

## Outputs

- `results/metrics.json`: RMSE and training losses.
- `results/quality_predictions.csv`: measured and predicted quality traces with uncertainty bounds.
- `results/trace_signals.csv`: trace-level process signals and modeled force/stiffness states.
- `results/digital_twin_dashboard.png`: visual state of the digital twin.
- `results/digital_twin_simulation.gif`: simulated online twin behavior over the machining path.
- `results/physical_digital_twin_simulation.gif`: physical + digital machining twin animation.
- `results/physical_digital_twin_final.png`: final surface state comparison snapshot.

## Process steps (paper-aligned)

1. Build a feature-linked digital thread (design/planning/process/quality data in one coordinate context).
2. Estimate cutting force with domain model (physics part, Eq. 4).
3. Learn force uncertainty from machine signals using hybrid residual model (Eq. 5).
4. Estimate stiffness and propagate uncertainty to TCP deviation.
5. Compute virtual quality with VMME-style process model and uncertainty bands (Eq. 6-7).
6. Learn residual between virtual quality and inspection quality via VMMNet-like model (Eq. 9).
7. Visualize and validate: quality profile, uncertainty bands, residual map, and online simulation state.

## Notes

- The real industrial data and machine integration stack from the paper are not publicly distributed in this folder, so this implementation provides a faithful and executable pipeline with synthetic but structured digital-thread data.
- The code is dependency-light (NumPy-only learning stack) to run in constrained environments.
