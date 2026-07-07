# Paper pipelines

Executable pipeline scripts are grouped by stage:

- `data_generation/`: flux sampling, KO datasets, and isotope-label simulation
- `training/`: informative-feature selection and AI model training
- `inference/`: AI and traditional MFA runs on experimental data
- `benchmarks/`: AI/MFA accuracy benchmarks and calibration checks
- `workflows/`: launchers that coordinate multiple stages

Run scripts from the repository root using their full path. Cluster entry points
live in `paper/scripts/` and reference these locations directly.
