# Paper pipelines

The training and inference stages ship with the package as console scripts — see
`sigmaflux.pipeline` for the full table (`sigmaflux-sample-flux`,
`sigmaflux-generate-data`, `sigmaflux-select-features`, `sigmaflux-train`,
`sigmaflux-workflow`, `sigmaflux-infer`, `sigmaflux-mfa-data`,
`sigmaflux-sample-ko-flux`, `sigmaflux-build-ko-split`).

What remains here is paper-specific and not part of the installed package:

- `benchmarks/`: AI/MFA accuracy benchmarks and calibration checks

Run the benchmark scripts from the repository root using their full path.
Cluster entry points live in `paper/submit/` and invoke the packaged stages as
`python -u -m sigmaflux.pipeline.<stage>`.
