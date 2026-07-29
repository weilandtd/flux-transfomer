# Flux Sampling Configuration

Fields under the `sampling:` key in any pipeline config.

## Core fields

| Field | Type | Default | Description |
|---|---|---|---|
| `n_samples` | int | — | Number of flux samples to generate |
| `thinning` | int | 100 | Steps between returned samples (higher = less correlated) |
| `processes` | int | — | Parallel Markov chains |
| `method` | str | `hopsy` | Sampler backend: `hopsy`, `optgp`, or `achr` |
| `fva` | bool | `false` | Constrain model by FVA before sampling (removes zero-flux reactions) |
| `ko` | bool | `false` | Also sample each non-essential single-reaction knockout |

## Hopsy-specific fields

Only used when `method: hopsy`. Requires dev cobrapy + hopsy installed.

| Field | Type | Default | Description |
|---|---|---|---|
| `hopsy_proposal` | str | `uniform_coordinate_hit_and_run` | Proposal distribution (see below) |
| `rounding` | bool | `true` | Offline rounding preconditioning — leave on unless you have a reason not to |
| `nproj` | int or null | `null` | Re-projection interval into feasibility space; lower values reduce constraint violations at the cost of speed |

`seed` is read from the top-level config key and passed automatically — do not duplicate it under `sampling:`.

### Available proposals

| `hopsy_proposal` value | Notes |
|---|---|
| `uniform_coordinate_hit_and_run` | **Default.** Fast, good mixing for metabolic networks |
| `uniform_hit_and_run` | Full-space hit-and-run |
| `gaussian_coordinate_hit_and_run` | Gaussian steps per coordinate |
| `gaussian_hit_and_run` | Gaussian full-space |
| `billiard_walk` | Bounces off polytope boundaries |
| `billiard_adaptive_metropolis` | Billiard walk with adaptive covariance tuning — better mixing in thin/elongated polytopes |
| `adaptive_metropolis` | Adaptive Metropolis (learns proposal covariance online) |
| `dikin_walk` | Geometry-aware; better for ill-conditioned polytopes |
| `ball_walk` | Uniform ball steps |

## Stratified sampling

Replaces the old single-pool + subsample approach. For each (reaction, bin) pair, a short
MCMC chain is run with that reaction constrained to its bin, guaranteeing every bin is
seeded. IPF then draws `n_samples` quasi-uniformly from the stratified pool.

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Enable stratified sampling |
| `n_subsamples` | int | — | MCMC samples per (reaction, bin) stratum |
| `n_bins` | int | `5` | Equal-width bins along each measured reaction's FVA range |
| `reactions` | list or null | `null` | Reactions to stratify over; null → `training.measured_fluxes` |

`n_samples` at the top level sets the **final output count** (not a pool size).

## Example

```yaml
seed: 42

sampling:
  n_samples: 5000          # final output
  thinning: 100
  processes: 8
  method: hopsy
  hopsy_proposal: uniform_coordinate_hit_and_run
  rounding: true
  nproj: null
  fva: true
  ko: false
  growth:
    lb: 0.01
    ub: 0.5
  media:
    EX_glc__D_e: {lb: -12.0, ub: -1.0}
    EX_o2_e:     {lb: -100.0, ub: 0.0}
    # ...
  stratified_sampling:
    enabled: true
    n_subsamples: 500
    n_bins: 5
    reactions: null
```

To fall back to optgp (e.g. on a machine without hopsy), change `method: optgp` and remove the hopsy-specific fields.
