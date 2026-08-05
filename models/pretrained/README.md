# Pretrained models

Each `.sfx` is a self-contained ensemble: weights, vocabularies, null-space basis,
flux names, the prepared COBRA network, and validation metrics. Load with
`sigmaflux.ai.load_model`; inspect without torch via `sigmaflux-inspect <file>`.

| Artifact | Members | Val. flux R² (median) | Status |
|---|---|---|---|
| `i_orientalis-v1.0.0.sfx` | 20 | 0.982 | released |
| `s_cerevisiae-v1.0.0.sfx` | 20 | 0.982 | released |
| `r_toruloides-v1.0.0.sfx` | 2 | 0.53–0.61 | **provisional** |

## ⚠️ r_toruloides-v1.0.0 is provisional

**Driven by only 6 labeling tokens. Not suitable for quantitative flux inference.**

Feature selection kept 6 of 336 candidates — `f6p_c` M+3/M+4, `gam6p_c` M+3/M+4,
`fdp_c` M+3, `s7p_c` M+4 — all in one hexose-phosphate/PPP neighbourhood, so much
of the network is driven by no labeling evidence. The other two organisms train on
87 and 35 tokens.

The cause is the permutation null floor in `rank_from_samples`
(`src/sigmaflux/ai/fisher_selection.py`): the **maximum** over
`n_permutations × n_tokens` null draws, 7.4487 nats here, above 98% of the real
tokens. It is an extreme-value statistic and is not comparable across organisms —
I. orientalis' floor was 2.98 on a near-identical token distribution, and widening
the pool from 336 to 2260 tokens left this one unchanged. Resolving that (a high
quantile is the obvious candidate) changes selection for *every* organism, so it
needs a deliberate decision before a v2.

Consequences: validation flux R² median 0.53/0.61 vs >0.98 elsewhere; bimodal with
~205 reactions at median 0.978 and ~160 below 0.2; 54–55 mislearned (R² < 0); and
the ensemble is 2 members rather than 20.

`tests/test_pretrained_models.py::test_reported_validation_is_healthy` requires
every member above R² 0.9, so it does not pass for this artifact. The test is doing
its job.

The packaged network is **not** the published Maranas imRt GEM — it adds two carbon
overflow drains, and still has no carotenoid pathway. See `add_rt_overflow_routes`
in `models/atomistic/convert_maranas_models.py` for the stoichiometry, the citation
and the rationale.
