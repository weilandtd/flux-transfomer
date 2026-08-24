# MoMA/FBApro KO prediction is NOT MID-compliant with re-optimization

Date: 2026-08-24. Script: `paper/analysis/ko_moma/mid_compliance.py`
(I. orientalis, raw annotated model, pFBA-optimal WT, net-flux-only MIDs, noise=0.01).

## Question
If we predict a knockout flux from the wild-type solution (FBApro/bounded MoMA =
"minimal adjustment"), does it reproduce the labeling a real KO cell would show — or
would the ¹³C MIDs catch it being wrong?

## Method
Common WT = pFBA-optimal flux. Per non-essential central-carbon KO, predict the flux two
ways — `bounded_moma` (minimal adjustment) and `pfba` (re-optimization) — simulate each
prediction's MIDs (SigmaFlux forward model, measurable metabolites only), and compare.

## Numbers (10 KOs, RMS MID, ×0.01 noise)
- median KO effect, pFBA vs WT : 0.051 (5.1× noise)  — KOs move the labeling
- median KO effect, MoMA vs WT : 0.071 (7.1× noise)
- median prior disagreement    : 0.042 (4.2× noise)  — MoMA vs pFBA MIDs
- max prior disagreement        : 0.097 (9.7× noise), individual mass fractions up to 0.5
- clean single-reaction tests   : TALA_c, ME2_m — pFBA gives 0 labeling change, MoMA ~3× noise

## Reading
The two KO hypotheses predict labeling that differs by ~4× the measurement noise (up to
~10×). So ¹³C labeling can distinguish "lazy cell" from "re-optimizing cell" — measuring
KO strains is informative and the FBApro/MoMA prediction cannot substitute for it. This
justifies KO ¹³C experiments and the KO training pipeline.

## Caveats
pFBA is the alternative prior, not ground truth (definitive test needs real KO ¹³C data);
net-flux-only MIDs; pFBA-optimal (not sampled MFA) WT; 10 central-carbon KOs on IO.
