# SigmaFlux — Methods, Step by Step

*A detailed walkthrough of the SigmaFlux method: data generation, flux sampling, Fisher
token selection, model architecture, the null‑space algebra, the loss, training, and how we
handle noise, missing values, and known fluxes. Written as a continuous story so it can be
read top to bottom. Concrete numbers come from the* I. orientalis *run (`i_orientalis.yaml`,
387 reactions).*

---

## 0. The biology and the problem we are solving

### 0.1 What a metabolic flux is, and why we want it
A living cell runs thousands of biochemical reactions. The **flux** of a reaction is the rate
at which it converts substrate into product (mmol · gDW⁻¹ · h⁻¹). The full vector of fluxes
**v** is the cell's metabolic "state" — it tells us how carbon and energy actually move through
the network (how much goes to growth, to ethanol, to the TCA cycle, etc.). Knowing **v** is the
goal of metabolic engineering and of understanding cellular physiology.

The trouble: fluxes are **not directly measurable**. We can measure a handful of exchange rates
(glucose uptake, ethanol secretion, growth) at the boundary of the cell, but the internal
fluxes are hidden.

### 0.2 ¹³C metabolic flux analysis (the classical inverse problem)
The classical trick is **¹³C labeling**. We feed the cell a glucose whose carbons are
isotopically labeled in a known pattern (e.g. uniformly ¹³C, or labeled only at positions 1–2).
As that labeled carbon flows through the network, each reaction rearranges, splits, and
recombines carbon skeletons. Every internal metabolite ends up as a *mixture* of isotope
variants ("isotopomers") differing by how many heavy carbons they carry. Mass spectrometry
measures, for each metabolite, the fraction of molecules with 0, 1, 2, … extra mass units — its
**Mass Isotopomer Distribution (MID)**.

The key fact: **the MID pattern depends on the fluxes.** A reaction carrying high flux imprints
its labeling signature strongly; a near‑silent reaction leaves little trace. So the MIDs are an
indirect, nonlinear fingerprint of **v**.

Classical ¹³C‑MFA solves the **inverse problem**: guess fluxes → simulate the MIDs they would
produce → compare to measured MIDs → iterate (nonlinear least squares) until the simulation
matches. This works but is slow, requires expert setup per organism/condition, and gives one
local fit at a time.

### 0.3 The SigmaFlux idea
**SigmaFlux replaces the per‑experiment optimization with a trained neural network.** We
generate a large, diverse library of (flux vector → simulated MID) pairs *in silico*, then train
a transformer to invert the map: given measured MIDs (plus the few known boundary fluxes), it
directly *reads off* the full flux vector. Once trained, inference is a single forward pass —
fast, automatic, and reusable across conditions.

The method has four offline stages, then an inference step:

1. **Sample fluxes** — draw thousands of feasible steady‑state flux vectors that broadly cover
   the space the cell could occupy. *(`sample_flux.py`)*
2. **Simulate labeling** — for each flux vector, run the ¹³C forward model (EMU) to compute the
   MIDs it would produce. *(`generate_training_data.py`)*
3. **Select informative tokens** — use Fisher information / optimal‑experiment‑design theory to
   keep only the MID measurements that actually carry flux information. *(`extract_informative_features.py`)*
4. **Train** the labeling→flux transformer on the simulated pairs. *(`train_ai_model.py`)*

Then at inference, real measured MIDs are fed in and the network predicts **v**.

The rest of this document develops each stage in detail, but we first need the shared
mathematical backbone — the steady‑state algebra and the labeling forward model — because every
stage is built on it.

---

## 1. The mathematical backbone

### 1.1 Steady state and the null space of the stoichiometric matrix
Let the network have $m$ metabolites and $n$ reactions, encoded in the **stoichiometric
matrix** $S \in \mathbb{R}^{m\times n}$ (entry $S_{ij}$ = how many molecules of metabolite $i$
reaction $j$ produces (+) or consumes (−)). On the timescale of a labeling experiment the cell
is at **metabolic steady state**: internal metabolite pools are constant, so production equals
consumption for every metabolite:

$$ S\,v = 0 . $$

This is a hard physical constraint. It says the feasible flux vectors are exactly the **null
space** of $S$. Because there are more reactions than independent mass balances
($\operatorname{rank}(S)=r<n$), the null space has dimension

$$ d = n - r > 0 , $$

the number of true **degrees of freedom** of the network. (For *I. orientalis*: $n = 387$
reactions, and after the constructions in §6 the working free dimension is $d = 44$.)

Any feasible flux can therefore be written with a basis $N\in\mathbb{R}^{n\times d}$ whose
columns span $\operatorname{null}(S)$:

$$ v = v_{\text{particular}} + N\,y , \qquad y \in \mathbb{R}^{d}. $$

The vector $y$ holds the **free coordinates** — the few numbers that, once known, fix every flux
in the network. **This is the quantity the model will actually predict** (§6–§7). Because the
homogeneous constraint $Sv=0$ (not $Sv=b$) is what we enforce, the implementation uses
$v_{\text{particular}}=0$, i.e. $v = N y$, which satisfies $Sv=0$ identically.

> **Intuition.** Instead of predicting 387 fluxes that must magically satisfy 343 balance
> equations, we predict 44 free numbers and *reconstruct* the 387 fluxes from them — mass balance
> is then automatic, by construction. This bakes the biology/physics into the model's output
> layer and removes 343 constraints the network would otherwise have to learn.

### 1.2 Reversible reactions: net flux + exchange
Many reactions run both directions. A net flux $v$ is split into forward and backward parts
using a **reversibility** $r\ge 0$ (the bidirectional exchange magnitude on top of the net):

$$ v_f = \tfrac12\big(|v| + v + r\big), \qquad v_b = \tfrac12\big(|v| - v + r\big). $$

For an irreversible reaction $r$ is forced to 0, so $v_b=0$. Reversibility matters for labeling:
exchange flux scrambles isotopes back and forth even when the *net* flux is small, so $r$ has a
real effect on the MIDs. The model predicts $r$ as a second output (§6).

### 1.3 The labeling forward model: EMU simulation
Given $v$ (hence $v_f,v_b$) and the tracer input, how do we compute the MIDs? We use the
**Elementary Metabolite Unit (EMU)** method (Antoniewicz et al., 2007), the standard efficient
algorithm for simulating isotope labeling.

The idea: to know the MID of a target metabolite fragment, you only need the MIDs of the smaller
carbon fragments that feed into it. EMU decomposition walks the atom‑transition map *backward*
from each measured fragment and collects the minimal set of carbon sub‑units ("EMUs") needed,
**grouped by size** (number of carbons). The labeling of size‑1 EMUs is computed first, then
size‑2, and so on — each size depends only on equal‑or‑smaller ones already solved.

For each EMU size $s$, the balance is a **linear system**

$$ A\,X = B\,Y , $$

where

$$ A = -\big(S_c\,\mathrm{diag}(v)\,S_+^{\,c}\big)^{\!\top}, \qquad
   B = \big(S_b\,\mathrm{diag}(v)\,S_+^{\,c}\big)^{\!\top}, $$

* $X$ is the matrix of unknown MIDs of the *core* EMUs of this size (each row an EMU, each of
  length $s+1$: the fractions at mass+0 … mass+$s$),
* $Y$ stacks the MIDs of the *boundary* (input) EMUs — either tracer inputs or already‑solved
  smaller EMUs,
* $S_c, S_b$ are the core/boundary rows of the size‑$s$ EMU incidence matrix and $S_+^{\,c}$ a
  source selector; $\mathrm{diag}(v)$ places the (reversible) edge fluxes on the diagonal.

When a larger EMU is produced by *condensing* two smaller fragments (e.g. a C–C bond formed from
two precursors), its boundary MID is the **convolution** of the two precursor MIDs — convolution
implements the fact that masses add:

$$ (\mathrm{MID}_1 * \mathrm{MID}_2)[\alpha] = \sum_{\beta=0}^{\alpha}\mathrm{MID}_1[\alpha-\beta]\,\mathrm{MID}_2[\beta]. $$

Solving the cascade of $A X = B Y$ systems from smallest to largest size yields every measured
metabolite's MID. The whole map $v \mapsto \mathrm{MID}$ is implemented as a differentiable
PyTorch operation (custom autograd `Function`), so gradients $\partial\,\mathrm{MID}/\partial v$
are available — important both for the classical solver and for our sensitivity analysis (§5).

> **Why this matters for SigmaFlux:** this forward model is the **simulator** that manufactures
> our training data. It is exact physics; the network's job is to learn its *inverse*.

---

## 2. Stage 1 — Generating diverse feasible flux vectors (sampling)

To teach the network to invert "MID → flux," we need a training set of flux vectors that covers
the range of metabolic states the real cell might be in. We obtain them by **sampling the
feasible flux polytope**. *(`sample_flux.py`)*

### 2.1 Preparing the model
Before sampling we lock down the feasible region so that sampled fluxes are biologically
plausible:

1. **Media bounds.** Every exchange reaction (`EX_*`) is first closed (bounds set to $0$), then
   only the nutrients/products in the config's `sampling.media` are reopened with explicit
   bounds. Sign convention: positive = secretion, negative = uptake. For *I. orientalis*:
   glucose uptake `EX_glc__D_e ∈ [−12, −1]` (mandatory carbon source), oxygen `[−100, 0]`,
   secretions (CO₂, ethanol, acetate, glycerol, succinate) `[0, 100]`, minerals `[−100, 0]`.
2. **Growth bounds.** The biomass reaction is bounded to `growth ∈ [0.01, 0.5]` h⁻¹, so we
   sample states from slow to fast growth.
3. **Feasibility + drop the objective.** We confirm the model is feasible, then set the
   objective to zero — we do **not** want to sample only optimal (max‑growth) states; we want the
   *whole* feasible polytope.
4. **FVA reduction.** Flux Variability Analysis over the entire feasible space
   (`fraction_of_optimum=0`) computes each reaction's true min/max. Reactions that can only be
   zero are removed; the rest have their bounds tightened to their actual FVA range (±1e‑6).
   Orphaned metabolites are pruned. This shrinks the polytope to its genuinely variable
   dimensions and makes sampling well‑conditioned.
5. **`skip_constraints` (BIOMASS).** After FVA (run *with* biomass active, so its coupling
   informs the other bounds), the biomass reaction is removed, along with the now production‑only
   biomass‑precursor pseudo‑metabolites it created. This frees the precursor pseudo‑reactions
   (`PROTEIN_c`, `RNA_c`, `DNA_c`, `LIPID_c`, `CARB_c`) to vary independently.
6. The fully prepared, reduced model is saved as **`model_prepared_for_mfa.json`** — the *exact*
   model the sampled fluxes (and all downstream code) refer to, guaranteeing consistent reaction
   ordering and bounds across every stage.

### 2.2 Why naive sampling is not enough — stratified sampling
A uniform Markov‑chain sampler over a high‑dimensional polytope concentrates near the centroid:
the *marginal* distributions of the few **measured** reactions (the ones the model will be
conditioned on) come out clustered around typical values. But the model must learn to invert
labeling across the *full range* of those measured fluxes. So we use **stratified sampling** to
force broad coverage of the measured‑flux marginals.

The stratified set is the list of measured reactions (`training.measured_fluxes`): the 10
reactions
`EX_glc__D_e, PROTEIN_c, RNA_c, DNA_c, LIPID_c, CARB_c, EX_etoh_e, EX_ac_e, EX_glyc_e, EX_succ_e`.

For each such reaction:

* split its FVA range into `n_bins = 10` equal‑width bins;
* for each bin, temporarily constrain that one reaction to the bin and run a short MCMC chain
  (`n_subsamples = 500`, `method = optgp`, `thinning = 100`, `processes = 8`), discarding bins
  that make the model infeasible.

This guarantees every bin of every measured reaction is seeded with samples — i.e. the pool
spans the whole measured‑flux range, not just the middle.

### 2.3 Quasi‑uniform draw via Iterative Proportional Fitting (IPF)
The stratified pool over‑represents some regions. We re‑weight it so that, for *every* measured
reaction, each of its bins carries equal probability mass — making the final draw
**quasi‑uniform** over the measured‑flux polytope.

Each sample is assigned a bin index per measured reaction. **IPF** then finds per‑sample weights
$w_i$ (initialized uniform) by repeatedly, for each reaction $f$, rescaling so the weighted mass
in each bin equals the target $1/\text{n\_bins}$:

$$ w_i \;\leftarrow\; w_i \cdot \frac{1/\text{n\_bins}}{\sum_{j:\,\text{bin}(j)=\text{bin}(i)} w_j}, \qquad
   \text{(cycle over reactions, renormalize, repeat 30×).} $$

This is classical iterative proportional fitting: the fixed point makes every (reaction, bin)
marginal uniform simultaneously. Finally we draw `n_samples = 10000` samples (without
replacement) with probability $\propto w_i$. The output is **`flux_samples.csv`**: 10 000 full
flux vectors (all reactions as columns), broadly and evenly covering the measured‑flux space.

> **Intuition.** Stratification + IPF is "manufactured diversity": it ensures the training set
> contains, e.g., low‑glucose *and* high‑glucose states, low‑ *and* high‑ethanol states, in equal
> measure, so the trained model is calibrated across the whole physiological range rather than
> only near the average.

---

## 3. Stage 2 — Simulating the isotope labeling (building MID targets)

Each sampled flux vector is now pushed through the EMU forward model (§1.3) to produce the MIDs
it would generate. *(`generate_training_data.py`)*

### 3.1 Tracers
The labeling experiment is defined by the tracers in `labeling.tracers`. *I. orientalis* uses
**two parallel glucose tracers**, each simulated as its own independent experiment:

* **U‑¹³C glucose** — all six carbons labeled (`[1,1,1,1,1,1]`), 50% enrichment;
* **1,2‑¹³C glucose** — first two carbons labeled (`[1,1,0,0,0,0]`), 50% enrichment.

Using two complementary labeling patterns resolves more fluxes than either alone (different
patterns disambiguate different pathways). Each metabolite with a defined carbon atom‑transition
map becomes an EMU target.

### 3.2 Reversibilities are sampled, not read
The CSV provides only **net** fluxes. The exchange/reversibility $r$ of each reaction is drawn
fresh **per sample** from an exponential prior and scaled by the net flux magnitude:

$$ r_i \;\approx\; \mathrm{Exp}(\lambda{=}1)\cdot|v_i|, \quad\text{clamped at }10\,|v_i|, \quad\text{($r_i=0$ if irreversible).} $$

So each training example carries a different, physically‑motivated exchange pattern. This
injects exchange variability into the labeling that is independent of the FBA sampling and makes
the model robust to it. (For *I. orientalis*, 33 of 387 reactions have variable reversibility.)

### 3.3 Forward pass and outputs
For each of the 10 000 samples: the net fluxes are loaded, reversibility is re‑drawn, and the
EMU cascade $A X = B Y$ (§1.3) is solved (in `float64`, TF32 disabled for determinism) to give
the full set of metabolite MIDs. Degenerate samples (singular $A$) are skipped with a warning.
The simulated MID vector is gathered in a fixed index order over
`(tracer, metabolite, M+i)` for $i = 1\ldots$(#carbons).

Three aligned HDF5 tables are written (sliceable by sample index for parallel jobs, suffix
`_<start>_<end>`):

* `wb_model_labeling_data*.h5` (key `labeling`) — the **MIDs** (model *inputs/observables*),
* `wb_model_flux_data*.h5` (key `fluxes`) — the **net fluxes** (regression *targets*),
* `wb_model_reversibility_data*.h5` (key `reversibility`) — the **reversibilities** (second
  target).

**No measurement noise is added at generation time** — the data are exact physics. Noise is
injected later, during training, as augmentation (§9). For *I. orientalis* the raw labeling
table is $4536$ MID features × $10000$ samples.

---

## 4. How a measurement becomes a model input — tokens

Before the selection and model stages, we define how data enters the network. SigmaFlux is a
**transformer over a set of tokens**. There are two token types, and crucially they are projected
into the same hidden space so they can live in one attention sequence.

### 4.1 Labeling tokens (one per measured isotopomer)
Each labeling token represents a single measured number: the fraction of metabolite $X$ found at
mass M+$k$ under tracer $T$. A token therefore carries four pieces of identity/value:

* **compound** (which metabolite, e.g. `glu__L`),
* **label** (which isotopomer, M+1, M+2, …),
* **tracer** (which labeling experiment),
* **fraction** — the measured scalar value $\in[0,1]$.

The three categorical parts are turned into vectors by **radial embeddings** (§4.3) and
concatenated into an "entity" vector; a **token‑type embedding** (id 0 for labeling) is appended.
The scalar fraction is *not* concatenated — instead it is encoded to the full hidden width by a
small scalar encoder and **added** to the projected entity vector, then layer‑normed:

$$ x_{\text{lab}} = \mathrm{LayerNorm}\big(\,W_{\text{in}}[\,e_{\text{comp}}\,\|\,e_{\text{lab}}\,\|\,e_{\text{trac}}\,\|\,e_{\text{type0}}\,] + \mathrm{Enc}_{\text{frac}}(\text{fraction})\big). $$

### 4.2 Flux ("bypass") tokens — how known fluxes enter
The few **known/measured fluxes** (boundary exchanges, growth) are provided to the model as a
second kind of token. Each carries:

* **reaction** identity (radial embedding of the reaction id), zero‑padded to the same width as
  the labeling entity vector,
* **token‑type embedding** id 1 (marking it a flux token),
* the **measured flux scalar**, per‑reaction z‑scored and then added through the *same* shared
  projection + a scalar encoder:

$$ \tilde v = \frac{v_{\text{meas}} - \mu_v}{\sigma_v}, \qquad
   x_{\text{flux}} = \mathrm{LayerNorm}\big(W_{\text{in}}[\,e_{\text{rxn}}\,\|\,e_{\text{type1}}\,] + \mathrm{Enc}_{\text{flux}}(\tilde v)\big). $$

Both token types are then concatenated into a single sequence
$x = [\,x_{\text{lab}};\,x_{\text{flux}}\,]$. **This is the entirety of how known fluxes are
provided: as ordinary tokens in the input set.** The network fuses them with the MID tokens
purely through attention (the token‑type embedding lets it tell them apart). There is *no* gate
and *no* separate "known‑flux shortcut" in the current architecture — an earlier design had a
soft fusion gate, but it was refactored out in favor of this simpler uniform‑token scheme. For
*I. orientalis*, a pre‑refactor run (§5, greedy‑trimmed selection, since removed) used 196
labeling tokens (98 per tracer × 2 tracers) + 10 flux tokens; today's significance‑only selection
(§5.4) keeps a different, larger token count.

### 4.3 Radial embeddings (and the scalar encoder)
Categorical ids (compound, label, tracer, reaction, token‑type) are embedded with a **fixed,
non‑learned radial table**. For a vocabulary of size $N$ and $H$ harmonics, index $k$ maps to a
$2H$‑dimensional vector of sines and cosines at $H$ angular frequencies:

$$ e(k) = \big[\sin\!\big(2\pi \tfrac{k}{N} h\big),\ \cos\!\big(2\pi \tfrac{k}{N} h\big)\big]_{h=1\ldots H}. $$

These are Fourier features of the index — like sinusoidal positional encodings, but over the
vocabulary axis. Two reasons we prefer this to a learned `nn.Embedding` lookup: (i) it is a
*deterministic* function of the id, so it **generalizes to ids never seen in training** (the
compound vocabulary is the *entire* metabolic model — 356 metabolites — not just the measured
ones); (ii) it gives a smooth, fixed geometry that doesn't need to be learned. The dimensions
used: compound $2H{=}32$, label $10$, tracer $2$, plus the fraction scalar → the printed
"Token dim 45" for the legacy feature layout. The scalar encoder featurizes a value $x$ as
$[\,x,\ \log(|x|{+}1),\ |x|,\ \mathrm{sign}(x)\,]$ before a linear projection to the hidden width
— giving the network easy access to magnitude, log‑magnitude, and sign.

---

## 5. Stage 3 — Fisher information token selection

The raw simulated MID has thousands of entries (4536 for *I. orientalis*), but most are
redundant or uninformative about the fluxes. Feeding all of them is wasteful and hurts learning.
We select an informative subset using **optimal‑experiment‑design theory**.
*(`extract_informative_features.py`, `fisher_selection.py`)*

### 5.1 The local linear model of "MID responds to flux"
Work in the free coordinates $\theta = y$ (§1.1; optionally the variable reversibilities,
log1p‑z‑scored, are appended). Linearize the simulator about the sample distribution: a token's
MID value responds to the free coordinates through a sensitivity (Jacobian) $J_m = \partial
L_m/\partial\theta$. Under i.i.d. Gaussian measurement noise of std $\sigma$, token $m$
contributes **Fisher information**

$$ F_m = \frac{1}{\sigma^2}\,J_m^{\top}J_m \quad (d\times d). $$

The Fisher matrix quantifies how sharply that measurement pins down the free fluxes, and in which
directions. In the linear‑Gaussian regime, Fisher information and *mutual information* coincide:

$$ I(\theta; L_S) = \tfrac12\,\log\det\!\big(I + F_S\,\Sigma_{\text{prior}}\big), $$

so selecting high‑Fisher tokens is the same as selecting maximally informative measurements.

### 5.2 Estimating sensitivities from samples (no simulator re‑runs)
Rather than finite‑difference the simulator, we estimate each token's sensitivity directly by
**least‑squares regressing its MID column onto $\theta$ across the generated samples** — a
global‑linear sensitivity. To avoid the positive bias of an in‑sample Gram, we use
**cross‑fitting**: split samples into folds A/B, estimate sensitivities $B_A,B_B$ separately, and
take the symmetrized cross‑product, averaged over several random splits, then projected to the
nearest PSD matrix:

$$ G_m = \mathrm{sym}\!\big(B_A^{\top}B_B\big), \qquad F_m = \tfrac{1}{\sigma^2}G_m. $$

Spurious gradients differ across folds and cancel in expectation, so a genuinely uninformative
token yields $G_m\approx 0$.

### 5.3 The standalone information score
The flux prior contributes a precision $F_0 = \mathrm{cov}(\theta)^{-1}$ (with a tiny ridge). Each
token's standalone information — the information it would give *on its own* — is

$$ \texttt{standalone\_nats} = \log\det(F_0 + \tfrac1{\sigma^2}G_m) - \log\det F_0 . $$

An earlier version of this pipeline also ran a greedy D‑optimal forward selection on top (adding
tokens by largest redundancy‑discounted marginal gain, stopping at a relative elbow) to prune
`significant` down to a smaller non‑redundant `selected` subset. It was tried and found not to
help, and has been removed: only the standalone significance test below decides what the model
trains on.

### 5.4 Significance, and what the model trains on
* **Significance floor.** Because the empirical Fisher has finite‑sample bias, we calibrate a
  null by permuting $\theta$ against the tokens many times and taking the largest standalone gain
  achievable by chance. A token is **`significant`** iff its standalone information exceeds that
  floor — and, with no redundancy trimming, **`included` = `significant`** always.
* **Exact vs. approximate.** `features.token_selection_method` picks how $G_m$ is estimated:
  `fisher` cross‑fits it (bias‑corrected, costlier), `fisher_fast` uses a single‑shot fit (biased,
  cheaper). Everything above runs identically either way.

`features.csv` columns: `metabolite_id, isotopomer, included, significant, rank,
standalone_nats`.

> **Why this is satisfying.** Stage 3 applies a significance test to choose *which mass‑spec peaks
> carry real information* versus what chance alone produces, keeping only tokens that clear a
> permutation‑calibrated floor — and, as an interpretability check, the trained model's attention
> later turns out to concentrate on exactly these Fisher‑informative metabolites. (The ~20×
> compression figure below was specific to the since‑removed greedy trim; significance‑only
> selection keeps a larger, differently‑sized set.)

---

## 6. The null‑space reparameterization — the algebra the model predicts in

The model predicts the $d$ free coordinates $y$ and reconstructs $v = Z y$. *Which* basis $Z$ we
use matters a great deal — a raw SVD basis spreads weight over nearly every reaction, making each
coordinate an uninterpretable, statistically entangled mixture. SigmaFlux therefore *reparameterizes*
the null space into a basis whose coordinates are sparse, identifiable, and aligned to meaningful
flux modes. *(`src/sigmaflux/ai/utils.py`, selected by `null_space_type`.)*

### 6.1 Varimax‑sparsified basis
Starting from an orthonormal SVD null‑space basis, a **Varimax rotation** $R$ (an orthogonal
rotation that maximizes the variance of squared loadings) concentrates each column onto a few
reactions:

$$ \max_R \sum_j\Big[n\sum_i z_{ij}^4 - \big(\textstyle\sum_i z_{ij}^2\big)^2\Big]. $$

Solved by Jacobi sweeps of closed‑form $2\times2$ rotations, which preserve orthonormality
exactly. Result: sparse, more pathway‑aligned free coordinates spanning the same space.

### 6.2 Targeted basis — give each measured reaction its own coordinate
For the **measured reactions** we want their flux to *be* a coordinate (so $y_j$ = that flux
directly). `compute_targeted_nullspace` builds a basis with an **identity block** on the target
rows: $Z[t, j_t]=1$. The construction:

1. **Separability check** — the target rows of the raw basis must be full rank; otherwise the
   targets are stoichiometrically coupled and cannot each get an independent coordinate (fall
   back to ICA).
2. Pick $r$ independent metabolite balances and a "dependent" block of reactions via pivoted QR;
   solve $S_D X = -S_{\text{free}}$ so that the resulting $Z$ satisfies $SZ=0$ exactly.
3. The free columns are: the targets first (identity), then the remaining independent reactions.
4. **Normalize columns** (so the Gram $G=Z^\top Z$ has unit diagonal — bounds the loss scale),
   and **Varimax‑rotate the non‑target free columns** for sparsity.

### 6.3 ICA‑informed and Targeted‑ICA (the production basis)
The non‑target free directions are still arbitrary. We make them **statistically meaningful** by
running **FastICA** on the *sampled* fluxes projected into the free subspace — finding directions
that are maximally independent flux modes (after PCA‑whitening; directions the sampler never
explored are detected via the covariance eigen‑spectrum and excluded so ICA doesn't fit noise).
Any leftover dimensions are filled with the SVD complement.

The default **Targeted‑ICA** basis combines §6.2 and ICA: identity coordinates for the measured
reactions, ICA‑aligned coordinates for the rest. For *I. orientalis* this yields the headline
decomposition

$$ d = 44 = \underbrace{10}_{\text{identity DOF (measured)}} + \underbrace{34}_{\text{ICA free modes}} + \underbrace{0}_{\text{SVD complement}}, \quad \text{free rank } 34/34, $$

i.e. all 34 non‑target directions are well‑sampled and ICA‑aligned (no padding needed).

### 6.4 The basis $Z$, its dual $Z_{\text{proj}}$, and normalization
* The chosen basis becomes $Z_{\text{full}}\in\mathbb{R}^{n\times d}$, stored in the model.
* Reconstruction: $v = y\,Z_{\text{full}}^{\top}$ (satisfies $Sv=0$).
* Because the targeted basis is **not orthonormal** ($G=Z^\top Z\ne I$), recovering coordinates
  from a flux uses the **dual basis**

$$ Z_{\text{proj}} = Z_{\text{full}}\,G^{-1}, \qquad y = v\,Z_{\text{proj}}, $$

  the pseudo‑inverse coordinate map (for an orthonormal basis $G=I$ and $Z_{\text{proj}}=Z_{\text{full}}$).
* **Target normalization.** True targets $y = v\,Z_{\text{proj}}$ are computed on the training set
  and z‑scored: $y_{\text{mean}}, y_{\text{std}}$ (clamped $\ge10^{-6}$). The network outputs a
  standard‑normal $\hat y_{\text{raw}}$, denormalized inside the model
  $\hat y = \hat y_{\text{raw}}\,y_{\text{std}} + y_{\text{mean}}$, then mapped to fluxes.

### 6.5 Observability — which directions the data can actually constrain
Not every free coordinate leaves a fingerprint in the labeling. For each free coordinate $y_i$
and each MID feature $L_j$ we compute the Pearson correlation across the training set, and take

$$ \text{obs}_i = \max_j |r_{ij}|. $$

A coordinate with $\text{obs}_i\approx0$ is **unobservable from labeling** — no measurement
constrains it. The diagnostics report how many directions exceed $|r|>0.1$ using labeling alone
vs labeling+known‑fluxes (the latter is higher, because a measured flux can pin a direction the
labeling cannot). These observability weights can down‑weight the loss on directions the data
fundamentally cannot resolve, so the model is not penalized for the impossible.

> **Why all this algebra.** Predicting in a sparse, identifiable, normalized basis means: (i)
> mass balance is automatic; (ii) the measured fluxes are literally coordinates (so the network's
> known‑flux inputs map cleanly onto its outputs); (iii) each predicted number is a meaningful,
> decorrelated flux mode rather than an arbitrary mixture; and (iv) we know up front which modes
> are even learnable.

---

## 7. Stage 4 — The model architecture

SigmaFlux is a **Perceiver‑IO‑style cross‑attention decoder** (`FluxDecoder`). It reads a set of
tokens (variable size) and produces a fixed set of outputs (the free coordinates +
reversibilities). For *I. orientalis* the model is ~895k parameters.

### 7.1 Encoder over the token set
The labeling and flux tokens (§4) are concatenated into one sequence and passed through a
**self‑attention transformer encoder** (default 4 layers, 4 heads, hidden width $d_{\text{model}}=64$,
feed‑forward $4\times$, GELU, pre‑norm, dropout). Padding/masking hides absent or dropped tokens.
Here every measurement can attend to every other and to the known fluxes — the model builds a
context‑aware representation of the whole experiment.

### 7.2 Cross‑attention to flux queries
A fixed bank of $R = d_{\text{free}} + n_{\text{rev}}$ **learned query vectors** (one per output)
attends over the encoded tokens (queries = the outputs, keys/values = the tokens). Each query
gathers, from across all MID and flux tokens, the information relevant to its output coordinate.
A single shared linear head maps each query's output to one scalar logit.

### 7.3 Outputs: fluxes (via the basis) and reversibilities
The $R$ logits split into:

* **Free coordinates** ($d_{\text{free}}$): denormalized with $y_{\text{std}},y_{\text{mean}}$ and
  reconstructed to the full flux vector

$$ v = \hat y\,Z_{\text{full}}^{\top} \quad\Rightarrow\quad S v = 0 \text{ exactly.} $$

* **Reversibilities** ($n_{\text{rev}}$): output in z‑scored‑log1p space; converted to physical
  $r = \mathrm{expm1}(\,\hat r\,\sigma_{\log r} + \mu_{\log r})$ (clamped $\ge0$) at inference.

For *I. orientalis*: 387 reconstructed fluxes and 33 reversibilities, from 44 + 33 = 77 query
outputs.

> **Architectural summary.** Inputs are an unordered *set* of measurements + known fluxes →
> self‑attention contextualizes them → learned flux queries cross‑attend to extract each free
> coordinate → the null‑space basis turns coordinates into a physically valid flux vector. The
> known fluxes participate as ordinary tokens; the basis output layer enforces the steady‑state
> physics.

---

## 8. The loss function

The network is trained by supervised regression against the simulated targets. The loss has two
terms — fluxes and reversibilities — summed with equal weight, computed in **normalized target
space**:

$$ \mathcal{L} = \mathcal{L}_{\text{flux}}(\hat v, v) + \mathcal{L}_{\text{rev}}(\hat r, r). $$

The default per‑term loss is a **variance‑normalized MSE** (`loss='variance'`): each feature's
squared error is divided by that feature's training variance,

$$ \mathcal{L} = \frac{1}{F}\sum_{f=1}^{F}\frac{\mathbb{E}_{\text{batch}}\big[(\hat z_f - z_f)^2\big]}{\mathrm{Var}(z_f) + \varepsilon}, $$

which is $\approx 1-R^2$ per feature. This makes every reaction contribute comparably regardless
of its flux magnitude — without it, a few high‑flux reactions would dominate the gradient and the
many small but biologically important fluxes would be ignored. (A plain MSE option exists, as
does an optional small term that directly matches the measured fluxes, off by default.)

**Why log1p for reversibilities.** Reversibilities span ~4 orders of magnitude (median ≈ 5, up to
thousands). We transform targets as $\tilde r = (\log(1+r) - \mu_{\log r})/\sigma_{\log r}$. The
`log1p` compresses the heavy tail (and handles $r=0$ gracefully); the z‑score puts it on the same
$\approx\mathcal N(0,1)$ scale as the flux coordinates, so neither output head dominates the loss.

> **Note (contrast with classical MFA).** The standalone classical ¹³C‑MFA solver in this
> codebase minimizes a *per‑block reduced chi‑squared* (labeling block vs flux block, each
> weighted by real measurement uncertainties and divided by its measurement count). That objective
> belongs to the optimization solver, **not** to the neural network — the network learns the
> inverse map from simulated data and uses the variance‑normalized MSE above.

---

## 9. Training details

* **Optimizer / schedule.** Adam, learning rate $10^{-3}$; a linear **warmup** over the first
  `warmup_epochs = 5`, then **ReduceLROnPlateau** (factor 0.5, patience 5, floor $10^{-6}$) on the
  validation loss. Gradient‑norm clipping at 1.0.
* **Budget.** `epochs = 150`, `batch_size = 256` (from the config). For *I. orientalis* this took
  ~6 min on one A100.
* **Split.** Seeded shuffle into train/val (default 80/20 → 8000 train / 2000 val). Validation
  reuses the training normalization statistics (no leakage).
* **Best‑model checkpointing.** The best‑validation state is kept in memory and restored at the
  end; only the final best model is saved.
* **What the checkpoint stores** (so inference is self‑contained): `model_state_dict`,
  `model_config`, the radial `vocabs` and `tracer_combo_vocab`, the full **`token_index`** (the
  fixed token skeleton), `norm_stats`, train/val indices, the loss/val logs, per‑reaction
  validation **R²** (`val_correlations`), `flux_names`/`reversibility_names`, `measured_fluxes`,
  the `null_space_data` (basis $Z$, `free_idx`/`fixed_idx`, observability weights,
  $y_{\text{mean}/\text{std}}$, $r_{\log\text{‑}\text{mean}/\text{std}}$), `flux_norm_stats`, and the
  paths `data_dir`/`model_json`. Per‑reaction R² drives a "reaction learning asymmetry" diagnostic
  (which fluxes the model learns well vs poorly).

---

## 10. Handling measurement noise

Real MIDs and real measured fluxes are noisy. We make the model robust by **augmenting with
Gaussian noise during training** (the simulated data themselves are noise‑free). Two independent
noise channels, applied per batch:

1. **Labeling (MID) noise** — *absolute, sum‑preserving.* For each metabolite×tracer group, the
   implicit M+0 is reconstructed ($M{+}0 = 1-\sum_{k\ge1}M{+}k$), Gaussian noise of std
   `labeling_noise_std` is added to every fraction *including* M+0, then the group is renormalized
   to sum to 1 and clipped to $[0,1]$. This mimics mass‑spec error while preserving the physical
   constraint that an MID is a probability distribution.

2. **Measured‑flux noise** — *relative (multiplicative).* Each measured flux is scaled by
   $(1 + \text{flux\_noise\_std}\cdot\epsilon)$, $\epsilon\sim\mathcal N(0,1)$ — a constant
   relative‑error model appropriate for exchange‑rate measurements.

Additionally, **token dropout** (randomly hiding some labeling fractions and some flux tokens,
always keeping at least one flux token) trains the model to cope with incomplete measurement sets.
Uncertainty is *not* fed as an explicit feature; it enters only through these augmentation
distributions. (Noise levels are config‑set; 0 disables a channel.)

---

## 11. Handling missing and variable measurements

Real datasets never measure every metabolite, and different experiments measure different things.
SigmaFlux handles this through a **fixed token skeleton + zero‑filling**, not variable‑length
hacks:

* **Fixed skeleton.** The set of tokens the model expects is exactly `token_index` (the
  Fisher‑selected tokens from training). At inference we build a zero‑filled table over this
  skeleton and write the user's measured fractions into the matching `(tracer, metabolite,
  isotopomer)` rows. **A missing measurement is simply a fraction of 0** — the sequence length is
  constant and no special masking is required for absent values.
* **Drop M+0.** M+0 is determined by the others (fractions sum to 1) and is never a model input;
  M+0 rows in user data are dropped.
* **Ignore unknowns.** Metabolites or tracers not seen in training are dropped/warned. Because the
  compound vocabulary is the *entire* metabolic model, any in‑network metabolite still gets a
  valid radial embedding.
* **Variable known fluxes.** The number of flux tokens is just however many measured fluxes are
  provided; a padding mask hides any that are absent, in both the encoder and the cross‑attention.
* **Constant‑reversibility reactions** (no variability in training) are dropped from the
  reversibility output head.

---

## 12. Inference — putting it together on real data

At prediction time (`inference.py`): load the checkpoint (which carries the basis, vocabs, token
skeleton, and all normalization), read the experimental MIDs and measured exchange fluxes, drop
M+0 / unknowns, build the fixed‑skeleton token set, and run one forward pass. The network outputs
the free coordinates and reversibilities; the basis reconstructs the full, mass‑balanced flux
vector $v = \hat y\,Z_{\text{full}}^{\top}$, and reversibilities are converted to physical units.
What took a bespoke nonlinear optimization per condition in classical MFA is now a single,
automatic forward pass — across many conditions and growth rates at once.

---

## 13. Why it works — and how we know

Two design choices do the heavy lifting:

1. **Physics in the output layer.** By predicting in the null‑space basis, every prediction
   *automatically* satisfies $Sv=0$. The network never has to learn 343 balance equations; it only
   has to learn the 44 free coordinates the data can constrain.

2. **Information‑guided inputs.** Fisher‑significance selection feeds the network only the
   measurements that actually carry flux information (beat the permutation‑null floor), without
   losing identifiability.

And the trained model behaves sensibly, which we verified post‑hoc on the attention maps:

* **Fisher‑aligned attention.** The model's attention per metabolite correlates with that
  metabolite's Fisher information (Spearman $r\approx0.58$, $p<10^{-3}$) — it independently learned
  to rely on the optimal‑design measurements.
* **Mechanistically local attention.** The metabolites a reaction attends to are closer in the
  metabolic network than chance (attention‑weighted distance below the uniform baseline in ~95% of
  reactions, $p\approx10^{-56}$; the single most‑attended metabolite a median 5 vs 7 hops away).

Together with strong recovery of expert‑curated reference fluxes across growth conditions, this
indicates SigmaFlux learns the *mechanistic* inverse of the labeling model rather than spurious
correlations.

---

### Appendix — concrete dimensions for the *I. orientalis* run

Token counts below are from a pre‑refactor run using the since‑removed greedy trim (§5.3); the
significance‑only selection now in use keeps a different, larger labeling‑token count.

| Quantity | Value |
|---|---|
| Reactions $n$ | 387 |
| Metabolites (compound vocab) | 356 |
| Raw MID features | 4536 → 4500 (extracellular dropped) |
| Fisher‑selected labeling tokens | **196** (98 per tracer × 2 tracers, pre‑refactor) |
| Known‑flux (bypass) tokens | 10 |
| Null‑space dimension $d$ | **44** = 10 identity DOF + 34 ICA free modes |
| Reversibility outputs | 33 (variable) of 387 |
| Query outputs $R$ | 77 (= 44 free + 33 rev) |
| Tracers | 2 (U‑¹³C and 1,2‑¹³C glucose, 50% enrichment) |
| Training samples | 10 000 (8000 train / 2000 val) |
| Model size | ~895k parameters |
| Training | 150 epochs, batch 256, Adam(1e‑3), warmup 5 + ReduceLROnPlateau |
| Hidden width / layers / heads | 64 / 4 / 4 |

*Pipeline scripts:* `sample_flux.py` (sampling) → `generate_training_data.py` (EMU labeling
simulation) → `extract_informative_features.py` (Fisher selection) → `train_ai_model.py`
(training); inference via `sigmaflux.ai.inference`. Core physics in `sigmaflux/core/` (stoichiometry,
EMU), model in `sigmaflux/ai/model.py`, null‑space constructions in `sigmaflux/ai/utils.py`,
Fisher selection in `sigmaflux/ai/fisher_selection.py`.
