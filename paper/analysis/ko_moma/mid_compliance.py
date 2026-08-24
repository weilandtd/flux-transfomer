"""MID-compliance check for the MoMA/FBApro knockout prior.

Two competing knockout hypotheses, from a common pFBA-optimal wild-type flux:
  MoMA  (bounded_moma) : the cell stays as close to WT as balance/bounds allow.
  pFBA                 : the cell re-optimizes growth under the knockout.
We simulate the 13C MIDs each predicted flux would produce (SigmaFlux forward model,
zero exchange so MIDs depend only on the net fluxes both methods predict) and ask:
do the two priors give *distinguishable* labeling? If yes, measuring KO MIDs can tell
them apart (KO 13C is worth doing); if no, either prior suffices.

Metrics per knockout (RMS over the measurable-metabolite MIDs, and in units of the
0.01 measurement noise):
  ko_effect_pFBA = || MID(pFBA_KO) - MID(WT) ||     does re-optimization move labeling?
  ko_effect_MoMA = || MID(MoMA_KO) - MID(WT) ||     does minimal adjustment move it?
  disagreement   = || MID(MoMA_KO) - MID(pFBA_KO) || the money number
"""
import os
import sys

import numpy as np
import pandas as pd
import yaml
from cobra.io.json import load_json_model
from cobra.flux_analysis import pfba

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_REPO, "src"))
from sigmaflux.pipeline._benchmark import build_mfa_context, forward_labeling  # noqa: E402
from sigmaflux.core.fbapro import knockout_flux_from_model  # noqa: E402

CFG = yaml.safe_load(open(os.path.join(_REPO, "paper/configs/i_orientalis.yaml")))
MODEL_JSON = os.path.join(_REPO, "models/atomistic/I_orientalis/imIsor_annotated_hierarchical.json")
MAPPING = os.path.join(_REPO, "models/atomistic/I_orientalis/I_orientalis_mapping.csv")
NOISE = 0.01
CANDIDATES = [  # central-carbon reactions to try (filtered to those present + non-essential)
    "PGI_c", "PFK_c", "FBA_c", "TPI_c", "GAPD_c", "PGK_c", "PGM_c", "ENO_c", "PYK_c",
    "G6PDH2r_c", "PGL_c", "GND_c", "RPE_c", "RPI_c", "TKT1_c", "TKT2_c", "TALA_c",
    "PDH_m", "CS_m", "ACONTa_m", "ACONT_m", "ICDHxm_m", "ICDHy_m", "AKGDm_m", "AKGDH_m",
    "SUCOASm_m", "SUCD_m", "FUM_m", "FUMm_m", "MDH_m", "MDHm_m", "ME1_c", "ME2_m",
    "PPC_c", "PPCK_c", "PC_m", "TKT1_c",
]


def apply_media(m):
    media = CFG["sampling"]["media"]
    for r in m.reactions:
        if r.id.startswith("EX_"):
            if r.id in media:
                r.bounds = (media[r.id]["lb"], media[r.id]["ub"])
            else:
                r.bounds = (0.0, r.upper_bound if r.upper_bound > 0 else 1000.0)


def main():
    m = load_json_model(MODEL_JSON)
    apply_media(m)
    m.objective = "BIOMASS"
    wt_growth = m.slim_optimize()
    v_wt = pfba(m).fluxes                                   # common WT reference
    print(f"WT growth={wt_growth:.4f}; WT pFBA total flux computed", flush=True)

    # measurable metabolites (what a real experiment would see)
    measured = set(pd.read_csv(MAPPING)["model_id"].dropna())

    print("Building MID forward model (EMU expansion, a few minutes) ...", flush=True)
    mfa_model, reaction_ids, _, labeling_to, labeling_index = build_mfa_context(
        m, CFG["labeling"]["tracers"], v_wt, "moma_mid_check", with_backward=False)
    # mask of tokens whose metabolite is measurable
    lab_met = [str(t[1]) if isinstance(t, tuple) else str(t).split("|")[0] for t in labeling_index]
    meas_mask = np.array([mt in measured for mt in lab_met])
    print(f"labeling tokens: {len(labeling_index)} total, {int(meas_mask.sum())} on measured metabolites",
          flush=True)

    rev0 = np.zeros(len(reaction_ids))                      # net-flux-only MIDs

    def mid(flux_series):
        arr = flux_series.reindex(reaction_ids).fillna(0.0).to_numpy(float)
        return forward_labeling(mfa_model, arr, rev0, labeling_to)[meas_mask]

    mid_wt = mid(v_wt)
    v_wt_dict = v_wt.to_dict()

    rows = []
    for ko in CANDIDATES:
        if ko not in reaction_ids or abs(v_wt.get(ko, 0.0)) < 1e-6:
            continue                                        # skip absent or WT-inactive
        with m:
            m.reactions.get_by_id(ko).bounds = (0.0, 0.0)
            g = m.slim_optimize()
            if not (g and g > 0.05 * wt_growth):
                continue                                    # essential-ish, skip
            v_pfba = pfba(m).fluxes
        v_moma = pd.Series(knockout_flux_from_model(m, v_wt_dict, [ko], bounded=True))
        mid_pfba, mid_moma = mid(v_pfba), mid(v_moma)
        rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
        rows.append(dict(
            ko=ko, ko_growth=round(g, 3),
            eff_pfba=rms(mid_pfba - mid_wt),
            eff_moma=rms(mid_moma - mid_wt),
            disagree=rms(mid_moma - mid_pfba),
            disagree_max=float(np.max(np.abs(mid_moma - mid_pfba))),
            flux_l2=float(np.linalg.norm(v_moma.reindex(reaction_ids).fillna(0) -
                                         v_pfba.reindex(reaction_ids).fillna(0))),
        ))
        r = rows[-1]
        print(f"  {ko:10s} g={r['ko_growth']:.2f}  eff(pFBA)={r['eff_pfba']:.4f} "
              f"eff(MoMA)={r['eff_moma']:.4f}  disagree={r['disagree']:.4f} "
              f"({r['disagree']/NOISE:.1f}x noise, max={r['disagree_max']:.3f})", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(__file__), "mid_compliance_results.csv")
    df.to_csv(out, index=False)
    print("\n===== SUMMARY (RMS MID over measured metabolites) =====", flush=True)
    print(f"knockouts evaluated: {len(df)}", flush=True)
    if len(df):
        print(f"median KO effect (pFBA vs WT)     : {df.eff_pfba.median():.4f}  "
              f"({df.eff_pfba.median()/NOISE:.1f}x noise)", flush=True)
        print(f"median KO effect (MoMA vs WT)     : {df.eff_moma.median():.4f}  "
              f"({df.eff_moma.median()/NOISE:.1f}x noise)", flush=True)
        print(f"median prior disagreement         : {df.disagree.median():.4f}  "
              f"({df.disagree.median()/NOISE:.1f}x noise)", flush=True)
        print(f"max prior disagreement (any KO)   : {df.disagree.max():.4f}  "
              f"({df.disagree.max()/NOISE:.1f}x noise)", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
