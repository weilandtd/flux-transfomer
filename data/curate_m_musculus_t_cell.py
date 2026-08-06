"""
Curate the raw M. musculus T-cell data into the canonical SigmaFlux layout.

The raw drop is three files that no loader can read:

    measured MID.xlsx                    sheets note/UGlc/UGln/12Glc, columns A1..N7
    MM_TCELL_exterinal_measurements.csv  tab-separated, long, uptake-positive, umol
    MFA_Fluxes.csv                       tab-separated, the paper's full flux table

This produces the four workbooks the pipeline expects, mirroring data/I_orientalis:

    M_musculus_T_Cell_A.xlsx        activated, one sheet per tracer   (cf. I_orientalis_batch.xlsx)
    M_musculus_T_Cell_N.xlsx        naive,     one sheet per tracer
    M_musculus_T_Cell_ExtFlux.xlsx  sheet 'Constraints'               (cf. I_orientalis_ExtFlux.xlsx)
    MFA_flux_data_MM.xlsx           reference fluxes                  (cf. Shen_et_al_flux_data_IO.xlsx)

Run from the repository root:

    python data/curate_m_musculus_t_cell.py

The three raw files have since been deleted — only the curated workbooks are kept, and
they are what the pipeline reads. This script is retained as the record of how they
were derived; restore the sources beside it to re-run.

UNITS AND SIGNS
---------------
The measurement CSV is umol/1E6 cells/h and uniformly uptake-positive.  The published
flux table is nmol/1E6 cells/h in the raw Maranas per-reaction direction, which is
inconsistent between exchanges.  SigmaFlux standardises every exchange to
secretion-positive at conversion time (_standardize_exchange_directions), so:

    measurements   ->  value = -1000 * mean            (one uniform rule)
    reference flux ->  negate the 21 exchanges that conversion flips

Both then live in the same convention.  Cross-check: activated EX_ala__L_e is
-1000 * 0.004153 = -4.153 from the measurements and -1 * 4.1375 from the paper.

VALIDATION
----------
Feeding the curated activated flux vector through the converted model gives
||S.v||inf = 4.7e-07 against a peak flux of 701, i.e. mass balance to machine
precision.  That checks the whole conversion chain at once: the dropped dilution
network, the _out removal, the 21 flipped exchanges and the transketolase mapping.

The naive vector balances everywhere except protonleak_m, which is off by 12.84.
ATPS_m produces 1.0 protonleak_m and Ht_c_m consumes 0.2663 per unit; activated
satisfies that exactly (0.2663 * 701.52 = 186.81 = ATPS_m), naive does not
(22.53 against 9.69).  This is an inconsistency in the published naive flux
vector, not in the conversion, and it is left as published.

19 reactions carry NA for the naive state - deoxyribonucleotide synthesis
(RNDR*, TMDS_c, DTMPK_c, ...) and de novo lipogenesis (ACITL_c, SK_accoa_c).
Naive T cells neither divide nor run lipogenesis, so NA means the reaction was
absent from the naive model and is curated as blank, which reads back as zero.
"""
import pathlib
import re
import sys
import unicodedata

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'src'))

from sigmaflux.core.utils import BIOMASS_COMPONENT_IDS, derive_biomass_fractions  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / 'data' / 'M_musculus_T_Cell'
MODEL_DIR = REPO_ROOT / 'models' / 'atomistic' / 'M_musculus_T_Cell'

RAW_MID = DATA_DIR / 'measured MID.xlsx'
RAW_EXT = DATA_DIR / 'MM_TCELL_exterinal_measurements.csv'
RAW_MFA = DATA_DIR / 'MFA_Fluxes.csv'

# Raw tracer sheet -> the display name the config's labeling.tracers[].name must match.
# read_xlsx_mid_data uses the sheet name verbatim as 'Tracer Name'.
TRACERS = {
    'UGlc':  'U-13C glucose',
    'UGln':  'U-13C glutamine',
    '12Glc': '1,2-13C glucose',
}

# Activated and naive are two conditions of one organism, so they get one condition
# letter each and a single growth condition, exactly like I_orientalis_batch.xlsx's B1.
STATES = {'activated': 'A', 'naive': 'N'}

# Measurement name -> exchange reaction.  The source mixes casing (Ala__L vs arg__L)
# and uses a bracketed compartment for oxygen; both are normalised here.
MEASUREMENT_TO_EXCHANGE = {
    'Ala__L': 'EX_ala__L_e', 'arg__L': 'EX_arg__L_e', 'asp__L': 'EX_asp__L_e',
    'gln__L': 'EX_gln__L_e', 'glu__L': 'EX_glu__L_e', 'gly': 'EX_gly_e',
    'ile__L': 'EX_ile__L_e', 'leu__L': 'EX_leu__L_e', 'lys__L': 'EX_lys__L_e',
    'met__L': 'EX_met__L_e', 'phe__L': 'EX_phe__L_e', 'pro__L': 'EX_pro__L_e',
    'ser__L': 'EX_ser__L_e', 'thr__L': 'EX_thr__L_e', 'val__L': 'EX_val__L_e',
    'glc__D': 'EX_glc__D_e', 'lac__L': 'EX_lac__L_e',
}
# Total oxygen uptake is the mitochondrial plus the cytosolic term.  o2 [c] is NOT
# mapped to O2USE1_c: the published fit puts that reaction at 0.51 against a measured
# 16.43 +/- 2.45, i.e. the original MFA left it unconstrained.
OXYGEN_TERMS = ('o2 [m]', 'o2 [c]')

# Biomass synthesis rate -> component reaction of the hierarchical model.
BIOMASS_TO_COMPONENT = {
    'protein.synthesis': 'PROTEIN_c',
    'RNA.synthesis':     'RNA_c',
    'DNA.synthesis':     'DNA_c',
    'g3p.synthesis':     'LIPID_c',
}

UMOL_TO_NMOL = 1000.0

# Measurements whose magnitude does not clear their own error bar carry no
# information, and their sign is noise.  Naive glucose is -0.004287 +- 0.006163,
# which reads as glucose secretion — but T cells lack glucose-6-phosphatase, the
# model forbids it, and the published fit resolves it to uptake.  Blanking the value
# keeps the honest statement (unmeasured) instead of feeding the model an
# impossible one; read_xlsx_extflux treats a missing flux as unknown.
BELOW_DETECTION = {('naive', 'glc__D')}


def _ascii(value):
    """Fold the source's non-ASCII text (naive is written with a diaeresis)."""
    if not isinstance(value, str):
        return value
    return unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode().strip()


def curate_mid():
    """Split the MID workbook into one file per state, one sheet per tracer."""
    written = {}
    for state, letter in STATES.items():
        out = DATA_DIR / f'M_musculus_T_Cell_{letter}.xlsx'
        with pd.ExcelWriter(out) as writer:
            for raw_sheet, tracer_name in TRACERS.items():
                df = pd.read_excel(RAW_MID, raw_sheet)
                # Replicate columns are the state letter plus a digit.  The digits
                # differ per sheet (A1-A3, A4-A6, A5-A7) because each tracer is a
                # separate culture, so they are renumbered positionally.
                cols = [c for c in df.columns if re.fullmatch(rf'{letter}\d+', str(c))]
                if len(cols) != 3:
                    raise ValueError(f"{raw_sheet}/{state}: expected 3 replicates, got {cols}")
                keep = df[['Compound', 'C_Label'] + cols].copy()
                keep.columns = ['Compound', 'C_Label'] + [
                    f'{letter}1_Rep{i}' for i in range(1, 4)
                ]
                keep['Compound'] = keep['Compound'].map(_ascii)
                keep.to_excel(writer, sheet_name=tracer_name, index=False)
        written[letter] = out
        print(f"  {out.name}: {len(TRACERS)} tracer sheets, 3 replicates")
    return written


def _read_measurements():
    """Long measurement table with the section column forward-filled and text folded."""
    df = pd.read_csv(RAW_EXT, sep='\t')
    df = df.rename(columns={df.columns[0]: 'section'})
    df['section'] = df['section'].ffill()
    for col in ('section', 'state', 'measurement', 'unit'):
        df[col] = df[col].map(_ascii)
    return df


def _biomass_fractions():
    """Component mass fractions of the hierarchical model, i.e. its flux units."""
    from cobra.io import load_json_model
    model = load_json_model(str(MODEL_DIR / 'mmTcell_annotated.json'))
    mets = pd.read_excel(MODEL_DIR / 'mmTcell_GEM_completeMFA.xlsx',
                         sheet_name='Metabolites', usecols=['metid', 'metformula'])
    formulas = dict(zip(mets['metid'].astype(str), mets['metformula']))
    return derive_biomass_fractions(model, formulas)


def curate_extflux():
    """Build the Constraints sheet: one row per state, model reaction IDs as columns."""
    meas = _read_measurements()
    fractions = _biomass_fractions()

    media = meas[meas['section'] == 'media exchange']
    biomass = meas[meas['section'] == 'biomass']

    # The published MFA sets BIOMASS to 1.0 for the activated state, and the GEM's
    # BIOMASS coefficients are that state's synthesis fluxes.  So the activated
    # component fluxes are exactly `fraction`, and naive scales by the measured ratio.
    activated = biomass[biomass['state'] == 'activated'].set_index('measurement')

    rows = {}
    for state, letter in STATES.items():
        row, err = {}, {}

        m = media[media['state'] == state].set_index('measurement')
        for name, reaction in MEASUREMENT_TO_EXCHANGE.items():
            below = (state, name) in BELOW_DETECTION
            row[reaction] = float('nan') if below else -UMOL_TO_NMOL * m.loc[name, 'mean']
            err[reaction] = UMOL_TO_NMOL * m.loc[name, 'error']
            if below:
                print(f"    {state} {reaction}: below detection, left blank")
        row['EX_o2_e'] = -UMOL_TO_NMOL * sum(m.loc[t, 'mean'] for t in OXYGEN_TERMS)
        err['EX_o2_e'] = UMOL_TO_NMOL * (
            sum(m.loc[t, 'error'] ** 2 for t in OXYGEN_TERMS) ** 0.5
        )

        b = biomass[biomass['state'] == state].set_index('measurement')
        for name, component in BIOMASS_TO_COMPONENT.items():
            scale = b.loc[name, 'mean'] / activated.loc[name, 'mean']
            row[component] = fractions[component] * scale
            rel = (b.loc[name, 'error'] / b.loc[name, 'mean']
                   if b.loc[name, 'mean'] else 0.0)
            err[component] = abs(row[component]) * rel

        rows[f'{letter}1'] = (row, err)

    columns = list(MEASUREMENT_TO_EXCHANGE.values()) + ['EX_o2_e'] + [
        c for c in BIOMASS_COMPONENT_IDS if c in BIOMASS_TO_COMPONENT.values()
    ]
    records = []
    for condition, (row, err) in rows.items():
        record = {'Condition': condition}
        for col in columns:
            record[col] = row[col]
            record[f'{col}_err'] = err[col]
        records.append(record)

    out = DATA_DIR / 'M_musculus_T_Cell_ExtFlux.xlsx'
    pd.DataFrame(records).to_excel(out, sheet_name='Constraints', index=False)
    print(f"  {out.name}: {len(records)} conditions x {len(columns)} reactions")
    return out


# Exchanges whose raw direction is uptake-positive, which conversion flips.  The
# published flux table is in the raw direction, so these must be negated.
FLIPPED_EXCHANGES = {
    'EX_glc__D_e', 'EX_o2_e', 'EX_pi_e', 'EX_ala__L_e', 'EX_gln__L_e', 'EX_gly_e',
    'EX_pro__L_e', 'EX_ser__L_e', 'EX_met__L_e', 'EX_val__L_e', 'EX_phe__L_e',
    'EX_ile__L_e', 'EX_thr__L_e', 'EX_leu__L_e', 'EX_arg__L_e', 'EX_lys__L_e',
    'EX_asn__L_e', 'EX_cys__L_e', 'EX_his__L_e', 'EX_trp__L_e', 'EX_tyr__L_e',
}

# The paper reports lumped transketolase; the GEM uses the ping-pong half reactions
# sharing a tktc2_c carrier.  Solving the carrier balance gives v_h = v_1 + v_2.
TKT_LUMPED_TO_HALF = {'TKT1_c': 'TKT1h_c', 'TKT2_c': 'TKT2h_c'}

METRICS = ('mfa', 'mfaLB', 'mfaUB', 'pfba')


def curate_reference_flux():
    """Map the paper's flux table onto converted-model IDs and the model's sign convention."""
    raw = pd.read_csv(RAW_MFA, sep='\t')
    for col in ('id', 'name', 'reaction', 'subsystem', 'gpr'):
        raw[col] = raw[col].map(_ascii)

    out = pd.DataFrame({
        'id': raw['id'], 'name': raw['name'], 'reaction': raw['reaction'],
        'gpr': raw['gpr'], 'subsystem': raw['subsystem'],
    })
    for letter, suffix in (('A', 'a'), ('N', 'n')):
        for metric in METRICS:
            out[f'{letter}_{metric}'] = pd.to_numeric(
                raw[f'{metric}.{suffix}'], errors='coerce')

    out['id'] = out['id'].replace(TKT_LUMPED_TO_HALF)

    # TKThalf_c carries the sum of the two half reactions.  Its bounds are left blank:
    # the summands are correlated, so adding their intervals would fabricate one.
    half = out[out['id'].isin(TKT_LUMPED_TO_HALF.values())]
    tkt = {'id': 'TKThalf_c', 'name': 'First half reaction of transketolase',
           'reaction': 'xu5p__D_c <=> tktc2_c + g3p_c', 'gpr': '',
           'subsystem': 'Pentose phosphate pathway'}
    for letter in ('A', 'N'):
        for metric in ('mfa', 'pfba'):
            tkt[f'{letter}_{metric}'] = half[f'{letter}_{metric}'].sum()
    out = pd.concat([out, pd.DataFrame([tkt])], ignore_index=True)

    # BIOMASS is one reaction in the model but two rows here, one per state.
    biomass = out[out['id'].isin(('BIOMASS_activated', 'BIOMASS_naive'))].set_index('id')
    merged = {'id': 'BIOMASS', 'name': 'Biomass reaction', 'reaction': '', 'gpr': '',
              'subsystem': 'Pseudoreaction'}
    for letter, source in (('A', 'BIOMASS_activated'), ('N', 'BIOMASS_naive')):
        for metric in METRICS:
            merged[f'{letter}_{metric}'] = biomass.loc[source, f'{letter}_{metric}']
    out = out[~out['id'].isin(('BIOMASS_activated', 'BIOMASS_naive'))]
    out = pd.concat([out, pd.DataFrame([merged])], ignore_index=True)

    # The published table predates the hierarchical split, so it has no component
    # rows. Activated is `fraction` by construction (its BIOMASS is fit to 1.0 and
    # the GEM's coefficients are that state's fluxes); naive scales by the measured
    # ratio of synthesis rates.
    extflux = pd.read_excel(DATA_DIR / 'M_musculus_T_Cell_ExtFlux.xlsx',
                            sheet_name='Constraints').set_index('Condition')
    components = []
    for component in BIOMASS_TO_COMPONENT.values():
        row = {'id': component, 'name': f'{component} synthesis', 'reaction': '',
               'gpr': '', 'subsystem': 'Pseudoreaction'}
        for letter, condition in (('A', 'A1'), ('N', 'N1')):
            value = extflux.loc[condition, component]
            error = extflux.loc[condition, f'{component}_err']
            row[f'{letter}_mfa'] = value
            row[f'{letter}_mfaLB'] = value - error
            row[f'{letter}_mfaUB'] = value + error
            row[f'{letter}_pfba'] = value
        components.append(row)
    out = pd.concat([out, pd.DataFrame(components)], ignore_index=True)

    flip = out['id'].isin(FLIPPED_EXCHANGES)
    for letter in ('A', 'N'):
        for metric in METRICS:
            out.loc[flip, f'{letter}_{metric}'] *= -1
        # Negating turns the lower bound into the upper one.
        lb, ub = f'{letter}_mfaLB', f'{letter}_mfaUB'
        out.loc[flip, [lb, ub]] = out.loc[flip, [ub, lb]].values

    path = DATA_DIR / 'MFA_flux_data_MM.xlsx'
    out.to_excel(path, index=False)
    print(f"  {path.name}: {len(out)} reactions, {flip.sum()} exchanges negated")
    return path


if __name__ == '__main__':
    print("MID:")
    curate_mid()
    print("External fluxes:")
    curate_extflux()
    print("Reference fluxes:")
    curate_reference_flux()
    print("\nDone.")
