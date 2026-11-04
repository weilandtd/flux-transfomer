"""Build the flat ``*_annotated.json`` models from their GEM + AMM workbooks.

Paths resolve against the repository root, not the working directory, since
``convert_mfa_maranas`` passes them straight to ``pd.read_excel``.

``measurement_pools`` is declared per organism here, and that is the only place it is
declared: whether an LC/MS extract sees a metabolite's compartments as one measurable
species is a statement about the organism model, so the models ship carrying it.

Two compartments are never pool members. ``_e`` is the medium, which is not in a cell
extract -- and pooling it would mix the tracer inlet (``glc__D_e``) into a measured pool.
``_p`` is not a real compartment either: the outer membrane is porous, so the periplasm
equilibrates with the medium rather than being its own labeled space. The E. coli models
therefore get no pools at all and their mapping CSVs name intracellular ``_c`` ids only.
That is also why the Pennycress plastid is ``_h`` (curate_pennycress_workbooks.py).

Usage:
    python models/atomistic/convert_maranas_models.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / 'src'))

from sigmaflux.core.utils import convert_mfa_maranas  # noqa: E402

_MODELS = pathlib.Path(__file__).parent


def _print_model_summary(label: str, model) -> None:
    """Print reaction/metabolite counts and BIOMASS presence for a converted model.

    Args:
        label: organism name to print as the summary heading.
        model: the converted COBRA model.
    """
    rxn_ids = {r.id for r in model.reactions}
    annotated = [r for r in model.reactions if 'atom_mapping' in r.annotation]
    print(f"\n{label}")
    print(f"  Reactions: {len(model.reactions)} total, {len(annotated)} annotated")
    print(f"  Metabolites: {len(model.metabolites)}")
    # These models carry a single aggregate BIOMASS; build_hierarchical_models.py
    # splits it into the five component reactions afterwards.
    print(f"  BIOMASS present (expected True): {'BIOMASS' in rxn_ids}")


def _annotate_compartments(model, names, out):
    """Set each metabolite's compartment from its id suffix and re-save *model*.

    Args:
        model: the converted COBRA model, modified in place.
        names: ``{suffix letter: compartment name}``; other ids stay compartment-less.
        out: output JSON path, relative to this file's directory.
    """
    import cobra

    from sigmaflux.core.measurement_pools import _compartment

    for met in model.metabolites:
        suffix = _compartment(met.id)
        met.compartment = suffix if suffix in names else ''
    model.compartments = names
    cobra.io.save_json_model(model, str(_MODELS / out))


def _check_mappings(by_organism):
    """Report every mapping row its model cannot honour; raise if any organism has one.

    Args:
        by_organism: ``{label: (model, [mapping csv paths relative to this file])}``.
    """
    import csv

    from sigmaflux.core.measurement_pools import check_mapping_csv

    failed = {}
    for label, (model, paths) in by_organism.items():
        for rel in paths:
            path = _MODELS / rel
            if not path.exists():
                continue
            with path.open() as fh:
                rows = [(r['experimental_id'], (r.get('model_id') or '').strip())
                        for r in csv.DictReader(fh) if (r.get('model_id') or '').strip()]
            problems = check_mapping_csv(model, rows)
            if problems:
                failed[f"{label}/{path.name}"] = problems
    if failed:
        detail = "\n".join(f"  {where}:\n" + "\n".join(f"    {p}" for p in problems[:10])
                           for where, problems in failed.items())
        raise ValueError(f"{len(failed)} mapping file(s) do not match their model:\n{detail}")
    print("\nMapping files: every model_id names a token its model measures")


def convert(gem, amm, out, **kwargs):
    """Convert one organism's GEM/AMM workbook pair to the flat annotated JSON model.

    Args:
        gem: GEM workbook path, relative to this file's directory.
        amm: AMM workbook path, relative to this file's directory.
        out: output JSON path, relative to this file's directory.
        kwargs: forwarded to convert_mfa_maranas (e.g. drop_subsystems, biomass_id).
    """
    return convert_mfa_maranas(str(_MODELS / gem), str(_MODELS / amm),
                               str(_MODELS / out), **kwargs)


# I_orientalis
model_lo = convert('I_orientalis/imIsor_GEM.xlsx', 'I_orientalis/imIsor_AMM.xlsx',
                   'I_orientalis/imIsor_annotated.json',
                   measurement_pools=True)
_print_model_summary("I_orientalis", model_lo)

# S_cerevisiae
model_sc = convert('S_cerevisiae/scGEM.xlsx', 'S_cerevisiae/scAMM.xlsx',
                   'S_cerevisiae/sc_annotated.json',
                   measurement_pools=True)
_print_model_summary("S_cerevisiae", model_sc)

# R_toruloides — completeMFA workbook, so the explicit dilution sub-network is dropped
model_rt = convert('R_toruloides/imRt_GEM.xlsx', 'R_toruloides/imRt_AMM.xlsx',
                   'R_toruloides/imRt_annotated.json',
                   drop_subsystems=('dilution',),
                   measurement_pools=True)
_print_model_summary("R_toruloides", model_rt)

# M_musculus_T_Cell — completeMFA workbook; biomass is named BIOMASS_c here
model_mm = convert('M_musculus_T_Cell/mmTcell_GEM_completeMFA.xlsx',
                   'M_musculus_T_Cell/mmTcell_AMM_completeMFA.xlsx',
                   'M_musculus_T_Cell/mmTcell_annotated.json',
                   drop_subsystems=('dilution',),
                   biomass_id='BIOMASS_c',
                   measurement_pools=True)
_print_model_summary("M_musculus_T_Cell", model_mm)

# T_arvense_L — Pennycress embryo FTBL model; the workbooks are canonicalised by
# curate_pennycress_workbooks.py first. No BIOMASS: the FTBL drains each measured
# product separately, so there is no hierarchical step either. No pools: every GC/MS
# ion is read from a compartment-less *_bm pool, so a whole-cell pool would only add
# mixing fractions no measurement can see.
model_ta = convert('T_arvense_L/taGEM.xlsx', 'T_arvense_L/taAMM.xlsx',
                   'T_arvense_L/ta_annotated.json',
                   measurement_pools=False)
_annotate_compartments(model_ta, {'c': 'cytosol', 'h': 'plastid', 'm': 'mitochondrion',
                                  'e': 'extracellular'},
                       'T_arvense_L/ta_annotated.json')
_print_model_summary("T_arvense_L", model_ta)

# E_coli_core / E_coli — the workbooks are normalised in place by
# curate_e_coli_workbooks.py first; run that before this if either xlsx changes.
# Both spell biomass 'biomass_out', which the configs' measured_fluxes and the
# single lumped BIOMASS both refer to as 'BIOMASS'.
model_ec_core = convert('E_coli_core/example_GEM.xlsx', 'E_coli_core/example_AMM.xlsx',
                        'E_coli_core/e_coli_core_annotated.json',
                        biomass_id='biomass_out',
                        measurement_pools=False)
_print_model_summary("E_coli_core", model_ec_core)

model_ec = convert('E_coli/iAF1260_GEM.xlsx', 'E_coli/iAF1260_AMM.xlsx',
                   'E_coli/iAF1260_annotated.json',
                   biomass_id='biomass_out',
                   measurement_pools=False)
_print_model_summary("E_coli", model_ec)

# The mapping CSVs are curated by hand, so they are only checked here, never rewritten.
# A row naming a compartment the model now pools is stale, and would otherwise go
# unnoticed until feature selection quietly dropped the token.
_check_mappings({
    'I_orientalis': (model_lo, ['I_orientalis/I_orientalis_mapping.csv',
                                'I_orientalis/I_orientalis_experiment_mapping.csv']),
    'S_cerevisiae': (model_sc, ['S_cerevisiae/S_cerevisiae_mapping.csv',
                                'S_cerevisiae/S_cerevisiae_experiment_mapping.csv']),
    'R_toruloides': (model_rt, ['R_toruloides/R_toruloides_mapping.csv']),
    'M_musculus_T_Cell': (model_mm,
                          ['M_musculus_T_Cell/M_musculus_T_Cell_mapping.csv',
                           'M_musculus_T_Cell/M_musculus_T_Cell_experiment_mapping.csv']),
    # E_coli_core_fragment_mapping.csv names fragment pools, so build_gcms_fragment_model.py
    # checks it against the GC/MS model it builds.
    'E_coli_core': (model_ec_core, ['E_coli_core/E_coli_core_mapping.csv']),
    'E_coli': (model_ec, ['E_coli/E_coli_mapping.csv',
                          'E_coli/E_coli_fragment_mapping.csv']),
})
