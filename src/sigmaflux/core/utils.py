import os
import string

import cobra
import warnings

import pandas as pd

import numpy as np


from scipy.optimize import linprog

from sigmaflux.expansion.utils import atom_pattern_regex
import re

import time
import logging
import torch

logger = logging.getLogger(__name__)

def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Function {func.__name__} took {elapsed_time:.4f} seconds to complete.")
        return result
    return wrapper



_AMINO_ACID_IDS: frozenset = frozenset({
    'ala__L_c', 'arg__L_c', 'asn__L_c', 'asp__L_c', 'cys__L_c',
    'gln__L_c', 'glu__L_c', 'gly_c',    'his__L_c', 'ile__L_c',
    'leu__L_c', 'lys__L_c', 'met__L_c', 'phe__L_c', 'pro__L_c',
    'ser__L_c', 'thr__L_c', 'trp__L_c', 'tyr__L_c', 'val__L_c',
})
_RNA_NTP_IDS: frozenset = frozenset({'ctp_c', 'gtp_c', 'utp_c'})  # atp_c handled separately
_DNA_NTP_IDS: frozenset = frozenset({'datp_c', 'dctp_c', 'dgtp_c', 'dttp_c'})
_CARB_IDS: frozenset = frozenset({
    '13BDglucan_c', '16BDglucan_c', 'chtn_c', 'glycogen_c', 'mannan_c', 'tre_c',
})
_SHARED_MET_IDS: frozenset = frozenset({
    'atp_c', 'adp_c', 'amp_c', 'pi_c', 'ppi_c', 'h2o_c', 'h_c',
    'ca2_c', 'cu2_c', 'fe2_c', 'k_c', 'mg2_c', 'mn2_c', 'zn2_c', 'so4_c',
})
_BIOMASS_ENERGY_FRACS: dict = {
    'PROTEIN_c': 0.50, 'CARB_c': 0.20, 'LIPID_c': 0.15, 'RNA_c': 0.12, 'DNA_c': 0.03,
}
BIOMASS_COMPONENT_IDS: tuple = ('PROTEIN_c', 'RNA_c', 'DNA_c', 'LIPID_c', 'CARB_c')
_COMPONENT_TO_BM: dict = {
    'PROTEIN_c': 'bm_protein_c', 'RNA_c': 'bm_rna_c', 'DNA_c': 'bm_dna_c',
    'LIPID_c': 'bm_lipid_c', 'CARB_c': 'bm_carb_c',
}
#: Macromolecule mass fractions of the two published yeast models, at the precision
#: the released models carry.
#:
#: These are measured compositions from the source papers, not something to
#: re-derive: ``derive_biomass_fractions`` reads them back out of the BIOMASS
#: stoichiometry, which is a different quantity that happens to be close. Passing
#: these to ``split_biomass_hierarchical`` is what lets the two models be rebuilt
#: from their workbooks -- picking up every atom-map and Terminal-sheet fix -- while
#: their composition stays exactly what was published and trained on.
PUBLISHED_BIOMASS_FRACTIONS: dict = {
    'S_cerevisiae': {'PROTEIN_c': 0.4888, 'RNA_c': 0.0874, 'DNA_c': 0.00354,
                     'LIPID_c': 0.02529, 'CARB_c': 0.3949},
    'I_orientalis': {'PROTEIN_c': 0.4722603271367565, 'RNA_c': 0.11097742313505278,
                     'DNA_c': 0.008573098060797861, 'LIPID_c': 0.04562162376171028,
                     'CARB_c': 0.3625675279056826},
}
# Maranas biomass formulas use placeholder tokens for the variable acyl chains and
# the sphingoid backbone (e.g. tag_c = C3H5O3Acyl3), so those masses are nominal:
# a palmitoyl chain and a C18 phytoceramide core.
_PLACEHOLDER_MASSES: dict = {'Acyl': 239.4, 'Cer': 537.9}
_ATOMIC_MASSES: dict = {
    'C': 12.011, 'H': 1.008, 'O': 15.999, 'N': 14.007, 'P': 30.974, 'S': 32.06,
}


def _build_atom_labels():
    r"""One printable, delimiter-free character per tracked atom (~5700 of them via alphanumeric Unicode, since ASCII alnum's 62 is too few for some reactions)."""
    base = string.ascii_lowercase + string.ascii_uppercase + string.digits
    extra = [chr(c) for c in range(0xC0, 0x2000)
             if chr(c).isalnum() and chr(c).isprintable() and chr(c) not in base]
    return base + ''.join(extra)


ATOM_LABELS = _build_atom_labels()


def _biomass_component(met_id: str) -> str | None:
    """Which macromolecule a BIOMASS building block belongs to (None = shared cofactor)."""
    if met_id in _SHARED_MET_IDS:
        return None
    if met_id in _AMINO_ACID_IDS:
        return 'PROTEIN_c'
    if met_id in _RNA_NTP_IDS:
        return 'RNA_c'
    if met_id in _DNA_NTP_IDS:
        return 'DNA_c'
    if met_id in _CARB_IDS:
        return 'CARB_c'
    # Everything unrecognised is called lipid. The id sets above are BiGG-spelled and
    # organism-agnostic, so a metabolite this model spells differently lands here on a
    # technicality rather than on its chemistry -- which is also why the T cell appears
    # to have no carbohydrate. Worth saying out loud, since the assignment decides both
    # the mass fractions and which component reaction carries the block.
    warnings.warn(f"{met_id} matches no component id set; assigned to LIPID_c",
                  RuntimeWarning, stacklevel=3)
    return 'LIPID_c'


def _formula_mass(formula) -> float:
    """Molecular weight of a Maranas metabolite formula, or 0.0 if unparseable."""
    if not isinstance(formula, str):
        return 0.0
    mass, remainder = 0.0, formula
    for token, token_mass in _PLACEHOLDER_MASSES.items():
        # A trailing digit multiplies the placeholder, e.g. Acyl3 in C3H5O3Acyl3
        for match in re.finditer(re.escape(token) + r'(\d*)', remainder):
            mass += token_mass * (int(match.group(1)) if match.group(1) else 1)
        remainder = re.sub(re.escape(token) + r'\d*', '', remainder)
    for element, count in re.findall(r'([A-Z][a-z]?)(\d*)', remainder):
        if not element:
            continue
        if element not in _ATOMIC_MASSES:
            return 0.0
        mass += _ATOMIC_MASSES[element] * (int(count) if count else 1)
    return mass


def derive_biomass_fractions(model, formulas: dict) -> dict:
    """Macromolecule mass fractions implied by the aggregate BIOMASS stoichiometry, normalised to sum to 1.

    Approximate (placeholder lipid tokens, per-monomer polysaccharides); prefer
    measured values when available.

    Args:
        model: cobra model with the aggregate BIOMASS reaction.
        formulas: metabolite id -> chemical formula.
    """
    biomass = model.reactions.get_by_id('BIOMASS')
    masses = {cid: 0.0 for cid in BIOMASS_COMPONENT_IDS}
    for met, coeff in biomass.metabolites.items():
        component = _biomass_component(met.id) if coeff < 0 else None
        if component is None:
            continue
        masses[component] += -coeff * _formula_mass(formulas.get(met.id))

    total = sum(masses.values())
    if total <= 0.0:
        raise ValueError("BIOMASS carries no building blocks with a parseable formula")
    return {cid: mass / total for cid, mass in masses.items()}


def split_biomass_hierarchical(model, fractions: dict) -> None:
    """Split the aggregate BIOMASS into 5 macromolecule component reactions (PROTEIN_c, RNA_c, DNA_c, LIPID_c, CARB_c) exposed as separately measurable fluxes.

    Each component consumes its building blocks at `coefficient / fraction` and feeds
    BIOMASS through one `bm_*_c` pseudo-metabolite, leaving BIOMASS's composition
    unchanged. Also calls `redistribute_biomass_energy`, without which the components
    would become energy-decoupled once BIOMASS is dropped during sampling.

    Args:
        model: cobra model whose BIOMASS reaction is split, modified in place.
        fractions: macromolecule component id -> mass fraction.
    """
    import cobra

    rxn_ids = {r.id for r in model.reactions}
    if 'BIOMASS' not in rxn_ids:
        raise ValueError("model has no BIOMASS reaction to split")
    already_split = [c for c in BIOMASS_COMPONENT_IDS if c in rxn_ids]
    if already_split:
        raise ValueError(f"model already has component reactions {already_split}")

    biomass = model.reactions.get_by_id('BIOMASS')
    building_blocks: dict = {cid: {} for cid in BIOMASS_COMPONENT_IDS}
    for met, coeff in biomass.metabolites.items():
        component = _biomass_component(met.id) if coeff < 0 else None
        if component is not None:
            building_blocks[component][met] = coeff

    for component_id, blocks in building_blocks.items():
        # A model need not carry all five macromolecules — the M. musculus T-cell
        # biomass has no carbohydrate.  Creating CARB_c anyway would leave a
        # bm_carb_c that BIOMASS consumes at coefficient zero, i.e. a free-floating
        # generator, so skip the component entirely.
        if not blocks:
            print(f"split_biomass_hierarchical: no building blocks for {component_id}, skipping")
            continue
        fraction = fractions[component_id]
        bm_met = cobra.Metabolite(id=_COMPONENT_TO_BM[component_id],
                                  name=f"{component_id} biomass precursor", compartment='c')
        model.add_metabolites([bm_met])
        component = cobra.Reaction(component_id, lower_bound=0.0, upper_bound=1000.0)
        model.add_reactions([component])
        component.add_metabolites({met: coeff / fraction for met, coeff in blocks.items()})
        component.add_metabolites({bm_met: 1.0})
        # BIOMASS consumes the pseudo-metabolite in place of the building blocks
        biomass.add_metabolites({met: -coeff for met, coeff in blocks.items()})
        biomass.add_metabolites({bm_met: -fraction})

    redistribute_biomass_energy(model)


def redistribute_biomass_energy(model) -> None:
    """Move shared cofactor stoichiometry (ATP, H2O, ions) from BIOMASS into the 5 component reactions, proportionally to _BIOMASS_ENERGY_FRACS, then strip BIOMASS to only bm_* pseudo-metabolites.

    Needed because component reactions otherwise carry only building blocks and become
    energy-decoupled drains once BIOMASS is dropped during sampling. Missing components
    have their energy fraction renormalised over the ones present.
    """
    rxn_ids = {r.id for r in model.reactions}
    if 'BIOMASS' not in rxn_ids:
        print("redistribute_biomass_energy: BIOMASS not found, skipping")
        return
    present = [c for c in BIOMASS_COMPONENT_IDS if c in rxn_ids]
    if not present:
        print("redistribute_biomass_energy: no component reactions, skipping")
        return

    total_frac = sum(_BIOMASS_ENERGY_FRACS[c] for c in present)
    energy_fracs = {c: _BIOMASS_ENERGY_FRACS[c] / total_frac for c in present}
    if len(present) < len(BIOMASS_COMPONENT_IDS):
        missing = [c for c in BIOMASS_COMPONENT_IDS if c not in rxn_ids]
        print(f"redistribute_biomass_energy: absent components {missing}, "
              f"renormalised energy fractions over {present}")

    biomass = model.reactions.get_by_id('BIOMASS')
    stoich = {m.id: c for m, c in biomass.metabolites.items()}

    # Identify bm_* pseudo-metabolite stoichiometry (consumed by BIOMASS)
    bm_fracs = {mid: abs(c) for mid, c in stoich.items() if mid.startswith('bm_')}

    # Shared energy cofactors to redistribute (everything except bm_* pseudo-metabolites)
    energy_stoich = {mid: c for mid, c in stoich.items() if not mid.startswith('bm_')}

    # ATP is distributed by energy fraction like every other shared cofactor. An
    # RNA-specific share was once carved out using UTP as its proxy, but this runs
    # inside split_biomass_hierarchical, which has already moved utp_c out of BIOMASS,
    # so the proxy was always 0.0 and the carve-out never happened. The shipped models
    # were built without it; reinstating it would be a change to their stoichiometry,
    # not a bug fix, so it is removed rather than repaired.
    for comp_id, frac in energy_fracs.items():
        bm_id = _COMPONENT_TO_BM[comp_id]
        bm_frac = bm_fracs.get(bm_id, 0.0)
        if bm_frac == 0.0:
            continue
        rxn = model.reactions.get_by_id(comp_id)
        # Every shared metabolite goes out by energy fraction, per unit component flux.
        additions = {mid: coeff * frac / bm_frac for mid, coeff in energy_stoich.items()}

        met_map = {model.metabolites.get_by_id(mid): coeff
                   for mid, coeff in additions.items() if coeff != 0.0}
        rxn.add_metabolites(met_map)

    # Strip all non-bm_* metabolites from BIOMASS
    remove_from_biomass = {m: -c for m, c in biomass.metabolites.items()
                           if not m.id.startswith('bm_')}
    biomass.add_metabolites(remove_from_biomass)
    print(f"redistribute_biomass_energy: distributed energy to {present}")


def annotate_model(model, annotation_file,index='id',atom_mapping='atom_mapping'):
    """Load atom_mapping annotations from a CSV and attach them to the model's reactions.

    Args:
        model: cobra model to annotate, modified in place.
        annotation_file: path to a CSV with reaction ids and atom-mapping strings.
        index: annotation_file column used to look up the reaction id.
        atom_mapping: annotation_file column holding the atom-mapping string.
    """

    atom_mapping_annotation = pd.read_csv(annotation_file)
    atom_mapping_annotation.index = atom_mapping_annotation[index]

    for met, row in atom_mapping_annotation.iterrows():
        try: 
            if isinstance(row[atom_mapping], str):
                model.reactions.get_by_id(met).annotation['atom_mapping'] = row[atom_mapping]
            else:
               # print("no atom_mapping", met, row)
                model.reactions.get_by_id(met).annotation['atom_mapping'] = ''
        except Exception:
            pass


def annotate_atom_patterns(model):
    """Give every metabolite an ``atom_pattern`` (a distinct label per tracked atom) sized by its reaction atom mappings, which ``build_targets`` needs to decide which EMUs to simulate."""
    for met in model.metabolites:
        counts = {}
        for rxn in met.reactions:
            amm_str = rxn.annotation.get("atom_mapping", "")
            for match in re.findall(atom_pattern_regex(met.id), amm_str):
                counts.setdefault(len(match), []).append(rxn.id)
        if not counts:
            continue
        # The largest wins, but two reactions disagreeing on a metabolite's carbon
        # count is a defect in one of them: whichever loses is left with an atom map
        # shorter than the pool it feeds, and the surplus carbon goes to a void pool.
        if len(counts) > 1:
            detail = ", ".join(f"{n} in {sorted(set(rxns))[:3]}"
                               for n, rxns in sorted(counts.items()))
            warnings.warn(f"{met.id}: reactions disagree on its carbon count "
                          f"({detail}); taking the largest",
                          RuntimeWarning, stacklevel=2)
        met.annotation["atom_pattern"] = ATOM_LABELS[:max(counts)]
    return model


def read_terminal_sheet(path_to_atom_map_xls):
    """``(terminal pool ids, terminal reaction ids)`` declared by the AMM workbook's Terminal sheet, marking where the model stops following the label. The two columns are independently ragged, so each is read on its own."""
    try:
        term = pd.read_excel(path_to_atom_map_xls, sheet_name='Terminal')
    except ValueError:
        # A workbook without the sheet declares no terminals, which is a legitimate
        # state -- but it is indistinguishable from one that should have declared some
        # and does not. R. toruloides sat in exactly that gap: hierarchical, carrying
        # the same lump chemistry as the other two yeasts, and silently terminal-free.
        print(f"    {os.path.basename(str(path_to_atom_map_xls))}: no Terminal sheet, "
              f"so no pool or reaction is declared terminal")
        return [], []

    if 'Metab_name' not in term.columns and 'Rxn_name' not in term.columns:
        raise ValueError(f"{path_to_atom_map_xls}: the Terminal sheet needs a "
                         f"'Metab_name' or 'Rxn_name' column, found "
                         f"{list(term.columns)}")
    # A terminal *reaction* is the escape hatch for a lump whose atom map cannot carry
    # its own carbon: lumpIPC_c condenses C16/C24/C26 acyl-CoA into an 8-carbon
    # ipcbiom_c, stranding 164 of its 212 carbons. Declaring the reaction terminal says
    # that carbon leaves the labelled network -- the same statement a terminal pool
    # makes -- without inventing chemistry to say it.
    return tuple([str(n).strip() for n in term[col].dropna() if str(n).strip()]
                 if col in term.columns else []
                 for col in ('Metab_name', 'Rxn_name'))


def apply_terminal_annotations(model, path_to_atom_map_xls, created_later=frozenset()):
    """Mark the pools and reactions the AMM's Terminal sheet declares terminal.

    Applied both before and after :func:`split_biomass_hierarchical`, since the ``bm_*``
    pools it may name don't exist until that split creates them; *created_later* lists
    those pools so the first pass doesn't warn about them.

    Args:
        model: cobra model to annotate, modified in place.
        path_to_atom_map_xls: AMM workbook containing the Terminal sheet.
        created_later: pool/reaction ids not yet in the model, to skip silently.
    """
    pool_ids, rxn_ids = read_terminal_sheet(path_to_atom_map_xls)
    for idents, lookup, what in ((pool_ids, model.metabolites, 'pool'),
                                 (rxn_ids, model.reactions, 'reaction')):
        for ident in idents:
            try:
                lookup.get_by_id(ident).annotation['terminal'] = True
            except KeyError:
                if ident in created_later:
                    continue
                # Naming something the model does not contain means the declaration
                # silently does nothing, the failure this sheet exists to avoid.
                warnings.warn(f"Terminal sheet names {what} {ident}, which is not "
                              f"in the model", RuntimeWarning, stacklevel=2)
    return model


def convert_mfa_maranas(path_to_cobra, path_to_atom_map_xls, path_to_output=None, atom_type='C',
                        drop_subsystems=(), biomass_id=None, measurement_pools=False):
    """Load a Maranas-lab GEM xlsx + AMM xlsx and return a COBRApy model annotated with SigmaFlux atom_mapping strings, ready for MFAModel().

    Converts the Maranas AMM format (one row per metabolite per reaction, atoms as
    integers) to SigmaFlux string format, e.g.
    ``"accoa_c (ab) + accoa_c (cd) -> aacoa_c (abcd) + coa_c"``.

    Args:
        path_to_cobra: Maranas-lab GEM xlsx.
        path_to_atom_map_xls: Maranas-lab AMM xlsx.
        path_to_output: path to save the annotated model as JSON, if given.
        atom_type: AMM startNodeSymbol to filter atom transitions on.
        drop_subsystems: subsystems to exclude (dropped before parsing, since some
            carry zero-coefficient reactants COBRA would silently discard).
        biomass_id: non-canonical biomass reaction id to rename to 'BIOMASS' on load.
        measurement_pools: declare a whole-cell pool for every metabolite the model holds
            in more than one compartment. Whether an LC/MS extract sees the compartments
            as one species is a property of the organism model, so it is stated here
            rather than assembled per run; see
            :mod:`sigmaflux.core.measurement_pools`.
    """
    import cobra

    # Load GEM xlsx into a cobra model
    gem = pd.read_excel(path_to_cobra, sheet_name=None)
    reactions, dropped_formulas = gem['Reactions'], []
    if drop_subsystems:
        dropped = reactions['subsystem'].isin(drop_subsystems)
        dropped_formulas = reactions.loc[dropped, 'Reaction Formula'].astype(str).tolist()
        reactions = reactions[~dropped]
        print(f"Dropped {len(dropped_formulas)} reactions in subsystems {list(drop_subsystems)}")

    model = cobra.Model()
    for _, row in gem['Metabolites'].iterrows():
        name = '' if pd.isna(row['metname']) else str(row['metname'])
        model.add_metabolites([cobra.Metabolite(id=str(row['metid']), name=name)])
    for _, row in reactions.iterrows():
        rxn = cobra.Reaction(str(row['Rxn ID']), lower_bound=row['Lower bound'], upper_bound=row['Upper bound'])
        model.add_reactions([rxn])
        rxn.build_reaction_from_string(str(row['Reaction Formula']))

    _remove_metabolites_orphaned_by(model, dropped_formulas)

    if biomass_id is not None:
        model.reactions.get_by_id(biomass_id).id = 'BIOMASS'
        model.repair()
        print(f"Renamed biomass reaction {biomass_id} -> BIOMASS")

    # Annotate with atom mappings
    amm = pd.read_excel(path_to_atom_map_xls, sheet_name='Atom Transition')
    amm = amm[amm['startNodeSymbol'] == atom_type]
    amm = amm[amm['rxnAbbreviation'].isin({r.id for r in model.reactions})]

    # An AMM row is (reaction, metabolite, side, atoms); repeating one verbatim
    # states the same transition twice and silently doubles that metabolite's
    # stoichiometry in the mapping string. It is not inert: the duplicated educts
    # make the origin set larger than its target, and EMUExpansion only adds an
    # edge when the sizes match (expansion/emu_expansion.py), so the transition is
    # dropped entirely rather than doubled. GLYCL_m in the M. musculus T-cell AMM
    # is exactly this -- four rows repeated, and its forward edge went missing.
    duplicated = amm.duplicated(
        subset=['rxnAbbreviation', 'metAbbreviation', 'ReactantProductFlag', 'maps'])
    if duplicated.any():
        offenders = sorted(amm.loc[duplicated, 'rxnAbbreviation'].unique())
        warnings.warn(
            f"{int(duplicated.sum())} duplicate atom-transition row(s) in "
            f"{os.path.basename(str(path_to_atom_map_xls))}, affecting "
            f"{offenders}; keeping one copy of each. A repeated row doubles the "
            f"metabolite in the mapping string, which makes the EMU origin set "
            f"outgrow its target and drops the transition.",
            RuntimeWarning, stacklevel=2)
        amm = amm[~duplicated]

    _chars = ATOM_LABELS

    for rxn_id, grp in amm.groupby('rxnAbbreviation', sort=False):
        try:
            rxn = model.reactions.get_by_id(rxn_id)

            # Map sorted unique integer atom IDs → single chars (a-z, A-Z, 0-9)
            all_ids = sorted({int(x) for maps in grp['maps'] for x in str(maps).split(',')})
            lbl = {aid: _chars[i] for i, aid in enumerate(all_ids)}

            reactants, products = [], []
            for _, row in grp.iterrows():
                ids = [int(x) for x in str(row['maps']).split(',')]
                entry = f"{row['metAbbreviation']} ({''.join(lbl[i] for i in ids)})"
                (reactants if row['ReactantProductFlag'] == 'reactant' else products).append(entry)

            amm_mets = set(grp['metAbbreviation'])
            # A metabolite the AMM names but the reaction does not contain is dropped
            # by both mapping parsers, which iterate over cobra's stoichiometry -- so
            # its carbon vanishes with nothing said. That is a workbook error.
            stray = amm_mets - {m.id for m in rxn.metabolites}
            if stray:
                warnings.warn(f"{rxn_id}: the AMM maps {sorted(stray)}, which the "
                              f"reaction does not contain; that carbon is dropped",
                              RuntimeWarning, stacklevel=2)

            # A metabolite the AMM does not map is named once, unlabelled, however
            # many copies the stoichiometry asks for. That is how an unmapped educt at
            # coefficient 4 contributes one copy's worth of letters and strands the
            # rest, so say when the two disagree.
            for m, coef in rxn.metabolites.items():
                if m.id not in amm_mets and abs(coef) > 1 and m.annotation.get('atom_pattern'):
                    warnings.warn(f"{rxn_id}: {m.id} is unmapped at coefficient "
                                  f"{coef:g} but can only be named once",
                                  RuntimeWarning, stacklevel=2)
            reactants += [m.id for m in rxn.reactants if m.id not in amm_mets]
            products  += [m.id for m in rxn.products  if m.id not in amm_mets]

            rxn.annotation['atom_mapping'] = ' + '.join(reactants) + ' -> ' + ' + '.join(products)
        except KeyError:
            # The AMM names a reaction the GEM does not contain; nothing to annotate.
            pass
        except Exception as exc:
            # Anything else means the workbook described a mapping we failed to encode,
            # and leaving the reaction silently unmapped hides carbon from the fit.
            warnings.warn(f"could not build the atom mapping for {rxn_id}: "
                          f"{type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)

    # Symmetries. A dropped one is invisible in the finished model -- the pool simply
    # never gets a 'symmetries' annotation -- yet it changes every MID downstream of it,
    # so each failure is reported rather than swallowed. ATOM_LABELS rather than
    # ascii_lowercase because a symmetric pool may carry more than 26 atoms.
    try:
        sym = pd.read_excel(path_to_atom_map_xls, sheet_name='Symmetry')
    except ValueError:
        sym = None               # a workbook without the sheet declares no symmetries
    if sym is not None:
        for _, row in sym.iterrows():
            name = str(row['Metab_name']).strip()
            try:
                met = model.metabolites.get_by_id(name)
                default = [int(x) for x in str(row['default mapping']).split()]
                sym_pos = [int(x) for x in str(row['symmetry mapping']).split()]
                p2l = {p: ATOM_LABELS[i] for i, p in enumerate(default)}
                entry = ''.join(p2l[p] for p in default) + '.' + ''.join(p2l[p] for p in sym_pos)
            except Exception as exc:
                warnings.warn(f"Symmetry sheet: dropped {name}: "
                              f"{type(exc).__name__}: {exc}",
                              RuntimeWarning, stacklevel=2)
                continue
            met.annotation.setdefault('symmetries', []).append(entry)

    # The bm_* pools this may name do not exist yet -- split_biomass_hierarchical creates
    # them -- so build_hierarchical_models.py applies the sheet a second time afterwards,
    # and that second pass is where a mistyped bm_* name gets reported. Warning about
    # them here too would bury the genuine misses under five expected ones per model.
    apply_terminal_annotations(model, path_to_atom_map_xls,
                               created_later=frozenset(_COMPONENT_TO_BM.values()))

    annotate_atom_patterns(model)

    _remove_out_metabolites(model)
    _standardize_exchange_directions(model)
    audit_carbon_balance(model)

    # After the atom patterns, since a pool member has to carry one, and after the
    # exchange/out cleanup, since those decide which compartments survive.
    if measurement_pools:
        from .measurement_pools import add_measurement_pools, pool_members
        added = add_measurement_pools(model)
        pools = pool_members(model)
        print(f"Added {len(pools)} whole-cell measurement pools ({len(added)} reactions): "
              f"{', '.join(sorted(pools))}")

    if path_to_output is not None:
        cobra.io.save_json_model(model, str(path_to_output))

    return model


def audit_carbon_balance(model):
    """Report every mapped internal reaction whose atom map loses or invents carbon (an atom map must be a bijection).

    Boundaries are exempt: carbon should leave at an exchange or demand, and a pool
    the AMM declares terminal is one the model has decided to stop tracking.
    """
    def carbon(met):
        if met.annotation.get('terminal'):
            return 0
        return len(met.annotation.get('atom_pattern', ''))

    unbalanced = []
    for rxn in model.reactions:
        if not rxn.annotation.get('atom_mapping') or rxn.annotation.get('terminal'):
            continue
        if rxn.boundary or rxn.id.startswith(('EX_', 'DM_', 'SK_')):
            continue
        gap = sum(carbon(m) * -c for m, c in rxn.metabolites.items() if c < 0) \
            - sum(carbon(m) * c for m, c in rxn.metabolites.items() if c > 0)
        if abs(gap) > 0.5:
            unbalanced.append((rxn.id, gap))

    if unbalanced:
        detail = ', '.join(f"{rid} ({gap:+.0f} C)" for rid, gap in unbalanced[:10])
        warnings.warn(f"{len(unbalanced)} mapped reaction(s) do not conserve carbon: "
                      f"{detail}", RuntimeWarning, stacklevel=2)
    else:
        print("    carbon balance: every mapped internal reaction conserves carbon")
    return unbalanced


def _remove_metabolites_orphaned_by(model, dropped_formulas):
    """Remove metabolites left without reactions by a `drop_subsystems` filter.

    Only metabolites mentioned in the dropped formulas are removed; metabolites that
    already carried no reaction are left alone.

    Args:
        model: cobra model to remove orphaned metabolites from, modified in place.
        dropped_formulas: reaction formula strings removed by the subsystem filter.
    """
    if not dropped_formulas:
        return

    mentioned = set()
    for formula in dropped_formulas:
        mentioned.update(re.findall(r'[A-Za-z0-9_\[\]-]+', formula))

    orphaned = [m for m in model.metabolites if not m.reactions and m.id in mentioned]
    model.remove_metabolites(orphaned)
    print(f"Removed {len(orphaned)} metabolites orphaned by the dropped reactions")


def _remove_out_metabolites(model):
    """Convert Maranas _out boundary metabolites to proper COBRA exchange reactions that simply deplete the _e metabolite (``glc__D_e -> glc__D_out`` becomes ``glc__D_e ->``), updating atom_mapping annotations and dropping orphaned _out metabolites."""
    for reaction in list(model.reactions):
        mets = list(reaction.metabolites.keys())
        out_mets = [m for m in mets if m.id.endswith("_out")]

        # Remove _out metabolites from the reaction stoichiometry
        reaction.subtract_metabolites({m: reaction.metabolites[m] for m in out_mets})

        # An exchange has to be a dead end: it depletes its _e metabolite and nothing
        # produces it. A surviving _out educt makes the exchange a producer instead,
        # so the tracer inlet is a mixing node rather than an EMU network input. The
        # name is stripped whether or not the pool is still in the stoichiometry --
        # a workbook that already rewrote the formula leaves it only in the mapping.
        if "atom_mapping" in reaction.annotation:
            reaction.annotation["atom_mapping"] = _strip_out_from_atom_mapping(
                reaction.annotation["atom_mapping"]
            )

    # Remove metabolites that are now orphaned (no reactions left)
    orphaned = [m for m in model.metabolites if m.id.endswith("_out") and not m.reactions]
    model.remove_metabolites(orphaned)
    print(f"Removed {len(orphaned)} _out metabolites; EX_ reactions now deplete _e metabolites directly")


def _standardize_exchange_directions(model):
    """Normalise every EX_ reaction to the depletion direction (met_e coeff -1), so positive flux is secretion and negative flux is uptake; Maranas models mix both conventions."""
    flipped = []
    for reaction in model.reactions:
        if not reaction.id.startswith("EX_"):
            continue
        e_mets = {m: c for m, c in reaction.metabolites.items() if m.id.endswith("_e")}
        if not e_mets:
            continue
        # If the _e metabolite is a product (+1), flip the reaction direction
        if all(c > 0 for c in e_mets.values()):
            # Negate all stoichiometric coefficients
            reaction.subtract_metabolites({m: 2 * c for m, c in reaction.metabolites.items()})
            lb, ub = reaction.bounds
            reaction.bounds = (-ub, -lb)
            # Flip atom_mapping: swap reactants and products across the '->' arrow
            if "atom_mapping" in reaction.annotation:
                amm = reaction.annotation["atom_mapping"]
                if "->" in amm:
                    lhs, rhs = amm.split("->", 1)
                    reaction.annotation["atom_mapping"] = rhs.strip() + " -> " + lhs.strip()
            flipped.append(reaction.id)

    print(f"Standardized {len(flipped)} EX_ reactions to depletion direction: {flipped}")


def _strip_out_from_atom_mapping(amm_str):
    """Remove all _out metabolite entries (optionally carrying an atom label) from an atom_mapping string, at any position on either side of '->'."""
    if "->" not in amm_str:
        return amm_str

    def _strip_terms(half):
        # Split on '+', strip _out entries, rejoin
        terms = [t.strip() for t in half.split("+")]
        kept = [t for t in terms if not re.match(r'\S+_out\b', t)]
        return " + ".join(kept)

    lhs, rhs = amm_str.split("->", 1)
    return _strip_terms(lhs) + " -> " + _strip_terms(rhs)


def atommap_boundary(model, strict=True):
    """Give every boundary reaction the atom map of the metabolite it exchanges, copied from an interior reaction that already maps it. Modifies model in place.

    If strict, raises when an exchanged metabolite appears in no mapped reaction;
    otherwise returns the ids of boundary reactions left unmapped.

    Args:
        model: cobra model whose boundary reactions are annotated, modified in place.
        strict: raise instead of skipping an unmappable boundary reaction.
    """
    all_rxns = set(model.reactions)
    unmapped = []
    for r in model.boundary:
        metabolite = list(r.metabolites.keys())[0]
        metabolite_reaction = None
        for reaction in metabolite.reactions:
            if reaction in all_rxns and "atom_mapping" in reaction.annotation:
                metabolite_reaction = reaction
                break

        if metabolite_reaction is None:
            if strict:
                raise ValueError(
                    f"No atom-mapped reaction found for {metabolite.id}, so the "
                    f"boundary reaction {r.id} cannot be annotated. Pass "
                    f"strict=False to skip such boundaries."
                )
            unmapped.append(r.id)
            continue

        pattern = r'(?<![\w-])' + re.escape(metabolite.id) + r'\s*(\([^)]+\))'
        match = re.search(pattern, metabolite_reaction.annotation['atom_mapping'])
        if match:
            atom_map = match.group(1)
            r.annotation['atom_mapping'] = insert_mapping_into_reaction(
                r.reaction, metabolite.id, atom_map)
        else:
            unmapped.append(r.id)

    return unmapped


def insert_mapping_into_reaction(reaction, compound_id, mapping):
    """Insert an atom-mapping string after every occurrence of a compound id in a reaction string.

    Args:
        reaction: reaction formula string to modify.
        compound_id: metabolite id to insert the mapping after.
        mapping: atom-mapping string, e.g. '(abc)'.
    """
    # Define a regular expression pattern to find the compound ID
    # and capture the surrounding context
    pattern = re.compile(rf'(\b{re.escape(compound_id)}\b)')

    # Use a replacement function in re.sub()
    def replacement(match):
        # Include the mapping after the compound ID
        return f'{match.group(1)} {mapping}'

    # Replace the pattern in the reaction string with the new string including the mapping
    new_reaction = pattern.sub(replacement, reaction)
    return new_reaction



def solve_for_theta(K,V):
    """Solve K @ theta = V for the minimum-sum-of-coefficients theta via linear programming.

    Args:
        K: constraint matrix.
        V: target values, K @ theta = V.
    """
    c = np.ones(K.shape[1])
    res = linprog(c=c, A_eq = K, b_eq = V, bounds=(None,None))
    # TODO Chatch stupid erorrs 
    # TODO maybe also make this a tensor? 
    if res.x is None:
        raise ValueError(f"No solution found for theta {res.status}")
    return res.x

### Sparse utils ### 

def select_sparse_column(sparse_matrix, col_idx):
    """Select a single column from a sparse matrix, setting all other columns to zero.

    Args:
        sparse_matrix: sparse COO tensor to select from.
        col_idx: column index to keep.
    """
    rows, cols = sparse_matrix.shape

    # Extract indices and values from the original sparse matrix
    indices = sparse_matrix._indices()
    values = sparse_matrix._values()

    # Filter indices to only keep the desired column
    col_indices = indices[1] == col_idx
    filtered_indices = indices[:, col_indices]
    filtered_values = values[col_indices]

    # Adjust the column indices in filtered_indices to all be col_idx
    filtered_indices[1, :] = col_idx

    # Create a new sparse matrix with the filtered indices and values
    new_sparse_matrix = torch.sparse_coo_tensor(filtered_indices, filtered_values, (rows, cols))

    return new_sparse_matrix

def flatten_sparse_matrix(sparse_matrix):
    """Flatten a 2D sparse matrix into a 1D sparse vector."""
    rows, cols = sparse_matrix.shape
    indices = sparse_matrix._indices()
    values = sparse_matrix._values()

    # Compute linear indices for the 1D flattened vector
    flat_indices = indices[0] * cols + indices[1]

    # Create the flattened 1D sparse vector
    flat_sparse_matrix = torch.sparse_coo_tensor(flat_indices.unsqueeze(0), values, (rows * cols,))

    return flat_sparse_matrix

def reshape_sparse_tensor_to_3D(sparse_tensor, new_shape):
    """Reshape a 2D sparse tensor of shape (N, X*Y) to a 3D sparse tensor of shape new_shape = (N, X, Y).

    Args:
        sparse_tensor: 2D sparse COO tensor to reshape.
        new_shape: target (N, X, Y) shape.
    """
    if len(new_shape) != 3:
        raise ValueError("New shape must be a tuple of 3 dimensions (N, X, Y)")

    N, M = sparse_tensor.shape
    Y,X = new_shape[1], new_shape[2]

    if M != X * Y:
        raise ValueError("The product of X and Y must be equal to M")

    # Extract original indices and values
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()

    # Recalculate indices for the new shape
    new_indices = torch.zeros(3, indices.size(1), dtype=torch.long)
    new_indices[0] = indices[0]  # N dimension remains the same
    new_indices[1] = indices[1] % Y # New X dimension
    new_indices[2] = indices[1] // Y 

    # Create the new sparse tensor
    new_sparse_tensor = torch.sparse_coo_tensor(new_indices, values, new_shape)
    return new_sparse_tensor

def multiply_3d_sparse_with_2d(sparse_3d, tensor_2d, transpose_2nd = False, flip_mult = False):
    """Multiply each 2D slice of a 3D sparse tensor [N, C, B] (along dim 0) with a 2D dense tensor [B, P], giving [N, C, P].

    Args:
        sparse_3d: 3D sparse tensor [N, C, B] to multiply slice-wise.
        tensor_2d: dense 2D tensor multiplied against each slice.
        transpose_2nd: transpose each sparse slice before multiplying.
        flip_mult: multiply tensor_2d @ slice instead of slice @ tensor_2d.
    """

    # Fast path for the only combination the EMU backward pass uses: every slice
    # multiplied on the right by the same dense matrix.
    #
    # Slice-by-slice, this is ~400 tiny sparse matmuls per call and the per-call
    # overhead swamps the arithmetic — profiling an MFA fit put this function at 71%
    # of total runtime, with 1.08M torch.mm invocations. Stacking the slices
    # vertically turns it into one matmul: row (n, c) of the flattened [N*C, B]
    # operand is row c of slice n, so [N*C, B] @ [B, P] gives exactly the stacked
    # per-slice products, and a view restores [N, C, P]. Same arithmetic, same
    # result, one kernel launch.
    if not transpose_2nd and not flip_mult:
        sp = sparse_3d.coalesce()
        n_idx, c_idx, b_idx = sp.indices()
        n_slices, n_rows, n_cols = sp.shape
        flat = torch.sparse_coo_tensor(
            torch.stack([n_idx * n_rows + c_idx, b_idx]),
            sp.values().to(tensor_2d.dtype),
            (n_slices * n_rows, n_cols),
            device=tensor_2d.device,
        ).coalesce()
        return torch.mm(flat, tensor_2d).view(n_slices, n_rows, tensor_2d.shape[1])

    # Placeholder for the results
    results = []


    # Loop through the slices of the sparse matrix
    for i in range(sparse_3d.shape[0]):
        # Extract the i-th slice as a 2D sparse tensor
        if transpose_2nd: 
            slice_2d = sparse_3d[i].T.coalesce()
        else: 
            slice_2d = sparse_3d[i].coalesce()
        
        # Perform the sparse-dense multiplication
        if flip_mult: 
            result_slice = torch.mm(tensor_2d, slice_2d)
        else: 
            result_slice = torch.mm(slice_2d, tensor_2d)

        # Store the result slice
        results.append(result_slice)

    # Concatenate the results along the first dimension
    result_tensor = torch.cat([r.unsqueeze(0) for r in results], dim=0)

    return result_tensor

def find_dependent_rows(matrix):
    rank_full = torch.linalg.matrix_rank(matrix)
    dependent_rows = []

    for i in range(matrix.shape[0]):
        matrix_minus_i = torch.cat([matrix[:i], matrix[i+1:]])
        rank_minus_i = torch.linalg.matrix_rank(matrix_minus_i)

        if rank_minus_i == rank_full:
            dependent_rows.append(i)

    return dependent_rows

def add_dilution_inputs(model, metabolite_ids, suffix="_env"):
    """Give each named metabolite an explicit unlabeled (M+0) environmental input pool (``met_env -> met``), so it dilutes in the EMU network even when it is also produced internally.

    Needed for pools like ``co2_e`` that are both an internal mixing node and an
    exchange whose uptake direction alone yields no EMU edge and thus no dilution.

    Args:
        model: cobra model to add dilution inputs to, modified in place.
        metabolite_ids: ids of metabolites to give a dilution input.
        suffix: suffix appended to build each new environmental pool's id.
    """
    added = []
    for met_id in metabolite_ids:
        if met_id not in model.metabolites:
            raise KeyError(f"{met_id} is not in the model")
        met = model.metabolites.get_by_id(met_id)
        pattern = met.annotation.get("atom_pattern", "")
        if not pattern:
            raise ValueError(f"{met_id} carries no atom_pattern, so it cannot dilute")
        source_id = f"{met_id}{suffix}"
        if source_id in model.metabolites:
            continue
        source = cobra.Metabolite(source_id, name=met.name, compartment=met.compartment)
        source.annotation = {"atom_pattern": pattern}
        model.add_metabolites([source])

        # The env pool needs its own supply or it is a dead end: nothing produces
        # it, so steady state forces the dilution flux to zero and constrain_by_fva
        # drops the reaction as unused. This mirrors how a medium amino acid works
        # -- EX_ser__L_e makes ser__L_e from nothing, which is what makes it an EMU
        # network input carrying the medium's labeling.
        supply = cobra.Reaction(f"EX_{source_id}")
        supply.lower_bound, supply.upper_bound = 0.0, 1000.0
        model.add_reactions([supply])
        supply.add_metabolites({source: 1})
        supply.annotation = {"atom_mapping": f" -> {source_id} ({pattern})"}

        rxn = cobra.Reaction(f"DILUT_{met_id}")
        rxn.lower_bound, rxn.upper_bound = 0.0, 1000.0
        model.add_reactions([rxn])
        rxn.add_metabolites({source: -1, met: 1})
        rxn.annotation = {"atom_mapping": f"{source_id} ({pattern}) -> {met_id} ({pattern})"}
        added.extend([supply.id, rxn.id])
    return added

def media_dilution_targets(model, media_cfg):
    """Carbon-carrying media pools that the cell also produces internally, and so are candidates for `add_dilution_inputs`.

    A medium component the cell cannot synthesise is already an EMU network input and
    dilutes by construction; one it also produces is a mixing node whose exchange
    uptake alone yields no EMU edge, so it needs an explicit dilution input instead.

    Args:
        model: cobra model to check for internal production.
        media_cfg: exchange reaction ids that define the medium.
    """
    targets = []
    for exchange_id in media_cfg:
        if exchange_id not in model.reactions:
            continue
        exchange = model.reactions.get_by_id(exchange_id)
        for met in exchange.metabolites:
            if not met.annotation.get("atom_pattern") or met.id in targets:
                continue
            # Both directions, matching can_make_met: a reversible transporter like
            # ALAt_c_e (ala__L_e <=> ala__L_c) only ever has ala__L_e as a reactant,
            # so it produces ala__L_e exclusively via its *reverse* bound -- the
            # forward-only check below missed this class of internal producer
            # entirely, leaving reversible-transporter media components undiluted.
            produced_internally = any(
                rxn is not exchange
                and "atom_mapping" in rxn.annotation
                and ((rxn.metabolites[met] > 0 and rxn.upper_bound > 0)
                     or (rxn.metabolites[met] < 0 and rxn.lower_bound < 0))
                for rxn in met.reactions)
            if produced_internally:
                targets.append(met.id)
    return targets
