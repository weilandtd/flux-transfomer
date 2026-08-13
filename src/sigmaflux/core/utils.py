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
# Maranas biomass formulas use placeholder tokens for the variable acyl chains and
# the sphingoid backbone (e.g. tag_c = C3H5O3Acyl3), so those masses are nominal:
# a palmitoyl chain and a C18 phytoceramide core.
_PLACEHOLDER_MASSES: dict = {'Acyl': 239.4, 'Cer': 537.9}
_ATOMIC_MASSES: dict = {
    'C': 12.011, 'H': 1.008, 'O': 15.999, 'N': 14.007, 'P': 30.974, 'S': 32.06,
}


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
    """Macromolecule mass fractions implied by the aggregate BIOMASS stoichiometry.

    `formulas` maps metabolite id -> chemical formula (the GEM's `metformula`
    column).  Each building block contributes coefficient x molecular weight to its
    macromolecule, and the five totals are normalised to sum to 1.

    These fractions are a units convention, not a physical constraint: they cancel
    out of the net stoichiometry, so they only set the scale of the five component
    fluxes.  They are approximate — the lipid formulas carry placeholder tokens and
    the polysaccharides are written per monomer — so prefer measured values when a
    biomass composition is available for the organism.
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
    """Split the aggregate BIOMASS into the 5 macromolecule component reactions.

    Each component reaction consumes its building blocks at `coefficient / fraction`
    and produces one `bm_*_c` pseudo-metabolite; BIOMASS is left consuming only those
    pseudo-metabolites, at the mass fractions.  The composition through BIOMASS is
    therefore unchanged — the split exposes PROTEIN_c, RNA_c, DNA_c, LIPID_c and
    CARB_c as separately measurable fluxes.

    Energy is then moved out of BIOMASS by `redistribute_biomass_energy`, without
    which the components become energy-decoupled drains once BIOMASS is dropped
    during sampling and labeling loses all sensitivity to them.
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
    """Move ATP/cofactor stoichiometry from BIOMASS into the 5 component reactions.

    The SC hierarchical model keeps all energy (ATP, H2O, ions) in the BIOMASS
    reaction while component reactions (PROTEIN_c etc.) only carry building-block
    metabolites.  After skip_constraints drops BIOMASS during sampling, these
    reactions become energy-decoupled drains and labeling loses all sensitivity
    to their fluxes — causing feature selection to fail.

    This function redistributes the shared cofactors from BIOMASS to each
    component reaction proportionally via _BIOMASS_ENERGY_FRACS, scaled so the
    net stoichiometric flux balance is unchanged.  The BIOMASS reaction is then
    stripped to only bm_* pseudo-metabolites.

    Models that carry only some of the five macromolecules get the energy fractions
    renormalised over the components that exist.  The fractions must sum to 1 or the
    energy left in BIOMASS would be discarded by the strip at the end rather than
    moved, silently unbalancing the model.
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

    # RNA-specific ATP: the portion of ATP that goes with RNA NTPs (proxy: UTP stoich)
    utp_stoich = stoich.get('utp_c', 0.0)   # e.g. -0.1256 for SC
    atp_total = energy_stoich.get('atp_c', 0.0)
    atp_rna_specific = utp_stoich            # ATP co-consumed with RNA NTP synthesis
    atp_energy = atp_total - atp_rna_specific  # energy ATP to distribute by fracs

    for comp_id, frac in energy_fracs.items():
        bm_id = _COMPONENT_TO_BM[comp_id]
        bm_frac = bm_fracs.get(bm_id, 0.0)
        if bm_frac == 0.0:
            continue
        rxn = model.reactions.get_by_id(comp_id)
        additions = {}
        # Distribute all shared metabolites by energy fraction, scaled per component flux unit
        for mid, coeff in energy_stoich.items():
            if mid == 'atp_c':
                # Energy ATP share
                additions[mid] = additions.get(mid, 0.0) + atp_energy * frac / bm_frac
            else:
                additions[mid] = additions.get(mid, 0.0) + coeff * frac / bm_frac
        # RNA-specific ATP goes entirely to RNA_c
        if comp_id == 'RNA_c' and atp_rna_specific != 0.0:
            additions['atp_c'] = additions.get('atp_c', 0.0) + atp_rna_specific / bm_frac

        met_map = {model.metabolites.get_by_id(mid): coeff
                   for mid, coeff in additions.items() if coeff != 0.0}
        rxn.add_metabolites(met_map)

    # Strip all non-bm_* metabolites from BIOMASS
    remove_from_biomass = {m: -c for m, c in biomass.metabolites.items()
                           if not m.id.startswith('bm_')}
    biomass.add_metabolites(remove_from_biomass)
    print(f"redistribute_biomass_energy: distributed energy to {present}")


def annotate_model(model, annotation_file,index='id',atom_mapping='atom_mapping'):

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
    """Give every metabolite an ``atom_pattern`` sized by its reaction atom mappings.

    ``build_targets`` reads these patterns to decide which EMUs to simulate, so a
    model carrying only reaction-level ``atom_mapping`` strings is not yet usable
    for MFA. The pattern is just a distinct label per tracked atom.
    """
    chars = string.ascii_lowercase + string.ascii_uppercase + string.digits
    for met in model.metabolites:
        max_n = 0
        for rxn in met.reactions:
            amm_str = rxn.annotation.get("atom_mapping", "")
            for match in re.findall(atom_pattern_regex(met.id), amm_str):
                max_n = max(max_n, len(match))
        if max_n > 0:
            met.annotation["atom_pattern"] = (
                chars[:max_n] if max_n <= len(chars)
                else "".join(f"a{i}" for i in range(1, max_n + 1))
            )
    return model


def convert_mfa_maranas(path_to_cobra, path_to_atom_map_xls, path_to_output=None, atom_type='C',
                        drop_subsystems=(), biomass_id=None):
    """
    Load a Maranas-lab GEM xlsx + AMM xlsx and return a COBRApy model annotated
    with SigmaFlux atom_mapping strings, ready for MFAModel().

    Maranas AMM format (one row per metabolite per reaction, atoms as integers)
    is converted to SigmaFlux string format, e.g.:
        reaction.annotation['atom_mapping'] = "accoa_c (ab) + accoa_c (cd) -> aacoa_c (abcd) + coa_c"

    :param path_to_cobra: path to Maranas GEM xlsx (Reactions + Metabolites sheets)
    :param path_to_atom_map_xls: path to Maranas AMM xlsx (Atom Transition + Symmetry sheets)
    :param path_to_output: if given, save annotated model as JSON
    :param atom_type: element to track, default 'C'
    :param drop_subsystems: subsystems to exclude, e.g. ('dilution',) for the
        completeMFA workbooks whose explicit dilution sub-network SigmaFlux does
        not model.  Their formulas are dropped before parsing — some carry
        zero-coefficient reactants that COBRA would silently discard, leaving atom
        mappings that reference metabolites no longer in the reaction.
    :param biomass_id: id of the GEM's biomass reaction when it is not already the
        canonical 'BIOMASS', e.g. 'BIOMASS_c' in the M. musculus workbook.  It is
        renamed on load so the flat model, the hierarchical split and the configs'
        skip_constraints all agree on one name.
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
        model.add_metabolites([cobra.Metabolite(id=str(row['metid']), name=str(row['metname']))])
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

    _chars = string.ascii_lowercase + string.ascii_uppercase + string.digits

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
            reactants += [m.id for m in rxn.reactants if m.id not in amm_mets]
            products  += [m.id for m in rxn.products  if m.id not in amm_mets]

            rxn.annotation['atom_mapping'] = ' + '.join(reactants) + ' -> ' + ' + '.join(products)
        except Exception:
            pass

    # Symmetries
    try:
        sym = pd.read_excel(path_to_atom_map_xls, sheet_name='Symmetry')
        for _, row in sym.iterrows():
            try:
                met = model.metabolites.get_by_id(str(row['Metab_name']).strip())
                default = [int(x) for x in str(row['default mapping']).split()]
                sym_pos = [int(x) for x in str(row['symmetry mapping']).split()]
                p2l = {p: string.ascii_lowercase[i] for i, p in enumerate(default)}
                entry = ''.join(p2l[p] for p in default) + '.' + ''.join(p2l[p] for p in sym_pos)
                met.annotation.setdefault('symmetries', []).append(entry)
            except Exception:
                pass
    except Exception:
        pass

    annotate_atom_patterns(model)

    _remove_out_metabolites(model)
    _standardize_exchange_directions(model)

    if path_to_output is not None:
        cobra.io.save_json_model(model, str(path_to_output))

    return model


def _remove_metabolites_orphaned_by(model, dropped_formulas):
    """Remove metabolites left without reactions by a `drop_subsystems` filter.

    Only metabolites that the dropped formulas actually mention are removed.
    Metabolites that carry no reaction in the workbook to begin with are kept —
    the I. orientalis GEM lists ~200 such dead `*_i` / `sink_*` entries, and
    pruning them would silently change a model that already ships as a release.
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
    """Convert Maranas _out boundary metabolites to proper COBRA exchange reactions.

    In Maranas MFA models, EX_ reactions connect an _e metabolite to an _out
    metabolite (the external environment pool).  This converts them to standard
    COBRA boundary reactions that simply deplete the _e metabolite:

        Before:  EX_glc__D_e : glc__D_e → glc__D_out
        After:   EX_glc__D_e : glc__D_e →

    Atom-mapping annotations are updated to remove _out references, and all
    orphaned _out metabolites are removed from the model.
    """
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
    """Ensure all EX_ reactions are in the depletion direction: met_e → (coeff -1).

    In Maranas models the stoichiometric direction of exchange reactions is
    inconsistent — some produce the _e metabolite (+1, uptake-positive) while
    others consume it (-1, secretion-positive).  This normalises every EX_
    reaction so that:

        positive flux = secretion  (metabolite leaves the system)
        negative flux = uptake     (metabolite enters the system)

    Reactions that already follow the convention are left untouched.
    For reactions where the _e metabolite has coefficient +1, the stoichiometry
    is flipped and the bounds are negated and swapped.
    """
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
    """Remove all _out metabolite entries from an atom_mapping string.

    Handles entries at any position: first term, after '+', and on either
    side of '->'.  Entries may optionally carry an atom label, e.g.
    ``glc__D_out (abcdef)``.
    """
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
    """Give every boundary reaction the atom map of the metabolite it exchanges.

    Boundary (exchange/sink/source) reactions carry no atom mapping of their
    own, but the EMU expansion needs one to treat them as label inputs. The
    mapping is copied from any interior reaction that already maps the exchanged
    metabolite.

    Parameters
    ----------
    model : cobra.Model
        Modified in place.
    strict : bool
        Raise when an exchanged metabolite appears in no mapped reaction. Set
        False to skip those boundaries instead — useful for models where some
        exchanged species carry no tracked atoms (ions, water, protons).

    Returns
    -------
    list of str
        Ids of boundary reactions left without a mapping.
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
    c = np.ones(K.shape[1]) 
    res = linprog(c=c, A_eq = K, b_eq = V, bounds=(None,None))
    # TODO Chatch stupid erorrs 
    # TODO maybe also make this a tensor? 
    if res.x is None:
        raise ValueError(f"No solution found for theta {res.status}")
    return res.x

### Sparse utils ### 

def select_sparse_column(sparse_matrix, col_idx):
    """
    Select a single column from a sparse matrix, setting all other columns to zero.

    Args:
    sparse_matrix (torch.Tensor): The input sparse matrix.
    col_idx (int): The index of the column to select.

    Returns:
    torch.Tensor: A new sparse matrix with only the selected column.
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
    """
    Flatten a 2D sparse matrix into a 1D sparse vector.

    Args:
    sparse_matrix (torch.Tensor): A 2D sparse matrix.

    Returns:
    torch.Tensor: A 1D sparse vector.
    """
    rows, cols = sparse_matrix.shape
    indices = sparse_matrix._indices()
    values = sparse_matrix._values()

    # Compute linear indices for the 1D flattened vector
    flat_indices = indices[0] * cols + indices[1]

    # Create the flattened 1D sparse vector
    flat_sparse_matrix = torch.sparse_coo_tensor(flat_indices.unsqueeze(0), values, (rows * cols,))

    return flat_sparse_matrix

def reshape_sparse_tensor_to_3D(sparse_tensor, new_shape):
    """
    Reshape a 2D sparse tensor to a 3D sparse tensor.
    
    Args:
    sparse_tensor (torch.Tensor): A 2D sparse tensor.
    new_shape (tuple): The desired shape (N, X, Y).

    Returns:
    torch.Tensor: A 3D sparse tensor.
    
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
    """
    Multiply each 2D slice of a 3D sparse tensor (along the first dimension) 
    with a 2D dense tensor, leveraging sparse-dense multiplication optimizations.

    Args:
    sparse_3d (torch.Tensor): A 3D sparse tensor of shape [N, C, B].
    tensor_2d (torch.Tensor): A 2D sparse tensor of shape [B, P].
    transpose_2nd (bool): Whether to transpose the second dimension of the sparse matrix.
    flip_mult: whether tensor_2D is multiplied on the right or left of sparse_3d

    Returns:
    torch.Tensor: A 3D tensor of shape [N, C, P] containing the results with type of the tensor 2d.
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
    """Give each named metabolite an explicit unlabeled environmental input pool.

    A pool that exchanges with the environment is diluted by it, but the exchange
    is invisible to the EMU network unless the environment is a *node*. ``co2_e``
    is the case that matters: it is produced by ``CO2t_c_e`` from internal
    decarboxylations, so it is a core mixing node, while the uptake direction of
    ``EX_co2_e`` produces it from nothing and yields an empty origin set -- no edge,
    no dilution. Its MID then reflects only internal carbon, and since ``PC_m``
    refixes CO2 at ~40 flux units that error propagates into oxaloacetate and the
    whole TCA/aspartate branch.

    Adding ``co2_e_env -> co2_e`` fixes it with no new machinery: the env pool has
    no producer, so it is an ordinary EMU network input, it carries no tracer and
    is therefore M+0, and the fit identifies how much of the pool comes from it.
    This is exactly the shape the medium amino acids already have.

    The source is pure M+0: the measured MIDs are already corrected for natural
    abundance, so carrying the 1.1% here would double-count it.
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
        source = cobra.Metabolite(source_id, compartment=met.compartment)
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
    """Carbon-carrying media pools that the cell also makes itself.

    A medium component the cell cannot synthesise is already an EMU network input:
    nothing produces it, so the medium alone sets its MID and it dilutes by
    construction. One the cell *also* produces is a mixing node, and the uptake
    direction of its exchange makes it from nothing -- an empty origin set, no
    edge -- so the medium contributes no label and cannot dilute it at all.

    That is wrong regardless of net direction, and most wrong where the net flux is
    outward: a T cell secreting lactate still sits in medium containing unlabeled
    lactate, and the exchange dilutes the intracellular pool. On the T-cell medium
    this selects lac__L_e, co2_e, urea_e, asp__L_e and glu__L_e.

    Pools the cell cannot make are excluded because they are already medium
    boundaries; a second feed would only dilute the tracer. How much of the medium
    label actually reaches the cytosol is not decided here either way -- that is
    the transporter's exchange flux, which the fit identifies.
    """
    targets = []
    for exchange_id in media_cfg:
        if exchange_id not in model.reactions:
            continue
        exchange = model.reactions.get_by_id(exchange_id)
        for met in exchange.metabolites:
            if not met.annotation.get("atom_pattern") or met.id in targets:
                continue
            produced_internally = any(
                rxn is not exchange
                and "atom_mapping" in rxn.annotation
                and rxn.metabolites[met] > 0 and rxn.upper_bound > 0
                for rxn in met.reactions)
            if produced_internally:
                targets.append(met.id)
    return targets
