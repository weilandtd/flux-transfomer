"""Convert the prepared cobra model into an x3cflux NetworkData.

The hard part is that x3cflux's ``Reaction`` carries no stoichiometric
coefficients -- multiplicity is expressed by repeating a metabolite name -- while
x3cflux derives the steady-state matrix S automatically from the declared pools.
Our cobra S has nullity 25; restricted to the 155 carbon pools it has nullity 50.
So the non-carbon balances must be reproduced too, or Ipopt gets a flux space
twice as large as the one our fit searches.

Three devices make that work:

  * **0-atom pools.** ``Metabolite(name, 0, {Carbon: 0})`` and 0-atom reactions
    are legal and enter S as ordinary balance rows, so all 188 cobra pools and all
    203 reactions can be declared.
  * **Name repetition** for integer coefficients (ATPS_m 3x adp/atp, CBPS_c 2x atp).
  * **Splitting** for non-integer coefficients, which name repetition cannot
    express: each participant becomes a unit half-reaction and the pieces are tied
    back together by equality LinearConstraints carrying the exact cobra ratios
    (emitted by :mod:`~sigmaflux.x3cflux.build_config`). Affects PROTEIN_c, RNA_c,
    DNA_c, LIPID_c and Ht_c_m. FECOOR_m and NADPHBAL_c instead become all-integer
    when scaled by 2, which is cheaper and keeps them as single reactions.

Two further deviations from a naive reading of the cobra model, both required to
match our EMU forward model rather than to satisfy x3cflux:

  * **Backward-only reactions** (ub <= 0; 15 of them, all EX_ uptakes) are flipped,
    because x3cflux with ``bidirectional=False`` only builds forward labeling edges
    while our AtomMapping keys off ``lower_bound < 0`` and builds reverse ones.
    Flipping records scale = -1.
  * **Symmetric pools** (fum_c, fum_m, succ_m) have no native x3cflux flag. Our
    expansion adds both the natural and the reversed origin set with
    ``symmetry_weight = 1/(1+len(symmetries))`` = 1/2 each, so every reaction
    producing such a pool is split into two half-flux copies, one per orientation.
"""

import json
import re
import warnings
from collections import OrderedDict

import x3cflux as x

CARBON = x.TracerElement.Carbon
INT_TOL = 1e-9
SRC_SUFFIX = "__src"

# SK_accoa_c is the one reaction that carries carbon but has no atom_mapping, so
# it cannot be declared with num_atoms = 0 the way the carbon-free unmapped
# reactions are: x3cflux would compute atom offsets for accoa_c/hco3_c/co2_c
# against a 0-atom permutation and walk off the end (a segfault, not an
# exception). Our expansion instead skips the reaction entirely, so co2_c receives
# nothing from it. x3cflux has no way to express "this reaction moves mass but not
# label", so the minimal defensible stand-in maps the chemically identical carbon
# hco3_c -> co2_c and voids acetyl-CoA's two. This is a real deviation from our
# forward model and G2 measures its size.
#
# This override cannot be retired by any fix on our side: adding the real mapping
# is a biology call on the atom-mapping spreadsheet (hco3_c (a) -> co2_c (a) in
# mmTcell_AMM_completeMFA.xlsx), not a code change.
MAPPING_OVERRIDES = {
    "SK_accoa_c": "accoa_c (ab) + hco3_c (c) + atp_c (defghijklm) "
                  "-> co2_c (c) + adp_c (defghijklm)",
}


# ── atom-mapping parsing ──────────────────────────────────────────────────────

def parse_atom_mapping(map_str, stoich):
    """Split an ``atom_mapping`` annotation into educt and product letter strings.

    Mirrors :func:`sigmaflux.expansion.atom_mapping.parse_mapping_string`: for each
    cobra reactant/product, ``re.findall`` collects *every* parenthesised group
    naming it, so a metabolite listed twice yields two mappings. Metabolites named
    in the string but absent from the stoichiometry are ignored, and metabolites in
    the stoichiometry with no parenthesised group contribute no atoms.
    """
    if not map_str.startswith(" "):
        map_str = " " + map_str
    educts, products = [], []
    for met, coef in stoich.items():
        pattern = r"\b{}\s*\(([^)]+)\)".format(re.escape(met))
        for m in re.findall(pattern, map_str):
            (educts if coef < 0 else products).append((met, m))
    return educts, products


def _bit_positions(entries):
    """letter -> global bit index, over a concatenation of (met, pattern) entries.

    x3cflux indexes atoms by *bitset bit index* and bit 0 is the rightmost
    character, so a pool with n atoms whose pattern is s (s[0] = C1, the FluxML
    ``cfg=`` convention) carries atom s[j] at bit n-1-j. Offsets accumulate over
    ``educt_names`` / ``product_names`` in storage order.
    """
    pos, off = {}, 0
    for _, pat in entries:
        n = len(pat)
        for j, letter in enumerate(pat):
            pos[letter] = off + (n - 1 - j)
        off += n
    return pos, off


def build_permutation(educts, products, rxn_id):
    """``perm[educt_bit] = product_bit``, matched by letter.

    Returns ``(perm, n_atoms, n_void)``. ``n_void`` counts educt atoms with no
    matching product letter: 5 reactions (ACITL_c, MDH_c, GHMT2r_m, GLYCL_m,
    ASPTA_c) drop carbon into products that the *prepared* model does not contain
    -- their maps name oaa_c / mlthf_m, which the FVA pruning removed. Our
    expansion simply lets those letters fail to propagate; x3cflux needs a
    bijection, so the caller absorbs them into a per-reaction void pool that is
    drained by a dedicated efflux. That adds one row and one column to S, so the
    nullspace dimension is unchanged.

    Since ``da6c709`` a stranded pool gets an explicit demand instead of being
    deleted, so on regenerated data ``n_void`` should be 0 everywhere and the void
    machinery becomes dead weight -- that is what gate G2 is used to confirm before
    removing it.
    """
    e_pos, n_e = _bit_positions(educts)
    p_pos, n_p = _bit_positions(products)

    orphan_products = sorted(set(p_pos) - set(e_pos))
    if orphan_products:
        raise ValueError(f"{rxn_id}: product letters {orphan_products} have no educt; "
                         f"this would need a void *source*, which is not implemented")

    # Unmatched educt letters, in educt-concatenation order, land in the void pool.
    unmatched = sorted((ei, letter) for letter, ei in e_pos.items() if letter not in p_pos)
    for k, (_, letter) in enumerate(unmatched):
        p_pos[letter] = n_p + (len(unmatched) - 1 - k)

    perm = [0] * n_e
    for letter, ei in e_pos.items():
        perm[ei] = p_pos[letter]
    return perm, n_e, len(unmatched)


def apply_symmetry(products, sym_map):
    """Rewrite product letter strings for one symmetry orientation.

    ``sym_map`` maps metabolite id -> (orig, image) e.g. ('abcd', 'dcba'). The
    orientation is applied by permuting the letters of that product's pattern the
    same way ``compute_origin_emu_set`` does: ``index_map[i] = image.index(orig[i])``.
    """
    out = []
    for met, pat in products:
        if met in sym_map:
            orig, image = sym_map[met]
            if len(orig) != len(pat):
                raise ValueError(f"symmetry {orig}.{image} does not fit pattern {pat} "
                                 f"of {met}")
            idx = [image.index(c) for c in orig]
            pat = "".join(pat[i] for i in idx)
        out.append((met, pat))
    return out


# ── stoichiometry helpers ─────────────────────────────────────────────────────

def expand_names(stoich, sign):
    """Metabolite names repeated |coef| times, for integer coefficients."""
    names = []
    for met, coef in stoich.items():
        if coef * sign <= 0:
            continue
        names.extend([met] * int(round(abs(coef))))
    return names


def integer_scale(stoich):
    """Smallest factor in {1, 2} making every coefficient integral, else None."""
    for f in (1, 2):
        if all(abs(c * f - round(c * f)) < INT_TOL for c in stoich.values()):
            return f
    return None


# ── the converter ─────────────────────────────────────────────────────────────

class Converted:
    """NetworkData plus the bookkeeping needed to map fluxes in and out.

    Attributes
    ----------
    network_data : x3cflux.NetworkData
    scale : dict[str, float]
        x3cflux flux ``u_r`` for a plain reaction equals ``scale[r] * v_r``.
        -1 for flipped backward-only reactions, 2 for the rescaled ones.
    parents : OrderedDict[str, list[tuple[str, float]]]
        cobra reaction -> [(x3c sub-reaction, coefficient), ...] for split and
        symmetry-halved reactions. The coefficient is what the sub-flux equals when
        the parent flux is 1.
    reversible : dict[str, bool]
        cobra reaction -> whether an exchange parameter should stay free.
    """

    def __init__(self, network_data, scale, parents, reversible, x3c_reactions,
                 void_pools, symmetric, sources, atom_free_reverse, extra_sources):
        self.network_data = network_data
        self.scale = scale
        self.parents = parents
        self.reversible = reversible
        self.x3c_reactions = x3c_reactions
        self.void_pools = void_pools
        self.symmetric = symmetric
        self.sources = sources
        self.atom_free_reverse = atom_free_reverse
        self.extra_sources = extra_sources


def input_source_pools(facts):
    """EMU input metabolite -> upstream source pool that carries its labeling.

    A pool declared as a ConstantSubstrate must be a pure input: x3cflux drops it
    from the balance and stops tracing there. Our 17 input pools are not pure
    inputs -- each is produced by its EX_ uptake, which after flipping is a
    from-nothing influx. Declaring the pool as a substrate *and* leaving a producer
    attached makes ``create_simulator_from_data`` walk off the end and SIGSEGV
    (minimal repro: EX_glc__D_e + GLCt_c_e + HEX1_c, measuring g6p_c).

    So the substrate is moved one step upstream: ``glc__D_e__src -> glc__D_e`` via
    the existing EX_ reaction, with ``glc__D_e__src`` carrying the tracer profile.
    ``glc__D_e`` is then an ordinary balanced pool, EX_glc__D_e keeps its identity
    and flux variable (it is one of the 22 measured fluxes), and the cobra balance
    row is preserved. The source pool adds no balance row of its own, so the
    nullspace dimension is unchanged.

    This is independent of the input-pool MID fix (``c4f7012``) and stays whatever
    that fix does: it is an x3cflux structural requirement, and the single biggest
    crash source.
    """
    sources = {}
    for m in facts["input_metabolites"]:
        for r in facts["reactions"]:
            coef = r["stoichiometry"].get(m, 0)
            if not coef:
                continue
            produces = (coef > 0 and r["upper_bound"] > 0) or \
                       (coef < 0 and r["lower_bound"] < 0)
            others = [k for k in r["stoichiometry"] if k != m]
            if produces and not others:
                sources[m] = m + SRC_SUFFIX
                break
    return sources


def convert(facts, symmetry=True):
    """Build a :class:`Converted` from ``model_facts.json``.

    *symmetry* emulates our symmetric pools with half-flux orientation pairs. It
    can be turned off for debugging: x3cflux segfaults deep in
    ``ReductionOrdering::getSystemIndex`` when the pairs are present in the full
    network, though not in any small replica of it.
    """
    mets_by_id = {m["id"]: m for m in facts["metabolites"]}
    atoms = {m["id"]: len(m["atom_pattern"]) for m in facts["metabolites"]}
    sources = input_source_pools(facts)
    for met, src in sources.items():
        atoms[src] = atoms[met]

    metabolites = []
    for m in facts["metabolites"]:
        n = len(m["atom_pattern"])
        metabolites.append(x.Metabolite(m["id"], n, {CARBON: n}))

    # Symmetric pools: 'abcd.dcba' -> ('abcd', 'dcba')
    symmetric = {m["id"]: tuple(m["symmetries"][0].split("."))
                 for m in facts["metabolites"] if m["symmetries"]}

    reactions, scale, parents, reversible = [], {}, OrderedDict(), {}
    void_pools, void_effluxes, atom_free_reverse = {}, [], []
    extra_sources = {}

    for r in facts["reactions"]:
        rid, stoich = r["id"], dict(r["stoichiometry"])
        lb, ub = r["lower_bound"], r["upper_bound"]
        rev = r["reversible"]
        reversible[rid] = rev

        # Backward-only: flip so x3cflux's forward labeling edges are the ones our
        # AtomMapping builds via lower_bound < 0.
        flip = ub <= 0
        sgn = -1.0 if flip else 1.0

        f = integer_scale(stoich)
        if f is None:
            # Non-integer: split into unit half-reactions, tied by build_config.
            parents[rid] = []
            for met, coef in stoich.items():
                sub = f"{rid}__{met}"
                n = len(mets_by_id[met]["atom_pattern"])
                # Consumed -> efflux (empty products); produced -> influx.
                educts, products = ([met], []) if coef * sgn < 0 else ([], [met])
                if products and n:
                    # A from-nothing influx of a carbon-carrying pool has no
                    # upstream to trace, which is the same unsourced-pool crash as
                    # the EX_ uptakes. Give it a dedicated unlabeled source. Every
                    # pool this affects (amp_c from PROTEIN_c/RNA_c/DNA_c/LIPID_c)
                    # carries only inert 1001-block carbons that stay at M+0, so
                    # feeding it unlabeled material changes no MID.
                    src = sub + SRC_SUFFIX
                    extra_sources[src] = n
                    atoms[src] = n
                    educts = [src]
                reactions.append(x.Reaction(sub, n, False, x.Permutation(n),
                                            educts, products))
                parents[rid].append((sub, abs(coef)))
                reversible[sub] = False
            scale[rid] = sgn
            continue

        # Coefficients were multiplied by f, so the flux must be divided by it:
        # sum(f*coef * u) == sum(coef * v)  =>  u = v/f.
        scale[rid] = sgn / f
        scaled = {m: c * f * sgn for m, c in stoich.items()}
        educt_names = expand_names(scaled, -1)
        product_names = expand_names(scaled, +1)

        mapping = MAPPING_OVERRIDES.get(rid, r["atom_mapping"])
        if mapping:
            educts, products = parse_atom_mapping(mapping, stoich)
            educts = _cap_to_stoichiometry(educts, stoich, rid)
            products = _cap_to_stoichiometry(products, stoich, rid)
            if flip:
                educts, products = products, educts
            # Put the atom-carrying names first, in the storage order x3cflux will
            # index against, then the atom-free ones (no indices of their own).
            mapped_e = [m for m, _ in educts]
            mapped_p = [m for m, _ in products]
            educt_names = mapped_e + _remainder(educt_names, mapped_e)
            product_names = mapped_p + _remainder(product_names, mapped_p)

            # Re-attach patterns over the FULL name lists, so every atom x3cflux
            # will index has a letter (see assign_patterns).
            atom_count = dict(atoms)
            educts, spare = assign_patterns(educt_names, educts, atom_count)
            products, _ = assign_patterns(product_names, products, atom_count, spare)
            educts = [(m, p) for m, p in educts if p]
            products = [(m, p) for m, p in products if p]

            if not educts and not products:
                # Annotated, but nothing on either side carries atoms -- the toy
                # model's water shuttle (``H2O_src -> H2O``) is written this way.
                # Identical to an unmapped reaction: keep the balance rows and trace
                # no label. Falling through would index an empty educt list.
                reactions.append(x.Reaction(rid, 0, rev, x.Permutation(0),
                                            educt_names, product_names))
                continue

            if not educts and products:
                # Pure influx: draw from the upstream source pool so the produced
                # pool stays balanced and traceable (see input_source_pools).
                (met, pat) = products[0]
                src = sources.get(met)
                if src is None:
                    raise ValueError(f"{rid} produces {met} from nothing but {met} "
                                     f"is not an EMU network input")
                n_atoms = len(pat)
                reactions.append(x.Reaction(rid, n_atoms, rev, x.Permutation(n_atoms),
                                            [src], product_names))
                continue

            if not products:
                # Pure efflux: nothing produced, so there is nothing to permute.
                n_atoms = sum(len(p) for _, p in educts)
                (met, _) = educts[0]
                src = sources.get(met)
                if src and len(educts) == 1:
                    # An input whose EX_ is written as secretion (met ->, e.g.
                    # EX_ser__L_e). It must still be emitted in *influx* form,
                    # src -> met: a ConstantSubstrate pool may never be produced by
                    # any reaction, or the EMU builder dies in
                    # ReductionOrdering::getSystemIndex. Writing it as met -> src
                    # would produce the substrate on forward flux. The reaction
                    # direction is therefore reversed with respect to cobra, hence
                    # scale = -1, and it is unidirectional for the same reason MDH_c
                    # is: our expansion builds no edge for the reverse either (empty
                    # origin set), so the exchange flux is unidentifiable in both.
                    scale[rid] = -sgn
                    if rev:
                        atom_free_reverse.append(rid)
                        reversible[rid] = False
                    reactions.append(x.Reaction(rid, n_atoms, False,
                                                x.Permutation(n_atoms),
                                                [src], educt_names))
                    continue
                names_p = product_names
                # A reaction whose product side carries no atoms must be declared
                # unidirectional even when cobra calls it reversible: running it
                # backwards would mean permuting n_atoms out of a side that has none,
                # which is an out-of-bounds read (MDH_c is the case that found this
                # -- mal__L_c in, only h_c/nadh_c out, lb < 0 < ub). Our expansion
                # reaches the same place from the other side: the reverse origin set
                # is empty, so no edge is built either way, and the exchange flux is
                # unidentifiable in both models.
                one_sided = not names_p or all(atoms[m] == 0 for m in names_p)
                if one_sided and rev:
                    atom_free_reverse.append(rid)
                    reversible[rid] = False
                reactions.append(x.Reaction(rid, n_atoms, rev and not one_sided,
                                            x.Permutation(n_atoms),
                                            educt_names, names_p))
                continue

            # Our expansion symmetrises on *production* of a symmetric pool (the
            # target EMU's own symmetries), which for a bidirectional reaction
            # includes the reverse direction -- FUM_m produces fum_m when run
            # backwards. So a symmetric educt of a reversible reaction counts too.
            #
            # Exactly one side is flipped. SUCDq9_m has symmetric pools on both
            # sides (succ_m -> fum_m) and flipping both would compose to the
            # identity, collapsing the two orientations into one. Flipping one is
            # also sufficient: the other pool's own producers already symmetrise it,
            # and mirroring an already-symmetric MID is a no-op.
            sym_p = ({m: symmetric[m] for m, _ in products if m in symmetric}
                     if symmetry else {})
            sym_e = ({m: symmetric[m] for m, _ in educts if m in symmetric}
                     if rev and symmetry else {})
            orientations = [(educts, products)]
            if sym_p:
                orientations.append((educts, apply_symmetry(products, sym_p)))
            elif sym_e:
                orientations.append((apply_symmetry(educts, sym_e), products))
            sym_here = sym_p or sym_e

            for k, (educts_k, prods) in enumerate(orientations):
                perm, n_atoms, n_void = build_permutation(educts_k, prods, rid)
                names_p = list(product_names)
                if n_void:
                    void = f"__void_{rid}"
                    if void not in void_pools:
                        void_pools[void] = n_void
                        # Drain it, so the pool balances and S keeps its rank.
                        void_effluxes.append((void, n_void))
                    names_p = names_p + [void]
                if sym_here:
                    # Two half-flux orientations, matching our
                    # symmetry_weight = 1/(1+len(symmetries)) = 1/2.
                    sub = f"{rid}__sym{k}"
                    # Both copies must be the same reaction stoichiometrically.
                    assert len(orientations) == 2
                    parents.setdefault(rid, []).append((sub, 0.5))
                    reversible[sub] = rev
                else:
                    sub = rid
                reactions.append(x.Reaction(sub, n_atoms, rev, x.Permutation(perm),
                                            educt_names, names_p))
        else:
            # No atom mapping: stoichiometry only, no labeling edges. Our expansion
            # skips such reactions outright, which is why pools produced solely by
            # them end up as EMU network inputs. Declaring 0 atoms reproduces that
            # while keeping the balance rows.
            reactions.append(x.Reaction(rid, 0, rev, x.Permutation(0),
                                        educt_names, product_names))

    for met, src in sources.items():
        n = atoms[met]
        metabolites.append(x.Metabolite(src, n, {CARBON: n}))
    for src, n in extra_sources.items():
        metabolites.append(x.Metabolite(src, n, {CARBON: n}))
    for void, n in void_pools.items():
        metabolites.append(x.Metabolite(void, n, {CARBON: n}))
    for void, n in void_effluxes:
        drain = f"{void}__drain"
        reactions.append(x.Reaction(drain, n, False, x.Permutation(n), [void], []))
        reversible[drain] = False

    return Converted(x.NetworkData(metabolites, reactions), scale, parents,
                     reversible, [rr.name for rr in reactions],
                     sorted(void_pools), list(symmetric), sources,
                     atom_free_reverse, extra_sources)


def assign_patterns(names, parsed, atoms, spare=None):
    """Attach an atom pattern to every name in ``names``, in storage order.

    x3cflux computes atom offsets from the *full* name list, so every copy of a
    metabolite contributes its atoms whether or not the atom map mentions it.
    CBPS2_m is the case that matters: its stoichiometry has 2 atp_m / 2 adp_m but
    the map names each once, leaving 10 atoms per side outside the permutation --
    an out-of-bounds read inside x3cflux, i.e. a segfault rather than an error.

    Uncovered copies get freshly minted private-use letters. On the educt side
    those are collected into ``spare``; on the product side they are drawn from
    ``spare`` in order, so the k-th uncovered product atom pairs with the k-th
    uncovered educt atom. For CBPS2_m that pairs the second atp_m with the second
    adp_m -- both inert 1001-block carbons that stay at M+0, so the pairing is
    immaterial. Educt atoms with no product partner fall through to the void pool;
    the reverse would need a void *source* and is rejected.
    """
    pool = {}
    for met, pat in parsed:
        pool.setdefault(met, []).append(pat)
    minting = spare is None
    spare = [] if minting else list(spare)
    out, minted = [], []
    for met in names:
        n = atoms[met]
        if n == 0:
            out.append((met, ""))
            continue
        if pool.get(met):
            out.append((met, pool[met].pop(0)))
            continue
        if minting:
            pat = "".join(chr(0x100 + len(minted) + i) for i in range(n))
            minted.extend(pat)
        else:
            if len(spare) < n:
                raise ValueError(f"uncovered product atoms for {met} have no educt "
                                 f"partner; this would need a void source")
            pat, spare = "".join(spare[:n]), spare[n:]
        out.append((met, pat))
    return out, (minted if minting else spare)


def _cap_to_stoichiometry(entries, stoich, rxn_id=None):
    """Keep at most |coefficient| mapped copies of each metabolite.

    GLYCL_m's atom map was duplicated verbatim while its stoichiometry said 1x, so
    without the cap the name repetition would double its stoichiometry. Since
    ``0fb33f9`` de-duplicates such rows at conversion time, a cap that actually
    drops something means the atom-mapping spreadsheet still disagrees with the
    stoichiometry -- worth a warning rather than silence, and the warning is what
    tells us this can be turned into a hard error.
    """
    seen, out, dropped = {}, [], []
    for met, pat in entries:
        limit = max(int(round(abs(stoich[met]))), 1)
        if seen.get(met, 0) < limit:
            seen[met] = seen.get(met, 0) + 1
            out.append((met, pat))
        else:
            dropped.append(met)
    if dropped:
        warnings.warn(
            f"{rxn_id or '<reaction>'}: atom map names {sorted(set(dropped))} more "
            f"times than the stoichiometry allows; extra copies dropped. The map and "
            f"the stoichiometry disagree.", RuntimeWarning, stacklevel=2)
    return out


def _remainder(all_names, used):
    """Names in *all_names* left over after removing one copy of each in *used*."""
    rest = list(all_names)
    for u in used:
        if u in rest:
            rest.remove(u)
    return rest


def load_facts(path):
    """Read a ``model_facts.json`` written by :mod:`sigmaflux.x3cflux.facts`."""
    with open(path) as fh:
        return json.load(fh)
