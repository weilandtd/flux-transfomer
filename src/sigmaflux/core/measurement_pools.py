"""Whole-cell measurement pools: compartment-less pseudo-metabolites for measured tokens.

An LC/MS platform measures a whole-cell extract, so it cannot tell ``pyr_c`` from
``pyr_m``. Writing the measured MID into each compartment separately -- what this
pipeline did before -- does not model a mixture, it asserts ``MID(pyr_c) == MID(pyr_m)``,
which is false whenever the compartments differ. The correct model is a mixture, and the
standard MFA way to write one is a pseudo-metabolite fed by *zero-stoichiometry*
reactions::

    0 pyr_c (abc) -> pyr (abc)     POOL_pyr_c,  [0, 1]
    0 pyr_m (abc) -> pyr (abc)     POOL_pyr_m,  [0, 1]
      pyr (abc) ->                 POOL_SK_pyr, [1, 1]

The zero coefficient keeps ``S.v`` untouched, so the pool observes the compartments
without depleting them, while the atom map still carries label -- so the EMU balance of
``pyr`` becomes the flux-weighted mixture. ``S.v = 0`` on the pool's own row forces
``v_c + v_m = 1``, which makes the mixing fluxes *the relative compartment abundances*:
sampled freely and inferred like any other flux instead of assumed.

COBRA cannot hold a zero coefficient -- ``add_metabolites`` drops the entry outright, in
memory, so there is nothing for the JSON round-trip to preserve either. The educt is
therefore carried in ``annotation['zero_stoichiometry']`` and read back through
:func:`sigmaflux.expansion.utils.labeling_educts`.

**Pooling is a modeling choice, not a processing choice.** A model either states that its
compartmental copies of a metabolite are one measurable species or it does not, and that
statement belongs in the model file -- built by the organism conversion script, shipped in
``*_annotated_hierarchical.json``, and named directly by the mapping CSVs. Nothing here
reads a config or a mapping file: :func:`plan_measurement_pools` derives the plan from the
model's own compartment structure, so there is no list to keep in step with anything.

``POOL_`` is the marker. Every reaction whose id starts with it is observation machinery
rather than metabolism, and every one of them is what the x3cflux converter has to rewrite
into a shape that engine can express (see :mod:`sigmaflux.x3cflux.convert`).
"""

import re
import warnings

import cobra

#: Prefix marking every measurement-pool reaction -- the marker for the whole block, and
#: the id of a zero-stoichiometry mixing leg is exactly ``POOL_<member>``.
POOL_PREFIX = "POOL_"
#: Prefix of a pool's pinned drain, ``POOL_SK_<pool>``. Deliberately inside the ``POOL_``
#: namespace rather than the conventional bare ``SK_``: the mouse T-cell model already
#: carries a metabolic ``SK_accoa_c``, which a bare prefix would silently reclassify as
#: observation machinery.
POOL_DRAIN_PREFIX = POOL_PREFIX + "SK_"

#: Metabolite annotation listing a pool's members.
POOL_MEMBERS_KEY = "measurement_pool"

#: Compartment suffixes that are never pool members. ``e`` is the medium, which a cell
#: extract does not contain -- and pooling it would mix the tracer inlet (``glc__D_e``)
#: into a measured pool. ``p`` is the periplasm, which is not a real labeled space: the
#: outer membrane is porous, so it equilibrates with the medium.
EXCLUDED_COMPARTMENTS = ("e", "p")

_COMPARTMENT_SUFFIX = re.compile(r"_([a-z])$")


def _compartment(met_id):
    """The single-letter compartment suffix of *met_id*, or None if it carries none."""
    match = _COMPARTMENT_SUFFIX.search(met_id)
    return match.group(1) if match else None


def _stem(met_id):
    """*met_id* without its compartment suffix -- the pool id it belongs to."""
    return _COMPARTMENT_SUFFIX.sub("", met_id)


def is_pool_reaction(rxn):
    """Whether *rxn* is a measurement-pool mixing leg or drain.

    These are observation machinery, not metabolism: they carry no mass, are trivially
    non-essential, and their flux is a compartment fraction rather than a rate. Callers
    that knock reactions out, or that score predicted fluxes, must skip them.
    """
    return rxn.id.startswith(POOL_PREFIX)


def pool_reaction_ids(model):
    """Ids of every measurement-pool reaction in *model*."""
    return [r.id for r in model.reactions if is_pool_reaction(r)]


def pool_members(model):
    """``{pool_id: [member ids]}`` read back from *model*'s own annotations."""
    return {m.id: list(m.annotation[POOL_MEMBERS_KEY])
            for m in model.metabolites if POOL_MEMBERS_KEY in m.annotation}


def measurement_token(model, met_id, exclude_compartments=EXCLUDED_COMPARTMENTS):
    """The id *met_id* is actually measured under, or None if it is not measurable.

    A metabolite the model pools is measured as its pool -- an extract cannot resolve the
    compartments, so naming one of them would assert a labeling the measurement does not
    distinguish. A metabolite in an excluded compartment is not in the extract at all.
    Everything else is measured as itself.

    This is what a mapping CSV's ``model_id`` column has to hold, so it is applied when
    the generated CSVs are built and checked by :func:`check_mapping_csv`.

    Args:
        model: the cobra model, carrying its pool annotations.
        met_id: a model metabolite id.
        exclude_compartments: compartment suffixes that are not in the measured extract.
    """
    for pool, members in pool_members(model).items():
        if met_id in members:
            return pool
    if _compartment(met_id) in set(exclude_compartments):
        return None
    return met_id


def check_mapping_csv(model, rows):
    """Report every mapping row *model* cannot honour, as a list of complaints.

    Mapping files are curated by hand and checked by hand, so this only ever reports --
    but a stale row is otherwise invisible until feature selection quietly drops a token.

    Args:
        model: the cobra model the mapping is written against.
        rows: ``(experimental_id, model_id)`` pairs.
    """
    problems = []
    for experimental_id, met_id in rows:
        if met_id not in model.metabolites:
            problems.append(f"{experimental_id}: {met_id!r} is not in the model")
            continue
        token = measurement_token(model, met_id)
        if token is None:
            problems.append(f"{experimental_id}: {met_id!r} is in a compartment the "
                            f"extract does not contain")
        elif token != met_id:
            problems.append(f"{experimental_id}: {met_id!r} is pooled into {token!r}; "
                            f"name the pool instead")
    return problems


def plan_measurement_pools(model, exclude_compartments=EXCLUDED_COMPARTMENTS):
    """``{pool_id: [member ids]}`` for every metabolite *model* holds in >1 compartment.

    Derived from the model alone -- a metabolite the network carries in several
    compartments is one species to a whole-cell extract, whether or not today's platform
    happens to report it. That is what keeps the statement a property of the organism
    model rather than of one experiment, and it is why no list of pools exists anywhere.

    Three filters, and nothing else:

    * ``exclude_compartments`` members are dropped -- see
      :data:`EXCLUDED_COMPARTMENTS` for why ``_e`` and ``_p`` are never members.
    * members must carry an ``atom_pattern``. The yeast models hold untracked copies
      (``pyr_i``, ``pyr_d1``, ``pyr_d2``) that the atom mapping never reaches; they carry
      no label to contribute and must not become members.
    * a pool needs **two** members. A single-member pool is identically its member at any
      flux, so it would buy a metabolite, a leg, a drain and a scoring exclusion for
      nothing.

    Args:
        model: the cobra model the pools will be added to.
        exclude_compartments: compartment suffixes that are not in the measured extract.

    Raises:
        ValueError: members disagree on atom pattern or symmetry, or a pool id is taken.
    """
    excluded = set(exclude_compartments)
    members_by_pool = {}
    for met in model.metabolites:
        compartment = _compartment(met.id)
        if compartment is None or compartment in excluded:
            continue
        if not met.annotation.get("atom_pattern"):
            continue
        members_by_pool.setdefault(_stem(met.id), []).append(met.id)

    plan = {pool: sorted(members)
            for pool, members in sorted(members_by_pool.items()) if len(members) > 1}
    _validate_plan(model, plan)
    return plan


def _validate_plan(model, plan):
    """Reject a plan that cannot be realised as a labeling mixture."""
    for pool, members in plan.items():
        if pool in model.metabolites:
            raise ValueError(
                f"pool id {pool!r} is already a metabolite; it cannot also name the "
                f"whole-cell pool of {members}")
        patterns = {model.metabolites.get_by_id(m).annotation["atom_pattern"]
                    for m in members}
        if len({len(p) for p in patterns}) > 1:
            raise ValueError(
                f"pool {pool!r}: members {members} disagree on carbon count "
                f"{sorted(len(p) for p in patterns)}; they cannot be one measured pool")
        symmetries = {tuple(model.metabolites.get_by_id(m).annotation.get("symmetries", ()))
                      for m in members}
        if len(symmetries) > 1:
            raise ValueError(
                f"pool {pool!r}: members {members} disagree on symmetries {symmetries}")


def add_measurement_pools(model, plan=None):
    """Add each pool's metabolite, zero-stoichiometry mixing legs and pinned drain.

    The drain is pinned at 1 for two reasons: it normalises the mixing fluxes to
    fractions summing to one, and it keeps their sum away from zero -- the pool's EMU row
    has the sum of its inflows on the diagonal, so a zero sum would make it singular.
    :func:`~sigmaflux.pipeline.sample_flux.relax_measurement_pool_bounds` reopens the
    block once the samples that needed the simplex have been drawn.

    Args:
        model: cobra model, modified in place.
        plan: ``{pool_id: [member ids]}``; defaults to :func:`plan_measurement_pools`,
            which is what makes pooling an already-built model a one-call operation.

    Returns the ids of the reactions added.
    """
    plan = plan_measurement_pools(model) if plan is None else plan
    _reject_taken_prefixes(model)

    added = []
    for pool_id, members in plan.items():
        first = model.metabolites.get_by_id(members[0])
        pattern = first.annotation["atom_pattern"]

        pool = cobra.Metabolite(pool_id, compartment="")
        pool.name = f"{first.name} (whole-cell pool)" if first.name else pool_id
        # Deliberately not carrying the members' `symmetries`: a mixture of
        # already-symmetrised MIDs is symmetric, and re-declaring it would make the
        # x3cflux converter split every mixing leg into half-flux orientation pairs.
        pool.annotation = {"atom_pattern": pattern, POOL_MEMBERS_KEY: list(members)}
        model.add_metabolites([pool])

        for member in members:
            rxn = cobra.Reaction(f"{POOL_PREFIX}{member}")
            rxn.lower_bound, rxn.upper_bound = 0.0, 1.0
            model.add_reactions([rxn])
            # The member is deliberately absent from the stoichiometry: the pool
            # observes it without consuming it, and reaches the EMU network through
            # zero_stoichiometry instead.
            rxn.add_metabolites({pool: 1.0})
            rxn.annotation = {
                "atom_mapping": f"{member} ({pattern}) -> {pool_id} ({pattern})",
                "zero_stoichiometry": [member],
            }
            added.append(rxn.id)

        drain = cobra.Reaction(f"{POOL_DRAIN_PREFIX}{pool_id}")
        drain.lower_bound, drain.upper_bound = 1.0, 1.0
        model.add_reactions([drain])
        drain.add_metabolites({pool: -1.0})
        drain.annotation = {"atom_mapping": f" {pool_id} ({pattern}) -> "}
        added.append(drain.id)
    return added


def _reject_taken_prefixes(model):
    """Fail if the model already uses the pool prefixes for something else.

    ``POOL_`` is the marker the whole pipeline reads, so a metabolic reaction wearing it
    would be silently reclassified as observation machinery -- skipped by knockout
    screening, dropped from scoring, and rewritten by the x3cflux converter.
    """
    taken = [r.id for r in model.reactions if is_pool_reaction(r)]
    if taken:
        raise ValueError(
            f"{len(taken)} reaction(s) already use the measurement-pool prefix "
            f"{POOL_PREFIX!r}: {taken[:10]}. The prefix is the marker for the pool "
            f"block, so it cannot also name metabolism.")


def warn_unproducible_members(model, fva, tol=1e-6):
    """Warn about pool members the network cannot produce, and return their ids.

    Such a member has only zeros on the diagonal of its EMU row, so its MID comes back
    zeroed and the pool it feeds silently stops summing to one. The condition predates
    pooling -- the member's MID was already undefined -- but the pool is what makes it
    observable, so it is worth naming once FVA has run rather than only per sample.

    Args:
        model: the prepared cobra model, carrying the pool annotations.
        fva: the flux-variability frame from :func:`~sigmaflux.pipeline.sample_flux.constrain_by_fva`,
            reused rather than recomputed.
        tol: production reach below this counts as unproducible.
    """
    dead = []
    for members in pool_members(model).values():
        for member in members:
            met = model.metabolites.get_by_id(member)
            reach = 0.0
            for rxn in met.reactions:
                if is_pool_reaction(rxn) or rxn.id not in fva.index:
                    continue
                coefficient = rxn.metabolites[met]
                if coefficient > 0:
                    reach = max(reach, fva.loc[rxn.id, "maximum"])
                else:
                    reach = max(reach, -fva.loc[rxn.id, "minimum"])
            if reach < tol:
                dead.append(member)
    if dead:
        warnings.warn(
            f"{len(dead)} measurement-pool member(s) cannot be produced by the network, "
            f"so their MID is undefined and the pool they feed will not sum to one: "
            f"{sorted(dead)}", RuntimeWarning, stacklevel=2)
    return sorted(dead)


def degenerate_pool_members(model, mid_of, tol=1e-6):
    """Pool members whose simulated MID is degenerate at this flux state, if any.

    A metabolite nothing produces has an all-zero EMU balance row, so ``pinv`` returns a
    zeroed MID rather than raising. That is not a number the pool can mix: weighted by
    its still-nonzero mixing flux it drags the pool's labeling down, and because M+0 is
    never stored the deficit hides there instead of showing up as an invalid MID.
    Measured on I. orientalis, ``ala__L_m`` is unproducible in 3.3% of sampled flux
    states and ``adp_m`` in 0.2%.

    The condition is a property of the flux state, not of the model -- the same
    metabolite is produced normally in most samples -- so it has to be detected per
    sample rather than screened out at build time.

    Args:
        model: prepared cobra model carrying the pool annotations.
        mid_of: ``met_id -> MID array`` over M+0..M+n for this flux state.
        tol: how far the MID may fall short of summing to one.

    Returns the members whose degeneracy corrupts the mixture they feed.
    """
    corrupting = []
    for members in pool_members(model).values():
        for member in members:
            mid = mid_of.get(member)
            if mid is not None and abs(float(sum(mid)) - 1.0) > tol:
                corrupting.append(member)
    return corrupting
