"""Substrates, constraints and measurements -> one MeasurementConfiguration per tracer.

The constraint set is what restores our flux space inside x3cflux. x3cflux derives
S automatically from the declared pools, which gets us the cobra balances wherever
the coefficients are +-1 (and, via name repetition, wherever they are integral).
Three groups of equality constraints repair the rest:

  * **tie constraints** put back the reactions that :mod:`convert` had to split, at
    their exact cobra ratios, so each parent keeps one degree of freedom;
  * **input source pools** move each ConstantSubstrate one step upstream, so the
    input pools themselves stay balanced and keep their cobra rows;
  * **exchange ties** keep the two halves of a symmetry-split reversible reaction
    carrying equal exchange flux.

Together these must bring the net-flux nullity back to 25, matching cobra's. If it
comes out at 50 the non-carbon balances are missing; if it is 25 + k, k constraints
did not land.

``LinearConstraint`` semantics are ``constant + sum(c_i * p_i) {=,<=} 0`` -- note
this contradicts the class docstring, which puts the constant on the other side;
the sign here is the empirically verified one.
"""

import numpy as np
import x3cflux as x

CARBON = x.TracerElement.Carbon


def tracer_profile(labeling, n_atoms, enrichment):
    """ConstantSubstrate profile dict for one tracer.

    Profiles are not normalised by x3cflux, so the complete distribution must be
    given; the bitset string's leftmost character is atom C1 (FluxML ``cfg=`` convention).
    """
    pattern = "".join("1" if b else "0" for b in labeling)
    if len(pattern) != n_atoms:
        raise ValueError(f"tracer pattern {pattern} does not fit {n_atoms} atoms")
    if enrichment >= 1.0:
        return {pattern: 1.0}
    return {pattern: enrichment, "0" * n_atoms: 1.0 - enrichment}


def build_substrates(facts, conv, tracer):
    """One ConstantSubstrate per EMU network input, for a single tracer.

    Attaches each substrate to the input's upstream *source* pool, not the input pool
    itself (see :func:`convert.input_source_pools`); iterates ``conv.sources`` rather
    than ``facts["input_metabolites"]`` to match exactly the pools ``convert`` relies
    on already existing.
    """
    atoms = {m["id"]: len(m["atom_pattern"]) for m in facts["metabolites"]}
    subs = []
    for met, pool in conv.sources.items():
        n = atoms[met]
        if met == tracer["metabolite"]:
            profiles = tracer_profile(tracer["labeling"], n, tracer["enrichment"])
        else:
            profiles = {"0" * n: 1.0}          # unlabeled medium component
        subs.append(x.ConstantSubstrate(f"sub_{met}", pool, 0.0, profiles))
    # Unlabeled feeds for the biomass split's from-nothing influxes. They only
    # supply inert 1001-block carbon (amp_c), which is M+0 regardless.
    for src, n in conv.extra_sources.items():
        subs.append(x.ConstantSubstrate(f"sub_{src}", src, 0.0, {"0" * n: 1.0}))
    return subs


def build_constraints(facts, conv):
    """``(net_flux_constraints, exchange_flux_constraints)``.

    Returns ParameterConstraints objects; the parameter names inside carry no
    ``.n``/``.x`` suffix because the two collections are already type-scoped.
    """
    net_eq, xch_eq, xch_def = [], [], []

    # 1. Tie the pieces of every split/symmetry-halved reaction back together. For
    #    subs (s_i, c_i) of one parent, u_i / c_i is the same for all i, so
    #    u_0/c_0 - u_i/c_i = 0 pins everything to one degree of freedom.
    for parent, subs in conv.parents.items():
        (s0, c0) = subs[0]
        for (si, ci) in subs[1:]:
            net_eq.append(x.LinearConstraint(
                f"tie_{parent}_{si}", [s0, si], [1.0 / c0, -1.0 / ci], 0.0))

    # 2. Nothing to restore for the input pools: the substrate now sits on their
    #    upstream __src pool, so the input pools themselves stay balanced by
    #    x3cflux and keep their cobra rows. Only the __src pools are dropped from
    #    S, and they have no cobra row to lose.

    # 3. Exchange fluxes. bidirectional=False still creates a free .x that has no
    #    effect on the MIDs, so pin those to zero or they enter Ipopt as
    #    structurally non-identifiable parameters.
    for rxn, rev in conv.reversible.items():
        if rxn in conv.parents:          # parents are not x3cflux reactions
            continue
        if not rev:
            xch_def.append(x.DefinitionConstraint(f"xch0_{rxn}", rxn, 0.0))

    # 4. Both halves of a symmetry-split reversible reaction carry equal exchange.
    for parent, subs in conv.parents.items():
        if not conv.reversible.get(subs[0][0], False):
            continue
        (s0, _) = subs[0]
        for (si, _) in subs[1:]:
            xch_eq.append(x.LinearConstraint(
                f"xtie_{parent}_{si}", [s0, si], [1.0, -1.0], 0.0))

    return (x.ParameterConstraints([], net_eq, []),
            x.ParameterConstraints(xch_def, xch_eq, []))


def build_measurements(facts, conv, labeling_index, measured_flux_ids, noisy_labeling,
                       noisy_flux, flux_sigma, tracer_key, noise_std,
                       flux_sigma_scale=1.0):
    """MSMeasurements for the observable metabolites + the flux measurements.

    Only M+1..M+n are included (M+0 is never scored); ``flux_sigma_scale`` inflates
    the flux sigmas so x3cflux's raw SSR stays proportional to our reduced objective.
    """
    atoms = {m["id"]: len(m["atom_pattern"]) for m in facts["metabolites"]}
    rows = {}
    for (trc, met, iso), value in zip(labeling_index, noisy_labeling):
        if trc == tracer_key:
            rows.setdefault(met, {})[int(iso[2:])] = float(value)

    meas = []
    for met in sorted(rows):
        n = atoms[met]
        weights = sorted(rows[met])
        if weights != list(range(1, n + 1)):
            raise ValueError(f"{met}: weights {weights} != M+1..M+{n}")
        values = np.array([rows[met][w] for w in weights], dtype=float)
        data = x.MeasurementDataSet([0.0], [values],
                                    [np.full(n, noise_std, dtype=float)], [])
        meas.append(x.MSMeasurement(f"ms_{met}", False, met, n,
                                    x.MSSpecification("1" * n, weights), data))

    for rxn, value, sigma in zip(measured_flux_ids, noisy_flux, flux_sigma):
        names, formula = flux_expression(conv, rxn)
        meas.append(x.FluxMeasurement(
            f"flux_{rxn}", False, _expr(formula), x.FluxSpecification(names, True),
            float(value), float(sigma) * flux_sigma_scale, _expr("1")))
    return meas


def flux_expression(conv, rxn_id):
    """``(flux names, formula)`` giving the *cobra* net flux of ``rxn_id``.

    Rewrites split, flipped, and symmetry-halved reactions back into their cobra
    terms, undoing what :mod:`convert` did, so the formula matches the measured flux.
    """
    subs = conv.parents.get(rxn_id)
    scale = conv.scale[rxn_id]
    if subs is None:
        names = [rxn_id]
        formula = rxn_id if scale == 1 else f"({rxn_id})/({scale!r})"
    elif subs[0][0].endswith("__sym0"):
        # Halves sum back to the parent flux.
        names = [s for s, _ in subs]
        total = " + ".join(names)
        formula = total if scale == 1 else f"({total})/({scale!r})"
    else:
        # Every sub is parent_flux * coefficient; read it off the first one.
        s0, c0 = subs[0]
        names = [s0]
        formula = f"({s0})/({(c0 * scale)!r})"
    return names, formula


def _expr(s):
    import sympy
    return sympy.parse_expr(s)


def structural_configuration(facts, conv):
    """One measurement-free configuration, for inspecting the flux space alone.

    ``create_simulator_from_data`` rejects an empty configuration list, so this
    supplies one configuration with no measurements instead.
    """
    import x3cflux as x

    net_c, xch_c = build_constraints(facts, conv)
    tracer = facts["tracers"][0]
    return x.MeasurementConfiguration(
        tracer["key"], "", True, build_substrates(facts, conv, tracer), [],
        net_c, xch_c, x.ParameterConstraints([], [], []), [])


def build_configurations(facts, conv, labeling_index, measured_flux_ids, noisy_labeling,
                         noisy_flux, flux_sigma, noise_std, flux_sigma_scale=1.0):
    """One MeasurementConfiguration per tracer, sharing network and constraints."""
    net_c, xch_c = build_constraints(facts, conv)
    empty = x.ParameterConstraints([], [], [])
    configs = []
    for tracer in facts["tracers"]:
        meas = build_measurements(
            facts, conv, labeling_index, measured_flux_ids, noisy_labeling,
            noisy_flux, flux_sigma, tracer["key"], noise_std, flux_sigma_scale)
        configs.append(x.MeasurementConfiguration(
            tracer["key"], "", True, build_substrates(facts, conv, tracer), meas,
            net_c, xch_c, empty, []))
    return configs
