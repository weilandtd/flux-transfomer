"""Pre-flight validation of a NetworkData, because x3cflux ships none.

x3cflux has no model checker. ``create_simulator_from_data`` assumes a set of
invariants about a NetworkData and, when one is violated, reads out of bounds and
dies with SIGSEGV -- no Python exception, no log line, nothing to catch. The only
checks it does expose (``is_feasible``, ``compute_stoichiometry_error``,
``contains``) live on the parameter space, which only exists once the simulator
has already been built successfully.

So these are the invariants, enforced up front where a violation names the
offending reaction instead of killing the interpreter. ``CBPS2_m`` was the case
that motivated this: 2 atp_m / 2 adp_m in the stoichiometry against one mapped
copy each, so 10 atoms per side sat outside an 11-index permutation. (That same
reaction turned out to be mis-weighted in our own EMU stoichiometry -- see
``tests/test_emu_stoichiometry.py``.)
"""


class NetworkCheckError(AssertionError):
    """A NetworkData violates an invariant x3cflux would segfault on."""


def check_network(nd, verbose=True):
    """Return a list of problem strings; empty means the network is well-formed."""
    problems = []
    atoms = {m.name: m.num_atoms for m in nd.metabolites}

    dupes = _duplicates([m.name for m in nd.metabolites])
    if dupes:
        problems.append(f"duplicate metabolite names: {sorted(dupes)[:5]}")
    dupes = _duplicates([r.name for r in nd.reactions])
    if dupes:
        problems.append(f"duplicate reaction names: {sorted(dupes)[:5]}")

    for r in nd.reactions:
        unknown = [m for m in list(r.educt_names) + list(r.product_names)
                   if m not in atoms]
        if unknown:
            problems.append(f"{r.name}: undeclared metabolites {unknown}")
            continue

        n_e = sum(atoms[m] for m in r.educt_names)
        n_p = sum(atoms[m] for m in r.product_names)

        # A one-sided reaction has nothing to permute, so only a side that
        # actually carries atoms has to match num_atoms. MDH_c is one-sided in
        # this sense: it has products, but they are the 0-atom nad/nadh/h pools.
        if n_e and n_e != r.num_atoms:
            problems.append(f"{r.name}: num_atoms={r.num_atoms} but educt_names "
                            f"carry {n_e} atoms")
        if n_p and n_p != r.num_atoms:
            problems.append(f"{r.name}: num_atoms={r.num_atoms} but product_names "
                            f"carry {n_p} atoms")

        perm = r.atom_permutation
        if perm.num_indices != r.num_atoms:
            problems.append(f"{r.name}: permutation has {perm.num_indices} indices "
                            f"for num_atoms={r.num_atoms}")
            continue
        images = [perm(i) for i in range(r.num_atoms)]
        if any(not 0 <= v < r.num_atoms for v in images):
            problems.append(f"{r.name}: permutation image out of range")
        elif len(set(images)) != len(images):
            problems.append(f"{r.name}: permutation is not injective "
                            f"({len(images) - len(set(images))} collisions)")

    # A pool that is never produced and never consumed makes the network
    # degenerate, which is another documented way to segfault.
    touched = set()
    for r in nd.reactions:
        touched.update(r.educt_names)
        touched.update(r.product_names)
    orphans = [m.name for m in nd.metabolites if m.name not in touched]
    if orphans:
        problems.append(f"{len(orphans)} metabolites in no reaction: {orphans[:5]}")

    if verbose:
        if problems:
            print(f"check_network: {len(problems)} PROBLEMS")
            for p in problems[:20]:
                print(f"   - {p}")
            if len(problems) > 20:
                print(f"   ... and {len(problems) - 20} more")
        else:
            print(f"check_network: OK ({len(nd.metabolites)} metabolites, "
                  f"{len(nd.reactions)} reactions)")
    return problems


def assert_network(nd, verbose=True):
    """Raise :class:`NetworkCheckError` unless *nd* is well-formed.

    Always call this before ``create_simulator_from_data``: the alternative to an
    exception here is not a worse error message, it is a killed process.
    """
    problems = check_network(nd, verbose=verbose)
    if problems:
        raise NetworkCheckError(
            f"{len(problems)} NetworkData problem(s) x3cflux would segfault on:\n  "
            + "\n  ".join(problems[:20]))


def _duplicates(names):
    seen, dupes = set(), set()
    for n in names:
        (dupes if n in seen else seen).add(n)
    return dupes
