"""Boundary-decomposition null-space basis.

Splits null(S) into:
  - throughput block: columns visible at the exchange boundary (E v ≠ 0), each
    realising the sparsest physiological flux for one external-exchange coordinate.
  - cycle block: columns invisible at the boundary (E v = 0), kept dense/orthonormal.

Public entry point: compute_boundary_basis(model_or_S, ...)
"""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
import scipy.linalg
import scipy.sparse

try:
    import cobra
    from cobra.util.array import create_stoichiometric_matrix
    from cobra.util.solver import check_solver
    _COBRA_AVAILABLE = True
except ImportError:
    _COBRA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_boundary_basis(
    model_or_S: Union["cobra.Model", tuple],
    solver: str | None = None,
    tol: float = 1e-8,
    milp_time_limit: float = 120.0,
) -> tuple[np.ndarray, dict]:
    """Compute a boundary-decomposed null-space basis.

    Parameters
    ----------
    model_or_S:
        Either a COBRApy Model or a 3-tuple (S, reversibility, external_ids) where
        - S             ndarray (m, n) stoichiometric matrix
        - reversibility ndarray (n,) bool, True = reversible
        - external_ids  ndarray/list (n,) bool or str; truthy = external reaction
    solver:
        'glpk' | 'cplex' | 'gurobi'.  None → infer from cobra model or 'glpk'.
    tol:
        Numerical tolerance for rank decisions and feasibility checks.
    milp_time_limit:
        Per-coordinate MILP time limit in seconds (best-effort).

    Returns
    -------
    Z      : ndarray (n, k)  null-space basis, columns [throughput | cycle]
    status : dict            metadata
    """
    S, rev, ext_mask, rxn_ids, solver_name = _parse_input(model_or_S, solver)

    S_aug, rev_aug, ext_mask_aug, ext_ids_aug, split_map, orig_to_aug = (
        normalize_externals(S, rev, ext_mask, rxn_ids)
    )

    C, q = compute_cycle_block(S_aug, ext_mask_aug, tol=tol)

    b_targets, coord_meta = select_throughput_coordinates(
        S_aug, ext_mask_aug, ext_ids_aug, tol=tol
    )
    p = len(b_targets)

    iface = _get_solver_interface(solver_name)

    throughput_vecs = []
    for b_i, meta in zip(b_targets, coord_meta):
        big_m = _per_conversion_bigm(S_aug, b_i)
        v, fwd_feasible, level_penalties = solve_sparse_column(
            S_aug, b_i, rev_aug, ext_mask_aug, ext_ids_aug,
            iface, solver_name, big_m, tol=tol, time_limit=milp_time_limit,
        )
        meta["forward_feasible"] = fwd_feasible
        meta["level_penalties"] = level_penalties
        meta["support"] = [
            ext_ids_aug[j] for j in range(len(ext_ids_aug)) if abs(v[j]) > tol
        ]
        throughput_vecs.append(v)

    Z_aug, validation = assemble_and_validate(throughput_vecs, C, S_aug, tol=tol)

    # Unaugment the two blocks separately; rank-reduce the throughput block.
    # The augmented system has extra throughput modes (from split-external pivots)
    # that map to existing directions in null(S) after unaugmentation.  We keep
    # only the linearly independent subset, selected by QR with column pivoting.
    n_orig = S.shape[1]
    T_aug = Z_aug[:, :p]
    C_aug = Z_aug[:, p:]

    T_full = _unaugment_block(T_aug, n_orig, split_map, orig_to_aug)
    C_full = _unaugment_block(C_aug, n_orig, split_map, orig_to_aug)

    # Drop zero throughput columns first (pure split-external cycle modes).
    T_col_norms = np.linalg.norm(T_full, axis=0)
    T_nonzero = T_col_norms > tol
    T_nz = T_full[:, T_nonzero]
    meta_nz = [m for m, keep in zip(coord_meta, T_nonzero) if keep]

    # Rank-reduce: select linearly independent columns via QR pivot.
    if T_nz.shape[1] > 0:
        p_orig = int(np.linalg.matrix_rank(T_nz, tol=tol * 100))
        if p_orig < T_nz.shape[1]:
            _, _, perm = scipy.linalg.qr(T_nz, pivoting=True)
            T_nz = T_nz[:, perm[:p_orig]]
            meta_nz = [meta_nz[perm[i]] for i in range(p_orig)]
    Z_T = T_nz
    coord_meta_final = meta_nz

    # Cycle block: drop near-zero columns (numerical debris).
    C_keep = np.linalg.norm(C_full, axis=0) > tol
    Z_C = C_full[:, C_keep]

    parts = [arr for arr in [Z_T, Z_C] if arr.shape[1] > 0]
    Z = np.hstack(parts) if parts else np.zeros((n_orig, 0))

    # Normalize columns to unit norm — controls loss scale and gradient magnitude
    # during training (mirrors compute_targeted_nullspace / compute_ica_nullspace).
    if Z.shape[1] > 0:
        col_norms = np.linalg.norm(Z, axis=0)
        col_norms[col_norms < 1e-14] = 1.0
        Z = Z / col_norms
        # Sign convention: largest-|value| element positive per column.
        for j in range(Z.shape[1]):
            if Z[np.argmax(np.abs(Z[:, j])), j] < 0:
                Z[:, j] *= -1

    n_throughput = int(Z_T.shape[1])
    n_cycle = int(Z_C.shape[1])

    status = {
        "n_throughput":    n_throughput,
        "n_cycle":         n_cycle,
        "throughput_meta": coord_meta_final,
        "cycle_dim":       n_cycle,
        "solver_used":     solver_name,
        "validation":      validation,
        "fallback":        False,
    }
    return Z, status


# ---------------------------------------------------------------------------
# Step 0 — normalize externals
# ---------------------------------------------------------------------------

def normalize_externals(
    S: np.ndarray,
    reversibility: np.ndarray,
    ext_mask: np.ndarray,
    rxn_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict, dict]:
    """Replace each reversible external reaction with two irreversible legs.

    Each reversible external j is removed and replaced by:
      j_fwd (lb=0): same stoichiometry as column j   (e.g. uptake direction)
      j_rev (lb=0): negated stoichiometry             (e.g. secretion direction)

    Irreversible reactions (external or internal) are passed through unchanged.

    Parameters
    ----------
    S:
        Stoichiometric matrix (metabolites x reactions).
    reversibility:
        Per-reaction reversibility flags matching S columns.
    ext_mask:
        Per-reaction external/exchange flags matching S columns.
    rxn_ids:
        Reaction identifiers matching S columns.

    Returns
    -------
    S_aug        : (m, n_aug)
    rev_aug      : (n_aug,) bool — all False for externals
    ext_mask_aug : (n_aug,) bool
    ext_ids_aug  : list[str] length n_aug
    split_map    : {orig_idx -> (aug_fwd_idx, aug_rev_idx)}
    orig_to_aug  : {orig_idx -> aug_idx}  for non-split reactions
    """
    m, n = S.shape
    rev_ext_set = {j for j in range(n) if ext_mask[j] and reversibility[j]}

    cols, rev_out, ext_out, ids_out = [], [], [], []
    split_map: dict = {}
    orig_to_aug: dict = {}
    aug = 0

    for j in range(n):
        if j in rev_ext_set:
            split_map[j] = (aug, aug + 1)
            cols.append(S[:, j].copy())        # fwd leg (lb=0 in MILP)
            rev_out.append(False)
            ext_out.append(True)
            ids_out.append(rxn_ids[j] + "_fwd")
            cols.append(-S[:, j].copy())       # rev leg (lb=0 in MILP)
            rev_out.append(False)
            ext_out.append(True)
            ids_out.append(rxn_ids[j] + "_rev")
            aug += 2
        else:
            orig_to_aug[j] = aug
            cols.append(S[:, j].copy())
            rev_out.append(bool(reversibility[j]))
            ext_out.append(bool(ext_mask[j]))
            ids_out.append(rxn_ids[j])
            aug += 1

    S_aug = np.column_stack(cols).astype(float)
    return (
        S_aug,
        np.array(rev_out, dtype=bool),
        np.array(ext_out, dtype=bool),
        ids_out,
        split_map,
        orig_to_aug,
    )


# ---------------------------------------------------------------------------
# Step 1 — cycle block
# ---------------------------------------------------------------------------

def compute_cycle_block(
    S_aug: np.ndarray,
    ext_mask_aug: np.ndarray,
    tol: float = 1e-8,
) -> tuple[np.ndarray, int]:
    """Orthonormal basis of {v ∈ null(S_aug) : v_external = 0}.

    Parameters
    ----------
    S_aug:
        Augmented stoichiometric matrix (reversible externals split into legs).
    ext_mask_aug:
        Per-column external/exchange flags matching S_aug.
    tol:
        Numerical tolerance for null-space rank decisions.

    Returns (C, q) where C is (n_aug, q).
    """
    N = scipy.linalg.null_space(S_aug, rcond=tol)     # (n_aug, k0)
    ext_rows = np.where(ext_mask_aug)[0]

    if N.shape[1] == 0:
        return np.zeros((S_aug.shape[1], 0)), 0

    if len(ext_rows) == 0:
        return N, N.shape[1]

    A = N[ext_rows, :]                                  # (d, k0)
    null_A = scipy.linalg.null_space(A, rcond=tol)     # (k0, q)

    if null_A.shape[1] == 0:
        return np.zeros((S_aug.shape[1], 0)), 0

    C_raw = N @ null_A                                  # (n_aug, q)
    C, _ = np.linalg.qr(C_raw, mode='reduced')
    return C, C.shape[1]


# ---------------------------------------------------------------------------
# Step 2 — throughput coordinate selection
# ---------------------------------------------------------------------------

def select_throughput_coordinates(
    S_aug: np.ndarray,
    ext_mask_aug: np.ndarray,
    ext_ids_aug: list[str],
    tol: float = 1e-8,
) -> tuple[list[np.ndarray], list[dict]]:
    """Choose p independent exchange-coordinate targets by rank-revealing QR on A = E N.
    Secretion externals (column sum < 0, i.e. running forward removes metabolite from
    the system) are ordered before uptake externals so product-conversion modes become
    the primary coordinates.

    Parameters
    ----------
    S_aug:
        Augmented stoichiometric matrix (reversible externals split into legs).
    ext_mask_aug:
        Per-column external/exchange flags matching S_aug.
    ext_ids_aug:
        Reaction identifiers matching S_aug columns.
    tol:
        Numerical tolerance for rank decisions.

    Returns
    -------
    b_targets   : list of p vectors in R^d (target exchange bundles, d = #externals)
    coord_meta  : list of p dicts with 'external_id' and 'kind'
    """
    N = scipy.linalg.null_space(S_aug, rcond=tol)
    ext_rows = np.where(ext_mask_aug)[0]
    d = len(ext_rows)

    if d == 0 or N.shape[1] == 0:
        return [], []

    A = N[ext_rows, :]                          # (d, k0)
    p = int(np.linalg.matrix_rank(A, tol=tol))
    if p == 0:
        return [], []

    # Secretion: running the reaction at v>0 removes metabolite (column sum < 0).
    secretion_mask = np.array([
        np.sum(S_aug[:, idx]) < 0 for idx in ext_rows
    ])

    # Order: secretions first, then uptakes.
    order = np.concatenate([
        np.where(secretion_mask)[0],
        np.where(~secretion_mask)[0],
    ])
    A_ordered   = A[order, :]
    ext_ordered = ext_rows[order]
    ids_ordered = [ext_ids_aug[i] for i in ext_ordered]
    sec_ordered = secretion_mask[order]

    # Rank-revealing QR on A_ordered.T → pivots select external rows of A.
    _, _, pivot = scipy.linalg.qr(A_ordered.T, pivoting=True)
    pivot_p = pivot[:p]

    A_pivot = A_ordered[pivot_p, :]    # (p, k0)

    b_targets: list[np.ndarray] = []
    coord_meta: list[dict] = []
    for rank_i, row_i in enumerate(pivot_p):
        e_i = np.zeros(p)
        e_i[rank_i] = 1.0
        y, _, _, _ = np.linalg.lstsq(A_pivot, e_i, rcond=None)
        b_full = A_ordered @ y          # (d,) in ordered space

        # Repair any negative legs of split-external pseudo-reactions.
        b_full, repaired = _repair_negative_legs(b_full, ids_ordered)

        # Map back to original d-vector (undo secretion-first sort).
        b_orig = np.zeros(d)
        b_orig[order] = b_full
        b_targets.append(b_orig)

        kind = "production" if sec_ordered[row_i] else "substitution"
        coord_meta.append({
            "external_id":    ids_ordered[row_i],
            "kind":           kind,
            "_b_neg_repair":  repaired,
        })

    return b_targets, coord_meta


def _repair_negative_legs(b: np.ndarray, ids: list[str]) -> tuple[np.ndarray, bool]:
    """If a split-external leg has b < 0, transfer it to the mirror leg (fwd↔rev).

    Parameters
    ----------
    b:
        Target exchange bundle over the augmented external columns.
    ids:
        Reaction identifiers matching b, used to find each leg's mirror.
    """
    b = b.copy()
    repaired = False
    for i, (val, rid) in enumerate(zip(b, ids)):
        if val < -1e-10:
            mirror = rid[:-3] + "rev" if rid.endswith("fwd") else rid[:-3] + "fwd"
            if mirror in ids:
                j = ids.index(mirror)
                b[j] -= val    # move magnitude to mirror
                b[i] = 0.0
                repaired = True
    return b, repaired


# ---------------------------------------------------------------------------
# Step 3 — sparse canonical column via staged lexicographic MILP
# ---------------------------------------------------------------------------

def solve_sparse_column(
    S_aug: np.ndarray,
    b_i: np.ndarray,
    reversibility_aug: np.ndarray,
    ext_mask_aug: np.ndarray,
    ext_ids_aug: list[str],
    solver_iface,
    solver_name: str,
    big_m: float,
    tol: float = 1e-8,
    time_limit: float = 120.0,
) -> tuple[np.ndarray, bool, list[float]]:
    """Find the sparsest null-space flux realising exchange target b_i.

    Parameters
    ----------
    S_aug:
        Augmented stoichiometric matrix (reversible externals split into legs).
    b_i:
        Target exchange bundle to realise, over the augmented external columns.
    reversibility_aug:
        Per-column reversibility flags matching S_aug.
    ext_mask_aug:
        Per-column external/exchange flags matching S_aug.
    ext_ids_aug:
        Reaction identifiers matching S_aug columns.
    solver_iface:
        optlang-style solver interface (Model/Variable/Constraint/Objective).
    solver_name:
        Name of the backing solver ('glpk' | 'cplex' | 'gurobi').
    big_m:
        Big-M bound for the indicator/support constraints.
    tol:
        Numerical tolerance for feasibility and freezing constraints.
    time_limit:
        Per-stage MILP time limit in seconds (best-effort).

    Returns (v, forward_feasible, level_penalties).
    Level penalties: [stage1_directionality, stage2_cardinality, stage3_weighted_flux].
    """
    m, n_aug = S_aug.shape
    ext_rows = np.where(ext_mask_aug)[0]
    irrev_internal = np.where(~reversibility_aug & ~ext_mask_aug)[0]

    has_indicators = solver_name in ("cplex", "gurobi")

    lp = solver_iface.Model(name="sparse_col")
    _set_time_limit(lp, time_limit)

    # ── Variables ──────────────────────────────────────────────────────────
    # All flux variables are relaxed to [-big_m, big_m].  Irreversibility is
    # enforced as a soft stage-1 penalty (backward flux cost), not a hard lb=0.
    # This ensures the MILP is always feasible even when the exchange target b_i
    # would require reverse flow through an irreversible reaction — which is then
    # correctly flagged as forward_infeasible rather than causing solver failure.
    v_vars = [solver_iface.Variable(f"v_{j}", lb=-big_m, ub=big_m)
              for j in range(n_aug)]

    # Binary support indicators.
    z_vars = [solver_iface.Variable(f"z_{j}", lb=0, ub=1, type="binary")
              for j in range(n_aug)]

    # Absolute-value auxiliaries (for stage 3 weighted-flux objective).
    vabs_vars = [solver_iface.Variable(f"vabs_{j}", lb=0.0, ub=big_m)
                 for j in range(n_aug)]

    # Backward-flux auxiliaries for ALL irreversible reactions (including externals).
    # v_neg_j = max(0, -v_j) via: v_neg_j ≥ 0, v_j + v_neg_j ≥ 0.
    irrev_all = np.where(~reversibility_aug)[0]
    v_neg = {j: solver_iface.Variable(f"vneg_{j}", lb=0.0, ub=big_m)
             for j in irrev_all}

    all_vars = v_vars + z_vars + vabs_vars + list(v_neg.values())
    lp.add(all_vars)

    constraints = []

    # ── Mass balance: S_aug @ v = 0 ───────────────────────────────────────
    for i in range(m):
        nz = [(j, float(S_aug[i, j])) for j in range(n_aug) if S_aug[i, j] != 0]
        if not nz:
            continue
        expr = sum(c * v_vars[j] for j, c in nz)
        constraints.append(
            solver_iface.Constraint(expr, lb=0.0, ub=0.0, name=f"mb_{i}")
        )

    # ── External target: v[ext_rows[i]] = b_i[i] ─────────────────────────
    for rank_i, row_idx in enumerate(ext_rows):
        val = float(b_i[rank_i])
        constraints.append(
            solver_iface.Constraint(
                v_vars[row_idx], lb=val, ub=val, name=f"ext_{rank_i}"
            )
        )

    # ── Support coupling ───────────────────────────────────────────────────
    for j in range(n_aug):
        if has_indicators:
            constraints.append(solver_iface.Constraint(
                v_vars[j], indicator_variable=z_vars[j],
                active_when=0, lb=0.0, ub=0.0, name=f"ind_{j}"
            ))
        else:
            # v_j ≤ big_m * z_j
            constraints.append(solver_iface.Constraint(
                v_vars[j] - big_m * z_vars[j], ub=0.0, name=f"bm_ub_{j}"
            ))
            # -v_j ≤ big_m * z_j  (needed for reversible / relaxed-lb reactions)
            if not ext_mask_aug[j]:
                constraints.append(solver_iface.Constraint(
                    -v_vars[j] - big_m * z_vars[j], ub=0.0, name=f"bm_lb_{j}"
                ))

    # ── |v_j| linearisation: vabs_j ≥ v_j and vabs_j ≥ -v_j ─────────────
    for j in range(n_aug):
        constraints.append(solver_iface.Constraint(
            vabs_vars[j] - v_vars[j], lb=0.0, name=f"abs_u_{j}"
        ))
        constraints.append(solver_iface.Constraint(
            vabs_vars[j] + v_vars[j], lb=0.0, name=f"abs_l_{j}"
        ))

    # ── Stage-1 backward-flux auxiliaries ─────────────────────────────────
    for j, vn in v_neg.items():
        # v_neg_j ≥ -v_j  →  v_j + v_neg_j ≥ 0
        constraints.append(solver_iface.Constraint(
            v_vars[j] + vn, lb=0.0, name=f"neg_{j}"
        ))

    lp.add(constraints)

    # ── Stage 1: minimize backward use of irreversible internal reactions ──
    if v_neg:
        obj1_expr = sum(v_neg.values())
    else:
        _dummy = solver_iface.Variable("_s1_zero", lb=0.0, ub=0.0)
        lp.add([_dummy])
        obj1_expr = _dummy
    opt1 = _optimize_and_freeze(lp, solver_iface, obj1_expr, "frz_s1", tol)
    level_penalties = [opt1]
    forward_feasible = (opt1 <= tol)

    # ── Stage 2: minimize support cardinality ─────────────────────────────
    obj2_expr = sum(z_vars)
    opt2 = _optimize_and_freeze(lp, solver_iface, obj2_expr, "frz_s2", tol)
    level_penalties.append(opt2)

    # ── Stage 3: minimize weighted flux (reaction-index tie-break) ────────
    obj3_expr = sum((j + 1) * vabs_vars[j] for j in range(n_aug))
    opt3 = _optimize_and_freeze(lp, solver_iface, obj3_expr, "frz_s3", tol)
    level_penalties.append(opt3)

    v_sol = np.array([v_vars[j].primal for j in range(n_aug)])
    return v_sol, forward_feasible, level_penalties


def _optimize_and_freeze(lp, iface, obj_expr, freeze_name: str, tol: float) -> float:
    """Minimize obj_expr, then freeze it at its optimum as a named constraint.

    Parameters
    ----------
    lp:
        Solver model being optimized.
    iface:
        optlang-style solver interface, used to build the freeze constraint.
    obj_expr:
        Expression to minimize.
    freeze_name:
        Name for the constraint that freezes obj_expr at its optimum.
    tol:
        Slack added to the frozen upper bound.
    """
    lp.objective = iface.Objective(obj_expr, direction="min")
    lp.optimize()
    val = lp.objective.value if lp.objective.value is not None else 0.0
    lp.add([iface.Constraint(obj_expr, ub=val + tol, name=freeze_name)])
    return float(val)


# ---------------------------------------------------------------------------
# Step 4 — assemble and validate
# ---------------------------------------------------------------------------

def assemble_and_validate(
    throughput_vecs: list[np.ndarray],
    cycle_basis: np.ndarray,
    S_aug: np.ndarray,
    tol: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """Stack columns, orthogonalise cycle block against throughput, validate.

    Parameters
    ----------
    throughput_vecs:
        Per-coordinate sparse flux vectors from solve_sparse_column.
    cycle_basis:
        Orthonormal cycle-block basis from compute_cycle_block.
    S_aug:
        Augmented stoichiometric matrix (reversible externals split into legs).
    tol:
        Numerical tolerance for orthogonalisation and validation checks.

    Returns (Z_aug, validation_report).
    """
    n_aug = S_aug.shape[1]
    p = len(throughput_vecs)
    q_aug = cycle_basis.shape[1] if cycle_basis.ndim == 2 else 0

    if p == 0 and q_aug == 0:
        raise ValueError("Empty basis: both throughput and cycle blocks are empty.")

    # Gram-Schmidt: orthogonalise cycle columns against fixed throughput columns.
    if p > 0 and q_aug > 0:
        T = np.column_stack(throughput_vecs)    # (n_aug, p)
        C = cycle_basis.copy()
        for i in range(q_aug):
            col = C[:, i]
            for k in range(p):
                t = T[:, k]
                col = col - (col @ t) / (t @ t + 1e-30) * t
            C[:, i] = col
        # Re-QR to restore orthonormality after projection.
        if np.any(np.abs(C) > tol):
            C, _ = np.linalg.qr(C, mode='reduced')
        cycle_basis = C

    parts = []
    if p > 0:
        parts.append(np.column_stack(throughput_vecs))
    if q_aug > 0:
        parts.append(cycle_basis)
    Z_aug = np.hstack(parts)

    k = Z_aug.shape[1]
    mb_res = float(np.linalg.norm(S_aug @ Z_aug))
    actual_rank = int(np.linalg.matrix_rank(Z_aug, tol=tol))
    expected_nullity_aug = S_aug.shape[1] - int(np.linalg.matrix_rank(S_aug, tol=tol))

    validation = {
        "mass_balance_residual": mb_res,
        "mass_balance_ok":       mb_res < tol * max(k, 1) * 10,
        "rank_aug":              actual_rank,
        "expected_nullity_aug":  expected_nullity_aug,
        "full_rank_aug_ok":      actual_rank == expected_nullity_aug,
    }
    if not validation["mass_balance_ok"]:
        warnings.warn(
            f"boundary_basis: mass-balance residual {mb_res:.2e} exceeds tolerance.",
            RuntimeWarning, stacklevel=3,
        )
    if not validation["full_rank_aug_ok"]:
        warnings.warn(
            f"boundary_basis: augmented rank {actual_rank} ≠ "
            f"augmented nullity {expected_nullity_aug}.",
            RuntimeWarning, stacklevel=3,
        )
    return Z_aug, validation


# ---------------------------------------------------------------------------
# Map augmented columns back to original n reactions
# ---------------------------------------------------------------------------

def _unaugment_block(
    Z_aug: np.ndarray,
    n_orig: int,
    split_map: dict,
    orig_to_aug: dict,
) -> np.ndarray:
    """Map a block of augmented columns back to the original n_orig reactions: for
    each split reversible external j_orig, Z[j_orig] = Z_aug[j_fwd] - Z_aug[j_rev].
    Split-external pure-cycle modes become all-zero columns (caller drops them).

    Parameters
    ----------
    Z_aug:
        Basis block in augmented-column space.
    n_orig:
        Number of original (pre-augmentation) reactions.
    split_map:
        {orig_idx -> (aug_fwd_idx, aug_rev_idx)} for split reversible externals.
    orig_to_aug:
        {orig_idx -> aug_idx} for non-split reactions.
    """
    k = Z_aug.shape[1]
    Z = np.zeros((n_orig, k))
    for j in range(n_orig):
        if j in split_map:
            j_fwd, j_rev = split_map[j]
            Z[j, :] = Z_aug[j_fwd, :] - Z_aug[j_rev, :]
        else:
            Z[j, :] = Z_aug[orig_to_aug[j], :]
    return Z


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_input(model_or_S, solver):
    """Return (S, rev, ext_mask, rxn_ids, solver_name).

    Parameters
    ----------
    model_or_S:
        Either a COBRApy Model or a (S, reversibility, external_ids) tuple.
    solver:
        Solver name or interface override; None infers from the model.
    """
    if _COBRA_AVAILABLE and isinstance(model_or_S, cobra.Model):
        model = model_or_S
        S = create_stoichiometric_matrix(model, array_type="dense").astype(float)
        rev = np.array([r.reversibility for r in model.reactions], dtype=bool)
        ext_mask = np.array(
            [r.id.startswith("EX_") or r.boundary for r in model.reactions],
            dtype=bool,
        )
        rxn_ids = [r.id for r in model.reactions]
        raw_solver = solver or getattr(getattr(model, "solver", None), "interface", None)
        solver_name = _solver_name(raw_solver)
    elif isinstance(model_or_S, tuple) and len(model_or_S) == 3:
        S_raw, reversibility, external_ids = model_or_S
        S = (S_raw.toarray() if scipy.sparse.issparse(S_raw)
             else np.asarray(S_raw, dtype=float))
        rev = np.asarray(reversibility, dtype=bool)
        ext_arr = np.asarray(external_ids)
        if ext_arr.dtype == bool or ext_arr.dtype == np.bool_:
            ext_mask = ext_arr.astype(bool)
            rxn_ids = [f"r{j}" for j in range(S.shape[1])]
        else:
            # list of strings or None/empty strings
            ext_mask = np.array([bool(eid) for eid in ext_arr], dtype=bool)
            rxn_ids = [
                str(eid) if eid else f"r{j}"
                for j, eid in enumerate(ext_arr)
            ]
        solver_name = _solver_name(solver)
    else:
        raise TypeError(
            "model_or_S must be a cobra.Model or (S, reversibility, external_ids) tuple."
        )
    return S, rev, ext_mask, rxn_ids, solver_name


def _solver_name(raw) -> str:
    if raw is None:
        return "glpk"
    if isinstance(raw, str):
        return raw
    # optlang interface object or cobra solver
    name = getattr(raw, "__name__", None) or getattr(raw, "name", None) or ""
    if "glpk" in name:
        return "glpk"
    if "cplex" in name:
        return "cplex"
    if "gurobi" in name:
        return "gurobi"
    return "glpk"


def _get_solver_interface(solver_name: str):
    if _COBRA_AVAILABLE:
        try:
            return check_solver(solver_name)
        except Exception:
            pass
    import optlang.glpk_interface as glpk
    return glpk


def _per_conversion_bigm(S_aug: np.ndarray, b_i: np.ndarray) -> float:
    max_b = max(1.0, float(np.max(np.abs(b_i))))
    max_s = max(1.0, float(np.max(np.abs(S_aug))))
    return max_b * (max_s + 1.0) * S_aug.shape[1]


def _set_time_limit(model, time_limit: float):
    """Best-effort application of a solver time limit across optlang backends.

    Parameters
    ----------
    model:
        Solver model to configure.
    time_limit:
        Time limit in seconds.
    """
    for attr in ("timeout", "time_limit", "configuration"):
        try:
            obj = getattr(model, attr)
            if attr == "configuration":
                setattr(obj, "timeout", time_limit)
            else:
                setattr(model, attr, time_limit)
            return
        except (AttributeError, TypeError):
            pass
