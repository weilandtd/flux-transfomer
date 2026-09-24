"""
Read published reference fluxes and SigmaFlux predictions, and pair them for comparison.

Pure data access. Every path, condition, column convention and reaction set is passed in:
nothing here globs for files, infers a config from a directory name, or reaches outside the
package. How a project lays its runs out is the caller's business, and a library that
assumed it would break the moment it was installed rather than run from a checkout.

Two things are easy to get wrong and are handled explicitly rather than smoothed over:

* **Signs are kept.** Published workbooks commonly report exchange fluxes as unsigned
  magnitudes while the model is signed, so exchanges are oriented into the model's
  convention first, using orientation the caller supplies (see :func:`reference_orientation`,
  which derives it from a model's own declared media bounds). Whatever sign disagreement
  survives is real and is reported in ``sign_agrees``. Comparing magnitudes instead would
  let a flux predicted in the wrong direction score as perfect agreement.
* **Unbounded published bounds are not numbers.** A fit that cannot bound a flux at all
  writes a large sentinel (``+/-1e8`` in the workbooks this was written for). Those are
  statements of non-identifiability; kept as numbers they draw a 1e8-long error bar and
  dominate any statistic computed from them.
"""

import re

import numpy as np
import pandas as pd


def reference_orientation(media: dict) -> dict:
    """Sign mapping a published exchange magnitude onto a model's flux convention.

    An uptake-only exchange is one the model opens downwards and caps at or below zero.
    Deriving the direction from the model's own declared bounds keeps this a property of
    the model rather than something fitted to make a comparison look better.

    Args:
        media: exchange reaction id -> {'lb': float, 'ub': float}, as an organism config's
            media block provides. Reactions absent from it are left unflipped.
    """
    return {rxn: (-1.0 if float(b.get('ub', 0.0)) <= 0.0 < -float(b.get('lb', 0.0)) else 1.0)
            for rxn, b in (media or {}).items()}


def magnitude_bounds(lo, hi):
    """Map a signed interval onto the |flux| a log axis plots.

    While the interval stays on one side of zero its magnitude bounds are just its ends,
    reordered. When it spans zero the flux could be arbitrarily small, so the magnitude has
    no positive lower bound and the bar must be drawn open-ended -- closing it at
    ``min(|lo|, |hi|)`` would claim a precision the estimate does not have.

    Returns ``(mag_lo, mag_hi, spans_zero)``, with ``mag_lo`` NaN where the bar is
    open-ended.
    """
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    spans_zero = (lo < 0) & (hi > 0)
    mag_lo = np.where(spans_zero, np.nan, np.minimum(np.abs(lo), np.abs(hi)))
    return mag_lo, np.maximum(np.abs(lo), np.abs(hi)), spans_zero


def read_reference_fluxes(path, condition: str, suffix: str = '_mfa', *,
                          index_col: str = 'id',
                          unbounded: float | None = 1e6) -> pd.DataFrame | None:
    """Published values for one condition: index = reaction id, columns = growth condition.

    Returns None when no column matches, so a caller can treat "this workbook has no such
    column" as a first-class case rather than an error.

    Args:
        path: reference workbook.
        condition: condition letter to select (matches ``<condition><n><suffix>`` and a
            bare ``<condition><suffix>``, which maps to growth condition 1).
        suffix: which family of columns to read -- the point value by default; pass the
            workbook's lower/upper-bound suffixes for the published interval.
        index_col: column holding the reaction id.
        unbounded: magnitude at or beyond which a value is the fit's "unbounded" sentinel
            and is returned as NaN. Pass None to read the raw numbers instead.
    """
    ref = pd.read_excel(path).set_index(index_col)
    pat = re.compile(rf'^{re.escape(condition)}(\d*){re.escape(suffix)}$')
    cols, gcs = [], []
    for c in ref.columns:
        m = pat.match(str(c))
        if m:
            cols.append(c)
            gcs.append(int(m.group(1)) if m.group(1) else 1)
    if not cols:
        return None
    out = ref[cols].copy()
    out.columns = gcs
    return out if unbounded is None else out.mask(out.abs() >= unbounded)


def read_predicted_fluxes(path, value_col: str = 'Value', *,
                          kind: str = 'flux') -> pd.DataFrame | None:
    """Predicted values per (growth condition, reaction). Wide: index = reaction id.

    Inference reports one row per (condition, growth condition, reaction) -- the estimate at
    the maximum-likelihood input -- so there is nothing to aggregate over here. Returns None
    when the file has no such column, which is how a prediction CSV written before a column
    existed is handled.

    Args:
        path: predictions CSV.
        value_col: which column to read (the point estimate, or an interval bound).
        kind: value of the 'Type' column to keep.
    """
    df = pd.read_csv(path)
    df = df[df['Type'] == kind]
    if value_col not in df.columns:
        return None
    return (df.groupby(['GrowthCondition', 'Flux Name'])[value_col]
              .first().unstack('GrowthCondition'))


def pair_fluxes(pred: dict, ref: dict, *, condition: str, orientation: dict | None = None,
                reactions: set | None = None, growth_rates: dict | None = None,
                min_abs: float = 0.0) -> pd.DataFrame:
    """Join predicted and reference fluxes into one signed, tidy frame.

    Args:
        pred: {'value': frame, 'lo': frame or None, 'hi': frame or None}, as
            :func:`read_predicted_fluxes` returns.
        ref: same shape, as :func:`read_reference_fluxes` returns.
        condition: condition label to record on every row.
        orientation: reaction -> sign applied to the reference before comparing; reactions
            absent from it are left unflipped.
        reactions: restrict to these reactions; None keeps every shared one.
        growth_rates: growth condition -> dilution rate, recorded as 'D'.
        min_abs: drop rows where either side is at or below this magnitude.
    """
    orientation = orientation or {}
    growth_rates = growth_rates or {}
    ref_v, pred_v = ref['value'], pred['value']
    parts = []

    for gc in sorted(set(pred_v.columns) & set(ref_v.columns)):
        shared = list(pred_v.index.intersection(ref_v.index))
        if reactions is not None:
            shared = [r for r in shared if r in reactions]
        if not shared:
            continue

        def _col(frame):
            if frame is None or gc not in frame.columns:
                return np.full(len(shared), np.nan)
            return frame.reindex(shared)[gc].to_numpy(dtype=float)

        # Orient published exchange magnitudes into the model's convention before anything
        # is compared. Flipping swaps which end is the lower bound, so reorder afterwards.
        flip = np.array([orientation.get(r, 1.0) for r in shared])
        lo, hi = _col(ref['lo']) * flip, _col(ref['hi']) * flip

        parts.append(pd.DataFrame({
            'condition': condition,
            'gc':        gc,
            'D':         growth_rates.get(gc, float(gc)),
            'reaction':  shared,
            'ref':       _col(ref_v) * flip,
            'pred':      _col(pred_v),
            'ref_lo':    np.minimum(lo, hi),
            'ref_hi':    np.maximum(lo, hi),
            'pred_lo':   _col(pred['lo']),
            'pred_hi':   _col(pred['hi']),
        }))

    if not parts:
        return pd.DataFrame(columns=['condition', 'gc', 'D', 'reaction', 'ref', 'pred',
                                     'ref_lo', 'ref_hi', 'pred_lo', 'pred_hi',
                                     'sign_agrees'])
    df = pd.concat(parts, ignore_index=True)
    df['sign_agrees'] = np.sign(df['ref']) == np.sign(df['pred'])
    return df[(df['ref'].abs() > min_abs) & (df['pred'].abs() > min_abs)]
