

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""t01_route2_v3B.py

Shared utilities for route2_v3B T-series scripts.

Primary responsibilities
-----------------------
1) Differentiable soft ranking for rank-based show rules:
   - `soft_rank_pt(x, tau, mask)` returns a "soft rank" where
     smaller x -> rank≈1 and larger x -> rank≈n (monotone increasing).

2) Unordered-set likelihood for multi-elimination weeks (k<=3):
   - `logp_unordered_k_set_pt(weights, idxs)` returns log P({idxs} eliminated)
     under sequential without-replacement draw with item weights.

Design notes
------------
- Avoid deprecated/absent namespaces such as `pt.nnet.*`.
- Compatible with PyTensor where `pytensor.tensor` does NOT expose `.nnet`.
"""

from __future__ import annotations

from typing import List

import pytensor
import pytensor.tensor as pt

def rule_for_season(season):
    """
    Returns the rule for the given season:
    - 'rank' for rank-based seasons
    - 'percent' for percent-based seasons
    - Judges save flag for season >= 28
    """
    if season in [1, 2]:
        return {"mode": "rank", "judges_save": False}
    elif 3 <= season <= 27:
        return {"mode": "percent", "judges_save": False}
    elif season >= 28:
        return {"mode": "rank", "judges_save": True}  # Assuming rank-based for season 28+ with judges save
    else:
        raise ValueError(f"Season {season} not recognized. Must be >= 1.")


# -----------------------------
# Numerically stable primitives
# -----------------------------

def _sigmoid(x):
    """Logistic sigmoid implemented via exp; avoids pt.nnet.*"""
    return 1.0 / (1.0 + pt.exp(-x))


def _logsumexp(vec, axis=None, keepdims=False):
    m = pt.max(vec, axis=axis, keepdims=True)
    out = m + pt.log(pt.sum(pt.exp(vec - m), axis=axis, keepdims=True))
    if not keepdims and axis is not None:
        out = pt.squeeze(out, axis=axis)
    return out


# -----------------------------
# Soft ranking (mask-aware)
# -----------------------------

def soft_rank_pt(x, tau: float = 0.15, mask=None):
    """Soft rank for a 1D tensor (monotone increasing with x on active entries).

    rank_i = 1 + sum_{j!=i} sigmoid((x_i - x_j)/tau)

    - smallest x => rank ~ 1
    - largest x  => rank ~ n_active

    Inactive entries (mask=0) return 0.
    """
    x = pt.as_tensor_variable(x)
    if x.ndim != 1:
        raise ValueError("soft_rank_pt expects a 1D tensor")

    n = x.shape[0]
    m = pt.ones((n,), dtype="float64") if mask is None else pt.cast(mask, "float64")

    diff = (x[:, None] - x[None, :]) / pt.maximum(tau, 1e-12)

    pair_mask = (m[:, None] * m[None, :])
    pair_mask = pair_mask * (1.0 - pt.eye(n, dtype="float64"))

    sig = _sigmoid(diff) * pair_mask
    r = 1.0 + pt.sum(sig, axis=1)

    return r * m


def soft_rank_batch(x_2d, mask_2d, tau: float = 0.15):
    """Batch soft ranks computed row-wise via pytensor.scan."""
    def _row_fn(x_row, m_row):
        return soft_rank_pt(x_row, tau=tau, mask=m_row)

    res = pytensor.scan(fn=_row_fn, sequences=[x_2d, mask_2d], return_updates=False)
    return res[0] if isinstance(res, (tuple, list)) else res


# -----------------------------------------
# Unordered set likelihood (k = 2 or 3 only)
# -----------------------------------------

def logp_unordered_k_set_pt(weights, idxs: List[int], eps: float = 1e-30):
    """Log P(unordered eliminated set) for k=2 or 3 under sequential W/O replacement."""
    w = pt.as_tensor_variable(weights)
    k = len(idxs)
    if k not in (2, 3):
        raise ValueError("logp_unordered_k_set_pt supports k=2 or 3")

    W = pt.maximum(pt.sum(w), eps)

    ws = [pt.maximum(w[int(i)], eps) for i in idxs]

    if k == 2:
        wa, wb = ws
        term = (wa * wb / W) * (1.0 / pt.maximum(W - wa, eps) + 1.0 / pt.maximum(W - wb, eps))
        return pt.log(pt.maximum(term, eps))

    a, b, c = ws

    def order_prob(first, second, third):
        denom1 = pt.maximum(W - first, eps)
        denom2 = pt.maximum(W - first - second, eps)
        return (first / W) * (second / denom1) * (third / denom2)

    probs = [
        order_prob(a, b, c),
        order_prob(a, c, b),
        order_prob(b, a, c),
        order_prob(b, c, a),
        order_prob(c, a, b),
        order_prob(c, b, a),
    ]
    p = pt.sum(pt.stack(probs))
    return pt.log(pt.maximum(p, eps))


def _self_test():
    import numpy as np
    x = pt.as_tensor_variable(np.array([1.0, 2.0, 3.0, 4.0], dtype=float))
    m = pt.as_tensor_variable(np.array([1, 1, 1, 1], dtype=float))
    r = soft_rank_pt(x, tau=0.1, mask=m)
    f = pytensor.function([], r)
    rv = f()
    assert np.all(np.diff(rv) > 0), f"soft ranks not increasing: {rv}"


# NOTE: keep self-test off by default
# _self_test()
