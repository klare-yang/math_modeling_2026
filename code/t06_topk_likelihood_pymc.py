# code/t06_topk_likelihood_pymc.py
# -*- coding: utf-8 -*-

"""
T06 (Normalized) — Hierarchical Bayes model + multi-elimination likelihood.

Event-level model:
    logV[e,a] = alpha + X[e,a] @ beta + u[contestant_idx[e,a]] + w[event_week_idx[e]]

Elimination propensity: eta = -logV (lower votes => higher elimination propensity).

Likelihood options:
- "plackett_luce_set" (recommended): unordered eliminated set, exact marginalization for k<=3.
- "softmax_set_approx": i.i.d. approx with replacement.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pymc as pm
import pytensor.tensor as pt


@dataclass(frozen=True)
class ModelData:
    n_contestants: int
    n_events: int
    n_weeks: int
    p: int

    event_week_idx: np.ndarray          # (n_events,)

    max_active: int
    active_idx_pad: np.ndarray          # (n_events, max_active) padded with -1
    active_mask: np.ndarray             # (n_events, max_active)
    X_active_pad: np.ndarray            # (n_events, max_active, p)

    max_elim: int
    elim_pos_pad: np.ndarray            # (n_events, max_elim) padded with -1
    elim_mask: np.ndarray               # (n_events, max_elim)


def _masked_logits(logits: pt.TensorVariable, mask: pt.TensorVariable) -> pt.TensorVariable:
    neg_inf = pt.constant(-1.0e30, dtype="float64")
    return pt.where(mask, logits, neg_inf)


def _logsoftmax_gather(logits: pt.TensorVariable, idx: pt.TensorVariable) -> pt.TensorVariable:
    lsm = pt.nnet.logsoftmax(logits, axis=1)
    return lsm[pt.arange(logits.shape[0]), idx]


def _order_logp_plackett_luce(logits: pt.TensorVariable, order_pos: list[pt.TensorVariable]) -> pt.TensorVariable:
    lp = _logsoftmax_gather(logits, order_pos[0])
    neg_inf = pt.constant(-1.0e30, dtype="float64")
    current = logits
    for r in range(1, len(order_pos)):
        prev = order_pos[r - 1]
        oh = pt.extra_ops.to_one_hot(prev, current.shape[1])
        current = pt.where(oh > 0, neg_inf, current)
        lp = lp + _logsoftmax_gather(current, order_pos[r])
    return lp


def logp_plackett_luce_set_marginal(logits, elim_pos_pad, elim_mask):
    k = pt.sum(elim_mask, axis=1).astype("int32")
    total = pt.constant(0.0, dtype="float64")

    def add_group(K: int, perms: list[tuple[int, ...]]):
        nonlocal total
        maskK = pt.eq(k, K)
        idxK = pt.nonzero(maskK)[0]
        logitsK = logits[idxK, :]
        posK = elim_pos_pad[idxK, :K]

        perm_logps = []
        for perm in perms:
            order = [posK[:, j] for j in perm]
            perm_logps.append(_order_logp_plackett_luce(logitsK, order))

        stack = pt.stack(perm_logps, axis=0)
        lp_set = pt.logsumexp(stack, axis=0)
        total = total + pt.sum(lp_set)

    add_group(1, [(0,)])
    add_group(2, [(0, 1), (1, 0)])
    add_group(3, [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)])
    return total


def logp_softmax_set_approx(logits, elim_pos_pad, elim_mask):
    Kmax = elim_pos_pad.shape[1]
    total = pt.constant(0.0, dtype="float64")
    for r in range(Kmax):
        pos_r = elim_pos_pad[:, r]
        valid_r = elim_mask[:, r]
        lp_r = pm.logp(pm.Categorical.dist(logits=logits), pos_r)
        total = total + pt.sum(pt.where(valid_r, lp_r, 0.0))
    return total


def build_model(
    data: ModelData,
    likelihood: Literal["plackett_luce_set", "softmax_set_approx"] = "plackett_luce_set",
) -> pm.Model:
    with pm.Model() as model:
        event_week_idx = pm.Data("event_week_idx", data.event_week_idx, mutable=False)
        active_idx_pad = pm.Data("active_idx_pad", data.active_idx_pad, mutable=False)
        active_mask = pm.Data("active_mask", data.active_mask, mutable=False)
        X_active_pad = pm.Data("X_active_pad", data.X_active_pad, mutable=False)
        elim_pos_pad = pm.Data("elim_pos_pad", data.elim_pos_pad, mutable=False)
        elim_mask = pm.Data("elim_mask", data.elim_mask, mutable=False)

        alpha = pm.Normal("alpha", mu=0.0, sigma=5.0)
        beta = pm.Normal("beta", mu=0.0, sigma=10.0, shape=(data.p,))

        sigma_u = pm.HalfCauchy("sigma_u", beta=2.0)
        sigma_w = pm.HalfCauchy("sigma_w", beta=2.0)

        u = pm.Normal("u", mu=0.0, sigma=sigma_u, shape=(data.n_contestants,))
        w = pm.Normal("w", mu=0.0, sigma=sigma_w, shape=(data.n_weeks,))

        xb = pt.tensordot(X_active_pad, beta, axes=[2, 0])  # (E,A)

        idx_safe = pt.maximum(active_idx_pad, 0)
        u_term = u[idx_safe]
        w_term = w[event_week_idx][:, None]

        logV = alpha + xb + u_term + w_term
        pm.Deterministic("logV_active", logV)

        eta = -logV
        logits = _masked_logits(eta, active_mask)

        if likelihood == "plackett_luce_set":
            ll = logp_plackett_luce_set_marginal(logits, elim_pos_pad, elim_mask)
        elif likelihood == "softmax_set_approx":
            ll = logp_softmax_set_approx(logits, elim_pos_pad, elim_mask)
        else:
            raise ValueError(f"Unknown likelihood={likelihood}")

        pm.Potential("elimination_likelihood", ll)

    return model
