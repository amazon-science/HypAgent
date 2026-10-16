# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Simulator — a pure interpreter of an SCMSpec.

No LLM here. Given a spec it:
    - generate(n)            : sample a dataset (topological pass, mixed types, vectorized)
    - do(interventions)      : interventional distribution, propagate downstream + recompute outcome
    - ate(var, hi, lo, cond) : exact ground-truth average treatment effect
    - answer_key()           : the labeled subset with isolating-intervention recipes
    - auto_calibrate()       : set term thresholds from data quantiles (anti-memorization)

do() is valid on ANY node because we own every structural equation. That is the whole
point of going fully synthetic.
"""

from __future__ import annotations

import math
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from .spec import SCMSpec, NodeSpec, Term, Operand
except ImportError:  # allow running as a script
    from spec import SCMSpec, NodeSpec, Term, Operand


# =============================================================================
# helpers
# =============================================================================

def _standardize(v: np.ndarray, ref: tuple | None = None) -> np.ndarray:
    """Standardize with reference (mean, std). Under do() the intervened column
    is constant, so its own mean/std are degenerate — we must use the frozen
    observational stats (ref) or the term would vanish."""
    v = np.asarray(v, dtype=float)
    if ref is not None:
        mu, sd = ref
    else:
        mu, sd = np.nanmean(v), np.nanstd(v)
    v = np.where(np.isnan(v), mu, v)
    return (v - mu) / (sd if sd and sd > 0 else 1.0)


def _numeric(series) -> np.ndarray:
    """Coerce a column to float; encode strings as deviation-from-mode indicator."""
    arr = np.asarray(series)
    if arr.dtype.kind in ("U", "S", "O"):
        vals_u, counts = np.unique(arr, return_counts=True)
        mode = vals_u[counts.argmax()]
        return (arr != mode).astype(float)
    return arr.astype(float)


def _operand_values(df: pd.DataFrame, op: Operand) -> np.ndarray:
    """Resolve an operand (with its transform) to a numeric vector.

    Uses _numeric so a stray categorical operand degrades to a mode-indicator
    instead of crashing; validate() separately flags this as a repairable issue.
    """
    a = _numeric(df[op.var])
    if op.transform == "identity":
        return a
    if op.transform == "log":
        return np.log(np.clip(a, 1e-9, None))
    if op.transform == "ratio":
        b = _numeric(df[op.other])
        b = np.where(np.abs(b) < 1e-9, np.nan, b)
        r = a / b
        return np.where(np.isnan(r), np.nanmedian(r), r)
    if op.transform == "product":
        b = _numeric(df[op.other])
        return a * b
    raise ValueError(f"unknown transform {op.transform}")


# =============================================================================
# outcome score from terms  (the executable primitive library)
# =============================================================================

def score_from_terms(df: pd.DataFrame, terms: list[Term], base_noise: float, rng,
                     ref_stats: dict | None = None) -> np.ndarray:
    """Sum every term's contribution into a continuous latent score.

    ref_stats: {var_key: (mean, std) or {"q": {quantile: value}}} frozen from
    observational data. Required for correct do(): when a variable is clamped to
    a constant its own moments/quantiles are degenerate, so any standardization
    or quantile cut on it must use the frozen reference instead.
    """
    n = len(df)
    score = rng.normal(0.0, base_noise, n)
    rs = ref_stats or {}

    def moments(op_obj):
        key = op_obj.var + ("" if op_obj.transform == "identity"
                            else f"|{op_obj.transform}|{op_obj.other}")
        return rs.get(("m", key))

    def qcut(op_obj, q):
        key = op_obj.var + ("" if op_obj.transform == "identity"
                            else f"|{op_obj.transform}|{op_obj.other}")
        table = rs.get(("q", key))
        if table and q in table:
            return table[q]
        return None

    for t in terms:
        w, op = t.weight, t.op

        if op == "linear":
            v = _operand_values(df, t.operands[0])
            score += w * _standardize(v, moments(t.operands[0]))

        elif op == "threshold":
            v = _operand_values(df, t.operands[0])
            tau = t.threshold if t.threshold is not None else np.median(v)
            score += w * (v > tau).astype(float)

        elif op == "step":
            v = _operand_values(df, t.operands[0])
            lo = t.threshold_lo if t.threshold_lo is not None else np.quantile(v, 0.6)
            hi = t.threshold_hi if t.threshold_hi is not None else np.quantile(v, 0.85)
            eff = np.zeros(n)
            eff[v > lo] = 0.5 * w
            eff[v > hi] = w
            score += eff

        elif op == "bump":
            v = _operand_values(df, t.operands[0])
            lo = t.threshold_lo if t.threshold_lo is not None else np.quantile(v, 0.18)
            hi = t.threshold_hi if t.threshold_hi is not None else np.quantile(v, 0.82)
            eff = np.zeros(n)
            eff[v < lo] += w
            eff[v > hi] += 0.85 * w
            mid = (v >= lo) & (v <= hi)
            eff[mid] += -0.4 * w
            score += eff

        elif op == "interaction":
            v = _standardize(_operand_values(df, t.operands[0]), moments(t.operands[0]))
            modv = _numeric(df[t.moderator])
            mtau = (t.moderator_threshold if t.moderator_threshold is not None
                    else np.median(modv))
            grp = modv > mtau
            score += np.where(grp, w * v, t.reverse_sign * w * v)

        elif op == "gate":
            v = _operand_values(df, t.operands[0])
            tau = t.threshold if t.threshold is not None else np.median(v)
            modv = _numeric(df[t.moderator])
            mtau = (t.moderator_threshold if t.moderator_threshold is not None
                    else np.median(modv))
            score += w * ((v > tau) & (modv > mtau)).astype(float)

        elif op == "conjunction":
            mask = np.ones(n, dtype=bool)
            for o in t.operands:
                v = _operand_values(df, o)
                cut = qcut(o, t.conj_quantile)
                if cut is None:
                    cut = np.quantile(v, t.conj_quantile)
                mask &= (v > cut)
            score += w * mask.astype(float)

        elif op == "category":
            # categorical-level effect: each named level adds its own weight.
            # do()-able (clamp the categorical to a level) and recoverable by ==level.
            col = np.asarray(df[t.operands[0].var])
            for level, lw in t.level_weights.items():
                score += float(lw) * (col == level).astype(float)

    return score


def _compute_ref_stats(df: pd.DataFrame, terms: list[Term]) -> dict:
    """Freeze means/stds and needed quantile cuts from observational data."""
    rs: dict = {}
    for t in terms:
        for o in t.operands:
            key = o.var + ("" if o.transform == "identity"
                          else f"|{o.transform}|{o.other}")
            v = _operand_values(df, o)
            rs[("m", key)] = (float(np.nanmean(v)), float(np.nanstd(v)))
            if t.op == "conjunction":
                rs.setdefault(("q", key), {})[t.conj_quantile] = float(
                    np.quantile(v, t.conj_quantile))
    return rs


# =============================================================================
# simulator
# =============================================================================

class Simulator:
    """A fully-specified SCM with PERSISTED exogenous noise.

    Exogenous noise U is drawn ONCE (generate) and stored. All data is a
    deterministic function evaluate(U, overrides, frozen). This makes:
      - do()            an exactly-paired population intervention (same U, clamp X)
      - counterfactual() a per-unit query (hold that unit's U fixed, change X)
    which is what separates rung 2 from rung 3.
    """

    def __init__(self, spec: SCMSpec):
        self.spec = spec
        problems = spec.validate()
        if problems:
            raise ValueError("invalid SCMSpec:\n  " + "\n  ".join(problems))
        self._U = None            # persisted exogenous noise {node: array}, + "__score__"
        self._frozen = None       # frozen structural params (moments, thresholds, cutoff)
        # legacy attribute kept so existing callers/tests that read it still work
        self._cutoff = None
        self._ref_stats = None

    OUTCOME_U = "__score__"

    # ---- exogenous noise (drawn once, persisted) ----
    def _draw_noise(self, n: int, rng) -> dict:
        U = {}
        for node in self.spec.nodes:
            d = node.distribution
            if d == "uniform" or d == "categorical":
                U[node.name] = rng.uniform(0.0, 1.0, n)
            else:  # normal / lognormal / poisson-proxy / bernoulli-latent
                U[node.name] = rng.normal(0.0, 1.0, n)
        U[self.OUTCOME_U] = rng.normal(0.0, 1.0, n)
        return U

    def _base_from_noise(self, node: NodeSpec, u: np.ndarray):
        """Deterministic map exogenous-noise -> pre-parent base value."""
        p, d = node.params, node.distribution
        if d == "normal":
            return p.get("mean", 0.0) + p.get("std", 1.0) * u
        if d == "lognormal":
            return np.exp(p.get("mean", 0.0) + p.get("sigma", 0.5) * u)
        if d == "uniform":
            return p.get("low", 0.0) + (p.get("high", 1.0) - p.get("low", 0.0)) * u
        if d == "poisson":
            lam = p.get("lambda", 3.0)
            return lam + max(lam, 1.0) ** 0.5 * u
        if d == "bernoulli":
            return u  # latent normal, thresholded in _finalize
        if d == "categorical":
            cats = p.get("categories", ["A", "B", "C"])
            probs = p.get("probs")
            if probs is None:
                probs = [1.0 / len(cats)] * len(cats)
            edges = np.cumsum(probs)
            idx = np.searchsorted(edges, u, side="right")
            idx = np.clip(idx, 0, len(cats) - 1)
            return np.asarray(cats)[idx]
        return u

    def _finalize(self, node: NodeSpec, vals, frozen: dict) -> np.ndarray:
        """Map post-parent base value -> observed value, using FROZEN params so
        interventions/counterfactuals stay on the same structural equation."""
        dist, p = node.distribution, node.params
        if dist == "categorical":
            return vals  # already category strings
        if dist == "bernoulli":
            tau = frozen["bern_tau"].get(node.name)
            if tau is None:  # observational pass: compute & freeze
                tau = float(np.quantile(vals, 1 - p.get("p", 0.5)))
                frozen["bern_tau"][node.name] = tau
            return (vals > tau).astype(int)
        if dist == "poisson" or node.dtype == "count":
            return np.clip(np.round(vals), 0, None).astype(int)
        if dist == "uniform":
            return np.clip(vals, p.get("low", 0.0), p.get("high", 1.0))
        if dist == "lognormal":
            return np.clip(vals, 0.0, None)
        return vals

    def _evaluate(self, U: dict, overrides: dict, frozen: dict) -> pd.DataFrame:
        """Deterministic structural evaluation given noise U and interventions.

        frozen carries parent moments and bernoulli thresholds; on the first
        (observational) pass they are computed and stored, then reused so that
        do()/counterfactual() operate on the identical structural equations.
        """
        overrides = overrides or {}
        n = len(next(iter(U.values())))
        data: dict[str, np.ndarray] = {}
        by_name = {nd.name: nd for nd in self.spec.nodes}

        for name in self.spec.topological_order():
            node = by_name[name]
            if name in overrides:
                val = overrides[name]
                data[name] = np.full(n, val) if np.isscalar(val) else np.asarray(val)
                continue

            base = self._base_from_noise(node, U[name])
            if node.distribution != "categorical":
                base = base.astype(float)
                for parent, w in zip(node.parents, node.parent_weights):
                    if parent not in data:
                        continue
                    pv = _numeric(data[parent])
                    key = (name, parent)
                    mom = frozen["pmom"].get(key)
                    if mom is None:  # freeze parent moments on observational pass
                        mom = (float(pv.mean()), float(pv.std()))
                        frozen["pmom"][key] = mom
                    mu, sd = mom
                    if sd > 0:
                        base += w * (pv - mu) / sd
            data[name] = self._finalize(node, base, frozen)

        return pd.DataFrame(data)

    def _score_and_outcome(self, df, U, frozen) -> np.ndarray:
        """Compute latent score (using persisted outcome noise) and threshold it."""
        ref = frozen.get("ref_stats")
        if ref is None:
            ref = _compute_ref_stats(df, self.spec.terms)
            frozen["ref_stats"] = ref
        # deterministic base noise from persisted U
        score = self.spec.base_noise * U[self.OUTCOME_U]
        score = score + score_from_terms(
            df, self.spec.terms, 0.0, np.random.default_rng(0), ref_stats=ref)
        cutoff = frozen.get("cutoff")
        if cutoff is None:
            cutoff = float(np.quantile(score, 1 - self.spec.outcome_positive_rate))
            frozen["cutoff"] = cutoff
        return (score > cutoff).astype(int)

    def _new_frozen(self) -> dict:
        return {"pmom": {}, "bern_tau": {}, "ref_stats": None, "cutoff": None}

    # ---- generation (draws & persists U, freezes structural params) ----
    def generate(self, n: int | None = None, visible_only: bool = False) -> pd.DataFrame:
        """Generate the dataset. The full DGP (incl. hidden confounders) is always
        computed and persisted internally; visible_only=True returns the view a
        method/agent would actually see (hidden-role columns removed)."""
        n = n or self.spec.n_rows
        rng = np.random.default_rng(self.spec.seed)
        self._U = self._draw_noise(n, rng)
        self._frozen = self._new_frozen()
        df = self._evaluate(self._U, {}, self._frozen)
        df[self.spec.outcome_name] = self._score_and_outcome(df, self._U, self._frozen)
        # expose frozen values via legacy attributes
        self._cutoff = self._frozen["cutoff"]
        self._ref_stats = self._frozen["ref_stats"]
        if visible_only and self.spec.hidden_nodes:
            keep = self.spec.visible_columns() + [self.spec.outcome_name]
            return df[keep]
        return df

    def _ensure_generated(self, n: int):
        if self._U is None or len(self._U[self.OUTCOME_U]) != n:
            self.generate(n)

    # ---- interventions (rung 2) ----
    def do(self, interventions: dict, n: int | None = None,
           df: pd.DataFrame | None = None) -> pd.DataFrame:
        """do(interventions): exactly-paired population intervention.

        Reuses the persisted noise U and frozen structural params, so only the
        intervened variable (and its descendants) differ from the observational
        world — every other exogenous source is held identical.
        """
        n = n or (len(df) if df is not None else self.spec.n_rows)
        self._ensure_generated(n)
        feats = self._evaluate(self._U, interventions, self._frozen)
        feats[self.spec.outcome_name] = self._score_and_outcome(
            feats, self._U, self._frozen)
        return feats

    def ate(self, var: str, hi, lo, n: int | None = None,
            condition=None) -> dict:
        """Exact ATE = E[Y|do(var=hi)] - E[Y|do(var=lo)].

        condition: optional boolean callable(df)->mask to isolate a single
        structural component (used by the answer key to test one hypothesis
        while holding other terms on the same variable off).
        """
        n = n or min(200_000, self.spec.n_rows)
        self._ensure_generated(n)
        dt = self.do({var: hi}, n=n)
        dc = self.do({var: lo}, n=n)
        oc = self.spec.outcome_name
        # do(hi) and do(lo) share the same persisted noise U row-for-row, so the
        # per-row difference IS the individual treatment effect and its mean is the
        # exact ATE. That pairing lets us bootstrap a CI / p-value on the ATE directly.
        if condition is not None:
            mt, mc = condition(dt), condition(dc)
            yt = dt.loc[mt, oc].mean() if mt.any() else np.nan
            yc = dc.loc[mc, oc].mean() if mc.any() else np.nan
            # For the paired bootstrap CI we must compare the SAME units across arms.
            # If the condition variable is downstream of the intervened variable, the
            # intervention shifts it, so (mt & mc) can be empty. Define the subgroup on
            # the control (pre-intervention) world — a fixed set of units — and take the
            # paired diff there; both worlds share row identity and noise, so this is a
            # valid conditional-ATE estimator whose CI is well defined.
            mrows = np.asarray(mc)
            if not mrows.any():
                mrows = np.asarray(mt)
            diff = (dt.loc[mrows, oc].to_numpy() - dc.loc[mrows, oc].to_numpy()) if mrows.any() \
                else np.array([], dtype=float)
        else:
            diff = dt[oc].to_numpy() - dc[oc].to_numpy()
        # The reported ATE is the mean of the SAME paired per-row differences the
        # bootstrap resamples, so the point estimate always lies inside its own CI.
        ate = float(diff.mean()) if diff.size else float("nan")
        sig = self._bootstrap_ate(diff)

        # observational (naive) contrast for confounding detection
        base = self.do({}, n=n)   # identity intervention == observational world
        med = base[var].median() if base[var].dtype != object else None
        if med is not None:
            obs = float(base.loc[base[var] > med, oc].mean()
                        - base.loc[base[var] <= med, oc].mean())
        else:
            obs = np.nan
        return {
            "var": var, "hi": hi, "lo": lo,
            "ate_causal": ate,
            "ate_observational": obs,
            "confounding_bias": (obs - ate) if not np.isnan(obs) else np.nan,
            "is_confounded": (abs(obs - ate) > 0.03) if not np.isnan(obs) else False,
            "ci_low": sig["ci_low"], "ci_high": sig["ci_high"],
            "se": sig["se"], "p_value": sig["p_value"],
            "significant": sig["significant"],
        }

    @staticmethod
    def _bootstrap_ate(diff, n_boot: int = 1000, alpha: float = 0.05) -> dict:
        """Bootstrap CI + a normal-approx p-value for a paired ATE (mean of per-row
        treatment effects). `diff` is the row-wise Y_hi - Y_lo vector. With the large
        simulator n the SE is tiny, so 'significant' here means 'the effect estimate is
        distinguishable from 0', not 'the effect is large' (magnitude is judged elsewhere).
        """
        diff = np.asarray(diff, dtype=float)
        diff = diff[~np.isnan(diff)]
        if diff.size < 2:
            return {"ci_low": np.nan, "ci_high": np.nan, "se": np.nan,
                    "p_value": np.nan, "significant": False}
        mean = float(diff.mean())
        # analytic SE of the mean (paired) — exact and O(n); bootstrap CI for robustness
        se = float(diff.std(ddof=1) / np.sqrt(diff.size))
        rng = np.random.default_rng(12345)
        idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
        boots = diff[idx].mean(axis=1)
        ci_low = float(np.quantile(boots, alpha / 2))
        ci_high = float(np.quantile(boots, 1 - alpha / 2))
        # two-sided normal-approx p-value for H0: ATE == 0
        z = mean / se if se > 0 else (np.inf if mean != 0 else 0.0)
        p = float(math.erfc(abs(z) / math.sqrt(2)))  # 2-sided
        return {"ci_low": ci_low, "ci_high": ci_high, "se": se,
                "p_value": p, "significant": bool(ci_low > 0 or ci_high < 0)}

    # ---- counterfactuals (rung 3) ----
    def counterfactual(self, interventions: dict, rows=None,
                       n: int | None = None) -> pd.DataFrame:
        """P(Y_x | evidence) at the UNIT level via Pearl's 3 steps.

        Abduction: the units' exogenous noise U is already persisted (we drew it),
                   so inferring "what made this unit tick" is an exact lookup — the
                   step that on real data requires untestable structural assumptions.
        Action:    overwrite the intervened variables.
        Prediction: re-evaluate the SCM holding each unit's U fixed.

        rows: optional index array to restrict to specific units (else all).
        Returns the counterfactual world's dataframe (same rows, same U, new X).
        """
        n = n or self.spec.n_rows
        self._ensure_generated(n)
        U = self._U
        if rows is not None:
            U = {k: v[rows] for k, v in U.items()}
        cf = self._evaluate(U, interventions, self._frozen)
        cf[self.spec.outcome_name] = self._score_and_outcome(cf, U, self._frozen)
        return cf

    def ite(self, var: str, hi, lo, n: int | None = None) -> np.ndarray:
        """Individual Treatment Effect per unit: Y_{var=hi} - Y_{var=lo}, same U.

        E[ite] should equal the population ATE (sanity link between rungs 2 and 3).
        """
        n = n or min(200_000, self.spec.n_rows)
        self._ensure_generated(n)
        oc = self.spec.outcome_name
        yt = self.counterfactual({var: hi}, n=n)[oc].to_numpy()
        yc = self.counterfactual({var: lo}, n=n)[oc].to_numpy()
        return yt - yc

    def probabilities_of_causation(self, var: str, hi, lo,
                                   n: int | None = None) -> dict:
        """Rung-3 attribution among units that actually took var>=hi and had Y=1:
            PN (necessity)  = P(Y_{lo}=0 | var=hi, Y=1)  — would Y have flipped off?
            PS (sufficiency)= P(Y_{hi}=1 | var=lo, Y=0)  — would setting hi turn Y on?
        """
        n = n or min(200_000, self.spec.n_rows)
        self._ensure_generated(n)
        oc = self.spec.outcome_name
        obs = self.do({}, n=n)
        v = obs[var].to_numpy()
        y = obs[oc].to_numpy()
        col = self.spec.node(var)
        hi_mask = (v == hi) if (col and col.dtype == "categorical") else (v >= hi)
        lo_mask = (v == lo) if (col and col.dtype == "categorical") else (v <= lo)

        y_lo = self.counterfactual({var: lo}, n=n)[oc].to_numpy()
        y_hi = self.counterfactual({var: hi}, n=n)[oc].to_numpy()
        exposed_pos = hi_mask & (y == 1)
        unexposed_neg = lo_mask & (y == 0)
        pn = float((y_lo[exposed_pos] == 0).mean()) if exposed_pos.any() else float("nan")
        ps = float((y_hi[unexposed_neg] == 1).mean()) if unexposed_neg.any() else float("nan")
        return {"PN": pn, "PS": ps,
                "n_exposed_positive": int(exposed_pos.sum()),
                "n_unexposed_negative": int(unexposed_neg.sum())}

    # ---- answer key ----
    def answer_key(self) -> list[dict]:
        """Labeled hypotheses with the variables + archetype tag needed for scoring."""
        key = []
        for t in self.spec.labeled_terms:
            key.append({
                "id": t.id,
                "archetype": t.archetype,
                "op": t.op,
                "variables": sorted(t.variables),
                "moderator": t.moderator,
                "weight": t.weight,
                "description": t.description,
            })
        return key

    # ---- anti-memorization threshold calibration ----
    def auto_calibrate(self, df: pd.DataFrame | None = None):
        """Set term thresholds from DATA quantiles wherever *_quantile is given.

        Thresholds thus reflect the realized distribution, never a memorizable
        literal. Call before generate() if terms carry *_quantile fields.
        """
        if df is None:
            # draw a calibration sample of features (noise-only, no outcome needed)
            rng = np.random.default_rng(self.spec.seed)
            U = self._draw_noise(min(100_000, self.spec.n_rows), rng)
            df = self._evaluate(U, {}, self._new_frozen())
        for t in self.spec.terms:
            v0 = _operand_values(df, t.operands[0]) if t.operands else None
            if t.threshold_quantile is not None and v0 is not None:
                t.threshold = float(np.quantile(v0, t.threshold_quantile))
            if t.threshold_lo_quantile is not None and v0 is not None:
                t.threshold_lo = float(np.quantile(v0, t.threshold_lo_quantile))
            if t.threshold_hi_quantile is not None and v0 is not None:
                t.threshold_hi = float(np.quantile(v0, t.threshold_hi_quantile))
            if t.moderator_quantile is not None and t.moderator in df.columns:
                t.moderator_threshold = float(
                    np.quantile(_numeric(df[t.moderator]), t.moderator_quantile))
        # thresholds changed -> invalidate persisted state so next generate refreezes
        self._U = None
        self._frozen = None
        self._cutoff = None
        return self
