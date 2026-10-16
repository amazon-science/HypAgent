# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Value gates (steer funding toward unexplained structure) and validity gates
(check a confirmed hypothesis against the data before accepting it).

Value gates, run near-free per candidate:
  * ``residualize``           — remove all confirmed effects from the outcome.
  * ``liveness``              — nonlinear association of a variable with the residual.
  * ``conditional_liveness``  — super-additivity test for interaction proposals.

Validity gates, run only on the confirmation branch (~budget times per run):
  * ``recompute_claim_ok``    — reject a stated direction the data contradicts.
  * ``survives_adjustment``   — reject an effect that washes out under adjustment.

``hyp_signature`` gives each hypothesis a set-level identity for dedup.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from .config import (
    FIT_SUBSAMPLE_N, SUBSAMPLE_ROW_THRESHOLD, ALPHA_INTERACTION, INTERACTION_SCAN_N,
    CLAIM_SIG_P, ADJ_EFFECT_FLOOR, ADJ_RETAIN_FRAC, ADJ_TOPK_CONFOUNDERS,
)

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold
    _HAVE_ADJ = True
except Exception:  # pragma: no cover
    _HAVE_ADJ = False

try:
    import statsmodels.api as sm
    _HAVE_SM = True
except Exception:  # pragma: no cover
    _HAVE_SM = False


def numeric_view(df, cols):
    """Numeric matrix over ``cols``: category codes for objects, else float."""
    X = pd.DataFrame(index=df.index)
    for c in cols:
        col = df[c].astype("category").cat.codes if df[c].dtype == object else df[c].astype(float)
        # cat.codes returns -1 for NaN categories; map to 0 to keep the fit happy
        X[c] = col.replace(-1, 0).fillna(0)
    return X


def _num(s):
    """Numeric view of one column: category codes for objects, else float; NaN/-1 -> 0."""
    v = s.astype("category").cat.codes if s.dtype == object else s.astype(float)
    return v.replace(-1, 0).fillna(0).values.astype(float)


def hyp_signature(primary, partners=None, moderator=None):
    """Canonical identity of a hypothesis = the SET of variables it involves.

    Set-level dedup forbids only rediscovering the SAME variable set under a
    relabeled primary (e.g. {X,Y,Z} funded as X-primary then re-proposed as
    Y-primary). A genuinely new set that reuses a confirmed variable is allowed.
    """
    members = {primary}
    for p in (partners or []):
        if p:
            members.add(p)
    if moderator:
        members.add(moderator)
    return frozenset(members)


def _basis_specs_from_sample(df_sample, uniq):
    """Flexible-basis column specs (quantile-bin edges for numerics, one-hot levels
    for categoricals) computed from the FIT subsample, so the fit and the streamed
    full-row prediction share one basis definition. Each spec expands to one column
    via ``_basis_column``; specs hold only edges/levels, never full-row arrays."""
    specs = []
    for c in uniq:
        s = df_sample[c]
        if s.dtype == object:
            for lvl in s.dropna().unique():
                specs.append(("level", c, lvl))
        else:
            v = s.astype(float)
            edges = np.unique(np.quantile(v.dropna().values, np.linspace(0, 1, 6)))
            if len(edges) >= 3:  # >= 2 usable bins
                for b in range(len(edges) - 1):
                    specs.append(("bin", c, (edges, b)))
            else:  # degenerate (near-constant) column -> standardized fallback
                specs.append(("znum", c, (float(v.mean()), float(v.std() or 1))))
    return specs


def _basis_column(df, spec):
    """Materialize ONE flexible-basis column (full length) for a spec. Never stacks."""
    kind, c, payload = spec
    col = df[c]
    if kind == "level":
        return (col == payload).astype(float).values
    if kind == "znum":
        mean, std = payload
        v = col.astype(float).fillna(mean).values
        return (v - mean) / std
    edges, b = payload  # kind == "bin"
    v = col.astype(float).values
    idx = np.digitize(v, edges[1:-1])          # 0..nbins-1, matching np.quantile edges
    out = (idx == b).astype(float)
    out[np.isnan(v)] = 0.0
    return out


def residualize(df, outcome, confirmed_vars):
    """Return the outcome with all confirmed variables' effects removed —
    INCLUDING non-linear structure. A plain linear fit only removes the linear
    projection, leaving a confirmed threshold or U-shape effect in the residual
    (which would make the variable look unexplained and get re-mined). We expand
    each confirmed variable into quantile bins (a flexible step basis) so
    threshold / monotone / non-monotone effects are all absorbed, then regress
    those out.

    Large-n path (rows > SUBSAMPLE_ROW_THRESHOLD): the full-data stacked design
    matrix is ~5*confirmed_vars columns x n rows. Since this only residualizes (a
    gate, not a published estimate), fit the flexible basis on a fixed subsample
    and stream the full-row prediction one basis column at a time.
    """
    y = df[outcome].astype(float).values
    uniq = list(dict.fromkeys(c for c in confirmed_vars if c in df.columns))
    if not uniq:
        return y - y.mean()
    n = len(df)
    if n > SUBSAMPLE_ROW_THRESHOLD:
        idx = np.linspace(0, n - 1, FIT_SUBSAMPLE_N).astype(int)
        df_s = df.iloc[idx]
        specs = _basis_specs_from_sample(df_s, uniq)
        if not specs:
            return y - y.mean()
        Xs = np.nan_to_num(np.column_stack([_basis_column(df_s, sp) for sp in specs]),
                           nan=0.0, posinf=0.0, neginf=0.0)
        m = LinearRegression().fit(Xs, y[idx])
        # stream full-row prediction; never hold more than pred + one basis column
        pred = np.full(n, float(m.intercept_), dtype=float)
        for sp, coef in zip(specs, m.coef_):
            pred += coef * np.nan_to_num(_basis_column(df, sp), nan=0.0, posinf=0.0, neginf=0.0)
        return y - np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    cols = []
    for c in uniq:
        s = df[c]
        if s.dtype == object:
            for lvl in s.dropna().unique():  # one-hot the levels
                cols.append((s == lvl).astype(float).values)
        else:
            v = s.astype(float)
            try:  # 5 quantile-bin indicators = flexible basis (threshold/non-monotone)
                q = pd.qcut(v, 5, labels=False, duplicates="drop")
                for b in np.unique(q.dropna()):
                    cols.append((q == b).fillna(False).astype(float).values)
            except Exception:
                filled = v.fillna(v.median())
                cols.append(((filled - filled.mean()) / (filled.std() or 1)).values)
    X = np.nan_to_num(np.column_stack(cols), nan=0.0, posinf=0.0, neginf=0.0)
    m = LinearRegression().fit(X, y)
    pred = np.nan_to_num(m.predict(X), nan=0.0, posinf=0.0, neginf=0.0)
    return y - pred


def liveness(df, var, residual, confirmed_vars):
    """Nonlinear value signal: how much of the residual a flexible (quantile-binned)
    encoding of ``var`` explains. A plain linear correlation is blind to the hard
    archetypes — a U-shape or threshold has near-zero linear correlation, so a
    linear gate funds monotone distractors and misses non-monotone / categorical /
    weak effects. We bin ``var`` into quantiles, mean-encode the residual within
    bins, and correlate that encoding with the residual. Confirmed vars are first
    residualized out of both so already-explained paths don't inflate the score.

    Interaction effects have ~0 marginal signal by construction and are exempted
    from this gate upstream (see the triage in the run loop).
    """
    if var not in df.columns:
        return 0.0
    raw = df[var].astype("category").cat.codes if df[var].dtype == object else df[var].astype(float)
    x = raw.replace(-1, 0).fillna(0).values.astype(float)
    if x.std() == 0 or np.std(residual) == 0:
        return 0.0
    r = np.nan_to_num(np.asarray(residual, dtype=float), nan=0.0)
    ctrl = [c for c in confirmed_vars if c in df.columns and c != var]
    if ctrl:
        n = len(df)
        if n > SUBSAMPLE_ROW_THRESHOLD:
            # Large-n: fit the control projection on a subsample, stream the
            # full-row prediction one control column at a time.
            idx = np.linspace(0, n - 1, FIT_SUBSAMPLE_N).astype(int)
            Cs = np.nan_to_num(numeric_view(df.iloc[idx], ctrl).values.astype(float),
                               nan=0.0, posinf=0.0, neginf=0.0)
            m = LinearRegression().fit(Cs, r[idx])
            pred_c = np.full(n, float(m.intercept_), dtype=float)
            for c, coef in zip(ctrl, m.coef_):
                col = df[c].astype("category").cat.codes if df[c].dtype == object else df[c].astype(float)
                pred_c += coef * np.nan_to_num(col.replace(-1, 0).fillna(0).values.astype(float),
                                               nan=0.0, posinf=0.0, neginf=0.0)
            r = r - np.nan_to_num(pred_c, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            C = np.nan_to_num(numeric_view(df, ctrl).values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
            pred_c = np.nan_to_num(LinearRegression().fit(C, r).predict(C), nan=0.0, posinf=0.0, neginf=0.0)
            r = r - pred_c
        if np.std(r) == 0:
            return 0.0
    # flexible mean-encoding: E[residual | quantile-bin(var)]
    try:
        q = pd.qcut(pd.Series(x), 5, labels=False, duplicates="drop")
        enc = pd.Series(r).groupby(q).transform("mean").fillna(0).values
    except Exception:
        enc = x  # fallback: raw value
    if np.std(enc) == 0:
        return 0.0
    c = np.corrcoef(enc, r)[0, 1]
    return abs(c) if c == c else 0.0


def conditional_liveness(df, outcome, var, residual, confirmed_vars, max_scan=8):
    """Super-additivity gate for CONDITIONAL proposals (interaction/reversal/subgroup).

    These have ~0 marginal signal by construction, so the single-variable liveness
    gate is blind to them. Scan a handful of candidate moderators and test whether
    any pair (var x moderator) carries a significant super-additive term over the
    additive baseline: fit ``residual ~ A + M + A:M`` (subsampled for speed) and
    require the interaction coefficient's p < ALPHA_INTERACTION.

    Returns ``(best_pvalue, best_moderator, best_coef)``. Falls back to allowing the
    proposal (p=0) if statsmodels is unavailable, so behaviour only tightens.
    """
    if var not in df.columns:
        return 1.0, None, 0.0
    if not _HAVE_SM:
        return 0.0, None, 0.0  # can't test -> preserve exempt behaviour
    # Rank candidate moderators by a cheap bin-encoded correlation against the
    # (already-residualized) residual, then OLS-test only the top ``max_scan``.
    r = np.nan_to_num(np.asarray(residual, dtype=float), nan=0.0)
    cands = [c for c in df.columns
             if c != var and c != outcome and c not in confirmed_vars]
    scored = []
    for c in cands:
        try:
            x = _num(df[c])
            if x.std() == 0:
                continue
            q = pd.qcut(pd.Series(x), 5, labels=False, duplicates="drop")
            enc = pd.Series(r).groupby(q).transform("mean").fillna(0).values
            if np.std(enc) == 0:
                continue
            cc = np.corrcoef(enc, r)[0, 1]
            scored.append((c, abs(cc) if cc == cc else 0.0))
        except Exception:
            continue
    scored.sort(key=lambda t: t[1], reverse=True)
    mods = [c for c, _ in scored[:max_scan]]
    if not mods:
        return 1.0, None, 0.0

    n = len(df)
    idx = slice(None) if n <= INTERACTION_SCAN_N else np.linspace(0, n - 1, INTERACTION_SCAN_N).astype(int)
    y = np.asarray(residual, dtype=float)[idx]
    A = _num(df[var])[idx]
    A = (A - A.mean()) / (A.std() or 1)
    best = (1.0, None, 0.0)
    for m in mods:
        M = _num(df[m])[idx]
        if M.std() == 0:
            continue
        M = (M - M.mean()) / (M.std() or 1)
        X = np.column_stack([np.ones(len(A)), A, M, A * M])
        try:
            fit = sm.OLS(y, X).fit()
            p, coef = float(fit.pvalues[3]), float(fit.params[3])
        except Exception:
            continue
        if p < best[0]:
            best = (p, m, coef)
    return best


def recompute_claim_ok(df, outcome, var, hyp_text):
    """Reject a hypothesis whose stated direction the data confidently contradicts.

    Recompute the true monotone direction of ``var`` on ``outcome`` (top-vs-bottom
    quantile difference), parse the direction the text CLAIMS, and reject only on a
    confident contradiction: the recomputed effect is significant (p < CLAIM_SIG_P)
    AND points the opposite way. Ambiguous / non-monotone / interaction claims are
    left alone (no single stated direction to contradict). Returns ``(ok, info)``.
    """
    if var not in df.columns:
        return True, {"reason": "var not in df"}
    y = df[outcome].astype(float).values
    col = df[var]
    try:
        if col.dtype == object or col.nunique() <= 2:
            x = _num(col)
            hi = x >= np.median(x) if len(np.unique(x)) > 2 else x == x.max()
            lo = ~hi
        else:
            v = col.astype(float)
            q1, q3 = v.quantile([0.25, 0.75])
            hi, lo = (v >= q3).values, (v <= q1).values
        r_hi, r_lo = y[hi].mean(), y[lo].mean()
        n_hi, n_lo = hi.sum(), lo.sum()
        if n_hi < 30 or n_lo < 30:
            return True, {"reason": "too few rows to test"}
        se = np.sqrt(r_hi * (1 - r_hi) / n_hi + r_lo * (1 - r_lo) / n_lo)
        z = (r_hi - r_lo) / (se + 1e-12)
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        true_dir = "+" if r_hi >= r_lo else "-"
    except Exception:
        return True, {"reason": "recompute failed"}

    t = hyp_text.lower()
    pos_kw = ["increase", "higher", "more likely", "positively", "raises", "boost", "greater",
              "elevat", "drives up", "increases the", "positive effect"]
    neg_kw = ["decrease", "lower", "less likely", "negatively", "reduces", "reduc", "drop",
              "drives down", "diminish", "negative effect", "fewer"]
    has_pos = any(w in t for w in pos_kw)
    has_neg = any(w in t for w in neg_kw)
    # non-monotone / interaction claims have no single stated direction -> don't police
    if any(w in t for w in ["u-shape", "u shape", "non-monoton", "nonmonoton", "sweet spot",
                            "inverted", "interaction", "depends on", "only when", "reversal"]):
        return True, {"reason": "conditional/non-monotone claim — direction not policed",
                      "true_dir": true_dir, "p": float(p)}
    if has_pos == has_neg:
        return True, {"reason": "no unambiguous stated direction", "true_dir": true_dir, "p": float(p)}
    claimed_dir = "+" if has_pos else "-"

    contradiction = (claimed_dir != true_dir) and (p < CLAIM_SIG_P)
    return (not contradiction), {"claimed_dir": claimed_dir, "true_dir": true_dir,
                                 "p": float(p), "r_hi": float(r_hi), "r_lo": float(r_lo),
                                 "rejected": contradiction}


def survives_adjustment(df, outcome, var, confirmed_vars):
    """Validity gate: does the seed variable's effect survive adjusting for observed
    covariates? Partial the top-K covariates most correlated with ``var`` out of
    both the outcome and ``var`` via cross-fit gradient-boosted nuisance models
    (linear-DML with a nonlinear stage), on the same FIT_SUBSAMPLE_N subsample the
    other gates use. REJECT only when BOTH fire: the adjusted effect is negligible
    (|adj| < ADJ_EFFECT_FLOOR) AND adjustment washed out most of the marginal
    (|adj|/|marginal| < ADJ_RETAIN_FRAC). A real effect clears either bar. Fail-open
    when sklearn is absent or the fit errors, so behaviour only tightens.

    Only meaningful when confounders are OBSERVED; under a hidden confounder no
    observational adjustment can recover the truth. Returns ``(ok, info)``.
    """
    if not _HAVE_ADJ or var not in df.columns:
        return True, {"reason": "adjustment unavailable"}
    feats = [c for c in df.columns if c != outcome and c != var]
    if not feats:
        return True, {"reason": "no covariates to adjust for"}
    try:
        n = len(df)
        idx = (slice(None) if n <= FIT_SUBSAMPLE_N
               else np.linspace(0, n - 1, FIT_SUBSAMPLE_N).astype(int))

        def _z(a):
            a = np.asarray(a, dtype=float)
            s = a.std()
            return (a - a.mean()) / (s if s else 1.0)

        T = _z(_num(df[var])[idx])
        Y = _z(_num(df[outcome])[idx])
        if np.std(T) == 0 or np.std(Y) == 0:
            return True, {"reason": "degenerate column"}
        marginal = float(np.corrcoef(T, Y)[0, 1])
        # top-K covariates most correlated with the treatment = the confounders
        corrs = []
        for c in feats:
            xc = _num(df[c])[idx]
            if np.std(xc) == 0:
                continue
            cc = np.corrcoef(_z(xc), T)[0, 1]
            corrs.append((c, abs(cc) if cc == cc else 0.0))
        corrs.sort(key=lambda t: t[1], reverse=True)
        conf = [c for c, _ in corrs[:ADJ_TOPK_CONFOUNDERS]]
        if not conf:
            return True, {"reason": "no adjustable covariates"}
        X = np.column_stack([_z(_num(df[c])[idx]) for c in conf])
        # cross-fit residuals of Y and T on X (2-fold), then OLS = adjusted effect
        Yt = np.zeros(len(Y))
        Tt = np.zeros(len(T))
        kf = KFold(n_splits=2, shuffle=True, random_state=42)
        for tr, te in kf.split(X):
            my = HistGradientBoostingRegressor(max_iter=120, max_depth=4, learning_rate=0.1,
                                               random_state=42).fit(X[tr], Y[tr])
            mt = HistGradientBoostingRegressor(max_iter=120, max_depth=4, learning_rate=0.1,
                                               random_state=42).fit(X[tr], T[tr])
            Yt[te] = Y[te] - my.predict(X[te])
            Tt[te] = T[te] - mt.predict(X[te])
        denom = float(Tt @ Tt)
        if denom < 1e-9:
            return True, {"reason": "no residual variation in treatment"}
        adj = float(Tt @ Yt / denom)
        retain = abs(adj) / (abs(marginal) + 1e-9)
        washed_out = abs(adj) < ADJ_EFFECT_FLOOR and retain < ADJ_RETAIN_FRAC
        return (not washed_out), {"marginal": round(marginal, 4), "adjusted": round(adj, 4),
                                  "retain": round(retain, 3), "rejected": washed_out}
    except Exception as e:
        return True, {"reason": f"adjustment errored: {type(e).__name__}"}
