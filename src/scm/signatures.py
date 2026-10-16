# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""do()-verified structural signatures.

Every function here answers one question with a controlled experiment, never with
text: *is a given structural form actually present in the data for a given
variable?* The simulator owns every equation, so ``sim.ate`` / ``sim.do`` give the
exact interventional contrast that defines each archetype.

Two entry points:

* :func:`signature_present` takes a planted ground-truth entry (from
  ``Simulator.answer_key``) and checks the archetype-defining contrast. Used by the
  RECALL scorer to confirm a plant's signature exists before crediting a recovery.
* :func:`asserted_form_holds` takes a *method's own* claim (variable + form, no
  planted term) and checks whether that asserted form survives do(). Used by the
  PRECISION scorer. It reuses the same contrasts, handed a synthetic entry.

The archetype vocabulary matches ``spec.Term.archetype`` tags:
``monotone_threshold, non_monotone, derived_feature, interaction_reversal,
subpopulation_only, emergent_combination, categorical_level, simpsons_paradox,
mediation_chain``.
"""

from __future__ import annotations

import numpy as np

# do() effect bar shared across every signature: an interventional contrast below
# this in standardized-outcome units is treated as no real effect.
MIN_EFFECT = 0.02


def _get_term(spec, entry_id):
    """The planted Term for a labeled hypothesis id (or None)."""
    for t in spec.terms:
        if t.id == entry_id:
            return t
    return None


def signature_present(sim, key_entry, df, n=60_000, min_effect=MIN_EFFECT):
    """Run archetype-specific do() experiments to verify the structural signature of
    a labeled hypothesis is actually present in the data. Returns True if the
    archetype-defining contrast is significant.

    A proposal covers a labeled hypothesis only if it names the right variable AND
    the claimed structural form matches the do()-verified signature checked here.
    ``key_entry`` is a ``{"id", "archetype", "variables", "moderator"}`` dict as
    produced by ``Simulator.answer_key`` (or a synthetic one for precision).
    """
    arch = key_entry["archetype"]
    vars_ = key_entry["variables"]
    mod_v = key_entry.get("moderator")
    outcome = sim.spec.outcome_name

    if not vars_:
        return False

    primary = vars_[0]
    col = df[primary]

    try:
        if arch == "monotone_threshold":
            # signature: a real global effect AND the effect is concentrated in ONE arm
            # (either top-vs-mid or mid-vs-lo dominates, depending on where the threshold is)
            hi = col.quantile(0.85)
            lo = col.quantile(0.15)
            mid = col.quantile(0.50)
            r_full = sim.ate(primary, hi, lo, n=n)
            r_top_mid = sim.ate(primary, hi, mid, n=n)
            r_mid_lo = sim.ate(primary, mid, lo, n=n)
            full_real = abs(r_full["ate_causal"]) >= min_effect and r_full.get("significant", False)
            # threshold shape: one arm has >=2x the ATE of the other
            top_dom = abs(r_top_mid["ate_causal"]) >= 2 * abs(r_mid_lo["ate_causal"] or 1e-9)
            lo_dom = abs(r_mid_lo["ate_causal"]) >= 2 * abs(r_top_mid["ate_causal"] or 1e-9)
            return full_real and (top_dom or lo_dom)

        elif arch == "non_monotone":
            # signature: both extremes beat the middle (U-shape or inverted-U)
            hi = col.quantile(0.85)
            lo = col.quantile(0.15)
            mid_lo = col.quantile(0.40)
            mid_hi = col.quantile(0.60)
            r_hi_mid = sim.ate(primary, hi, mid_hi, n=n)
            r_lo_mid = sim.ate(primary, lo, mid_lo, n=n)
            hi_beats_mid = abs(r_hi_mid["ate_causal"]) >= min_effect
            lo_beats_mid = abs(r_lo_mid["ate_causal"]) >= min_effect
            # same sign means both extremes pull the same direction vs mid (U or inverted-U)
            same_sign = np.sign(r_hi_mid["ate_causal"]) == np.sign(r_lo_mid["ate_causal"])
            return hi_beats_mid and lo_beats_mid and same_sign

        elif arch == "derived_feature":
            # both component variables named, and the primary still has a real effect.
            # (ratio-vs-product structure is not directly testable without the formula,
            # so the signature requires BOTH variables named + a real primary effect.)
            if len(vars_) < 2:
                return False
            hi = col.quantile(0.85)
            lo = col.quantile(0.15)
            r = sim.ate(primary, hi, lo, n=n)
            return abs(r["ate_causal"]) >= min_effect and r.get("significant", False)

        elif arch == "interaction_reversal":
            # signature: effect of the treatment REVERSES sign between moderator-hi and
            # moderator-lo subpopulations. Try both variable orderings, since the answer
            # key may store (treatment, moderator) either way.
            all_vars = list(dict.fromkeys(vars_ + ([mod_v] if mod_v else [])))
            if len(all_vars) < 2:
                return False

            def _check_reversal(treat, mod):
                if treat not in df.columns or mod not in df.columns:
                    return False
                t_col = df[treat]
                t_hi = t_col.quantile(0.85)
                t_lo = t_col.quantile(0.15)
                m_col = df[mod]
                if m_col.nunique() <= 3:
                    m_max, m_min = m_col.max(), m_col.min()
                    if m_max == m_min:
                        return False
                    c_hi = lambda d, mv=mod, v=m_max: np.isclose(d[mv].astype(float), float(v))
                    c_lo = lambda d, mv=mod, v=m_min: np.isclose(d[mv].astype(float), float(v))
                else:
                    mhi = m_col.quantile(0.70)
                    mlo = m_col.quantile(0.30)
                    c_hi = lambda d, mv=mod, t=mhi: d[mv].astype(float) >= t
                    c_lo = lambda d, mv=mod, t=mlo: d[mv].astype(float) <= t
                r1 = sim.ate(treat, t_hi, t_lo, n=n, condition=c_hi)
                r2 = sim.ate(treat, t_hi, t_lo, n=n, condition=c_lo)
                a1 = r1.get("ate_causal", 0.0) or 0.0
                a2 = r2.get("ate_causal", 0.0) or 0.0
                return (np.sign(a1) != np.sign(a2) and
                        abs(a1) >= min_effect and abs(a2) >= min_effect)

            for potential_mod in all_vars:
                if potential_mod != primary and _check_reversal(primary, potential_mod):
                    return True
            for potential_treat in all_vars:
                for potential_mod in (v for v in all_vars if v != potential_treat):
                    if _check_reversal(potential_treat, potential_mod):
                        return True
            return False

        elif arch == "subpopulation_only":
            # the moderator GATES the treatment effect; the treatment is the other var.
            treat_var = primary
            gate_var = mod_v
            if mod_v == primary and len(vars_) > 1:
                treat_var = [v for v in vars_ if v != mod_v][0]
            elif not mod_v:
                gate_var = None

            treat_col = df[treat_var] if treat_var in df.columns else col
            t_hi = treat_col.quantile(0.85)
            t_lo = treat_col.quantile(0.15)

            r_global = sim.ate(treat_var, t_hi, t_lo, n=n)
            global_ate = abs(r_global["ate_causal"])

            if gate_var and gate_var in df.columns:
                gate_col = df[gate_var]
                term = _get_term(sim.spec, key_entry["id"])
                if term and term.moderator_quantile:
                    gate_thr = gate_col.quantile(term.moderator_quantile)
                elif gate_col.nunique() <= 3:
                    gate_thr = None
                else:
                    gate_thr = gate_col.quantile(0.70)

                if gate_thr is None:
                    gate_max = gate_col.max()
                    cond = lambda d, gv=gate_var, gm=gate_max: np.isclose(d[gv].astype(float), float(gm))
                else:
                    cond = lambda d, gv=gate_var, gt=gate_thr: d[gv].astype(float) >= gt

                r_cond = sim.ate(treat_var, t_hi, t_lo, n=n, condition=cond)
                cond_ate = abs(r_cond.get("ate_causal", 0.0) or 0.0)
                cond_real = cond_ate >= min_effect
                if gate_thr is not None:
                    inv_cond = lambda d, gv=gate_var, gt=gate_thr: d[gv].astype(float) < gt
                    r_inv = sim.ate(treat_var, t_hi, t_lo, n=n, condition=inv_cond)
                    inv_ate = abs(r_inv.get("ate_causal", 0.0) or 0.0)
                    cond_ate = max(cond_ate, inv_ate)
                # subpopulation: conditional effect amplified vs global by at least 1.3x
                amplified = cond_ate > global_ate * 1.3 or global_ate < min_effect
                return cond_real and amplified
            return global_ate >= min_effect

        elif arch == "emergent_combination":
            # three-variable conjunction: test v3's effect within the high region of v1,v2
            if len(vars_) < 3:
                return False
            v1, v2, v3 = vars_[:3]
            hi1 = df[v1].quantile(0.70)
            hi2 = df[v2].quantile(0.70)
            cond = (lambda d, v1=v1, v2=v2, h1=hi1, h2=hi2:
                    (d[v1].astype(float) >= h1) & (d[v2].astype(float) >= h2))
            col3 = df[v3]
            hi3 = col3.quantile(0.85)
            lo3 = col3.quantile(0.15)
            r_cond = sim.ate(v3, hi3, lo3, n=n, condition=cond)
            r_global = sim.ate(v3, hi3, lo3, n=n)
            cond_real = abs(r_cond.get("ate_causal", 0.0) or 0.0) >= min_effect
            amplified = abs(r_cond.get("ate_causal", 0.0) or 0.0) > abs(r_global["ate_causal"])
            return cond_real and amplified

        elif arch == "categorical_level":
            # a specific level drives the outcome — tested as a level-vs-base do() contrast
            if col.dtype != object and col.nunique() > 10:
                return False
            levels = list(col.unique())
            if len(levels) < 2:
                return False
            rates = {lv: df.loc[col == lv, outcome].mean() for lv in levels}
            best = max(rates, key=lambda lv: rates[lv])
            worst = min(rates, key=lambda lv: rates[lv])
            if best == worst:
                return False
            r = sim.ate(primary, best, worst, n=n)
            return abs(r["ate_causal"]) >= min_effect and r.get("significant", False)

        elif arch in ("simpsons_paradox", "mediation_chain"):
            # GRAPH archetypes: their signature is a do()-revealed gap between the
            # OBSERVATIONAL and CAUSAL effect on the treatment (Simpson's: a confounder
            # flips/inflates the marginal; mediation: the treatment acts through a
            # mediator). Needs the planted term to locate the treatment variable.
            term = _get_term(sim.spec, key_entry["id"])
            if term is None or not getattr(term, "operands", None):
                return False
            if arch == "simpsons_paradox":
                tv = term.operands[0].var
            else:  # mediation_chain: term sits on the mediator; treatment is its parent
                mv = term.operands[0].var
                node = sim.spec.node(mv)
                tv = node.parents[0] if (node and getattr(node, "parents", None)) else mv
            if tv not in df.columns or df[tv].dtype == object:
                return False
            r = sim.ate(tv, df[tv].quantile(0.85), df[tv].quantile(0.15), n=n)
            gap = abs(r.get("confounding_bias", 0) or 0)
            return gap > min_effect or (r.get("significant", False)
                                        and abs(r["ate_causal"]) > min_effect)

        else:
            return False

    except Exception:
        return False


# ---------------------------------------------------------------------------
# term-free verifiers for a METHOD'S asserted form (precision, no planted term)
# ---------------------------------------------------------------------------

def interaction_holds(sim, df, treat, mod, n=60_000, min_effect=MIN_EFFECT):
    """Moderation (amplification): the do()-effect of ``treat`` has a materially
    different magnitude across levels of ``mod`` (at least one arm a real effect).
    Covers 'stronger for subgroup' claims that do NOT flip sign — the common
    committed interaction that :func:`signature_present`'s reversal test misses."""
    if not mod or treat not in df.columns or mod not in df.columns:
        return False
    tcol = df[treat]
    if tcol.dtype == object:
        return False
    t_hi, t_lo = tcol.quantile(0.85), tcol.quantile(0.15)
    mcol = df[mod]
    try:
        if mcol.nunique() <= 3:
            m_max, m_min = mcol.max(), mcol.min()
            if m_max == m_min:
                return False
            c_hi = lambda d, mv=mod, v=m_max: np.isclose(d[mv].astype(float), float(v))
            c_lo = lambda d, mv=mod, v=m_min: np.isclose(d[mv].astype(float), float(v))
        else:
            mhi, mlo = mcol.quantile(0.70), mcol.quantile(0.30)
            c_hi = lambda d, mv=mod, t=mhi: d[mv].astype(float) >= t
            c_lo = lambda d, mv=mod, t=mlo: d[mv].astype(float) <= t
        a_hi = sim.ate(treat, t_hi, t_lo, n=n, condition=c_hi).get("ate_causal", 0.0) or 0.0
        a_lo = sim.ate(treat, t_hi, t_lo, n=n, condition=c_lo).get("ate_causal", 0.0) or 0.0
    except Exception:
        return False
    return (abs(a_hi - a_lo) >= min_effect) and (max(abs(a_hi), abs(a_lo)) >= min_effect)


def simpsons_holds(sim, df, treat, n=60_000, min_effect=MIN_EFFECT):
    """Simpson's / confounded-association: the observational and causal do() effects
    on the claimed variable disagree by more than the effect bar (a real confounding
    gap that adjustment would expose)."""
    if not treat or treat not in df.columns or df[treat].dtype == object:
        return False
    try:
        r = sim.ate(treat, df[treat].quantile(0.85), df[treat].quantile(0.15), n=n)
    except Exception:
        return False
    return abs(r.get("confounding_bias", 0.0) or 0.0) > min_effect


def mediation_holds(sim, df, treat, mediator, n=60_000, min_effect=MIN_EFFECT):
    """Mediation: the treatment has a real do()-effect on the outcome AND the named
    mediator also has a real do()-effect (the chain treat->M->Y needs both links).
    Term-free proxy; with no mediator parsed the mechanism is unverifiable."""
    if not treat or not mediator or treat not in df.columns or mediator not in df.columns:
        return False
    if df[treat].dtype == object or df[mediator].dtype == object:
        return False
    try:
        rt = sim.ate(treat, df[treat].quantile(0.85), df[treat].quantile(0.15), n=n)
        rm = sim.ate(mediator, df[mediator].quantile(0.85), df[mediator].quantile(0.15), n=n)
    except Exception:
        return False
    treat_real = abs(rt.get("ate_causal", 0.0) or 0.0) >= min_effect and rt.get("significant", False)
    med_real = abs(rm.get("ate_causal", 0.0) or 0.0) >= min_effect and rm.get("significant", False)
    return treat_real and med_real
