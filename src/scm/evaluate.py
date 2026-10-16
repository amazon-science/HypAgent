# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""do()-grounded evaluation of a method's hypotheses.

Given a fitted :class:`Simulator` (which owns the ground-truth SCM) and a list of
*structured proposals*, this scores two numbers, both defined entirely by
interventional experiments — never by keyword or text matching:

* **recall** — of the mechanisms the SCM actually planted (its ``answer_key``),
  what fraction did the method recover? A plant is recovered when a proposal names
  a variable in the plant's do()-causal set AND asserts the right functional-form
  *family*, with the plant's do()-signature confirmed present in the data.
* **claim precision** — of everything the method proposed, what fraction has its
  *full asserted claim* (variable + direction + the committed structural form)
  survive do() on the method's own variable? A bare-direction claim asserts no
  mechanism and so earns no structural credit.

Both rungs come from ONE pass of experiments per proposal (an
:class:`ExperimentRunner` do-gate plus a form-signature check), so the two metrics
can never disagree about whether a given claim's effect is real.

A **structured proposal** is a dict::

    {
      "variable":   "tenure_months",     # required: the primary cause named
      "form":       "threshold",         # one of FORMS (default "bare_direction")
      "direction":  "+" | "-" | "level", # default "+"
      "moderator":  "is_mobile" | None,  # a 2nd variable: moderator / component / mediator
      "level":      <category> | None,   # categorical: treated level
      "base":       <category> | None,   # categorical: baseline level
      "text":       "..."                # optional original free text (diagnostic only)
    }

Aliases accepted for interop with :mod:`scm.parse` output: ``asserted_form`` for
``form`` and ``secondary_variable`` for ``moderator``.

This module needs NO LLM. To score free-text hypotheses, first convert them with
:func:`scm.parse.parse_hypotheses` (which does use a provider), then pass the
resulting structured claims here.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .experiment import ExperimentRunner, Hypothesis
from . import signatures as sig

SCORE_N = 60_000     # reference sample size for every do() contrast (matches generation)
MIN_EFFECT = 0.02    # standardized-outcome effect bar shared with the signature checks

# The asserted-form vocabulary. Everything maps onto a do() signature check except
# "bare_direction" (judged only on the lenient directional gate).
FORMS = {
    "bare_direction", "threshold", "non_monotone", "interaction",
    "interaction_reversal", "subpopulation", "subpopulation_only",
    "derived_feature", "mediation", "simpsons", "categorical_level", "level",
}
# forms whose asserted primary effect carries a meaningful sign we also require
_DIRECTIONAL = {"threshold", "derived_feature", "categorical_level", "level", "mediation"}

# ---------------------------------------------------------------------------
# family mapping
# ---------------------------------------------------------------------------
# The SCM tags each planted term with an archetype; the archetype maps to a
# functional-form FAMILY. Recall credits a proposal that asserts the plant's family.
PLANT_FAMILY = {
    "monotone_threshold": "SHAPE", "non_monotone": "SHAPE",
    "interaction_reversal": "MODERATION", "subpopulation_only": "MODERATION",
    "derived_feature": "COMPOSITE",
    "simpsons_paradox": "CONFOUND", "mediation_chain": "CONFOUND",
}
# a proposal's asserted form -> the family it claims
ASSERTED_FAMILY = {
    "threshold": "SHAPE", "non_monotone": "SHAPE",
    "interaction": "MODERATION", "interaction_reversal": "MODERATION",
    "subpopulation": "MODERATION", "subpopulation_only": "MODERATION",
    "derived_feature": "COMPOSITE",
    "simpsons": "CONFOUND", "simpsons_paradox": "CONFOUND",
    "mediation": "CONFOUND", "mediation_chain": "CONFOUND",
    "categorical_level": "LEVEL", "level": "LEVEL",
}
# do()-grounded ASYMMETRIC family equivalence: a COMPOSITE (ratio/product) plant
# manifests under do() as a joint two-variable effect that a method naturally calls
# an interaction, so a MODERATION assertion recovers a COMPOSITE plant (but not the
# converse — the composite signature is strictly weaker).
_EQUIV = {"COMPOSITE": {"COMPOSITE", "MODERATION"}}
# CONFOUND (Simpson's / mediation) has NO token in any method's proposal vocabulary,
# so the literal family label is a false-negative machine. For CONFOUND plants recall
# is relaxed to the conditional/reversal/subgroup structural forms in which confounding
# is actually expressed, anchored on the plant's do() signature + a real do-gate effect.
NO_VOCAB_FAMILIES = {"CONFOUND"}
CONFOUND_STRUCT_FORMS = {"interaction", "interaction_reversal", "subpopulation",
                         "subpopulation_only", "emergent_combination"}

# the 7 scored archetypes (families with a planted mechanism)
ARCHETYPES = list(PLANT_FAMILY.keys())


def _family_credits(asserted_family, plant_family):
    """True if asserting ``asserted_family`` recovers a ``plant_family`` plant."""
    if asserted_family is None:
        return False
    return asserted_family in _EQUIV.get(plant_family, {plant_family})


# ---------------------------------------------------------------------------
# normalize a proposal + verify its asserted form under do()
# ---------------------------------------------------------------------------
def _norm(p):
    """Accept both native (``form``/``moderator``) and parse.py
    (``asserted_form``/``secondary_variable``) keys; return a canonical claim."""
    form = p.get("form", p.get("asserted_form", "bare_direction"))
    if form not in FORMS:
        form = "bare_direction"
    return {
        "variable": p.get("variable"),
        "form": form,
        "direction": p.get("direction", "+"),
        "moderator": p.get("moderator", p.get("secondary_variable")),
        "level": p.get("level"),
        "base": p.get("base"),
        "subpopulation_value": p.get("subpopulation_value"),
        "text": p.get("text", ""),
    }


def _build_condition(df, claim):
    """Subpopulation mask for the directional do-gate, if the claim scopes to a
    specific value of its second variable (mirrors the paper's lenient gate)."""
    sv = claim.get("moderator")
    sval = claim.get("subpopulation_value")
    if not sv or sv not in df.columns or sval is None or str(sval).strip() == "":
        return None, ""
    col = df[sv]
    if col.dtype != object:
        try:
            tgt = float(sval)
            cond = (lambda d, sv=sv, tgt=tgt: np.isclose(d[sv].astype(float), tgt))
        except (ValueError, TypeError):
            return None, ""
    else:
        cond = (lambda d, sv=sv, lv=str(sval): d[sv].astype(str) == lv)
    return (cond, sv) if cond(df).any() else (None, "")


def verify_form(sim, df, claim, n=SCORE_N):
    """Does the claim's asserted structural form hold under do() on its own variable?
    Returns True / False, or None for ``bare_direction`` (no mechanism asserted)."""
    form = claim["form"]
    if form == "bare_direction":
        return None
    prim = claim["variable"]
    sec = claim["moderator"]

    # forms with a term-free archetype signature: hand signatures a synthetic entry
    # built from the METHOD'S own claim (id=None), not a planted term.
    reuse = {
        "threshold": "monotone_threshold",
        "non_monotone": "non_monotone",
        "interaction_reversal": "interaction_reversal",
        "subpopulation": "subpopulation_only",
        "subpopulation_only": "subpopulation_only",
        "derived_feature": "derived_feature",
        "categorical_level": "categorical_level",
        "level": "categorical_level",
    }
    if form in reuse:
        entry = {"id": None, "archetype": reuse[form],
                 "variables": [v for v in (prim, sec) if v], "moderator": sec}
        return bool(sig.signature_present(sim, entry, df, n=n, min_effect=MIN_EFFECT))
    if form == "interaction":
        return sig.interaction_holds(sim, df, prim, sec, n=n)
    if form == "simpsons":
        return sig.simpsons_holds(sim, df, prim, n=n)
    if form == "mediation":
        return sig.mediation_holds(sim, df, prim, sec, n=n)
    return False


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def evaluate(sim, proposals, df=None, n=SCORE_N):
    """Score ``proposals`` against the SCM's ground truth with do() experiments.

    Parameters
    ----------
    sim : Simulator          the fitted DGP (owns ``answer_key`` + do()).
    proposals : list[dict]   structured proposals (see the module docstring).
    df : DataFrame, optional a reference sample; regenerated at size ``n`` if omitted.
    n : int                  sample size for every do() contrast.

    Returns a report dict with ``recall`` (family rung, plus confirmed/variable rungs
    and per-archetype breakdown), ``precision`` (claim-as-stated, plus directional),
    and ``per_proposal`` / ``per_plant`` traces.
    """
    if df is None:
        df = sim.generate(n)
    runner = ExperimentRunner(sim, n=n, min_effect=MIN_EFFECT)
    variables = [c for c in df.columns if c != sim.spec.outcome_name]

    # ---- one experiment pass per proposal ---------------------------------
    records = []
    for p in proposals:
        c = _norm(p)
        var = c["variable"]
        rec = {"variable": var, "form": c["form"], "direction": c["direction"],
               "text": c["text"], "supported": False, "direction_correct": False,
               "form_holds": None, "ate_causal": float("nan"), "evaluable": False}
        if not var or var not in variables:
            rec["reason"] = "no valid variable named"
            records.append(rec)
            continue
        cond, cond_desc = _build_condition(df, c)
        h = Hypothesis(variable=var, direction=c["direction"], level=c["level"],
                       base=c["base"], condition=cond, condition_desc=cond_desc,
                       text=c["text"][:120])
        v = runner.run(h)
        rec["evaluable"] = bool(getattr(v, "evaluable", True))
        rec["supported"] = bool(rec["evaluable"] and getattr(v, "supported", False))
        rec["direction_correct"] = bool(getattr(v, "direction_correct", False))
        rec["ate_causal"] = float(getattr(v, "ate_causal", float("nan")))
        rec["form_holds"] = verify_form(sim, df, c, n=n)
        records.append(rec)

    precision = _score_precision(records)
    recall = _score_recall(sim, df, records, n=n)
    return {"recall": recall, "precision": precision,
            "per_proposal": records, "n_proposals": len(proposals)}


def _score_precision(records):
    """Claim-as-stated: fraction of ALL proposals whose full asserted claim survives
    do(). A committed shape must hold under do() (plus direction for directional
    forms); a bare-direction claim asserts no mechanism and cannot score."""
    n_correct = n_directional = n_unparseable = 0
    by_form = defaultdict(lambda: [0, 0])   # form -> [correct, total]
    for r in records:
        form = r["form"]
        by_form[form][1] += 1
        if r.get("reason") == "no valid variable named":
            n_unparseable += 1
            r["correct"] = False
            continue
        supported, dir_ok, fh = r["supported"], r["direction_correct"], r["form_holds"]
        n_directional += int(supported)
        if fh is None:                       # bare_direction: no structural credit
            correct = False
        elif form in _DIRECTIONAL:
            correct = bool(fh and supported and dir_ok)
        else:
            correct = bool(fh)               # shape signature embeds its own effect test
        r["correct"] = correct
        n_correct += int(correct)
        by_form[form][0] += int(correct)
    n = len(records)
    return {
        "n_proposed": n, "n_correct": n_correct, "n_unparseable": n_unparseable,
        "claim_precision": n_correct / n if n else 0.0,
        "n_directional": n_directional,
        "directional_precision": n_directional / n if n else 0.0,
        "by_form": {f: {"correct": c[0], "total": c[1]} for f, c in by_form.items()},
    }


def _score_recall(sim, df, records, n=SCORE_N):
    """Structure-aware recall over the SCM's answer key. Per plant: does some proposal
    name a variable in the plant's do()-causal set AND assert its form family, with the
    plant's do()-signature confirmed present? Family rung is the headline; the confirmed
    rung additionally requires the recovering claim's own form to survive do()."""
    key = [e for e in sim.answer_key() if e["archetype"] in PLANT_FAMILY]
    per_plant = []
    fam_hits, conf_hits, var_hits = [], [], []
    arch_hits = defaultdict(list)
    arch_conf = defaultdict(list)

    for e in key:
        pfam = PLANT_FAMILY[e["archetype"]]
        V = set(e["variables"])
        has_sig = bool(sig.signature_present(sim, e, df, n=n, min_effect=MIN_EFFECT))

        effect_ok = fam_ok = conf_ok = var_ok = struct_ok = struct_holds_ok = False
        for r in records:
            v = r["variable"]
            if not (v and v in V):
                continue
            if r["supported"]:               # do-gate EFFECT rung
                effect_ok = True
            if not has_sig:                  # signature gate for the form rungs
                continue
            var_ok = True
            fa = ASSERTED_FAMILY.get(r["form"])
            if _family_credits(fa, pfam):
                fam_ok = True
                if r["form_holds"] is True:
                    conf_ok = True
            if r["form"] in CONFOUND_STRUCT_FORMS:   # relaxed CONFOUND recovery
                struct_ok = True
                if r["form_holds"] is True:
                    struct_holds_ok = True

        # RECOVERED = do-gate EFFECT and CORRECT FAMILY (an AND). For CONFOUND "correct
        # family" is the relaxed conditional-structure assertion; else the family rung.
        if pfam in NO_VOCAB_FAMILIES:
            rec_ok = struct_ok and effect_ok
            conf_rec_ok = struct_holds_ok and effect_ok
        else:
            rec_ok = fam_ok
            conf_rec_ok = conf_ok

        fam_hits.append(int(rec_ok))
        conf_hits.append(int(conf_rec_ok))
        var_hits.append(int(var_ok))
        arch_hits[e["archetype"]].append(int(rec_ok))
        arch_conf[e["archetype"]].append(int(conf_rec_ok))
        per_plant.append({"id": e["id"], "archetype": e["archetype"], "family": pfam,
                          "variables": sorted(V), "has_signature": has_sig,
                          "effect_ok": effect_ok, "recovered": rec_ok,
                          "confirmed": conf_rec_ok})

    def _m(xs):
        return float(np.mean(xs)) if xs else 0.0

    return {
        "n_plants": len(key),
        "recall": _m(fam_hits),               # HEADLINE: family rung
        "confirmed_recall": _m(conf_hits),    # + recovering claim survives its own do()
        "variable_recall": _m(var_hits),      # named the right variable (any form)
        "by_archetype": {a: {"recall": _m(arch_hits[a]),
                             "confirmed": _m(arch_conf[a]),
                             "n": len(arch_hits[a])}
                         for a in sorted(arch_hits)},
        "per_plant": per_plant,
    }
