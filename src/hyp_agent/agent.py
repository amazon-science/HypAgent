# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Value-gated cascade agent — an inference-time efficiency method for causal
hypothesis discovery.

Each round:
  1. PROPOSE   K candidate directions cheaply (one LLM call -> K variable + form seeds).
  2. TRIAGE    each near-free:
        novelty  = variable/region not already explained by confirmed effects (the residual)
        liveness = |quick association with the residualized outcome| >= eps
        value    = novelty AND liveness (liveness is measured on the RESIDUAL, so it
                   rewards structure not yet explained and steers toward hard archetypes)
  3. FUND      the expensive validation loop ONLY for the top candidate.
  4. VALIDATE  the confirmation with two data checks (stated direction, adjustment),
        then residualize the outcome so the same structure cannot be re-funded.

The budget is measured in EXPENSIVE VALIDATION LOOPS (the costly unit), giving a fair
discoveries-per-compute comparison against a flat propose->validate agent.
"""

import json
import time
from pathlib import Path

import numpy as np

from .config import ALPHA_INTERACTION, INT_EFFECT_FLOOR
from .gates import (
    residualize, liveness, conditional_liveness,
    recompute_claim_ok, survives_adjustment, hyp_signature,
)
from .llm import get_provider
from .proposer import propose, propose_analytical
from .validator import build_system_prompt, run_validation

#: components the ablation study can disable (each turns off exactly one part).
ABLATIONS = ("flat", "no_analytical", "no_interaction_gate",
             "no_claim_check", "no_adjustment", "no_dedup")


def _resume(hp, lp):
    """Load prior state from an interrupted run and dedup it.

    Repeated crash-resume cycles could otherwise re-fund a variable and append
    duplicate rows, inflating the spent-budget counter. Keep the first hypothesis
    per seed_var and the first log row per funded variable.
    """
    hypotheses = json.load(open(hp))
    seen = set()
    hypotheses = [h for h in hypotheses
                  if not (h["seed_var"] in seen or seen.add(h["seed_var"]))]
    log = []
    if lp.exists():
        fseen = set()
        for e in json.load(open(lp)):
            v = e.get("funded")
            if v and v in fseen:
                continue
            if v:
                fseen.add(v)
            log.append(e)
    return hypotheses, log


def run(df, meta, budget, out_dir, k_propose=6, live_eps=0.02,
        provider_name="bedrock", analytical=True, ablate=None):
    """Run the cascade agent on ``df`` (which must have a binary ``outcome`` column).

    Args:
        df: the dataset. Must contain an ``outcome`` column.
        meta: dict with ``domain``, ``scenario``, ``outcome_positive_rate``,
            ``context`` (see :mod:`hyp_agent.data`).
        budget: number of expensive validation loops to spend.
        out_dir: directory for the incremental result + log JSON.
        k_propose: candidate directions proposed per round.
        live_eps: minimum liveness for a marginal proposal to be fundable.
        provider_name: LLM backend ("bedrock", "anthropic", or "openai").
        analytical: use the code-analyzing proposer (recommended) vs schema-only.
        ablate: names from :data:`ABLATIONS` to DISABLE (default: full method).

    Writes ``<stem>.json`` (confirmed hypotheses) and ``<stem>_log.json`` (per-round
    trace) to ``out_dir`` incrementally, and resumes from them if present.
    """
    ablate = set(ablate or [])
    unknown = ablate - set(ABLATIONS)
    if unknown:
        raise ValueError(f"unknown ablation(s): {sorted(unknown)}; valid: {ABLATIONS}")

    name = meta.get("domain", "dataset")
    outcome = "outcome"
    provider = get_provider(provider_name)
    system_prompt = build_system_prompt(meta)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # each variant writes to a separate file so it never clobbers the full-method run
    if ablate:
        stem = "agent_ablate_" + "_".join(sorted(ablate))
    elif analytical:
        stem = "agent_cascade_analytical"
    else:
        stem = "agent_cascade"
    use_analytical = analytical and ("no_analytical" not in ablate)

    confirmed_vars, hypotheses, log = [], [], []
    confirmed_sets = set()   # set of frozensets — variable sets already confirmed

    hp = out_dir / f"{stem}.json"
    lp = out_dir / f"{stem}_log.json"
    if hp.exists():
        hypotheses, log = _resume(hp, lp)
        confirmed_vars = [h["seed_var"] for h in hypotheses]
        confirmed_sets = {hyp_signature(h["seed_var"], moderator=h.get("moderator"))
                          for h in hypotheses}
        # rewrite the deduped state so the counter and saved output are clean
        json.dump(hypotheses, open(hp, "w"), indent=2, default=str)
        json.dump(log, open(lp, "w"), indent=2, default=str)
        print(f"[{name}] resuming from {len(hypotheses)} hypotheses "
              f"({len([l for l in log if l.get('funded')])} loops already spent)")

    tok = {"in": 0, "out": 0}
    spent = len([l for l in log if l.get("funded")])  # expensive loops already used
    t0 = time.time()

    while spent < budget:
        # --- residualize: remove all confirmed effects from the outcome ---
        try:
            if "flat" in ablate:
                # value-gate ABLATION: never residualize, so no steering toward
                # unexplained structure — a plain propose->validate agent.
                y = df[outcome].astype(float).values
                residual = y - y.mean()
            else:
                residual = residualize(df, outcome, confirmed_vars)
        except Exception:
            y = df[outcome].astype(float).values
            residual = y - y.mean()

        # --- propose K candidate directions ---
        if use_analytical:
            cands = propose_analytical(provider, df, meta, hypotheses, residual, k_propose, tok)
        else:
            cands = propose(provider, meta, hypotheses, k_propose, tok)

        # --- triage (near free): score fundability of each candidate ---
        scored = []
        # standardize the interaction coef so INT_EFFECT_FLOOR stays comparable
        # across rounds as the residual shrinks.
        resid_sd = float(np.std(np.asarray(residual, dtype=float))) or 1.0
        for c in cands:
            v = c.get("variable")
            if not v or v not in df.columns:
                continue
            try:
                live = liveness(df, v, residual, confirmed_vars)
            except Exception:
                live = 0.0
            # interaction/reversal proposals have ~0 marginal signal by construction,
            # so gate them on SUPER-ADDITIVITY instead of single-variable liveness.
            form = str(c.get("direction", "")).lower()
            is_conditional = any(w in form for w in
                                 ["interaction", "reversal", "subgroup", "moderat", "combination"])
            cond_p, cond_mod, cond_coef = (1.0, None, 0.0)
            if "flat" in ablate:
                fundable_flag = True
            elif is_conditional:
                if "no_interaction_gate" in ablate:
                    fundable_flag = True   # blind-fund conditionals (old behaviour)
                else:
                    try:
                        cond_p, cond_mod, cond_coef = conditional_liveness(df, outcome, v, residual, confirmed_vars)
                    except Exception:
                        cond_p, cond_mod, cond_coef = (0.0, None, 0.0)  # fail-open
                    # require BOTH a significant AND a non-microscopic interaction:
                    # a coef significant-but-tiny at large n is re-mining fuel, and
                    # the floor is what makes set-dedup safe. cond_coef is in raw
                    # residual units; standardize by the residual scale.
                    cond_coef_std = abs(cond_coef) / resid_sd
                    fundable_flag = (cond_p < ALPHA_INTERACTION) and (cond_coef_std >= INT_EFFECT_FLOOR)
            else:
                fundable_flag = live >= live_eps
            scored.append({**c, "liveness": round(float(live), 4),
                           "conditional": is_conditional, "value": fundable_flag,
                           "interaction_p": round(float(cond_p), 5) if is_conditional else None,
                           "moderator": cond_mod})

        # --- SET-LEVEL DEDUP: exclude a candidate only if its full variable SET was
        # already confirmed (not just its primary), so a genuine new interaction that
        # reuses a confirmed variable is still allowed. ---
        if "no_dedup" in ablate:
            fundable = [c for c in scored if c["value"]]
        else:
            fundable = [c for c in scored if c["value"]
                        and hyp_signature(c["variable"], c.get("partners"), c.get("moderator"))
                        not in confirmed_sets]
        # prioritize by liveness; equal-liveness marginals sort before conditionals
        # so the budget is not spent on speculative interactions. FLAT keeps the
        # model's raw proposal order (no value-based ranking).
        if "flat" not in ablate:
            fundable.sort(key=lambda c: (c["liveness"], not c["conditional"]), reverse=True)
        declined = [c for c in scored if not c["value"]]

        if not fundable:
            log.append({"round": spent, "funded": None, "declined": len(declined),
                        "note": "no fundable candidate this round"})
            json.dump(log, open(lp, "w"), indent=2, default=str)
            if len(log) > budget * 4:   # avoid an infinite loop of cheap rounds
                break
            continue

        # --- FUND the expensive validation loop for the top candidate ---
        top = fundable[0]
        seed_note = top["note"] + f" (variable: {top['variable']}"
        if top.get("conditional") and top.get("moderator"):
            seed_note += f"; candidate moderator: {top['moderator']}"
        seed_note += ")"
        hyp = run_validation(provider, df, meta, system_prompt, seed_note, tok)
        spent += 1

        # --- validity gate 1: reject a stated direction the data contradicts ---
        claim_ok, claim_info = (True, None)
        if hyp and "no_claim_check" not in ablate:
            try:
                claim_ok, claim_info = recompute_claim_ok(df, outcome, top["variable"], hyp)
            except Exception:
                claim_ok, claim_info = True, {"reason": "check errored"}
        # --- validity gate 2: reject an effect that washes out under adjustment ---
        # (skipped for conditional claims — their effect is joint, not a single slope)
        adj_ok, adj_info = (True, None)
        if hyp and claim_ok and not top.get("conditional") and "no_adjustment" not in ablate:
            try:
                adj_ok, adj_info = survives_adjustment(df, outcome, top["variable"], confirmed_vars)
            except Exception:
                adj_ok, adj_info = True, {"reason": "adjustment errored"}

        accept = bool(hyp) and claim_ok and adj_ok
        if accept:
            hypotheses.append({"id": len(hypotheses), "hypothesis": hyp,
                               "seed_var": top["variable"], "liveness": top["liveness"],
                               "moderator": top.get("moderator")})
            confirmed_vars.append(top["variable"])   # residualize it out next round
            confirmed_sets.add(hyp_signature(top["variable"], top.get("partners"),
                                             top.get("moderator")))
        elif hyp and not claim_ok:
            print(f"[{name}] REJECTED claim on {top['variable']}: "
                  f"claimed {claim_info.get('claimed_dir')} but data says {claim_info.get('true_dir')} "
                  f"(p={claim_info.get('p'):.1e})")
        elif hyp and claim_ok and not adj_ok:
            print(f"[{name}] REJECTED (confounded) {top['variable']}: "
                  f"marginal={adj_info.get('marginal')} adjusted={adj_info.get('adjusted')} "
                  f"(retains {100 * adj_info.get('retain', 0):.0f}% after adjustment)")
        log.append({"round": spent, "funded": top["variable"], "liveness": top["liveness"],
                    "conditional": top.get("conditional"), "moderator": top.get("moderator"),
                    "interaction_p": top.get("interaction_p"),
                    "declined_cheaply": len(declined), "got_hypothesis": bool(hyp),
                    "claim_ok": claim_ok, "claim_info": claim_info,
                    "adj_ok": adj_ok, "adj_info": adj_info, "accepted": accept})
        # incremental save so a mid-run interruption never loses work
        json.dump(hypotheses, open(hp, "w"), indent=2, default=str)
        json.dump(log, open(lp, "w"), indent=2, default=str)
        if spent % 5 == 0:
            print(f"[{name}] budget {spent}/{budget}  hyps={len(hypotheses)}  "
                  f"confirmed_vars={len(set(confirmed_vars))}  [{(time.time() - t0) / 60:.0f}m]")

    json.dump(hypotheses, open(hp, "w"), indent=2, default=str)
    json.dump(log, open(lp, "w"), indent=2, default=str)
    print(f"\n[{name}] cascade done: {len(hypotheses)} hypotheses on {budget} expensive loops")
    print(f"[{name}] distinct seed vars: {len(set(confirmed_vars))}  tokens: "
          f"{tok['in'] / 1e3:.0f}k in / {tok['out'] / 1e3:.0f}k out")
    print(f"[{name}] saved -> {hp}")
    return hypotheses
