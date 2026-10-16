# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Optional: turn free-text hypotheses into structured claims for :mod:`scm.evaluate`.

The evaluator scores *structured* proposals (variable + committed form + direction).
A method that emits English sentences needs one translation step first. This module
is the ONLY place a provider is used, and it does no scoring — it extracts, with a
strict commit-only rule, the causal claim and the single structural form the text
unambiguously asserts. Hedges ("non-linear", "complex", "affects") map to
``bare_direction`` so vague text is judged on the lenient bar, never penalized for a
shape it only gestured at. The verdict itself is still 100% do()-based downstream.

The output dict keys (``variable``, ``asserted_form``, ``direction``,
``secondary_variable``, ``level``, ``base``) are exactly what
:func:`scm.evaluate.evaluate` accepts.
"""

from __future__ import annotations

import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FTimeout

from .evaluate import FORMS

CLASSIFY_SYSTEM = ("You extract a causal claim AND the structural form it commits to. "
                   "Return JSON only. Be conservative about the form.")

CLASSIFY_PROMPT = """A method proposed this hypothesis about what drives the outcome '{outcome}':

\"\"\"{text}\"\"\"

Available variables (the ONLY valid names): {variables}

Return JSON:
{{
  "variable": "<primary variable from the list, or null>",
  "direction": "+" | "-" | "level",
  "level": "<category value if direction=level, else null>",
  "base": "<baseline category if direction=level, else null>",
  "secondary_variable": "<a SECOND named variable that is a moderator, interacting term, ratio component, or mediator — from the list, else null>",
  "threshold_value": "<the specific numeric cutoff/center the claim states for the primary variable, in its own units, or null>",
  "asserted_form": "bare_direction | threshold | non_monotone | interaction | interaction_reversal | subpopulation | derived_feature | mediation | simpsons | categorical_level"
}}

asserted_form — pick the ONE structural form the claim EXPLICITLY commits to. Default to
"bare_direction". Only choose a specific form when the text unambiguously asserts it:
- threshold: says "threshold", "cutoff", "step", "above/below X", "only once it exceeds", "kicks in beyond". A plain "higher X -> higher Y" is NOT a threshold.
- non_monotone: says "U-shape", "inverted-U", "sweet spot", "optimal range", "both extremes", "too much/too little".
- interaction: the effect's SIZE depends on / is moderated by / is amplified for another named variable (stronger for group X), WITHOUT claiming the sign flips.
- interaction_reversal: the effect explicitly REVERSES sign / flips direction depending on another variable.
- subpopulation: the effect exists ONLY within a named subgroup and is absent/negligible globally.
- derived_feature: the effect is of a RATIO, PRODUCT, or difference of two named variables (not either alone).
- mediation: the effect operates THROUGH a named intermediate/mediator variable (via, mediated by, indirectly through).
- simpsons: the association REVERSES or is explained away once a confounder is adjusted/controlled/stratified.
- categorical_level: a SPECIFIC category value (not the variable overall) drives the outcome.
Hedge words alone ("non-linear", "complex", "affects", "predicts") are NOT a committed form -> use bare_direction.
Return only the JSON."""

PARSE_WORKERS = int(os.environ.get("PARSE_WORKERS", "12"))
PARSE_CALL_TIMEOUT = int(os.environ.get("PARSE_CALL_TIMEOUT", "120"))


def _classify_one(provider, text, outcome, variables):
    prompt = CLASSIFY_PROMPT.format(text=text[:1200], outcome=outcome,
                                    variables=", ".join(variables))
    try:
        r = provider.generate(system=CLASSIFY_SYSTEM, user=prompt,
                              temperature=0.0, max_tokens=450)
        t = re.sub(r"```json\s*|```\s*", "", r.content)
        a, b = t.find("{"), t.rfind("}") + 1
        data = json.loads(t[a:b]) if a >= 0 else {}
    except Exception as e:
        return {"variable": None, "text": text, "_error": str(e)}
    if data.get("asserted_form") not in FORMS:
        data["asserted_form"] = "bare_direction"
    data["text"] = text
    return data


def parse_hypotheses(texts, outcome, variables, provider=None, provider_name="bedrock",
                     workers=PARSE_WORKERS):
    """Convert free-text hypotheses into structured claims (one LLM call each, run
    concurrently under a hard global deadline). ``variables`` is the list of valid
    column names; ``outcome`` the target name. Pass a ``provider`` to reuse one, or a
    ``provider_name`` to construct via :func:`scm.llm.get_provider`. Returns a list
    aligned with ``texts``; a timed-out or unparseable item comes back
    ``{"variable": None, ...}`` and scores as unparseable downstream."""
    if not texts:
        return []
    if provider is None:
        from .llm import get_provider          # lazy: keep evaluate/generate LLM-free
        provider = get_provider(provider_name)

    claims = [{"variable": None, "text": t} for t in texts]
    nw = min(workers, len(texts))
    ex = ThreadPoolExecutor(max_workers=nw)
    futs = {ex.submit(_classify_one, provider, texts[i], outcome, variables): i
            for i in range(len(texts))}
    deadline = max(300, PARSE_CALL_TIMEOUT * math.ceil(len(texts) / nw) * 2)
    n_to = 0
    try:
        for fut in as_completed(list(futs), timeout=deadline):
            i = futs[fut]
            try:
                claims[i] = fut.result() or claims[i]
            except Exception:
                n_to += 1
    except FTimeout:
        n_to += sum(1 for f in futs if not f.done())
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    if n_to:
        print(f"    [parse: {n_to}/{len(texts)} timed out -> unparseable]", flush=True)
    return claims
