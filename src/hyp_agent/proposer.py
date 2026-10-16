# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""The cheap PROPOSE step: turn the current residual + confirmed set into a batch
of candidate directions (variable + structural form). Two interchangeable
proposers share the JSON output schema consumed by the triage in ``agent.run``.
"""

import json
import re

import numpy as np
import pandas as pd
from scipy import stats

from .parsing import extract_block
from .sandbox import execute_code
from .prompts import (
    PROPOSE_SYS, PROPOSE_PROMPT,
    PROPOSE_ANALYTICAL_SYS, PROPOSE_ANALYTICAL_PROMPT,
    format_confirmed,
)


def _parse_json_list(text):
    """Pull the first JSON list out of a model response; [] on failure."""
    t = re.sub(r"```json|```", "", text)
    a, b = t.find("["), t.rfind("]") + 1
    try:
        return json.loads(t[a:b])
    except Exception:
        return []


def propose(provider, meta, confirmed_hyps, k, tok):
    """Schema-only proposer: one LLM call -> up to ``k`` candidate directions.

    A provider failure is allowed to propagate on purpose — the caller resumes
    from its incremental save, so a transient blip costs a restart, not a silently
    truncated run.
    """
    p = PROPOSE_PROMPT.format(domain=meta["domain"], scenario=meta["scenario"],
                              outcome="outcome", rate=meta["outcome_positive_rate"] * 100,
                              context=meta["context"], confirmed=format_confirmed(confirmed_hyps), k=k)
    r = provider.generate(system=PROPOSE_SYS, user=p, temperature=0.7, max_tokens=900)
    tok["in"] += r.usage.get("input_tokens", 0)
    tok["out"] += r.usage.get("output_tokens", 0)
    return _parse_json_list(r.content)


def propose_analytical(provider, df, meta, confirmed_hyps, residual, k, tok, max_turns=4):
    """Analytical proposer: the model writes code to ANALYZE the residual (fit a
    shallow tree, scan split-slopes, bin variables) BEFORE naming directions.

    This gives the proposal step eyes on the unexplained structure instead of
    guessing from column names — it surfaces the interaction / combination atoms a
    single-variable gate is blind to, then feeds them into the SAME downstream
    pursuit loop. Returns the same JSON schema as :func:`propose`.
    """
    prompt = PROPOSE_ANALYTICAL_PROMPT.format(
        domain=meta["domain"], scenario=meta["scenario"], outcome="outcome",
        rate=meta["outcome_positive_rate"] * 100, context=meta["context"],
        confirmed=format_confirmed(confirmed_hyps), k=k, n_rows=len(df), max_turns=max_turns)
    # `resid` is aligned to df's row order (residualize returns a df-row-aligned array).
    ns = {'df': df.copy(), 'resid': np.asarray(residual, dtype=float),
          'pd': pd, 'np': np, 're': re, 'stats': stats}
    convo = [{"role": "user", "content": prompt}]
    for turn in range(max_turns + 1):
        resp = provider.generate(system=PROPOSE_ANALYTICAL_SYS, user=json.dumps(convo),
                                 temperature=0.6, max_tokens=1600)
        tok["in"] += resp.usage.get("input_tokens", 0)
        tok["out"] += resp.usage.get("output_tokens", 0)
        bt, content = extract_block(resp.content, tags=['propose', 'code'])
        if bt == 'propose':
            return _parse_json_list(content)
        elif bt == 'code':
            out = execute_code(content, ns)
            convo.append({"role": "assistant", "content": resp.content})
            tag = "Last turn — emit <propose> now." if turn >= max_turns - 1 else f"{max_turns - turn} code turns left."
            convo.append({"role": "user", "content": f"Output:\n{out}\n{tag}"})
        else:
            convo.append({"role": "assistant", "content": resp.content})
            convo.append({"role": "user", "content": "Write <code> to analyze `resid`, or <propose> with the JSON list."})
    return []
