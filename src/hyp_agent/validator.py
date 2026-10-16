# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""The expensive VALIDATION step — the costly unit the budget is measured in.

Once triage funds a direction, the model runs a full explore->hypothesize loop
over the data (writing and executing code) and returns one specific, testable
hypothesis. This is the only step that spends the expensive per-loop budget.
"""

import json
import re
import time

import numpy as np
import pandas as pd
from scipy import stats

from .parsing import extract_block
from .sandbox import execute_code


def build_system_prompt(meta):
    """System prompt for the validation loop, describing the data and the task."""
    return (f"You are a data scientist finding causal patterns in a {meta['domain']} dataset.\n"
            f"`df` has binary `outcome`. Columns:\n{meta['context']}\n"
            f"Find non-linear, interaction, subpopulation, derived-feature, and combination effects.\n"
            f"Write <code> to explore, <hypothesis> with the pattern + evidence when ready.")


def run_validation(provider, df, meta, system_prompt, seed_note, tok, max_turns=8):
    """Run the funded explore->validate loop seeded at ``seed_note``.

    Hands the model the FULL frame (it may subsample for cheap exploration but is
    told to confirm the final numbers on all rows). Returns the hypothesis text, or
    None if it never emitted one within ``max_turns``.
    """
    n_rows = len(df)
    pos = meta["outcome_positive_rate"] * 100
    ns = {'df': df.copy(), 'pd': pd, 'np': np, 're': re, 'stats': stats}
    convo = [{"role": "user", "content":
              f"`df` has the FULL {n_rows:,} rows, binary `outcome` ({pos:.1f}% positive).\n"
              f"Investigate this direction and produce ONE specific testable hypothesis: {seed_note}\n"
              f"For speed you MAY explore on a random subsample (e.g. df.sample(100_000, random_state=0)) "
              f"while scanning cheaply, but CONFIRM the final effect size, cutoff, or subgroup on the "
              f"FULL df before writing your <hypothesis>.\n"
              f"{max_turns} turns. Use <code> to explore, <hypothesis> when ready."}]
    for turn in range(max_turns):
        try:
            resp = provider.generate(system=system_prompt, user=json.dumps(convo),
                                     temperature=0.5, max_tokens=2000)
            tok["in"] += resp.usage.get("input_tokens", 0)
            tok["out"] += resp.usage.get("output_tokens", 0)
        except Exception:
            time.sleep(20)
            continue
        bt, content = extract_block(resp.content)
        if bt == 'hypothesis':
            return content
        elif bt == 'code':
            out = execute_code(content, ns)
            convo.append({"role": "assistant", "content": resp.content})
            tag = "Last turn — <hypothesis> now." if turn >= max_turns - 2 else f"{max_turns - turn - 1} left."
            convo.append({"role": "user", "content": f"Output:\n{out}\n{tag}"})
        else:
            convo.append({"role": "assistant", "content": resp.content})
            convo.append({"role": "user", "content": "Write <code> or <hypothesis>."})
    return None
