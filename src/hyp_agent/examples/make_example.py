#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Generate a small synthetic dataset with planted causal effects, plus a meta
JSON, so the agent can be run end-to-end without any external data.

The outcome is driven by a handful of effects spanning several structural forms
(monotone, threshold, U-shape, categorical level, and an interaction) buried
among pure-noise columns. Running the agent on this file should recover them.

    python examples/make_example.py
    python run.py --data examples/example.parquet --meta examples/example_meta.json \
                  --budget 8 --out results/example --provider anthropic
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
N = 40_000
SEED = 0

# column name -> human description (also becomes the agent-visible context block)
DESCRIPTIONS = {
    "tenure_months": "months the customer has been active",          # monotone +
    "support_tickets": "number of support tickets filed",             # threshold (hurts past a cutoff)
    "discount_pct": "percent discount applied at checkout",           # U-shape (mid is best)
    "plan_tier": "subscription tier: basic / plus / premium",         # categorical (premium lifts)
    "email_opens": "marketing emails opened last month",              # interaction with is_mobile
    "is_mobile": "1 if the session was on mobile, else 0",            # moderates email_opens
    "page_views": "pages viewed in the session (noise)",              # noise
    "region_code": "geographic region id (noise)",                    # noise
    "account_flags": "internal account flags count (noise)",          # noise
    "referral_score": "referral quality score (noise)",               # noise
}


def main():
    rng = np.random.default_rng(SEED)

    tenure_months = rng.integers(0, 60, N).astype(float)
    support_tickets = rng.poisson(2.0, N).astype(float)
    discount_pct = rng.uniform(0, 40, N)
    plan_tier = rng.choice(["basic", "plus", "premium"], N, p=[0.5, 0.35, 0.15])
    email_opens = rng.integers(0, 20, N).astype(float)
    is_mobile = rng.integers(0, 2, N).astype(float)
    page_views = rng.integers(1, 30, N).astype(float)
    region_code = rng.integers(0, 8, N).astype(float)
    account_flags = rng.integers(0, 5, N).astype(float)
    referral_score = rng.normal(0, 1, N)

    logit = -1.2
    logit += 0.03 * tenure_months                                    # monotone +
    logit += -0.8 * (support_tickets >= 4)                           # threshold (hurts past 4)
    logit += -0.004 * (discount_pct - 20) ** 2                       # U-shape, best near 20%
    logit += 0.9 * (plan_tier == "premium")                          # categorical level
    logit += 0.06 * email_opens * (2 * is_mobile - 1)                # interaction: helps on mobile, hurts off
    logit += rng.normal(0, 0.5, N)                                   # noise

    p = 1 / (1 + np.exp(-logit))
    outcome = (rng.uniform(0, 1, N) < p).astype(int)

    df = pd.DataFrame({
        "tenure_months": tenure_months, "support_tickets": support_tickets,
        "discount_pct": discount_pct, "plan_tier": plan_tier,
        "email_opens": email_opens, "is_mobile": is_mobile,
        "page_views": page_views, "region_code": region_code,
        "account_flags": account_flags, "referral_score": referral_score,
        "outcome": outcome,
    })

    context = "\n".join(f"- {c} ({df[c].dtype}): {DESCRIPTIONS[c]}"
                        for c in df.columns if c != "outcome")
    meta = {
        "domain": "subscription",
        "scenario": "subscription retention (synthetic; 5 planted effects among noise columns)",
        "outcome_positive_rate": float(df["outcome"].mean()),
        "context": context,
    }

    try:
        data_path = HERE / "example.parquet"
        df.to_parquet(data_path)
    except Exception:
        data_path = HERE / "example.csv"
        df.to_csv(data_path, index=False)
    meta_path = HERE / "example_meta.json"
    json.dump(meta, open(meta_path, "w"), indent=2)

    print(f"wrote {data_path}  ({len(df):,} rows, outcome rate {meta['outcome_positive_rate']:.2%})")
    print(f"wrote {meta_path}")
    print("\nplanted effects to recover:")
    print("  tenure_months (monotone +), support_tickets (threshold >=4), "
          "discount_pct (U-shape), plan_tier=='premium' (categorical), "
          "email_opens x is_mobile (interaction)")


if __name__ == "__main__":
    main()
