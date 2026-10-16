#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Author the example SCMSpec and write it to specs/example.json.

Five planted mechanisms among noise columns, one per scored functional-form family:

    tenure_months     monotone_threshold   (SHAPE)      retention turns on past a cutoff
    discount_pct       non_monotone         (SHAPE)      a mid-range sweet spot; extremes hurt
    email_opens        interaction_reversal (MODERATION) helps on mobile, hurts off it
    support_tickets    subpopulation_only   (MODERATION) only bites among premium users
    logins x session   derived_feature      (COMPOSITE)  the product drives retention, not either alone

The spec is DATA — a graph of nodes plus a set of typed terms. The Simulator
interprets it; nothing here is code the outcome depends on.

    python examples/make_example_spec.py      # -> specs/example.json
    python generate.py specs/example.json --n 40000 --out out/example
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # run from anywhere: put the pkg root first

from scm import SCMSpec, NodeSpec, Term, Operand  # noqa: E402

SPEC_OUT = HERE.parent / "specs" / "example.json"


def build_spec() -> SCMSpec:
    nodes = [
        # --- real causes / components ---
        NodeSpec("tenure_months", "count", "uniform", {"low": 0, "high": 60},
                 role="cause", description="months the customer has been active"),
        NodeSpec("discount_pct", "continuous", "uniform", {"low": 0, "high": 40},
                 role="cause", description="percent discount applied at checkout"),
        NodeSpec("email_opens", "count", "poisson", {"lambda": 6.0},
                 role="cause", description="marketing emails opened last month"),
        NodeSpec("support_tickets", "count", "poisson", {"lambda": 2.0},
                 role="cause", description="number of support tickets filed"),
        NodeSpec("logins", "count", "poisson", {"lambda": 8.0},
                 role="cause", description="logins in the last month"),
        NodeSpec("session_len", "continuous", "lognormal", {"mean": 1.5, "sigma": 0.5},
                 role="cause", description="average session length in minutes"),
        # --- moderators (numeric/binary so interaction & gate can read them) ---
        NodeSpec("is_mobile", "binary", "bernoulli", {"p": 0.5},
                 role="moderator", description="1 if the session was on mobile, else 0"),
        NodeSpec("is_premium", "binary", "bernoulli", {"p": 0.3},
                 role="moderator", description="1 if on the premium plan, else 0"),
        # --- noise: a true null and a decoy correlated with nothing causal ---
        NodeSpec("page_views", "count", "uniform", {"low": 1, "high": 30},
                 role="null", description="pages viewed in the session (noise)"),
        NodeSpec("referral_score", "continuous", "normal", {"mean": 0.0, "std": 1.0},
                 role="decoy", description="referral quality score (noise)"),
    ]

    terms = [
        # SHAPE: retention kicks in only once tenure clears a cutoff (~40th pct).
        Term("H1", "threshold", [Operand("tenure_months")], weight=1.1,
             threshold_quantile=0.40, archetype="monotone_threshold",
             description="Retention turns on once tenure exceeds a cutoff (~mid tenure)."),
        # SHAPE: inverted-U in discount — a mid-range sweet spot, both extremes hurt.
        # bump adds +w at the extremes; a negative weight makes the MIDDLE best.
        Term("H2", "bump", [Operand("discount_pct")], weight=-0.9,
             threshold_lo_quantile=0.25, threshold_hi_quantile=0.75,
             archetype="non_monotone",
             description="Discount has a sweet spot: moderate discounts retain best; "
                         "too little or too much both hurt."),
        # MODERATION: email opens help on mobile, hurt off it (sign reversal).
        Term("H3", "interaction", [Operand("email_opens")], weight=0.8,
             moderator="is_mobile", moderator_threshold=0.5, reverse_sign=-0.85,
             archetype="interaction_reversal",
             description="Opening marketing emails helps retention on mobile but hurts "
                         "it off mobile."),
        # MODERATION: support tickets only bite retention among premium users.
        Term("H4", "gate", [Operand("support_tickets")], weight=-1.0,
             threshold_quantile=0.60, moderator="is_premium", moderator_threshold=0.5,
             archetype="subpopulation_only",
             description="High support-ticket counts hurt retention only among premium "
                         "customers; elsewhere they barely matter."),
        # COMPOSITE: engagement = logins x session length drives retention (not either alone).
        Term("H5", "linear", [Operand("logins", transform="product", other="session_len")],
             weight=0.9, archetype="derived_feature",
             description="Engagement (logins times session length) drives retention; "
                         "neither logins nor session length alone captures it."),
    ]

    return SCMSpec(
        domain="subscription",
        scenario="subscription retention (synthetic; 5 planted mechanisms among noise)",
        nodes=nodes, terms=terms,
        outcome_name="outcome", outcome_positive_rate=0.35,
        base_noise=0.35, n_rows=1_000_000, seed=7,
    )


def main():
    spec = build_spec()
    problems = spec.validate()
    if problems:
        raise SystemExit("spec is invalid:\n  " + "\n  ".join(problems))
    SPEC_OUT.parent.mkdir(parents=True, exist_ok=True)
    spec.to_json(str(SPEC_OUT))
    print(f"wrote {SPEC_OUT}")
    print(f"  {len(spec.nodes)} nodes, {len(spec.terms)} planted mechanisms")
    for t in spec.terms:
        print(f"  - {t.id}: {t.archetype:20s} vars={sorted(t.variables)}")


if __name__ == "__main__":
    main()
