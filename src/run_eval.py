#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Score a method's hypotheses against an SCMSpec's ground truth with do() experiments.

Structured proposals (no LLM needed):

    python run_eval.py specs/example.json proposals.json

Free-text hypotheses (one LLM parse per hypothesis, then do()-scored):

    python run_eval.py specs/example.json hypotheses.json --parse --provider bedrock

``proposals.json`` may be:
  * a list of structured claims: ``[{"variable": "...", "form": "...", ...}, ...]``
    (see :mod:`scm.evaluate` for the fields), or
  * a list of free-text strings, or ``[{"hypothesis": "..."}]`` / ``[{"text": "..."}]``
    — use ``--parse`` to turn these into structured claims first.

Everything scored is do()-based: recall over the SCM's planted mechanisms and
claim-as-stated precision. Prints a summary and writes the full report to ``--out``.
"""

import argparse
import json
from pathlib import Path

from scm import SCMSpec, Simulator, evaluate

SCORE_N = 60_000


def _load_proposals(path):
    """Return ``(items, is_text)``. ``items`` is either a list of structured claim
    dicts or a list of free-text strings."""
    data = json.load(open(path))
    if isinstance(data, dict):                       # {"proposals": [...]} / {"hypotheses": [...]}
        data = data.get("proposals") or data.get("hypotheses") or []
    if not data:
        return [], False
    if all(isinstance(x, str) for x in data):
        return data, True
    # list of dicts: structured if any carries a form/variable, else pull the text field
    if any(("form" in x or "asserted_form" in x or "variable" in x) for x in data
           if isinstance(x, dict)):
        return data, False
    texts = [x.get("hypothesis") or x.get("text") or "" for x in data]
    return texts, True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="path to the spec .json the data was generated from")
    ap.add_argument("proposals", help="path to proposals/hypotheses .json")
    ap.add_argument("--parse", action="store_true",
                    help="treat inputs as free text and LLM-parse to structured claims")
    ap.add_argument("--provider", default="bedrock", help="provider for --parse")
    ap.add_argument("--n", type=int, default=SCORE_N, help="sample size for do() contrasts")
    ap.add_argument("--out", default=None, help="write the full JSON report here")
    a = ap.parse_args()

    spec = SCMSpec.from_json(a.spec)
    sim = Simulator(spec)
    sim.auto_calibrate()
    df = sim.generate(a.n)

    items, is_text = _load_proposals(a.proposals)
    if not items:
        print("no proposals found — nothing to score")
        return

    if is_text and not a.parse:
        raise SystemExit("proposals look like free text; re-run with --parse to "
                         "LLM-classify them into structured claims first.")
    if a.parse:
        from scm.parse import parse_hypotheses
        variables = [c for c in df.columns if c != spec.outcome_name]
        proposals = parse_hypotheses(items, spec.outcome_name, variables,
                                     provider_name=a.provider)
    else:
        proposals = items

    report = evaluate(sim, proposals, df=df, n=a.n)
    rec, prec = report["recall"], report["precision"]

    print(f"\n=== {spec.domain} — do()-scored evaluation (N={a.n:,}) ===")
    print(f"proposals scored : {report['n_proposals']}")
    print(f"planted mechanisms: {rec['n_plants']}")
    print("\nRECALL (fraction of planted mechanisms recovered)")
    print(f"  family rung      : {rec['recall']:.3f}")
    print(f"  confirmed (do())  : {rec['confirmed_recall']:.3f}")
    print(f"  variable named   : {rec['variable_recall']:.3f}")
    print("  by archetype:")
    for arch, d in rec["by_archetype"].items():
        print(f"    {arch:20s} {d['recall']:.2f}  (n={d['n']})")
    print("\nPRECISION (fraction of proposals whose claim survives do())")
    print(f"  claim-as-stated  : {prec['claim_precision']:.3f}  "
          f"({prec['n_correct']}/{prec['n_proposed']})")
    print(f"  directional only : {prec['directional_precision']:.3f}")
    if prec["n_unparseable"]:
        print(f"  unparseable      : {prec['n_unparseable']}")

    if a.out:
        outp = Path(a.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(outp, "w"), indent=2, default=str)
        print(f"\nwrote full report to {outp}")


if __name__ == "__main__":
    main()
