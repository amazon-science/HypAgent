#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Command-line entry point for HypAgent.

Run from the repository's ``src/`` directory (or after ``pip install -e .`` use
the installed ``HypAgent`` command):

    python -m hyp_agent.cli --data example/example.parquet \
                  --meta example/example_meta.json --budget 25 --out results/example

See README.md for the meta format and provider configuration.
"""

import argparse

from hyp_agent import run, load
from hyp_agent.agent import ABLATIONS


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="path to a .parquet or .csv dataset")
    ap.add_argument("--meta", default=None,
                    help="path to a meta JSON (optional; fields are filled from the data if omitted)")
    ap.add_argument("--out", required=True, help="output directory for results + log")
    ap.add_argument("--budget", type=int, default=10,
                    help="number of expensive validation loops (default: 10)")
    ap.add_argument("--k", type=int, default=6, help="candidate directions proposed per round")
    ap.add_argument("--outcome-col", default="outcome",
                    help="name of the binary outcome column (renamed to 'outcome')")
    ap.add_argument("--provider", default="bedrock", choices=["bedrock", "anthropic", "openai"],
                    help="LLM backend (default: bedrock)")
    ap.add_argument("--proposer", default="analytical", choices=["analytical", "schema"],
                    help="analytical (code-analyzing, recommended) or schema-only proposer")
    ap.add_argument("--ablate", default="",
                    help=f"comma-separated components to DISABLE for an ablation study. "
                         f"Options: {', '.join(ABLATIONS)}. Writes to a separate file.")
    args = ap.parse_args()

    df, meta = load(args.data, args.meta, outcome_col=args.outcome_col)
    ablate = [a.strip() for a in args.ablate.split(",") if a.strip()]
    run(df, meta, budget=args.budget, out_dir=args.out, k_propose=args.k,
        provider_name=args.provider, analytical=(args.proposer == "analytical"), ablate=ablate)


if __name__ == "__main__":
    main()
