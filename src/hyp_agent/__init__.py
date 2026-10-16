# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Value-gated cascade agent for causal hypothesis discovery.

Public API:
    from hyp_agent import run, load
    df, meta = load("data.parquet", "meta.json")
    run(df, meta, budget=25, out_dir="results/")
"""

from .agent import run, ABLATIONS
from .data import load, build_context

__all__ = ["run", "load", "build_context", "ABLATIONS"]
