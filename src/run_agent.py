#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Convenience launcher for the HypAgent CLI.

Lets you run the agent the same way as the simulator CLIs, from the ``src/``
directory:

    python run_agent.py --data hyp_agent/examples/example.parquet \
        --meta hyp_agent/examples/example_meta.json --budget 25 --out ../results/example

Equivalent to ``python -m hyp_agent.cli`` and to the installed ``HypAgent`` command.
"""

from hyp_agent.cli import main

if __name__ == "__main__":
    main()
