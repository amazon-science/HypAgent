# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Generic data loading.

The agent needs only two things:

  * a DataFrame with a binary column named ``outcome``, and
  * a ``meta`` dict with four keys the prompts interpolate:
        domain                 short data-domain label (e.g. "ecommerce")
        scenario               one-line description of the setting
        outcome_positive_rate  P(outcome == 1), a float in [0, 1]
        context                a column reference block, one line per column:
                               "- <name> (<dtype>): <description>"

This module builds both from a data file (parquet/csv) and an optional meta JSON.
Any missing meta field is filled from the data so the agent runs on a bare table.
"""

import json
from pathlib import Path

import pandas as pd

REQUIRED_META_KEYS = ("domain", "scenario", "outcome_positive_rate", "context")


def _read_frame(data_path):
    p = Path(data_path)
    if p.suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    if p.suffix in (".csv", ".txt"):
        return pd.read_csv(p)
    raise ValueError(f"unsupported data file: {p.name} (use .parquet or .csv)")


def build_context(df, descriptions=None):
    """Render the column reference block from a DataFrame.

    ``descriptions`` optionally maps column name -> human description; columns
    without one are listed by name and dtype only.
    """
    descriptions = descriptions or {}
    lines = []
    for c in df.columns:
        if c == "outcome":
            continue
        desc = descriptions.get(c, "")
        suffix = f": {desc}" if desc else ""
        lines.append(f"- {c} ({df[c].dtype}){suffix}")
    return "\n".join(lines)


def load(data_path, meta_path=None, outcome_col="outcome"):
    """Load ``(df, meta)`` for the agent.

    Args:
        data_path: path to a ``.parquet`` or ``.csv`` file.
        meta_path: optional path to a JSON file with any of the meta keys above.
            Missing keys are filled from the data.
        outcome_col: name of the binary outcome column; renamed to ``outcome``.

    Returns:
        ``(df, meta)`` ready to pass to :func:`hyp_agent.agent.run`.
    """
    df = _read_frame(data_path)
    if outcome_col not in df.columns:
        raise ValueError(f"outcome column '{outcome_col}' not found; columns: {list(df.columns)}")
    if outcome_col != "outcome":
        df = df.rename(columns={outcome_col: "outcome"})

    vals = set(pd.unique(df["outcome"].dropna()))
    if not vals <= {0, 1, True, False}:
        raise ValueError(f"'outcome' must be binary (0/1); found values: {sorted(vals)[:10]}")
    df["outcome"] = df["outcome"].astype(float)

    meta = {}
    if meta_path:
        meta = json.load(open(meta_path))

    meta.setdefault("domain", Path(data_path).stem)
    meta.setdefault("scenario", meta["domain"])
    meta.setdefault("outcome_positive_rate", float(df["outcome"].mean()))
    if "context" not in meta:
        meta["context"] = build_context(df, meta.get("descriptions"))

    missing = [k for k in REQUIRED_META_KEYS if k not in meta]
    if missing:  # defensive; the setdefaults above should have filled everything
        raise ValueError(f"meta is missing required keys: {missing}")
    return df, meta
