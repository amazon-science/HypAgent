#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Generate a dataset from an SCMSpec.

    python generate.py specs/example.json --n 40000 --out out/example

Writes three files under ``--out``:

  * ``<name>.parquet``  — the dataset a method sees (hidden-role nodes removed),
                          with the binary ``outcome`` column. Falls back to CSV if
                          parquet support is unavailable.
  * ``<name>_meta.json`` — ``{domain, scenario, outcome_positive_rate, context}``,
                          the same shape the cascade agent's ``load()`` consumes, so
                          the generated data drops straight into a run.
  * ``<name>_key.json``  — the ground-truth answer key (planted terms + archetypes)
                          for :mod:`scm.evaluate`. NEVER hand this to a method.

Deterministic: same spec (with its seed) + same ``--n`` => byte-identical data.
"""

import argparse
import json
from pathlib import Path

from scm import SCMSpec, Simulator


def build_context(spec, df):
    """A human-readable column block for the method's prompt (dtype + description),
    over the visible feature columns only."""
    lines = []
    for name in spec.visible_columns():
        if name == spec.outcome_name:
            continue
        node = spec.node(name)
        desc = (node.description if node and node.description else "").strip()
        dtype = df[name].dtype if name in df.columns else (node.dtype if node else "?")
        lines.append(f"- {name} ({dtype}): {desc}" if desc else f"- {name} ({dtype})")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Generate a dataset from an SCMSpec.")
    ap.add_argument("spec", help="path to a spec .json")
    ap.add_argument("--n", type=int, default=40_000, help="rows to generate")
    ap.add_argument("--out", default="out/dataset", help="output path stem (dir/name)")
    a = ap.parse_args()

    spec = SCMSpec.from_json(a.spec)
    sim = Simulator(spec)
    sim.auto_calibrate()

    # visible_only drops hidden-role nodes (e.g. hidden confounders) from the data a
    # method sees; the simulator still uses them internally for do()/scoring.
    df = sim.generate(a.n, visible_only=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    name = out.name

    try:
        data_path = out.with_name(f"{name}.parquet")
        df.to_parquet(data_path)
    except Exception:
        data_path = out.with_name(f"{name}.csv")
        df.to_csv(data_path, index=False)

    meta = {
        "domain": spec.domain,
        "scenario": spec.scenario,
        "outcome_positive_rate": float(df[spec.outcome_name].mean())
        if spec.outcome_name in df.columns else None,
        "context": build_context(spec, df),
    }
    meta_path = out.with_name(f"{name}_meta.json")
    json.dump(meta, open(meta_path, "w"), indent=2)

    key = sim.answer_key()
    key_path = out.with_name(f"{name}_key.json")
    json.dump(key, open(key_path, "w"), indent=2, default=str)

    rate = meta["outcome_positive_rate"]
    print(f"wrote {data_path}  ({len(df):,} rows, {df.shape[1]} cols"
          + (f", outcome rate {rate:.2%}" if rate is not None else "") + ")")
    print(f"wrote {meta_path}")
    print(f"wrote {key_path}  ({len(key)} planted mechanisms)")
    for e in key:
        print(f"  - {e['id']}: {e['archetype']:20s} vars={e['variables']}")


if __name__ == "__main__":
    main()
