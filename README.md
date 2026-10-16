# HypAgent

This code is being released solely for academic and scientific reproducibility
purposes, in support of the methods and findings described in the associated
publication. Pull requests are not being accepted in order to maintain the code
exactly as it was used in the paper.

HypAgent studies one question: can an LLM agent look at a tabular dataset, propose
the *mechanisms* that drive an outcome, and be measured honestly against ground
truth? The package has two pillars.

- **A method** — a value-gated discovery agent that explores data through a code
  REPL, spends its expensive validation budget only where signal remains, and
  checks each hypothesis with a controlled experiment before accepting it. It is
  the `hyp_agent` package.
- **A benchmark** — a fully-owned structural causal simulator. Because it owns
  every equation, it can run exact interventions (`do()`), so every score is an
  experiment rather than a keyword match. It is the `scm` package.

The method proposes; the benchmark generates the data and grades the proposals.

## Where to go

| I want to… | Read |
|------------|------|
| run the discovery agent on my own dataset | [src/hyp_agent/README.md](src/hyp_agent/README.md) |
| generate benchmark data and evaluate a method's hypotheses | [src/scm/README.md](src/scm/README.md) |
| understand the whole package | this file |

Those two READMEs are the source of truth for their areas. This overview does not
repeat their usage; it explains how the pieces fit together.

## How the pieces fit

```
                 authors a spec (typed DATA, never code)
   SCMSpec  ─────────────────────────────────────────────┐
      │                                                   │
      ▼                                                   ▼
     scm    ──generate──▶  dataset + meta          answer key (ground truth)
      │                          │                        │
      │                          ▼                        │
      │                    hyp_agent  ────proposes────▶  hypotheses
      │                                                   │
      └────────────── evaluate (do() experiments) ◀───────┘
                          │
                          ▼
                 recall + claim precision
```

`generate.py` writes its `meta.json` in exactly the shape `hyp_agent`'s loader
consumes, so a generated dataset drops straight into an agent run, and the agent's
output hypotheses feed back into the evaluator (`run_eval.py --parse`).

## Layout

```
HypAgent/
  pyproject.toml     packaging, dependencies, and tool config
  setup.cfg          flake8 config
  requirements.txt   pinned runtime deps
  LICENSE, NOTICE    CC BY-NC 4.0 license and copyright notice
  CONTRIBUTING.md, CODE_OF_CONDUCT.md   security reporting and code of conduct
  src/
    hyp_agent/     the discovery METHOD  (see src/hyp_agent/README.md)
    scm/           the BENCHMARK simulator + evaluator  (see src/scm/README.md)
    run_agent.py   launcher for the agent CLI
    generate.py    simulator CLI: spec -> dataset + meta + answer key
    run_eval.py    simulator CLI: proposals -> recall / precision
    examples/      simulator spec author + example proposals
  test/            smoke tests
```

## Setup

Install the runtime stack once from the repository root, then run the CLIs from
`src/`:

```bash
pip install -r requirements.txt
```

The pins in `requirements.txt` are tested on Python 3.10–3.12. On Python 3.13 or
newer, use `pip install .` (below) instead.

`hyp_agent` and `scm` both import as top-level packages when you run from `src/`.
See the two READMEs for the exact commands.

Alternatively, `pip install .` installs both packages and the `HypAgent` command;
add extras for Parquet support and your LLM provider, e.g.
`pip install ".[parquet,bedrock]"` (extras: `parquet`, `bedrock`, `anthropic`,
`openai`).

## Evaluation principle

Scoring is always an experiment. A hypothesis is graded by running `do(hi)` vs
`do(lo)` (optionally within a subpopulation) and reading the exact interventional
effect off the simulator, never by matching words in the claim. Recall asks what
fraction of the planted mechanisms a method recovered; claim precision asks what
fraction of a method's claims survive their own `do()`. Both are defined in
[src/scm/README.md](src/scm/README.md).

## Tests

```bash
pip install pytest
pytest
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0
International License (CC BY-NC 4.0) for non-commercial research use. See
[LICENSE](LICENSE) and [NOTICE](NOTICE).
