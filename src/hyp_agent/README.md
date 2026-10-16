# HypAgent

HypAgent is an inference-time efficiency method for causal hypothesis discovery.
Given a tabular dataset with a binary outcome, it proposes candidate causal
directions, spends its expensive validation budget only on the ones that still
carry unexplained signal (a value-gated cascade), and checks each confirmed
hypothesis against the data before accepting it. The Python import package is
`hyp_agent`.

The budget is measured in expensive validation loops rather than wall-clock, so
the method is a fair discoveries-per-compute comparison against a flat
propose-then-validate agent.

## How it works

Each round runs four steps.

1. **Propose.** One cheap LLM call returns K candidate directions, each a primary
   variable plus a guessed structural form (monotone, threshold, U-shape, ratio,
   interaction, subgroup, combination, or categorical).
2. **Triage.** Every candidate is scored near-free. A marginal proposal is
   fundable when its variable still explains signal in the residual (the outcome
   with all confirmed effects removed). An interaction proposal has no marginal
   signal by construction, so it is gated instead on a significant super-additive
   term against a scanned moderator.
3. **Fund.** Only the top candidate enters the expensive explore-and-validate
   loop, where the model writes and runs code over the data and returns one
   specific hypothesis.
4. **Validate and residualize.** Two data checks run before acceptance. On
   acceptance the outcome is residualized against the new effect so the same
   structure cannot be re-funded.

### The gates

| gate | file | what it does |
|------|------|--------------|
| liveness | `gates.residualize`, `gates.liveness` | nonlinear signal of a variable in the residual, so U-shapes and thresholds are not missed |
| interaction | `gates.conditional_liveness` | super-additivity test for interaction, subgroup, and reversal proposals |
| claim check | `gates.recompute_claim_ok` | rejects a stated direction the data confidently contradicts |
| adjustment | `gates.survives_adjustment` | rejects an effect that washes out after controlling for observed covariates |
| dedup | `gates.hyp_signature` | forbids rediscovering the same variable set under a relabeled primary |

Every gate is conservative. Each one only ever tightens acceptance relative to
funding every proposal, and each fails open if an optional dependency is absent.

## Layout

Inside the repository the agent lives under `src/`:

```
src/
  run_agent.py           convenience launcher (same as python -m hyp_agent.cli)
  hyp_agent/
    cli.py               command-line entry point (installed as the HypAgent command)
    agent.py             the round loop (propose, triage, fund, validate)
    proposer.py          the two proposers (schema-only and code-analyzing)
    validator.py         the expensive explore-and-validate loop
    gates.py             value gates and validity gates
    config.py            gate thresholds
    prompts.py           prompt templates
    data.py              generic loader (parquet or csv plus a meta dict)
    llm.py               provider adapters (bedrock, anthropic, openai)
    sandbox.py           forked, timeout-bounded code execution
    parsing.py           tag extraction from model output
    examples/
      make_example.py    generates a synthetic dataset with planted effects
```

## Install

From the repository root:

```bash
pip install -r requirements.txt      # numpy, pandas, scipy, scikit-learn, statsmodels, pyarrow
```

Install the one provider SDK you plan to use: `boto3` for Bedrock, `anthropic`,
or `openai`.

## Quick start

Run from the `src/` directory. Generate the synthetic example, then run the agent
on it.

```bash
cd src
python hyp_agent/examples/make_example.py
python run_agent.py --data hyp_agent/examples/example.parquet \
    --meta hyp_agent/examples/example_meta.json \
    --budget 8 --out ../results/example --provider anthropic
```

The example plants five effects among noise columns (a monotone effect, a
threshold, a U-shape, a categorical level, and an interaction). A successful run
recovers them into `results/example/agent_cascade_analytical.json`.

## Running on your own data

The agent needs a table with a binary column named `outcome` and a small meta
dict. Pass a data file and an optional meta JSON.

```bash
python run_agent.py --data mydata.parquet --meta mymeta.json --budget 25 --out results/mine
```

The meta JSON holds four fields. Any field you omit is filled from the data, so
a bare table also works.

```json
{
  "domain": "ecommerce",
  "scenario": "online retail conversion",
  "outcome_positive_rate": 0.12,
  "context": "- account_age_months (float64): months since signup\n- plan_tier (object): basic / plus / premium\n..."
}
```

`context` is a column reference block, one line per column, in the form
`- <name> (<dtype>): <description>`. If you omit it, `data.build_context` builds
one from the column names and dtypes.

If your outcome column has a different name, pass `--outcome-col mycol`.

## Providers

Set credentials through the standard environment for your chosen backend.

| provider | key environment |
|----------|-----------------|
| bedrock | AWS credentials plus `BEDROCK_MODEL_ID`, `AWS_REGION`, optional `CREDENTIAL_REFRESH_CMD` |
| anthropic | `ANTHROPIC_API_KEY` |
| openai | `OPENAI_API_KEY` |

## Output

Two files are written to the output directory and updated after every round, so
an interrupted run resumes from where it stopped.

* `agent_cascade_analytical.json` is the list of confirmed hypotheses, each with
  its seed variable and liveness.
* `agent_cascade_analytical_log.json` is the per-round trace, including declined
  candidates and the reason for every rejection.

## Ablations

Disable one component at a time to measure its contribution. Each variant writes
to its own file so it never overwrites the full-method run.

```bash
python run_agent.py --data ... --out ... --ablate no_adjustment
```

Valid names are `flat`, `no_analytical`, `no_interaction_gate`, `no_claim_check`,
`no_adjustment`, and `no_dedup`.

## Programmatic use

```python
from hyp_agent import run, load

df, meta = load("mydata.parquet", "mymeta.json")
hypotheses = run(df, meta, budget=25, out_dir="results/mine", provider_name="anthropic")
```
