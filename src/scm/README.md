# SCM Simulator + do()-Grounded Evaluator

A fully-owned structural causal simulator for benchmarking causal hypothesis
discovery, plus the evaluator that scores a method's hypotheses against ground
truth using controlled experiments only. It answers the two questions a benchmark
needs: **how do we generate the data**, and **how do we evaluate a method on it**.

Because the simulator owns every structural equation, it can run exact
interventions (`do()`, `ate()`, counterfactuals) for free. Every score is an
interventional experiment, never a keyword or text match.

## The two halves

### 1. Generate

A dataset is authored as a declarative `SCMSpec`: the LLM (or a human) emits
**data**, a typed spec of a node graph and a set of typed structural terms, never
code. A fixed `Simulator` interprets it. Two axes compose into any archetype:

- **Graph** structure (`NodeSpec.parents`) creates realistic correlation,
  confounding, and mediation.
- **Term** shape (`Term.op`) maps a node to the outcome. Primitives are `linear`,
  `threshold`, `step`, `bump` (U-shape), `interaction`, `gate` (subpopulation-only),
  `conjunction`, and `category`, each optionally over a derived `Operand`
  (`log`, `ratio`, `product`).

Each labeled term carries an archetype tag (`monotone_threshold`, `non_monotone`,
`interaction_reversal`, `subpopulation_only`, `derived_feature`, `simpsons_paradox`,
`mediation_chain`) that the answer key uses for scoring. The interpreter itself
never needs to know an archetype name.

```python
from scm import SCMSpec, Simulator
sim = Simulator(SCMSpec.from_json("specs/example.json"))
sim.auto_calibrate()          # thresholds set from data quantiles (anti-memorization)
df = sim.generate(40_000)     # deterministic given the spec seed
```

### 2. Evaluate

Given the fitted simulator and a method's proposals, `evaluate` scores two numbers
with do() experiments:

- **recall** — of the mechanisms the SCM planted (its `answer_key`), what fraction
  did the method recover? A plant is recovered when a proposal names a variable in
  the plant's do()-causal set **and** asserts the right functional-form *family*,
  with the plant's do()-signature confirmed present.
- **claim precision** — of everything the method proposed, what fraction has its
  full asserted claim (variable + direction + committed structural form) survive
  do() on the method's own variable? A bare-direction claim asserts no mechanism
  and so earns no structural credit.

Both rungs come from one experiment pass per proposal, so they can never disagree
about whether a claim's effect is real.

```python
from scm import evaluate
report = evaluate(sim, proposals)          # proposals = structured claims (below)
report["recall"]["recall"]                 # family-rung recall
report["precision"]["claim_precision"]     # claim-as-stated precision
```

## Structured proposals

The evaluator scores structured claims (no LLM needed):

```json
{
  "variable":  "tenure_months",       // the primary cause named
  "form":      "threshold",           // see FORMS; default "bare_direction"
  "direction": "+",                    // "+" | "-" | "level"
  "moderator": "is_mobile",            // a 2nd variable: moderator / component / mediator
  "level":     null,                   // categorical: treated level
  "base":      null                    // categorical: baseline level
}
```

`form` is one of `bare_direction`, `threshold`, `non_monotone`, `interaction`,
`interaction_reversal`, `subpopulation`, `derived_feature`, `mediation`, `simpsons`,
`categorical_level`. A method that emits English sentences converts them to this
shape first with `scm.parse.parse_hypotheses`, the only step that touches a
provider. The verdict is still 100% do()-based.

## Quickstart

Install once from the repository root (`pip install -r requirements.txt`), then
run the CLIs from the `src/` directory:

```bash
cd src

# author the example spec and generate a dataset + meta + answer key
python examples/make_example_spec.py                       # -> specs/example.json
python generate.py specs/example.json --n 40000 --out out/example

# score a set of structured proposals against ground truth
python run_eval.py specs/example.json examples/example_proposals.json --out out/report.json

# score free-text hypotheses (LLM-parse them to structured claims first)
python run_eval.py specs/example.json hypotheses.json --parse --provider bedrock
```

The example plants five mechanisms among noise columns, one per scored family:
`tenure_months` (threshold), `discount_pct` (U-shape), `email_opens × is_mobile`
(interaction reversal), `support_tickets` gated by `is_premium` (subpopulation
only), and `logins × session_len` (derived feature). A correct method recovers all
five; a structure-blind ranker scores near zero on claim precision.

## Layout

The simulator is the `scm` package under `src/`, with its CLIs and examples as
siblings so `from scm import ...` resolves when run from `src/`:

```
src/
  generate.py            CLI: spec.json -> dataset.parquet + meta.json + answer_key.json
  run_eval.py            CLI: proposals + spec -> recall / precision report
  examples/              make_example_spec.py + example_proposals.json
  specs/                 authored specs (example.json is generated)
  scm/
    spec.py              SCMSpec / NodeSpec / Term / Operand (the declarative DGP)
    simulator.py         the interpreter: generate / do / ate / counterfactual / answer_key
    experiment.py        ExperimentRunner + Hypothesis + Verdict (the do() gate)
    signatures.py        per-archetype do() signature verifiers (no LLM)
    evaluate.py          do()-grounded recall + claim precision over structured proposals
    parse.py             OPTIONAL free-text -> structured claim (the only LLM-touching module)
    llm.py               provider adapters (bedrock, anthropic, openai)
```

## Metric definitions

- **Family recall** credits a plant when a proposal names a variable in its
  do()-causal set and asserts the plant's form family, with the plant's do()
  signature present. The COMPOSITE family accepts a MODERATION assertion (a
  ratio/product manifests under do() as a joint two-variable effect); CONFOUND
  (Simpson's, mediation) has no proposal-vocabulary token, so it is credited by
  the conditional/reversal/subgroup structural form it is actually expressed in,
  anchored on the plant's signature plus a real do-gate effect.
- **Confirmed recall** additionally requires the recovering claim's own form to
  survive do() (an anti-spray check).
- **Claim precision** is the fraction of all proposals whose committed shape holds
  under do() (plus direction for directional forms). Bare-direction claims score a
  diagnostic `directional_precision` but no structural credit.

This is the same do()-grounded scoring used for the paper's main results; the
`scm` package is the dependency-light generation and evaluation core.

## Relation to `hyp_agent`

`hyp_agent` (in `src/hyp_agent/`) is the discovery **method** (it proposes
hypotheses from a dataset). The `scm` package is the **benchmark** it is measured
on. `generate.py` writes a `meta.json` in exactly the shape `hyp_agent`'s loader
consumes, so a generated dataset drops straight into an agent run, and the agent's
output hypotheses feed back into `run_eval.py --parse`.
