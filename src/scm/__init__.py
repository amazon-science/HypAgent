# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""scm — a fully-owned structural causal simulator and its do()-grounded evaluator.

Two halves, matching the two things a benchmark needs:

**Generate data.** Author a declarative :class:`SCMSpec` (typed DATA, never code:
a graph of nodes and a set of typed structural terms), build a :class:`Simulator`,
and draw a dataset. The simulator owns every equation, so it can run exact
interventions — ``do()``, ``ate()``, counterfactuals — for free.

    from scm import SCMSpec, Simulator
    sim = Simulator(SCMSpec.from_json("specs/example.json"))
    sim.auto_calibrate()
    df = sim.generate(50_000)

**Evaluate hypotheses.** Score a method's proposals against the SCM's ground truth
with controlled experiments only (no keyword matching):

    from scm import evaluate
    report = evaluate(sim, proposals)   # proposals = structured claims
    print(report["recall"]["recall"], report["precision"]["claim_precision"])

Free-text hypotheses go through :func:`scm.parse.parse_hypotheses` first (the only
LLM-touching step); everything scored here is do()-based.
"""

from .spec import SCMSpec, NodeSpec, Term, Operand
from .simulator import Simulator
from .experiment import ExperimentRunner, Hypothesis, Verdict
from .evaluate import evaluate, verify_form, ARCHETYPES, FORMS
from . import signatures

__all__ = [
    "SCMSpec", "NodeSpec", "Term", "Operand",
    "Simulator", "ExperimentRunner", "Hypothesis", "Verdict",
    "evaluate", "verify_form", "signatures", "ARCHETYPES", "FORMS",
]
