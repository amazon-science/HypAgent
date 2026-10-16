# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
SCMSpec — the declarative contract between the spec author (an LLM or a human) and the Simulator.

Design decision: the LLM emits DATA (this typed spec as JSON), never code.
The Simulator is a fixed interpreter of these primitives. This gives us:
    - safety (no exec of model-written code)
    - verifiability (the verifier and scorer read the same structured terms)
    - reuse (numpy generates rows; the LLM only authors structure)

Two axes, deliberately separated:
    1. GRAPH structure  — how feature nodes wire together (NodeSpec.parents).
       This creates realistic correlation, confounding, mediation.
    2. TERM shape        — how a node maps to the outcome latent score (Term.op).
       A small library of functional primitives that COMPOSE into any archetype.

"Archetypes" (Simpson's, mediation, interaction reversal, ...) are NOT primitives.
They are named *configurations* over these two axes, attached to labeled terms only
as a reporting/coverage tag. The interpreter never needs to know an archetype name.

Role spectrum (every variable has a defined causal status):
    background  — root cause of other features, no direct outcome term
    confounder  — parents a treatment AND has an outcome term (the naive-marginal trap)
    mediator    — on path treatment->mediator->outcome; treatment has NO direct term
    moderator   — modifies another variable's effect (used by interaction/gate terms)
    cause       — labeled direct cause (enters the answer key)
    weak        — unlabeled small real cause (precision-under-noise test)
    decoy       — correlated with a real cause but NO outcome term (true null effect)
    null        — independent, no effect (baseline)
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, asdict, fields
from typing import Optional


def _tolerant(cls, d: dict):
    """Build a dataclass from a dict, dropping unknown keys with a warning.

    LLM-authored specs occasionally invent field names (e.g. 'bump_center_quantile').
    One stray field should not crash an otherwise-valid spec — we drop it and warn.
    """
    valid = {f.name for f in fields(cls)}
    unknown = set(d) - valid
    if unknown:
        warnings.warn(f"{cls.__name__}: dropping unknown fields {sorted(unknown)}")
    return cls(**{k: v for k, v in d.items() if k in valid})


# =============================================================================
# TERM PRIMITIVES — the executable vocabulary
# =============================================================================
# op values the Simulator knows how to evaluate:
PRIMITIVES = {
    "linear",       # w * standardize(v)                         (monotone linear)
    "threshold",    # w * 1[v > tau]                             (monotone step / threshold)
    "step",         # piecewise: 0 | 0.5w | w  across (lo, hi)   (phase transition)
    "bump",         # U-shape: +w at extremes, -0.4w in middle  (non-monotone / hormesis)
    "interaction",  # w*std(main) in one moderator group,        (interaction reversal)
                    #   reverse_sign*w*std(main) in the other
    "gate",         # w * 1[main > tau] * 1[in subpop]           (subpopulation-only)
    "conjunction",  # w * AND_i(v_i > quantile_i)                (emergent combination)
    "category",     # sum_k level_weights[k] * 1[cat == level_k]  (categorical-level effect)
}

# transforms an operand may carry (derived features):
TRANSFORMS = {"identity", "log", "ratio", "product"}

ROLES = {"background", "confounder", "mediator", "moderator",
         "cause", "weak", "decoy", "null", "treatment",
         # hidden_confounder: in the DGP (biases observational estimates, accounted
         #   for by do()) but WITHHELD from the agent-visible data -> the regime where
         #   observational identification is provably impossible.
         "hidden_confounder",
         # instrument: affects a treatment, no other path to outcome -> lets us author
         #   IV-solvable tasks so an observational agent CAN win when a valid instrument
         #   is present (a fair rung-2 comparison).
         "instrument"}

# roles whose nodes are removed from the data handed to methods/agents
HIDDEN_ROLES = {"hidden_confounder"}


@dataclass
class Operand:
    """A variable reference with an optional derived transform.

    ratio/product read a second variable `other`; identity/log read only `var`.
    """
    var: str
    transform: str = "identity"          # identity | log | ratio | product
    other: Optional[str] = None          # second var for ratio/product

    def __post_init__(self):
        if self.transform not in TRANSFORMS:
            raise ValueError(f"unknown transform {self.transform!r}")
        if self.transform in ("ratio", "product") and not self.other:
            raise ValueError(f"transform {self.transform} on {self.var} needs `other`")


@dataclass
class NodeSpec:
    """A feature node: a structural equation = distribution(base) + sum(weight*parent)."""
    name: str
    dtype: str                            # continuous | binary | categorical | count | ordinal
    distribution: str                     # normal|lognormal|uniform|bernoulli|poisson|categorical
    params: dict = field(default_factory=dict)
    parents: list[str] = field(default_factory=list)
    parent_weights: list[float] = field(default_factory=list)
    role: str = "background"
    description: str = ""
    unit: str = ""

    def __post_init__(self):
        if len(self.parents) != len(self.parent_weights):
            raise ValueError(
                f"node {self.name}: {len(self.parents)} parents but "
                f"{len(self.parent_weights)} weights")
        if self.role not in ROLES:
            raise ValueError(f"node {self.name}: unknown role {self.role!r}")


@dataclass
class Term:
    """One additive contribution to the outcome latent score."""
    id: str
    op: str                               # one of PRIMITIVES
    operands: list[Operand]
    weight: float

    # shape parameters (which apply depends on op)
    threshold: Optional[float] = None
    threshold_lo: Optional[float] = None
    threshold_hi: Optional[float] = None
    # for interaction / gate
    moderator: Optional[str] = None
    moderator_threshold: Optional[float] = None
    reverse_sign: float = -0.85           # interaction: multiplier in the "other" group
    # for conjunction: per-operand quantile cut (defaults to 0.65 each)
    conj_quantile: float = 0.65
    # for category: {level_value: effect_weight}; levels absent => 0 contribution
    level_weights: dict = field(default_factory=dict)

    # auto-calibration: if set, threshold(s) computed from data quantiles (anti-memorization)
    threshold_quantile: Optional[float] = None
    threshold_lo_quantile: Optional[float] = None
    threshold_hi_quantile: Optional[float] = None
    moderator_quantile: Optional[float] = None

    # metadata (scoring / reporting — interpreter ignores these)
    labeled: bool = True                  # True => enters the discovery answer key
    archetype: str = ""                   # reporting tag, e.g. "simpsons_paradox"
    role: str = "cause"                   # cause | confounder | weak
    description: str = ""

    def __post_init__(self):
        if self.op not in PRIMITIVES:
            raise ValueError(f"term {self.id}: unknown op {self.op!r}")
        self.operands = [o if isinstance(o, Operand) else Operand(**o)
                         for o in self.operands]

    @property
    def variables(self) -> set[str]:
        """All variable names this term reads (operands + moderator)."""
        vs = set()
        for o in self.operands:
            vs.add(o.var)
            if o.other:
                vs.add(o.other)
        if self.moderator:
            vs.add(self.moderator)
        return vs


@dataclass
class SCMSpec:
    domain: str
    scenario: str
    nodes: list[NodeSpec]
    terms: list[Term]
    outcome_name: str = "outcome"
    outcome_positive_rate: float = 0.20
    base_noise: float = 0.30
    n_rows: int = 1_000_000
    seed: int = 42

    def __post_init__(self):
        self.nodes = [n if isinstance(n, NodeSpec) else _tolerant(NodeSpec, n)
                      for n in self.nodes]
        self.terms = [t if isinstance(t, Term) else _tolerant(Term, t)
                      for t in self.terms]

    # ---- lookup helpers ----
    def node(self, name: str) -> Optional[NodeSpec]:
        return next((n for n in self.nodes if n.name == name), None)

    @property
    def node_names(self) -> list[str]:
        return [n.name for n in self.nodes]

    def nodes_by_role(self, role: str) -> list[NodeSpec]:
        return [n for n in self.nodes if n.role == role]

    @property
    def labeled_terms(self) -> list[Term]:
        return [t for t in self.terms if t.labeled]

    @property
    def hidden_nodes(self) -> list[str]:
        """Nodes generated in the DGP but withheld from the agent-visible data."""
        return [n.name for n in self.nodes if n.role in HIDDEN_ROLES]

    def visible_columns(self) -> list[str]:
        """Feature columns handed to methods/agents (hidden roles removed)."""
        hidden = set(self.hidden_nodes)
        return [n.name for n in self.nodes if n.name not in hidden]

    # ---- topological order over feature nodes ----
    def topological_order(self) -> list[str]:
        names = self.node_names
        parent_map = {n.name: [p for p in n.parents if p in names] for n in self.nodes}
        in_deg = {n: len(parent_map[n]) for n in names}
        queue = sorted([n for n in names if in_deg[n] == 0])
        order: list[str] = []
        while queue:
            cur = queue.pop(0)
            order.append(cur)
            for m in names:
                if cur in parent_map[m]:
                    in_deg[m] -= 1
                    if in_deg[m] == 0:
                        queue.append(m)
        # any leftover (cycle) appended so generation never silently drops a node
        order.extend(n for n in names if n not in order)
        return order

    def validate(self) -> list[str]:
        """Structural sanity checks. Returns a list of problems (empty == ok)."""
        problems = []
        names = set(self.node_names)
        if len(names) != len(self.nodes):
            problems.append("duplicate node names")
        # parents exist
        for n in self.nodes:
            for p in n.parents:
                if p not in names:
                    problems.append(f"node {n.name}: parent {p!r} not defined")
        # term variables exist as nodes
        cat_nodes = {n.name for n in self.nodes if n.dtype == "categorical"}
        for t in self.terms:
            for v in t.variables:
                if v not in names:
                    problems.append(f"term {t.id}: variable {v!r} not a node")
            # numeric-shape ops cannot read a categorical operand meaningfully;
            # the 'category' op is the sole exception (it reads levels directly).
            if t.op != "category":
                for o in t.operands:
                    if o.var in cat_nodes:
                        problems.append(
                            f"term {t.id}: operand {o.var!r} is categorical; numeric ops "
                            f"(threshold/bump/step/linear/gate/conjunction) need a numeric "
                            f"variable — use op 'category' or encode it as binary/ordinal")
            else:
                if not t.level_weights:
                    problems.append(f"term {t.id}: op 'category' requires level_weights")
            if t.op in ("interaction", "gate"):
                if not t.moderator:
                    problems.append(f"term {t.id}: op {t.op} requires a moderator")
                else:
                    mnode = next((n for n in self.nodes if n.name == t.moderator), None)
                    if mnode and mnode.dtype == "categorical":
                        problems.append(
                            f"term {t.id}: moderator {t.moderator!r} is categorical; "
                            f"use a numeric (binary/continuous) moderator")
        # acyclicity (topo order covers all only if acyclic)
        order = self.topological_order()
        # detect cycle: a node whose parent appears after it
        pos = {n: i for i, n in enumerate(order)}
        for n in self.nodes:
            for p in n.parents:
                if p in pos and pos[p] > pos[n.name]:
                    problems.append(f"cycle involving {n.name} <- {p}")
        # term ids unique
        ids = [t.id for t in self.terms]
        if len(set(ids)) != len(ids):
            problems.append("duplicate term ids")
        return problems

    # ---- (de)serialization ----
    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "scenario": self.scenario,
            "outcome_name": self.outcome_name,
            "outcome_positive_rate": self.outcome_positive_rate,
            "base_noise": self.base_noise,
            "n_rows": self.n_rows,
            "seed": self.seed,
            "nodes": [asdict(n) for n in self.nodes],
            "terms": [asdict(t) for t in self.terms],
        }

    def to_json(self, path=None, indent=2) -> str:
        s = json.dumps(self.to_dict(), indent=indent)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s

    @classmethod
    def from_dict(cls, d: dict) -> "SCMSpec":
        return cls(
            domain=d["domain"],
            scenario=d.get("scenario", ""),
            nodes=[_tolerant(NodeSpec, n) for n in d["nodes"]],
            terms=[_tolerant(Term, t) for t in d["terms"]],
            outcome_name=d.get("outcome_name", "outcome"),
            outcome_positive_rate=d.get("outcome_positive_rate", 0.20),
            base_noise=d.get("base_noise", 0.30),
            n_rows=d.get("n_rows", 1_000_000),
            seed=d.get("seed", 42),
        )

    @classmethod
    def from_json(cls, path) -> "SCMSpec":
        with open(path) as f:
            return cls.from_dict(json.load(f))
