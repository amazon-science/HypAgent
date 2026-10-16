# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Experiment harness — the payoff of a fully-owned SCM.

An agent proposes a hypothesis. We run a CONTROLLED EXPERIMENT via do() and return
a verdict grounded in exact ground truth. This is what a real scientist would do in a
lab, except the simulator makes it free, instant, and exact.

A Hypothesis is a structured claim:
    variable   : the cause the agent names
    direction  : "+" (increasing it raises the outcome) | "-" | "level" (categorical)
    hi/lo      : the two intervention arms (auto-chosen from data if omitted)
    condition  : optional callable(df)->mask, a subpopulation the claim is scoped to
    level/base : for categorical claims — which level vs baseline

The harness runs do(hi) vs do(lo) [optionally within `condition`], computes the exact
causal ATE, and also the naive observational contrast — so we can tell the agent not
just "right/wrong" but WHY (e.g. confounded, subpopulation-only, direction flipped).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

try:
    from .simulator import Simulator
except ImportError:
    from simulator import Simulator


@dataclass
class Hypothesis:
    variable: str
    direction: str = "+"                     # "+" | "-" | "level"
    hi: Optional[float] = None
    lo: Optional[float] = None
    level: Optional[str] = None              # categorical: treated level
    base: Optional[str] = None               # categorical: baseline level
    condition: Optional[Callable] = None     # subpopulation scope
    condition_desc: str = ""
    claimed_effect: Optional[float] = None   # agent's estimated ATE, if any
    text: str = ""                           # human-readable statement


@dataclass
class Verdict:
    hypothesis: Hypothesis
    ate_causal: float
    ate_observational: float
    supported: bool                          # real causal effect in claimed direction?
    direction_correct: bool
    confounded: bool                         # observational misleads vs causal
    subpop_only: bool                        # effect only under the condition
    magnitude_ok: Optional[bool]             # agent's claimed_effect within tolerance
    explanation: str
    ci_low: float = float("nan")             # 95% bootstrap CI on the causal ATE
    ci_high: float = float("nan")
    p_value: float = float("nan")            # 2-sided, H0: ATE == 0
    significant: bool = False                # CI on the relevant effect excludes 0
    evaluable: bool = True                   # could the claim be tested at all?
                                             # (False when a claimed subgroup is empty)

    def __str__(self):
        s = "SUPPORTED" if self.supported else "REFUTED"
        return (f"[{s}] {self.hypothesis.text or self.hypothesis.variable}\n"
                f"    causal ATE={self.ate_causal:+.3f}  observational={self.ate_observational:+.3f}\n"
                f"    {self.explanation}")


class ExperimentRunner:
    def __init__(self, sim: Simulator, n: int = 100_000, min_effect: float = 0.02,
                 mag_tol: float = 0.30):
        self.sim = sim
        self.n = n
        self.min_effect = min_effect
        self.mag_tol = mag_tol
        if sim._cutoff is None:
            sim.auto_calibrate()
            sim.generate(n)
        self._df = sim.generate(n)

    def _arms(self, h: Hypothesis):
        """Choose intervention arms from data if the agent didn't specify them."""
        col = self._df[h.variable]
        is_cat = col.dtype == object
        # a categorical variable can only be contrasted level-vs-level, regardless
        # of the stated direction (LLM sometimes labels it '+'/'-').
        if h.direction == "level" or is_cat:
            levels = list(col.unique())
            level = h.level if h.level in levels else max(
                levels, key=lambda L: self._df.loc[col == L, self.sim.spec.outcome_name].mean())
            base = h.base if h.base in levels else min(
                levels, key=lambda L: self._df.loc[col == L, self.sim.spec.outcome_name].mean())
            if level == base and len(levels) > 1:
                base = next(L for L in levels if L != level)
            self._forced_level = True
            return level, base
        self._forced_level = False
        hi = h.hi if h.hi is not None else col.quantile(0.85)
        lo = h.lo if h.lo is not None else col.quantile(0.15)
        return hi, lo

    def run(self, h: Hypothesis) -> Verdict:
        hi, lo = self._arms(h)

        # global experiment
        res = self.sim.ate(h.variable, hi, lo, n=self.n)
        ate = res["ate_causal"]
        obs = res["ate_observational"]

        # experiment within the claimed subpopulation, if any
        ate_cond = None
        rc = None
        if h.condition is not None:
            rc = self.sim.ate(h.variable, hi, lo, n=self.n, condition=h.condition)
            ate_cond = rc["ate_causal"]

        # which effect (and which significance stats) does the claim refer to?
        effect = ate_cond if h.condition is not None else ate
        sig = rc if h.condition is not None else res

        # a claim whose subpopulation is empty (mask matched no rows) cannot be tested —
        # ate_cond comes back NaN. That is NOT a refutation; mark it untestable so it is
        # excluded from precision rather than silently counted as a false positive.
        evaluable = not (effect is None or (isinstance(effect, float) and np.isnan(effect)))

        # direction check (a forced-level categorical is judged on effect existence)
        if not evaluable:
            direction_correct = False
        elif getattr(self, "_forced_level", False) or h.direction == "level":
            direction_correct = abs(effect) >= self.min_effect  # any real level effect
        elif h.direction == "+":
            direction_correct = effect > 0
        elif h.direction == "-":
            direction_correct = effect < 0
        else:
            direction_correct = abs(effect) >= self.min_effect

        # a claim is supported only if the effect is (a) large enough, (b) in the
        # claimed direction, AND (c) statistically distinguishable from zero — the
        # CI on the relevant ATE must exclude 0. On the large simulator (c) rarely
        # binds, but it is what makes precision honest on small/noisy static data.
        significant = bool(sig.get("significant", False))
        supported = (evaluable and abs(effect) >= self.min_effect
                     and direction_correct and significant)

        # confounding: observational sign/magnitude disagrees materially with causal
        confounded = (not np.isnan(obs)) and abs(obs - ate) > 0.03
        sign_flip = (not np.isnan(obs)) and (np.sign(obs) != np.sign(ate)) and abs(ate) > self.min_effect

        # subpopulation-only: no global effect but a real conditional one
        subpop_only = (h.condition is not None and abs(ate) < self.min_effect
                       and ate_cond is not None and abs(ate_cond) >= self.min_effect)

        # magnitude accuracy vs agent's estimate
        magnitude_ok = None
        if evaluable and h.claimed_effect is not None and abs(effect) > 1e-6:
            magnitude_ok = abs(h.claimed_effect - effect) / abs(effect) <= self.mag_tol

        # build explanation
        parts = []
        if not evaluable:
            parts.append(f"UNTESTABLE: claimed subpopulation '{h.condition_desc}' "
                         f"matched no rows — excluded from precision")
        elif not supported:
            if abs(effect) < self.min_effect:
                parts.append(f"no real causal effect (|ATE|={abs(effect):.3f} < {self.min_effect})")
            elif not direction_correct:
                parts.append(f"effect is opposite to the claimed '{h.direction}' direction")
            elif not significant:
                parts.append(f"effect not statistically significant "
                             f"(95% CI [{sig.get('ci_low', float('nan')):+.3f}, "
                             f"{sig.get('ci_high', float('nan')):+.3f}] includes 0)")
        else:
            parts.append(f"real causal effect in the claimed direction "
                         f"(95% CI [{sig.get('ci_low', float('nan')):+.3f}, "
                         f"{sig.get('ci_high', float('nan')):+.3f}], p={sig.get('p_value', float('nan')):.1e})")
        if sign_flip:
            parts.append(f"CONFOUNDED: observational sign ({obs:+.3f}) is OPPOSITE the "
                         f"true causal sign ({ate:+.3f}) — naive analysis would mislead")
        elif confounded:
            parts.append(f"confounded: observational ({obs:+.3f}) biased vs causal ({ate:+.3f})")
        if subpop_only:
            parts.append(f"SUBPOPULATION-ONLY: ~0 globally ({ate:+.3f}) but "
                         f"{ate_cond:+.3f} within '{h.condition_desc}'")
        if magnitude_ok is not None:
            parts.append(f"claimed magnitude {h.claimed_effect:+.3f} "
                         f"{'matches' if magnitude_ok else 'MISSES'} true {effect:+.3f}")

        # Report the ATE the claim is actually about (the conditional effect for a
        # subgroup claim, else the global one) so it matches the CI/p, which are
        # computed on that same `effect`. Otherwise a subgroup claim would show a
        # global ATE next to a conditional CI and look inconsistent.
        reported_ate = effect if not (isinstance(effect, float) and np.isnan(effect)) else ate
        return Verdict(
            hypothesis=h, ate_causal=reported_ate, ate_observational=obs,
            supported=supported, direction_correct=direction_correct,
            confounded=confounded, subpop_only=subpop_only,
            magnitude_ok=magnitude_ok, explanation="; ".join(parts),
            ci_low=sig.get("ci_low", float("nan")),
            ci_high=sig.get("ci_high", float("nan")),
            p_value=sig.get("p_value", float("nan")),
            significant=significant, evaluable=evaluable,
        )
