# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tunable constants for the value and validity gates.

Defaults were fixed on held-out scored domains; they are conservative (they only
ever *tighten* acceptance relative to funding every proposal). Grouped by the
gate that uses them.
"""

FIT_SUBSAMPLE_N = 50_000          # rows used to FIT the residualize/liveness gates at large n
SUBSAMPLE_ROW_THRESHOLD = 200_000  # only subsample above this row count
ALPHA_INTERACTION = 0.01          # p-value gate on the A:M interaction coefficient
INTERACTION_SCAN_N = 50_000       # rows subsampled for the near-free interaction scan
INT_EFFECT_FLOOR = 0.005          # min |standardized A:M coef| for a conditional proposal to be fundable
CLAIM_SIG_P = 1e-3                # recomputed effect must be this significant to overrule a claim
ADJ_EFFECT_FLOOR = 0.02           # standardized adjusted effect below this counts as negligible
ADJ_RETAIN_FRAC = 0.5             # adjusted effect retaining < this fraction of marginal = washed out
ADJ_TOPK_CONFOUNDERS = 12         # partial out this many most-correlated covariates
