# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Prompt templates for the proposal step.

Two proposers share the same output schema. ``PROPOSE_*`` asks the model to name
directions from the column descriptions alone; ``PROPOSE_ANALYTICAL_*`` first lets
the model write code to analyze the residual (the unexplained part of the outcome)
before naming directions, which surfaces interaction and combination structure a
column-name guess would miss.
"""

# --- schema-only proposer ----------------------------------------------------
PROPOSE_SYS = "You are a data scientist proposing directions to investigate. Return JSON only."

PROPOSE_PROMPT = """Dataset: {domain} — {scenario}. Binary outcome '{outcome}' ({rate:.0f}% positive).
Columns:
{context}

HYPOTHESES ALREADY CONFIRMED so far:
{confirmed}

Your job: propose {k} DISTINCT, promising NEW directions for what CAUSES the outcome.
The PRIMARY variable of each proposal MUST be a variable that does NOT already appear in the
confirmed list above — those effects are already captured and will be ignored. Target genuinely
UNEXPLORED variables. A confirmed variable may still appear as a *partner*/moderator of a new
primary, but never as the primary itself. Do NOT restate an already-confirmed hypothesis.

A causal effect can take any of these structural FORMS (illustrated with placeholder names, NOT
columns in this data — map the idea onto real columns):
  - monotone      : outcome rises/falls steadily with feat_a
  - threshold     : feat_a matters only past a cutoff (e.g. above ~the 60th percentile)
  - nonmonotone   : outcome is best at a mid-range of feat_a and worse at both extremes (U-shape)
  - ratio         : it is feat_a divided by feat_b (a rate/density) that matters, not either alone
  - interaction   : feat_a helps when feat_b is high but hurts when feat_b is low (sign flips)
  - subgroup      : feat_a matters ONLY within a subpopulation (e.g. only when cat_x == 'level_2')
  - combination   : outcome shifts only when feat_a AND feat_b AND feat_c are jointly high/low
  - categorical   : one LEVEL of a categorical column (cat_x == 'level_3') shifts the outcome vs others
Actively cover ALL of these forms across your proposals, not just monotone/threshold. Categorical
columns and multi-variable combinations are easy to overlook — deliberately propose some.

For each give the primary variable (and any partner variables the form needs) and a guess at the form.
Return JSON list: [{{"variable":"<col>","partners":["<col>",...],"direction":"monotone|threshold|nonmonotone|ratio|interaction|subgroup|combination|categorical","note":"<12 words: what's NEW about this>"}}]"""


# --- analytical (code-writing) proposer --------------------------------------
PROPOSE_ANALYTICAL_SYS = ("You are a data scientist. You explore data by writing Python, then "
                          "propose directions to investigate. Use <code>...</code> to run code and "
                          "<propose>...</propose> with a JSON list when ready.")

PROPOSE_ANALYTICAL_PROMPT = """Dataset: {domain} — {scenario}. Binary outcome '{outcome}' ({rate:.0f}% positive).
Columns:
{context}

HYPOTHESES ALREADY CONFIRMED so far:
{confirmed}

In your Python namespace you have:
  - `df`      : the FULL {n_rows:,} rows (all columns + the raw `outcome`).
  - `resid`   : a numpy array aligned to df rows = the outcome with EVERY confirmed effect above
                already subtracted out. This is the UNEXPLAINED part of the outcome. Structure that
                is still visible in `resid` is exactly what has NOT been discovered yet.

Your job: ANALYZE `resid` to find what structure remains, then propose {k} DISTINCT, promising NEW
directions for what still drives it. Do the analysis yourself with code — do not guess. Useful moves:
  - Fit a shallow tree on `resid` (e.g. sklearn DecisionTreeRegressor/GradientBoostingRegressor,
    max_depth=3) over the numeric columns and read the split variables + leaf paths. A deep leaf
    path (A>a AND B>b AND C>c with a large leaf value) is a 3-way combination effect.
  - For a candidate moderator M, split df on M's median and compare a variable's slope-vs-resid in
    each half; a large slope DIFFERENCE means M moderates it (interaction/subgroup), even when the
    marginal correlation is ~0.
  - Bin a variable into quantiles and look at mean(resid) per bin to spot thresholds / U-shapes.
  - For categorical columns, compare mean(resid) across levels to spot single-level effects.
For SPEED you MAY explore on a subsample (e.g. idx = df.sample(100_000, random_state=0).index;
sub = df.loc[idx]; r = resid[df.index.get_indexer(idx)]) — you only need to LOCATE structure here,
not measure it precisely (a later step validates on full data).

The PRIMARY variable of each proposal MUST be a variable that does NOT already appear in the confirmed
list above. A confirmed variable may appear only as a *partner*/moderator of a new primary.

You have {max_turns} code turns. When ready, emit <propose> with a JSON list:
[{{"variable":"<col>","partners":["<col>",...],"direction":"monotone|threshold|nonmonotone|ratio|interaction|subgroup|combination|categorical","note":"<12 words: what the data showed>"}}]"""


def format_confirmed(confirmed_hyps):
    """Render the confirmed-hypotheses block shared by both proposer prompts.

    Lists every confirmed primary variable (so the model can steer away from all of
    them) plus the 15 most recent one-line hypothesis summaries.
    """
    if not confirmed_hyps:
        return "  (none yet)"
    seedvars = sorted(set(h.get("seed_var", "?") for h in confirmed_hyps))
    recent = "\n".join(f"  - [{h.get('seed_var', '?')}] {h['hypothesis'].splitlines()[0][:100]}"
                       for h in confirmed_hyps[-15:])
    return (f"ALL confirmed primary variables (do NOT reuse as primary): "
            f"{', '.join(seedvars)}\n" + recent)
