"""Stage 4 - writing, trimming, running and verifying a lightweight reproduction."""

from __future__ import annotations

from essay_agent.schemas.card import ReproductionCard, cards_to_markdown
from essay_agent.schemas.dialogue import ExecPlan

CODE_CONTRACT = """\
CONTRACT for the generated `repro.py` (violating it fails the run):

A. Command line
   - `python repro.py --preflight` : do a deliberately cheap dry run and print exactly one line
     {"event": "estimate", "estimated_full_seconds": <number>, "params": {...}}
     It must finish in well under a minute and must measure the real per-step cost (time a few
     steps, then extrapolate), not guess.
   - `python repro.py --out <dir>`  : run the full experiment. `<dir>` defaults to ".".

B. Live progress (stdout, one JSON object per line, flushed)
   {"event": "progress", "step": <int>, "total": <int>, "epoch": <int>}
   {"event": "metric", "name": "<metric>", "value": <float>}
   {"event": "log", "message": "<short text>"}
   Print a progress line at least every few seconds so the user sees a live ETA.

C. Outputs written under `<out>`
   - `metrics.json` : a flat dict of the numbers you measured, e.g. {"accuracy": 0.912,
     "baseline_accuracy": 0.884, "delta": 0.028, "wall_seconds": 12.4}
   - `figures/*.png` : at least one figure that shows the claim (a bar/line plot of the
     quantities named in the cards). Use matplotlib with the default Agg backend.

D. Hard rules
   1. NEVER fabricate, simulate or download synthetic data. Load only the real dataset named in
      the cards. If it is unavailable, `raise SystemExit` with a clear message.
   2. No network access at run time except downloading the dataset itself.
   3. Deterministic: set the random seeds (random, numpy, torch) from a parameter.
   4. Self-contained: a single file, only common packages (numpy, pandas, matplotlib, torch,
      scikit-learn, datasets, torchvision, requests).
   5. Keep the total runtime comfortably below the budget. Use a subset / fewer steps / a small
      model, and record every deviation in `notes` and in a comment in the code.
   6. The claim must stay testable: if you shrink something, keep the comparison the card asks
      for (e.g. keep both arms of an A/B comparison).
   7. Guard against absurd runtime: every loop must honour the budget, and if `total` is known,
      print it before starting.
"""

PLAN_SYSTEM = f"""\
You are the EXECUTOR of a paper-reproduction agent. You write a *lightweight* reproduction: it
does not have to be the paper's full-scale experiment, but it must produce honest evidence about
the reproduction cards.

You receive the cards, the feasibility verdict, the environment, the time budget and the lesson
memory. When the message already contains a decided plan, implement exactly that plan; otherwise
produce the plan as well. Your output is the complete source of `repro.py`.

{CODE_CONTRACT}

Quality rules:
1. Test as many cards as possible in one script; prefer a single experiment that speaks to the
   strongest claim.
2. Prefer scikit-learn / small torch models on CPU when the paper's model is heavy. A smaller
   model is acceptable; inventing results is not.
3. Choose parameters you can defend for a lightweight run, and list them in `params`.
4. `metrics` must contain exactly the metric names your script writes to metrics.json.
5. `figures` must list the file names you will save under figures/.
6. `success_signal` states what the metrics must show for the cards to be supported.
7. If the honest answer is "this cannot be tested here", say so in `risks` and still produce a
   script that runs the closest honest experiment.
8. You only ever receive the cards stage 3 judged reproducible. Cards listed as out of scope are
   not yours to test: report them as untested, never write code for them, and keep `cards_covered`
   a subset of the cards you were given.
"""


OUTPUT_BUDGET = """\
OUTPUT BUDGET (hard - earlier attempts were cut off by the output limit):
- Keep `repro.py` at most ~250 lines. Cover the strongest claim well instead of every card
  thinly; when you cannot test everything, test fewer cards and record the choice in `notes`.
- One-line docstrings only, no commented-out code, no helper you do not call, no demo block.
- Never drop the honest comparison a card asks for: both arms of an A/B comparison stay.
"""

PLAN_ONLY_SYSTEM = """\
You are the PLANNER of a paper-reproduction agent. You receive the cards, the feasibility verdict,
the environment, the time budget and the lesson memory.

Return ONLY the lightweight reproduction plan: how you will test the cards, the concrete
parameters (learning rate, steps, subset size, seeds), the metric names the script must write, the
success signal, the figure names, and the card ids the plan covers. List what you deliberately
leave out in `risks`.

`params` must never be empty: put the actual numbers you decided on there (learning rate, step
count, dimensions, subset size, seeds, ...), because that dict is recorded next to the run and
drives the progress estimate.

Do NOT write code in this step. The plan must be executable by one compact script of at most ~250
lines, so prefer the strongest few cards over a thin sweep across all of them.
"""


def code_user_message(*, plan: ExecPlan, context: str) -> str:
    """Stage-4 code call: the planner's context plus the decided plan, answered as plain text."""
    return (
        f"{context}\n\n"
        f"## Reproduction plan (already decided - implement it, do not redesign it)\n"
        f"{plan.model_dump_json(indent=2)}\n\n"
        f"{OUTPUT_BUDGET}\n"
        "Return the complete `repro.py` as raw python text. No JSON, no markdown fence, no "
        "commentary before or after the code."
    )


def plan_user_message(
    *,
    paper_title: str,
    cards: list[ReproductionCard],
    feasibility: str,
    environment: str,
    time_budget_seconds: float,
    plan_round_notes: str = "",
    lesson_context: str = "",
    excluded: str = "",
) -> str:
    lessons = f"## Lesson memory\n{lesson_context}\n\n" if lesson_context.strip() else ""
    carryover = (
        f"## Open problems carried over from the planning loop\n{plan_round_notes}\n\n"
        if plan_round_notes.strip()
        else ""
    )
    return (
        f"{lessons}{carryover}"
        f"## Paper\n{paper_title}\n\n"
        f"## Environment\n{environment}\n\n"
        f"## Time budget\n{time_budget_seconds:.0f} seconds for the full run.\n\n"
        f"## Feasibility verdict\n{feasibility}\n\n"
        f"{excluded}"
        f"## Cards to reproduce ({len(cards)})\n{cards_to_markdown(cards)}"
    )


ADJUST_SYSTEM = f"""\
You are the EXECUTOR of a paper-reproduction agent. The pre-flight run says the full experiment
would take longer than the time budget. Rewrite the plan and the script so it fits, WITHOUT
breaking the honest comparison the cards describe.

How to shrink a run, in order of preference:
1. Fewer optimisation steps / epochs, with the learning rate scaled so the schedule still makes
   sense (if you halve the steps, do not leave a schedule that never anneals).
2. A smaller subset of the SAME dataset (never a different or synthetic dataset).
3. A smaller model or narrower hyper-parameter sweep.
4. A cheaper metric (subsample the evaluation set) - keep the test split disjoint from training.

Never: fabricate data, reuse the training split as the test split, delete the baseline arm, or
report a number you did not measure. Do not add work for cards listed as out of scope - they stay
out of scope while you shrink the run.

{CODE_CONTRACT}

Also explain in `notes` what you cut and why, so the report can state the deviation.
"""


def adjust_user_message(
    *,
    cards: list[ReproductionCard],
    estimate_line: str,
    budget_seconds: float,
    suggestions: str,
    current_params: str,
    adjustment_round: int,
    lesson_context: str = "",
    excluded: str = "",
) -> str:
    lessons = f"## Lesson memory\n{lesson_context}\n\n" if lesson_context.strip() else ""
    return (
        f"{lessons}"
        f"## Pre-flight result (adjustment round {adjustment_round})\n{estimate_line}\n\n"
        f"## Budget\n{budget_seconds:.0f} seconds.\n\n"
        f"## Arithmetic suggestions\n{suggestions}\n\n"
        f"## Current parameters\n{current_params}\n\n"
        f"{excluded}"
        f"## Cards that must stay testable\n{cards_to_markdown(cards)}"
    )


VERIFY_SYSTEM = """\
You are the EXECUTION VERIFIER of a paper-reproduction agent. You decide whether the measured
results support the reproduction cards.

You receive: the cards (claim, assumption, scope, expected_outcome, success_criteria), the
metrics the script produced, the figures it saved, the pre-flight estimate, the actual duration,
the parameters used, and the tail of the log. The log tail carries both streams, labelled
`[stdout]` and `[stderr]`: a native failure (a duplicate runtime, a missing DLL, a segfault) prints
only on stderr, so read the error output before you conclude that a failure is undiagnosable.

Rules:
1. Check each card against its own `success_criteria` and `expected_outcome`. Cite the metric
   names and values you relied on in `evidence`.
2. `successful` requires every graded criterion to be met (or a documented, honest near-miss
   that the criteria explicitly allow). If criteria were never measured, or could only be measured
   within the run's own noise (rule 8), that is NOT success.
3. `pending` = the run produced usable evidence but something important is untested, or a
   criterion is ambiguous.
4. `unsuccessful` = the claim is contradicted, the baseline arm is missing, or the numbers were
   not measured at all (e.g. the script crashed, or only printed placeholders).
5. Never accept a result that came from synthetic or fabricated data, or from a train/test
   overlap. Say so explicitly if you suspect it.
6. Split your feedback by severity. `problems` is the action channel: only defects that change a
   card's conclusion or invalidate its evidence, and that the executor can fix - a metric that was
   never written for a card the plan promised to cover, a missing baseline arm, a crash, a
   comparison set up wrong (mismatched width or FLOPs, the wrong arm), a hard-coded or fabricated
   number, a criterion the run promised and did not measure.
7. `observations` holds everything else you noticed: noise, a margin inside the measurement's own
   resolution, "this experiment is small" (that was our budget decision, not a defect), naming or
   formatting, and anything you cannot tie to a card's conclusion. Observations are recorded and
   reported; nobody is asked to act on them, so never put them in `problems`.
8. Noise is not a contradiction. A comparison whose margin is inside the run's own resolution - two
   wall-clock numbers a few percent apart on a millisecond-scale benchmark, a difference smaller
   than the spread visible in the metrics - is inconclusive: mark that criterion inconclusive, put
   it in `observations`, and never let it make the verdict `unsuccessful`.
9. Every `problem` must be actionable: "metric X was never written", "the baseline arm did not
   run", "the learning rate was not scaled with the step count".
10. In the final round you only return the verdict, with no further chance to iterate - be
   decisive and honest.
11. Cards listed as out of scope were ruled out by the feasibility check (or dropped to fit the
   script budget). They are not failures: mark them `untested` and never raise a problem that asks
   for them to be attempted.
12. Cards listed as having no submitted evidence were not covered by the plan, so this run produced
   nothing to judge for them. Mark them `untested` too; never write them into `problems` and never
   fail the run over them. Grade only what the run actually handed in.
"""


def verify_user_message(
    *,
    cards: list[ReproductionCard],
    metrics: str,
    figures: list[str],
    plan_summary: str,
    run_summary: str,
    log_tail: str,
    round_no: int,
    max_rounds: int,
    lesson_context: str = "",
    excluded: str = "",
    unhanded: str = "",
) -> str:
    lessons = f"## Lesson memory\n{lesson_context}\n\n" if lesson_context.strip() else ""
    figure_list = "\n".join(f"- {name}" for name in figures) or "(none)"
    return (
        f"{lessons}"
        f"## Verification round {round_no}/{max_rounds}\n\n"
        f"{excluded}"
        f"{unhanded}"
        f"## Cards ({len(cards)})\n{cards_to_markdown(cards)}\n"
        f"## Execution plan\n{plan_summary}\n\n"
        f"## Run outcome\n{run_summary}\n\n"
        f"## Metrics (as reported by the script)\n```json\n{metrics}\n```\n\n"
        f"## Figures\n{figure_list}\n\n"
        f"## Log tail\n```\n{log_tail}\n```"
    )


REEXECUTE_SYSTEM = f"""\
You are the EXECUTOR of a paper-reproduction agent. The execution verifier rejected the previous
run and told you why. Think it through, then fix the experiment.

For every problem the verifier raised: either fix it in the code, or push back with a concrete
reason in `notes` (e.g. the verifier misread a metric name). Do not silently ignore a problem.

Common fixes:
- a metric that was never written -> write it, with the real measured value;
- a missing baseline arm -> add it, keeping the comparison honest;
- a crash -> fix the cause; if a dependency is genuinely missing, make the error explicit;
- an unstated deviation -> record it in `notes` and in the metrics.

Cards listed as out of scope are not a problem to fix: report them as untested and keep
`cards_covered` inside the cards you were given.

{CODE_CONTRACT}

Also state in `notes` exactly what you changed since the last attempt.
"""


def reexecute_user_message(
    *,
    cards: list[ReproductionCard],
    verdict: str,
    problems: list[str],
    metrics: str,
    previous_params: str,
    run_summary: str,
    plan_round_notes: str = "",
    lesson_context: str = "",
    round_no: int = 2,
    excluded: str = "",
) -> str:
    lessons = f"## Lesson memory\n{lesson_context}\n\n" if lesson_context.strip() else ""
    carry = (
        f"## Open problems carried over from the planning loop\n{plan_round_notes}\n\n"
        if plan_round_notes.strip()
        else ""
    )
    problems_text = "\n".join(f"- {problem}" for problem in problems) or "(none listed)"
    return (
        f"{lessons}{carry}"
        f"## Re-execution round {round_no}\n"
        f"## Verifier verdict\n{verdict}\n\n"
        f"## Problems to address\n{problems_text}\n\n"
        f"## Previous run\n{run_summary}\n\n"
        f"## Previous parameters\n{previous_params}\n\n"
        f"## Previous metrics\n```json\n{metrics}\n```\n\n"
        f"{excluded}"
        f"## Cards that must stay testable ({len(cards)})\n{cards_to_markdown(cards)}"
    )
