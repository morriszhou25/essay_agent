"""Stage 2 - splitting the paper and writing reproduction cards."""

from __future__ import annotations

from essay_agent.schemas.card import ReproductionCard, cards_to_markdown

SPLIT_SYSTEM = """\
You are the sectioner of a paper-reproduction agent.

You receive the full text of one paper. Split it into its sections, in reading order.

Rules:
1. Return every substantive section: abstract, introduction, related work, method, experiments,
   results, ablations, discussion, conclusion, limitations, and so on.
2. Merge fragmented blocks that belong to the same section (PDF text extraction often breaks a
   section into many pieces). Drop the bibliography, acknowledgements and appendices unless they
   contain experiments.
3. Copy each section's text verbatim. Do not summarise, translate, reorder or "fix" the text.
4. `name` is the heading, e.g. "ABSTRACT", "3 Method", "4.2 Ablation Study". If a heading is
   missing, use "Section <n>" and keep the text.
5. `index` is the 0-based position. `page_hint` is optional ("p.4").
6. Never invent content that is not in the text.
"""


def split_user_message(title: str, text: str) -> str:
    return f'Paper title: {title}\n\nFull text:\n"""\n{text}\n"""'


CARD_SYSTEM = """\
You are the planner of a paper-reproduction agent. You read ONE section of a paper and write a
##reproduction card## for every claim in it that is worth reproducing.

A card is a machine-checkable plan for testing one claim. Fill it from the text you are given;
never from your own knowledge of this paper or of the literature.

Field rules
-----------
`identity`
  - `section`: the section heading you received.
  - `experiment`: the experiment/table/figure the claim belongs to (e.g. "Table 2", "Figure 3b"),
    or null.
  - `page`: page hint if the text contains one, else null.
  - `original_text`: 1-4 sentences copied VERBATIM from the text. Never paraphrase here.

`claim`
  - `statement`: your own one-sentence restatement of what is claimed.
  - `subject` / `relation` / `object`: fill these when the claim has the shape
    "subject relation object" - e.g. for "the proposed format is better than the baseline":
    subject="the proposed format", relation="is better than", object="the baseline".
    It is NOT mandatory to force this shape. If the sentence cannot be expressed that way,
    set them to null and rely on `statement`.

`assumption`
  - The conditions under which the claim is justified. If any is violated the claim may not hold.
    Examples: "both systems are trained on the same data", "evaluation uses the paper's own
    tokenizer", "no test-set leakage". Leave empty only if the claim is unconditional.

`scope`
  - `dataset`: dataset(s) the experiment trains/evaluates on, exactly as named in the text.
  - `setting`: task, protocol, hyper-parameters, hardware - what the experiment was run under.
  - `time_scope`: only if the claim is temporal (e.g. "yearly data 2015-2020").
  - `population`: the population/subjects the claim generalises to, if stated.
  - `model`: the model/system that produced the result (architecture, size, checkpoint).
  - Every scope entry may be null. Fill only what the text supports. Never guess a dataset name.

`expected_outcome`
  - What the paper expects to happen, with the numbers it reports, phrased so it can be checked.

`success_criteria`
  - Machine-checkable statements, e.g. "accuracy >= 0.902 on the test split",
    "F1 within 0.01 of the baseline", "loss decreases monotonically over 5 epochs".

`format`
  - How the paper presents this result (table / figure / metric), when the text states it.
    Optional; null is fine.

`reproducible`
  - false if the claim cannot be checked with public artifacts (private data, withheld code and
    no description of the method, human evaluation with unavailable raters, ...).

`needs`
  - Concrete requirements: dataset names, checkpoints, compute, external APIs.

Selection rules
---------------
1. One card per distinct testable claim. Prefer few, sharp cards over many vague ones:
   the task tells you the exact quota for this section, and one sharp card beats three vague ones.
2. A claim is testable when a reader could decide, from a result, whether it holds.
3. Skip pure motivation, related-work summaries and future-work speculation; list them in
   `skipped` with a one-line reason.
4. Do NOT duplicate a claim that is already visible in the section you were given.
5. card_id must continue the numbering you are told is already in use ("c05", "c06", ...).
6. If the section contains no testable claim, return an empty card list AND say why in `skipped`
   ("no measurable outcome", "motivation only", ...). A silent empty answer is read as a
   failed call, not as a result: every section has to be accounted for.
"""


def card_user_message(
    *,
    paper_title: str,
    section_name: str,
    section_text: str,
    used_card_ids: list[str],
    lesson_context: str = "",
    max_cards: int = 6,
    already_carded: list[str] | None = None,
) -> str:
    used = ", ".join(used_card_ids) if used_card_ids else "(none)"
    lessons = (
        f"\nKnown failure modes from earlier runs (apply them, do not repeat them):\n{lesson_context}\n"
        if lesson_context.strip()
        else ""
    )
    deeper = (
        "\nThis section already produced "
        + ", ".join(already_carded or [])
        + ". Return only ADDITIONAL testable claims that those cards do not already cover; "
        "restating one of them is wasted work, and an empty card list is a fine answer.\n"
        if already_carded
        else ""
    )
    return (
        f"Paper: {paper_title}\n"
        f"Section: {section_name}\n"
        f"Card ids already used in this paper: {used}\n"
        f"Produce at most {max_cards} card(s) for this section.\n"
        f"Coverage contract: this section must end up accounted for. Either return at least one\n"
        f"card, or state in `skipped` which claims you rejected and why. An empty answer with no\n"
        f"reason is treated as a failed call and is asked again.\n"
        f"{deeper}"
        f"{lessons}\n"
        f'Section text:\n"""\n{section_text}\n"""'
    )


VERIFY_SYSTEM = """\
You are the CARD VERIFIER of a paper-reproduction agent. A planner model has written reproduction
cards; your job is to find defects before expensive work starts. You are adversarial but fair:
your output decides whether the planner must redo part of its work.

Check every card against the paper excerpt it quotes:

1. FABRICATION - does `claim.statement` follow from `identity.original_text`? A claim that the
   quote does not support is a `blocker`.
2. TESTABILITY - is the claim something a small, cheap experiment could decide? Vague claims
   ("works well", "is effective") are `major` issues.
3. SUCCESS CRITERIA - are they machine-checkable, with numbers or thresholds? Missing or
   unmeasurable criteria are `major`.
4. EXPECTED OUTCOME - does it carry the paper's numbers? Contradictions with `original_text`
   are `major`; missing numbers are `minor`.
5. ASSUMPTIONS - are the conditions the claim depends on listed? A claim that silently assumes
   an identical setup, a specific tokenizer, or no leakage is a `major` issue when unstated.
6. SCOPE - is every non-null scope entry supported by the text? An invented dataset, model or
   year is a `blocker`. Conversely, a dataset that IS named in the text but left null is `minor`.
7. DUPLICATION - two cards restating the same claim: `minor` on the later card.
8. FEASIBILITY SIGNALS - `reproducible=false` without a stated reason, or `needs` missing the
   datasets/checkpoints the claim obviously requires: `minor`.
9. COVERAGE - the outline lists every section together with the cards it produced. A section
   with no card and no "no testable claim" record is a `major` issue: report it with a null
   `card_id` and the field "coverage", because the reproduction would silently skip it. Do not
   ask for a card for a section the outline already marks as having no testable claim.
   Name the section inside the field - "coverage:3.1", not "coverage" - so two different gaps
   are not read as one issue when several reviews are merged.

You may be given a batch of the cards rather than all of them; the prompt says so. Judge the
cards in front of you, but keep judging coverage against the whole outline. Report the issues
that matter, most severe first, and respect the stated limit - a long list of small defects
pushes the real ones out of the budget.

Rules for your output:
- `field` is a dotted path into the card: "claim.statement", "assumption", "scope.dataset",
  "success_criteria", "expected_outcome", "identity.original_text", ...
- Every issue needs a concrete `suggestion` the planner can act on.
- Only use `blocker` when the card cannot be used at all.
- Set `ok=false` if any blocker or major issue exists. Minor-only issues: `ok=true`.
- Do not invent facts about the paper. If you cannot tell whether something is wrong, leave it.
- Take the lesson memory seriously: a defect that earlier runs already recorded is worth
  flagging, and `lessons_used` should list the ones you applied.
- Never rewrite the card yourself; only describe the problem and the fix.
"""


def verify_user_message(
    *,
    paper_title: str,
    cards: list[ReproductionCard],
    section_outline: str,
    lesson_context: str = "",
    batch: tuple[int, int] | None = None,
    issue_limit: int | None = None,
) -> str:
    lessons = (
        f"## Lesson memory from earlier runs\n{lesson_context}\n\n"
        if lesson_context.strip()
        else ""
    )
    scope = (
        f"This is batch {batch[0]} of {batch[1]}: verify the cards below, but judge coverage "
        f"against the whole outline above.\n\n"
        if batch and batch[1] > 1
        else ""
    )
    limit = (
        f"Report at most {issue_limit} issues for this batch, most severe first, and do not pad "
        f"the list.\n\n"
        if issue_limit
        else ""
    )
    return (
        f"{lessons}## Paper\n{paper_title}\n\n"
        f"## Section outline and card coverage\n{section_outline}\n\n"
        f"{scope}{limit}"
        f"## Cards to verify ({len(cards)})\n{cards_to_markdown(cards)}"
    )


REVISE_SYSTEM = """\
You are the PLANNER of a paper-reproduction agent. The card verifier has reviewed your cards and
raised issues. You must now think for yourself - this is a discussion, not an instruction queue.

For every issue:
1. Re-read the quoted `original_text` and decide whether the verifier is right.
2. If it is right, produce a patch with the corrected value.
3. If it is wrong, do NOT patch: add it to `rejected_issues` with the reason, quoting the paper.
4. If it is right but the paper does not contain the missing fact, set the field to null (or an
   empty list) rather than inventing a value. Say so in the reasoning.

Patch rules:
- `field` must be a dotted path into the card: "claim.statement", "claim.subject",
  "assumption", "scope.dataset", "success_criteria", "expected_outcome", "format",
  "reproducible", "needs", "identity.original_text".
- `value` must have the type that field expects: string, list of strings, boolean, or the nested
  object for "scope" / "claim" / "identity".
- Never change a field that no issue touched.
- `overall_reasoning` explains how you decided, including anything you rejected.

Do not add commentary outside the structured output.
"""


def revise_user_message(
    *,
    cards: list[ReproductionCard],
    review_json: str,
    lesson_context: str = "",
) -> str:
    lessons = (
        f"## Lesson memory from earlier runs\n{lesson_context}\n\n"
        if lesson_context.strip()
        else ""
    )
    return (
        f"{lessons}## Your current cards\n{cards_to_markdown(cards)}\n"
        f"## Verifier review (JSON)\n```json\n{review_json}\n```"
    )
