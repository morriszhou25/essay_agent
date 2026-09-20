r"""Stage-4 script-generation experiment (NOT collected by pytest).

Problem
-------
Stage 4 asks the model for ONE JSON object that carries the plan *and* the
complete ``repro.py`` source (``schemas.dialogue.ReproScript``). With 24 cards
the prompt grows to ~58k chars and the model has to fit plan + ~200 lines of
python inside an 8192-token JSON envelope. Two failure modes follow:

1. the envelope is cut off, so the JSON never parses (the real failure was
   ``structured output failed for ReproScript: could not parse JSON from reply``);
2. the model under-escapes backslashes inside the code string (it writes
   ``"\d+"`` instead of ``"\\d+"``), so the JSON is invalid even when complete.

``nodes.execute.check_code`` cannot help with either: it only runs on a reply
that already parsed, and even then the retry loop asks again and finally returns
the broken script without complaining.

Prototype under test
--------------------
``prototype_generate_script`` splits the call in two:

* the plan alone, as JSON into ``ExecPlan`` (small, structured, easy to check);
* the script as *plain text* (no JSON escaping at all), then ``extract_code``
  plus ``check_code``;
* a repair round that feeds the static-check error back;
* the plan/params go to ``code/plan_<tag>.json`` so the script never has to
  re-emit JSON literals.

Offline mode replays the known failure shapes with a scripted LLM, no network.
Live mode (``--live``) replays the real 24-card Adam prompt from
``.essay_agent/runs/run-20260919-121905-7c867d`` against the configured
provider.

Usage:
    python tests/experiment_stage4_script.py
    python tests/experiment_stage4_script.py --live
    python tests/experiment_stage4_script.py --live --limit-cards 6
"""

# ruff: noqa: E402 -- the package is imported after the src/ path shim below.
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from essay_agent.errors import LLMError, LLMReplyError
from essay_agent.llm import extract_json, looks_cut_off, preview_reply
from essay_agent.nodes.execute import check_code
from essay_agent.schemas.dialogue import ExecPlan, ReproScript

RUN_DIR = ROOT / ".essay_agent" / "runs" / "run-20260919-121905-7c867d"
ARTIFACT_DIR = ROOT / ".essay_agent" / "experiments" / "stage4"


# --------------------------------------------------------------------- fixtures
GOOD_CODE = '''\
"""Lightweight reproduction of the Adam update rule (experiment fixture)."""
import argparse
import json
import os
import re
import sys
import time

TOTAL = 20


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--out", default=".")
    args = parser.parse_args()
    figures = os.path.join(args.out, "figures")
    os.makedirs(figures, exist_ok=True)
    pattern = re.compile(r"^step-(\\d+)-ok$")
    if args.preflight:
        start = time.perf_counter()
        for _ in range(2):
            time.sleep(0.001)
        per_step = (time.perf_counter() - start) / 2
        print(json.dumps({"event": "estimate", "estimated_full_seconds": per_step * TOTAL,
                          "params": {"steps": TOTAL}}), flush=True)
        return 0
    for step in range(1, TOTAL + 1):
        assert pattern.match(f"step-{step}-ok")
        print(json.dumps({"event": "progress", "step": step, "total": TOTAL}), flush=True)
    print(json.dumps({"event": "metric", "name": "adam_matches_reference", "value": 1.0}))
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump({"adam_matches_reference": 1.0, "wall_seconds": 0.1}, handle)
    with open(os.path.join(figures, "fig1.png"), "wb") as handle:
        handle.write(b"\\x89PNG\\r\\n\\x1a\\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

PLAN_FIXTURE = {
    "approach": "Numeric replication of the Adam update rule on a fixed gradient sequence.",
    "params": {"steps": 20, "seed": 0},
    "metrics": ["adam_matches_reference"],
    "success_signal": "adam_matches_reference == 1.0",
    "figures": ["fig1.png"],
    "cards_covered": ["c01"],
    "risks": [],
}
FULL_JSON_TEXT = json.dumps({"plan": PLAN_FIXTURE, "code": GOOD_CODE, "params": {"steps": 20}})
# What the model actually emits when it forgets to escape: a single backslash
# where the JSON grammar needs two.
UNESCAPED_JSON_TEXT = FULL_JSON_TEXT.replace(r"\\d", r"\d").replace(r"\\x89", r"\x89")
FENCED_CODE = f"Sure, here is the script you asked for:\n\n```python\n{GOOD_CODE}```\n"
FENCED_JSON_TEXT = json.dumps({"plan": PLAN_FIXTURE, "code": FENCED_CODE, "params": {}})


def _truncate_mid_code(text: str, extra: int = 400) -> str:
    """Cut the envelope inside the ``code`` string, like the real failure did."""
    marker = '"code": "'
    start = text.index(marker) + len(marker)
    return text[: start + extra]


TRUNCATED_JSON_TEXT = _truncate_mid_code(FULL_JSON_TEXT)


# --------------------------------------------------------------------- helpers
FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
CODE_START = ("import ", "from ", "def ", "class ", "if __name__", "#!", '"""', "'''")


def extract_code(reply: str) -> str:
    """Pull python source out of a plain-text reply (fenced or bare)."""
    text = (reply or "").strip()
    blocks = [block.strip() for block in FENCE.findall(text)]
    if blocks:
        return max(blocks, key=len)
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(CODE_START):
            return "\n".join(lines[index:]).strip()
    return text


class CollectorUI:
    """Minimal UI stand-in; keeps the warnings the real node would print."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warn(self, message: str, *_: Any, **__: Any) -> None:
        self.warnings.append(str(message))

    def __getattr__(self, _name: str) -> Any:  # every other UI call is a no-op
        return lambda *a, **k: None


class ScriptedLLM:
    """Deterministic LLM stand-in: one scripted answer per call.

    A ``str`` entry is fed through the real JSON parser (``extract_json``) so the
    failure is reproduced exactly as ``LLMClient`` would produce it.
    """

    def __init__(self, json_replies: list[Any], text_replies: list[Any]) -> None:
        self._json = list(json_replies)
        self._text = list(text_replies)
        self.calls: list[str] = []

    def model_name(self, role: str = "main") -> str:
        return "scripted"

    def json(self, system: str, user: str, schema: Any, **_: Any) -> Any:
        self.calls.append(f"json:{schema.__name__}")
        entry = self._pop(self._json)
        if isinstance(entry, str):
            return schema.model_validate(extract_json(entry))
        if isinstance(entry, schema):
            return entry
        return schema.model_validate(entry)

    def text(self, system: str, user: str, **_: Any) -> str:
        self.calls.append("text")
        return self._pop(self._text)

    def _pop(self, queue: list[Any]) -> Any:
        if not queue:
            raise LLMReplyError("no scripted reply left", reason="empty")
        entry = queue.pop(0)
        if isinstance(entry, Exception):
            raise entry
        return entry


def run_current_path(
    llm: Any, *, system: str, user: str, label: str = "exp_current"
) -> dict[str, Any]:
    """Path A: the pre-fix stage 4 - one ``ReproScript`` JSON call plus a blind repair loop.

    Kept here on purpose: the source now implements the split flow, so the "before"
    column has to carry its own copy of the old algorithm to stay honest.
    """
    deps = SimpleNamespace(llm=llm, ui=CollectorUI())
    try:
        script = legacy_generate_script(deps, system, user, label=label)
    except LLMError as exc:
        return {"ok": False, "raised": True, "why": f"{type(exc).__name__}: {str(exc)[:200]}"}
    error = check_code(script.code)
    return {"ok": error is None, "raised": False, "why": error or "static check passed"}


def legacy_generate_script(
    deps: Any, system: str, user: str, *, label: str, repair_attempts: int = 2
) -> Any:
    """The stage-4 algorithm before this change (one JSON envelope, blind repairs)."""
    script = deps.llm.json(system, user, ReproScript, label=label)
    for attempt in range(repair_attempts):
        error = check_code(script.code)
        if error is None:
            return script
        deps.ui.warn(f"generated code failed a static check ({error}); asking for a fix")
        script = deps.llm.json(
            system,
            f"{user}\n\n## Mandatory fix\nYour previous answer's `code` failed a static check:\n"
            f"{error}\nReturn the COMPLETE corrected file, honouring the contract.",
            ReproScript,
            label=f"{label}_repair{attempt + 1}",
        )
    return script


def prototype_generate_script(
    llm: Any,
    *,
    plan_system: str,
    plan_user: str,
    code_system: str,
    code_user: str,
    code_attempts: int = 2,
) -> dict[str, Any]:
    """Path B: small JSON plan first, then the script as plain text."""
    result: dict[str, Any] = {"ok": False, "raised": False, "why": "", "notes": [], "calls": 0}
    plan = llm.json(plan_system, plan_user, ExecPlan, label="exp_plan")
    result["calls"] += 1
    result["plan"] = plan.model_dump(mode="json")
    error = "the previous reply was empty"
    for attempt in range(1, code_attempts + 1):
        user = code_user
        if attempt > 1:
            user = (
                f"{code_user}\n\n## Repair {attempt}\n"
                f"Your previous answer was rejected: {error}\n"
                "Return the COMPLETE corrected file as raw python text, nothing else."
            )
        reply = llm.text(code_system, user, label=f"exp_code{attempt}")
        result["calls"] += 1
        result["reply_chars"] = len(reply)
        if looks_cut_off(reply):
            error = "the reply stopped mid-expression (cut off by the output limit)"
            result["notes"].append(f"code reply {attempt} looks cut off ({len(reply)} chars)")
            continue
        candidate = extract_code(reply)
        error = check_code(candidate) or ""
        if not error:
            result["ok"] = True
            result["why"] = "static check passed"
            result["code"] = candidate
            return result
        result["notes"].append(f"code reply {attempt} failed: {error}")
    result["why"] = error or "no usable code was produced"
    return result


def classify(result: dict[str, Any]) -> str:
    if result.get("ok"):
        return "ok"
    return "raised" if result.get("raised") else "broken"


# ---------------------------------------------------------------------- offline
CASES: list[dict[str, Any]] = [
    {
        "name": "truncated JSON envelope",
        "what": "reply cut off inside the code string (the real 24-card failure)",
        "current_json": [TRUNCATED_JSON_TEXT],
        "proto_text": [GOOD_CODE],
        "expect_current": "raised",
        "expect_proto": "ok",
    },
    {
        "name": "unescaped backslashes",
        "what": r"code needs \d and \x89; the model forgets to double them",
        "current_json": [UNESCAPED_JSON_TEXT],
        "proto_text": [GOOD_CODE],
        "expect_current": "raised",
        "expect_proto": "ok",
    },
    {
        "name": "prose + fence wrapper",
        "what": "the model wraps the script in a ```python fence",
        "current_json": [FENCED_JSON_TEXT, FENCED_JSON_TEXT, FENCED_JSON_TEXT],
        "proto_text": [FENCED_CODE],
        "expect_current": "broken",
        "expect_proto": "ok",
    },
    {
        "name": "cut-off plain-text script",
        "what": "the script stops mid-file; the repair round supplies it again",
        "current_json": [TRUNCATED_JSON_TEXT],
        "proto_text": [GOOD_CODE[: int(len(GOOD_CODE) * 0.45)], GOOD_CODE],
        "expect_current": "raised",
        "expect_proto": "ok",
    },
]


def run_offline() -> int:
    print("=" * 78)
    print("stage-4 script generation - offline replay of the known failure shapes")
    print("=" * 78)
    mismatches = 0
    for case in CASES:
        current = run_current_path(
            ScriptedLLM(case["current_json"], []),
            system="PLAN_SYSTEM (24-card envelope)",
            user="plan_user_message (58k chars)",
        )
        prototype = prototype_generate_script(
            ScriptedLLM([PLAN_FIXTURE], case["proto_text"]),
            plan_system="exp plan system",
            plan_user="exp plan user",
            code_system="exp code system",
            code_user="exp code user",
        )
        got_current, got_proto = classify(current), classify(prototype)
        for got, want in ((got_current, case["expect_current"]), (got_proto, case["expect_proto"])):
            if got != want:
                mismatches += 1
        print(f"\n- {case['name']}")
        print(f"    what        : {case['what']}")
        print(f"    path A today: {got_current:<6} ({current['why'][:110]})")
        print(f"    path B proto: {got_proto:<6} ({prototype['why'][:110]})")
        for note in prototype["notes"]:
            print(f"    note        : {note}")

    backslashes = GOOD_CODE.count("\\")
    print("\n" + "-" * 78)
    print(f"the fixture script holds {backslashes} single backslash(es).")
    print("  JSON transport : every one must be escaped by the model, or the JSON is invalid.")
    print("  text transport : zero escaping, the file is copied verbatim.")
    print("-" * 78)
    print(f"\nunexpected outcomes: {mismatches}")
    return 1 if mismatches else 0


# ------------------------------------------------------------------ prompt size
def load_run_material(limit: int = 0) -> dict[str, Any]:
    """The real stage-4 inputs of the 24-card Adam run."""
    from essay_agent.schemas.card import CardSet

    payload = json.loads((RUN_DIR / "cards" / "cards.json").read_text("utf-8"))
    if isinstance(payload, list):  # older runs dump the bare list
        payload = {"cards": payload, "skipped": []}
    cards = CardSet.model_validate(payload).cards
    if limit:
        cards = cards[:limit]
    paper = json.loads((RUN_DIR / "paper" / "paper.json").read_text("utf-8"))
    feasibility = json.loads((RUN_DIR / "cards" / "feasibility.json").read_text("utf-8"))
    environment = (RUN_DIR / "logs" / "environment.txt").read_text("utf-8")
    return {
        "title": paper.get("title", "unknown paper"),
        "cards": cards,
        "feasibility": feasibility.get("summary") or feasibility,
        "environment": environment,
    }


def card_digest(cards: list[Any]) -> str:
    lines = []
    for card in cards:
        criteria = "; ".join(card.success_criteria) or "(none)"
        lines.append(
            f"- {card.card_id} [{card.identity.section}] {card.claim.statement} | success: {criteria}"
        )
    return "\n".join(lines)


def prompt_size_report(limit: int) -> None:
    from essay_agent.prompts.execute import CODE_CONTRACT, plan_user_message

    material = load_run_material(limit)
    cards = material["cards"]
    current_user = plan_user_message(
        paper_title=material["title"],
        cards=cards,
        feasibility=material["feasibility"],
        environment=material["environment"],
        time_budget_seconds=600.0,
    )
    plan_user = current_user
    code_user = (
        f"## Paper\n{material['title']}\n\n"
        f"## Reproduction plan (already decided)\n```json\n{json.dumps(PLAN_FIXTURE, indent=2)}\n```\n\n"
        f"## Cards in scope ({len(cards)})\n{card_digest(cards)}\n\n"
        f"## Environment\n{material['environment']}\n\n"
        f"{CODE_CONTRACT}"
    )
    print("\n" + "=" * 78)
    print(f"prompt sizes with {len(cards)} cards")
    print("=" * 78)
    print(f"  path A single call : {len(current_user):,} chars in, plan+code+JSON out")
    print(f"  path B plan call   : {len(plan_user):,} chars in, plan only out")
    print(f"  path B code call   : {len(code_user):,} chars in, raw python out")
    print(
        f"  code-call shrink   : {100 * (1 - len(code_user) / max(1, len(current_user))):.0f}% smaller input"
    )


# ------------------------------------------------------------------------- live
PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _proxy_alive(url: str, timeout: float = 0.75) -> bool:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if not parsed.hostname:
        return False
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 80), timeout):
            return True
    except OSError:
        return False


def pin_process_proxies() -> list[str]:
    """Drop *dead* proxy env vars so the LLM client can reach the API.

    The langchain/openai clients honour ``HTTP_PROXY``/``ALL_PROXY`` (trust_env),
    and this machine exports ``http://127.0.0.1:9`` - a black hole. The paper and
    dataset tools already pin their sessions; the LLM client does not, so the
    live leg has to neutralise the dead values here. On a machine without a dead
    proxy this is a no-op.
    """
    notes: list[str] = []
    for name in PROXY_ENV_VARS:
        value = os.environ.get(name)
        if not value:
            continue
        if _proxy_alive(value):
            notes.append(f"kept {name}={value}")
        else:
            os.environ.pop(name, None)
            notes.append(f"dropped {name}={value} (nothing listens there)")
    return notes


EXPERIMENT_PLAN_SYSTEM = """\
You are the EXECUTOR of a paper-reproduction agent. You are given the reproduction cards, the
feasibility verdict, the environment and the time budget.

Return ONLY the lightweight reproduction plan: how you will test the cards, the concrete
parameters you will use (learning rate, steps, subset size, seeds), the metric names the script
must write, the success signal, the figure names, and the card ids the plan covers.

Do NOT write any code in this step. Do not describe the file layout beyond what the plan needs.
"""


def run_live(args: argparse.Namespace) -> int:
    from essay_agent.config import load_settings
    from essay_agent.llm import build_llm
    from essay_agent.prompts.execute import CODE_CONTRACT, PLAN_SYSTEM, plan_user_message

    material = load_run_material(args.limit_cards)
    cards = material["cards"]
    llm_overrides: dict[str, Any] = {}
    if args.model:
        llm_overrides["model"] = args.model
    if args.max_tokens:
        llm_overrides["max_tokens"] = args.max_tokens
    overrides = {"llm": llm_overrides} if llm_overrides else None
    settings = load_settings(env_file=ROOT / ".env", overrides=overrides)
    transcript: list[tuple[str, int, str]] = []

    def record(label: str, system: str, user: str, reply: str) -> None:
        transcript.append((label, len(user), reply))

    llm = build_llm(settings.llm, transcript=record)
    print("=" * 78)
    print("stage-4 script generation - live replay")
    print("=" * 78)
    print(
        f"  provider={settings.llm.provider} model={settings.llm.model} "
        f"max_tokens={settings.llm.max_tokens}"
    )
    print(f"  cards={len(cards)} run={RUN_DIR.name}")
    for note in pin_process_proxies():
        print(f"  proxy: {note}")

    user = plan_user_message(
        paper_title=material["title"],
        cards=cards,
        feasibility=material["feasibility"],
        environment=material["environment"],
        time_budget_seconds=settings.runtime.time_budget_seconds,
    )
    print(f"  path A prompt: {len(user):,} chars\n")

    if not args.skip_current:
        print("- path A (today): one ReproScript JSON call")
        start = time.perf_counter()
        try:
            script = llm.json(PLAN_SYSTEM, user, ReproScript, label="live_A")
            error = check_code(script.code)
            print(
                f"    parsed OK in {time.perf_counter() - start:.1f}s, code={len(script.code):,} chars"
            )
            print(f"    static check: {error or 'passed'}")
            _save("live_A_code.py", script.code)
        except Exception as exc:
            print(
                f"    FAILED after {time.perf_counter() - start:.1f}s: {type(exc).__name__}: {str(exc)[:300]}"
            )
            detail = getattr(exc, "preview", None)
            if detail:
                print(f"    preview: {preview_reply(str(detail), 120, 120)!r}")
        print(f"    reply chars: {transcript[-1][2].__len__() if transcript else 0:,}\n")

    print("- path B (prototype): plan JSON, then the script as plain text")
    start = time.perf_counter()
    try:
        plan = llm.json(EXPERIMENT_PLAN_SYSTEM, user, ExecPlan, label="live_B_plan")
        print(
            f"    plan OK in {time.perf_counter() - start:.1f}s: {len(plan.metrics)} metric(s), {len(plan.cards_covered)} card(s)"
        )
        _save(
            "live_B_plan.json",
            json.dumps(plan.model_dump(mode="json"), indent=2, ensure_ascii=False),
        )
    except Exception as exc:
        print(f"    plan FAILED: {type(exc).__name__}: {str(exc)[:300]}")
        return 1
    code_user = (
        f"## Paper\n{material['title']}\n\n"
        f"## Reproduction plan (already decided)\n```json\n{plan.model_dump_json(indent=2)}\n```\n\n"
        f"## Cards in scope ({len(cards)})\n{card_digest(cards)}\n\n"
        f"## Environment\n{material['environment']}\n\n"
        f"## Time budget\n{settings.runtime.time_budget_seconds:.0f} seconds\n\n"
        f"{CODE_CONTRACT}\n"
        "Return the complete `repro.py` as raw python text. No JSON, no markdown fence, "
        "no commentary before or after the code."
    )
    if args.compact:
        code_user += (
            "\n\n## Output budget (hard)\n"
            "The file must stay compact: at most 250 lines, at most 6 cards in scope, one-line "
            "docstring only, no commented-out code, no helper you do not call. Keep the honest "
            "comparison the cards ask for, drop everything else."
        )
    start = time.perf_counter()
    reply = llm.text(EXPERIMENT_PLAN_SYSTEM, code_user, label="live_B_code")
    code = extract_code(reply)
    error = check_code(code)
    print(
        f"    code reply in {time.perf_counter() - start:.1f}s: {len(reply):,} chars -> "
        f"{len(code.splitlines()):,} lines of python"
    )
    print(f"    static check: {error or 'passed'}")
    _save("live_B_code.py", code)
    return 0 if error is None else 1


def _save(name: str, text: str) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / name).write_text(text, encoding="utf-8")
    print(f"    saved .essay_agent/experiments/stage4/{name}")


def run_source_path(args: argparse.Namespace) -> int:
    """Run the *patched* stage 4 (`nodes.execute._generate_script`) on the real 24-card prompt."""
    from essay_agent.config import load_settings
    from essay_agent.llm import build_llm
    from essay_agent.nodes.execute import _generate_script, check_code
    from essay_agent.prompts.execute import PLAN_SYSTEM, plan_user_message

    material = load_run_material(args.limit_cards)
    llm_overrides: dict[str, Any] = {}
    if args.model:
        llm_overrides["model"] = args.model
    if args.max_tokens:
        llm_overrides["max_tokens"] = args.max_tokens
    settings = load_settings(
        env_file=ROOT / ".env", overrides={"llm": llm_overrides} if llm_overrides else None
    )
    replies: list[tuple[str, str]] = []

    def record(label: str, system: str, user: str, reply: str) -> None:
        replies.append((label, reply))

    llm = build_llm(settings.llm, transcript=record)
    ui = CollectorUI()
    print("=" * 78)
    print("stage-4 script generation - the patched source path")
    print("=" * 78)
    print(
        f"  provider={settings.llm.provider} model={settings.llm.model} "
        f"max_tokens={settings.llm.max_tokens} code_max_tokens={settings.llm.code_max_tokens}"
    )
    print(f"  cards={len(material['cards'])} run={RUN_DIR.name}")
    for note in pin_process_proxies():
        print(f"  proxy: {note}")
    user = plan_user_message(
        paper_title=material["title"],
        cards=material["cards"],
        feasibility=material["feasibility"],
        environment=material["environment"],
        time_budget_seconds=settings.runtime.time_budget_seconds,
    )
    deps = SimpleNamespace(llm=llm, ui=ui, settings=settings)
    start = time.perf_counter()
    try:
        script = _generate_script(deps, PLAN_SYSTEM, user, label="repro_script")
    except Exception as exc:
        print(
            f"  FAILED after {time.perf_counter() - start:.1f}s: {type(exc).__name__}: {str(exc)[:600]}"
        )
        for label, reply in replies:
            print(f"    {label}: {len(reply):,} chars, tail={reply[-90:]!r}")
        return 1
    print(
        f"  OK in {time.perf_counter() - start:.1f}s: {len(script.code.splitlines())} line(s), "
        f"static check: {check_code(script.code) or 'passed'}"
    )
    print(f"  params: {json.dumps(script.params)[:200]}")
    print(f"  calls: {[label for label, _ in replies]}")
    for warning in ui.warnings:
        print(f"  warn: {warning[:200]}")
    _save("source_plan.json", script.plan.model_dump_json(indent=2))
    _save("source_repro.py", script.code)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="stage-4 script-generation experiment")
    parser.add_argument("--live", action="store_true", help="replay the real 24-card prompt")
    parser.add_argument("--limit-cards", type=int, default=0, help="use only the first N cards")
    parser.add_argument("--model", default=None, help="override the configured model")
    parser.add_argument("--max-tokens", type=int, default=0, help="override llm.max_tokens")
    parser.add_argument("--compact", action="store_true", help="live: add a hard output-size cap")
    parser.add_argument("--skip-current", action="store_true", help="live: skip path A")
    parser.add_argument(
        "--source-path",
        action="store_true",
        help="run the patched nodes/execute.py on the real prompt",
    )
    args = parser.parse_args(argv)

    code = run_offline()
    try:
        prompt_size_report(args.limit_cards)
    except Exception as exc:
        print(f"\n(prompt-size report unavailable: {type(exc).__name__}: {exc})")
    if args.live:
        code = max(code, run_live(args))
    if args.source_path:
        code = max(code, run_source_path(args))
    return code


if __name__ == "__main__":
    sys.exit(main())
