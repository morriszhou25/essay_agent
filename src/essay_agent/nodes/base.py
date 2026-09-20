"""Shared node plumbing: dependency injection and small state helpers."""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from essay_agent.config import Settings
from essay_agent.connectivity import (
    DATASET_SOURCE,
    MODEL_API,
    PAPER_SOURCE,
    ConnectivityProbe,
    ConnectivityReport,
)
from essay_agent.console import UI, build_ui
from essay_agent.llm import LLM, build_llm
from essay_agent.memory.lesson import LessonStore, RunLessons
from essay_agent.runtime.runner import ScriptRunner
from essay_agent.schemas.card import ReproductionCard
from essay_agent.schemas.paper import PaperRecord
from essay_agent.state import RunState
from essay_agent.tools.dataset_probe import DatasetProber, ProbeResult
from essay_agent.tools.paper_search import PaperSearcher
from essay_agent.workspace import RunWorkspace

NodeReturn = dict[str, Any]

_PACKAGES = (
    "numpy",
    "pandas",
    "matplotlib",
    "torch",
    "sklearn",
    "datasets",
    "torchvision",
    "requests",
    "pypdf",
    "scipy",
)


@dataclass
class Deps:
    """Everything the nodes depend on, injected so tests can swap it out."""

    settings: Settings
    llm: LLM
    ui: UI
    searcher: PaperSearcher
    prober: DatasetProber
    runner: ScriptRunner
    lesson_store: LessonStore
    connectivity: ConnectivityProbe
    clock: Callable[[], float] = time.monotonic
    workspace_factory: Callable[[str, str], RunWorkspace] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- accessors
    def workspace(self, state: RunState) -> RunWorkspace:
        if self.workspace_factory is not None:
            return self.workspace_factory(state["run_id"], state.get("slug", ""))
        return RunWorkspace.create(
            self.settings.resolved_paths().workspace_root,
            state["run_id"],
            state.get("slug", ""),
        )

    def lessons(self, state: RunState) -> RunLessons:
        return RunLessons(self.workspace(state).subdir("lesson"), state["run_id"])

    def lesson_context(self, phase: str) -> str:
        return self.lesson_store.context(phase)

    def sink_factory(self) -> Callable[[str, int | None], Any]:
        return lambda label, total=None: self.ui.progress(label, total)


def build_deps(
    settings: Settings,
    ui: UI | None = None,
    *,
    llm: LLM | None = None,
    searcher: PaperSearcher | None = None,
    prober: DatasetProber | None = None,
    runner: ScriptRunner | None = None,
    lesson_store: LessonStore | None = None,
    connectivity: ConnectivityProbe | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Deps:
    """Assemble the dependency bundle for a session."""
    paths = settings.resolved_paths().ensure()
    active_ui = ui or build_ui(settings, interactive=False)
    transcript = None
    if settings.verbose:
        transcript = _make_transcript_logger()
    probe = connectivity or ConnectivityProbe(
        budget_seconds=settings.search.connectivity_budget_seconds
    )
    return Deps(
        settings=settings,
        # Pinned to the channel the probe verified for the model API, exactly like the
        # tools below: a dead proxy in the environment must not break every model call.
        llm=llm
        or build_llm(
            settings.llm,
            transcript=transcript,
            verbose=settings.verbose,
            route=probe.provider(settings, MODEL_API),
        ),
        ui=active_ui,
        # The clients are pinned to the channel the probe verified, so a broken or
        # missing proxy environment variable cannot hijack the actual requests.
        searcher=searcher
        or PaperSearcher(
            settings.search,
            cache_dir=paths.cache_dir,
            route=probe.provider(settings, PAPER_SOURCE),
        ),
        prober=prober or DatasetProber(route=probe.provider(settings, DATASET_SOURCE)),
        runner=runner
        or ScriptRunner(settings.runtime, python_executable=settings.python_executable()),
        lesson_store=lesson_store or LessonStore(paths.lesson_dir, settings.memory),
        connectivity=probe,
        clock=clock,
    )


def _make_transcript_logger() -> Callable[[str, str, str, str], None]:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = (__import__("pathlib").Path.cwd() / ".essay_agent" / f"llm-{stamp}.log").resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    def _log(label: str, system: str, user: str, reply: str) -> None:
        with target.open("a", encoding="utf-8") as handle:
            handle.write(
                f"\n{'=' * 78}\n[{label}]\n--- system ---\n{system}\n--- user ---\n{user}\n--- reply ---\n{reply}\n"
            )

    return _log


# --------------------------------------------------------------------- state
def cards_of(state: RunState) -> list[ReproductionCard]:
    return [ReproductionCard.model_validate(payload) for payload in state.get("cards", [])]


def dump_cards(cards: list[ReproductionCard]) -> list[dict[str, Any]]:
    return [card.model_dump(mode="json") for card in cards]


def paper_of(state: RunState) -> PaperRecord | None:
    payload = state.get("paper")
    return PaperRecord.model_validate(payload) if payload else None


def json_block(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


# ----------------------------------------------------------- stage-4 card scope
EXEC_CARD_LIMIT = 10
"""Most cards one lightweight script may be asked to cover."""


@dataclass(frozen=True)
class CardScope:
    """Which cards stage 4 may execute, and which are excluded (with the reason).

    Stage 3 already decides whether each card can be reproduced here. Stages 4 and 4.1 used to
    ignore that verdict and hand the whole card set to both the script writer and the execution
    verifier, which then graded - and demanded re-runs for - cards the feasibility check had just
    ruled out. This object is the single place that decision is made.
    """

    cards: list[ReproductionCard]
    excluded_blocked: list[str]
    excluded_budget: list[str]
    relaxed: bool = False

    @property
    def excluded(self) -> list[str]:
        return [*self.excluded_blocked, *self.excluded_budget]

    def describe(self) -> str:
        """One line for the console and the run log."""
        parts = [f"{len(self.cards)} card(s) in scope"]
        if self.excluded_blocked:
            parts.append(f"{len(self.excluded_blocked)} excluded by the feasibility check")
        if self.excluded_budget:
            parts.append(f"{len(self.excluded_budget)} dropped to fit the script budget")
        if self.relaxed:
            parts.append(
                "every card was flagged infeasible; keeping the first few as a best effort"
            )
        return "; ".join(parts)

    def summary(self) -> dict[str, Any]:
        """JSON-friendly form, recorded in the run state and the run log."""
        return {
            "in_scope": [card.card_id for card in self.cards],
            "excluded_blocked": list(self.excluded_blocked),
            "excluded_budget": list(self.excluded_budget),
            "relaxed": self.relaxed,
        }

    def context(self) -> str:
        """The out-of-scope section for the prompts (empty when nothing was excluded)."""
        if not self.excluded and not self.relaxed:
            return ""
        lines = [
            "## Cards out of scope (report these as untested - do NOT test, grade or re-execute "
            "them)"
        ]
        if self.relaxed:
            lines.append(
                "- the feasibility check flagged every card, so the scope was relaxed to a best "
                "effort; these are not graded either"
            )
        else:
            lines.append(
                f"- blocked by the stage-3 feasibility check ({len(self.excluded_blocked)}): "
                f"{_id_list(self.excluded_blocked)}"
            )
        if self.excluded_budget:
            lines.append(
                f"- dropped to fit the script output budget ({len(self.excluded_budget)}): "
                f"{_id_list(self.excluded_budget)}"
            )
        lines.append(
            "They are out of scope, not failed: state them as untested in the report and spend "
            "the run on the cards above."
        )
        return "\n".join(lines) + "\n\n"


def _id_list(ids: list[str]) -> str:
    return ", ".join(ids) if ids else "-"


UNHANDED_HEADER = "## Cards with no submitted evidence"


def covered_card_ids(state: RunState) -> list[str]:
    """Card ids the executor's plan promised to cover (``exec_plan.cards_covered``)."""
    return [str(card_id) for card_id in (state.get("exec_plan") or {}).get("cards_covered") or []]


def unhanded_cards(state: RunState, cards: list[ReproductionCard]) -> list[str]:
    """Cards the run never handed in: in scope, but not covered by the plan.

    There is no evidence to grade for these, so the verifier must mark them `untested` instead of
    failing the run over work it never attempted. Only a plan that *states* its coverage can
    shrink the graded set this way; a plan with no ``cards_covered`` leaves every card gradeable,
    so this can never widen what stage 4 was allowed to touch.
    """
    covered = covered_card_ids(state)
    if not covered:
        return []
    known = set(covered)
    return [card.card_id for card in cards if card.card_id not in known]


def unhanded_context(ids: list[str]) -> str:
    """The ``no submitted evidence`` section for the verifier prompt."""
    if not ids:
        return ""
    return (
        f"{UNHANDED_HEADER} (do NOT grade them)\n"
        f"- {_id_list(ids)}\n"
        "The plan does not cover them, so this run hands in no evidence for them: mark them "
        "`untested`, never write them into `problems`, and never fail the run over them - that "
        "would only demand work the run never attempted.\n\n"
    )


def feasibility_verdicts(state: RunState) -> dict[str, bool]:
    """``card_id -> feasible`` from the stage-3 report.

    ``feasibility["checks"]`` is stored as plain dicts in the state, but accept model objects too
    so the helper cannot break on a caller that passes the pydantic report straight through.
    """
    verdicts: dict[str, bool] = {}
    for check in (state.get("feasibility") or {}).get("checks") or []:
        payload = check if isinstance(check, dict) else check.model_dump()
        card_id = str(payload.get("card_id") or "").strip()
        if card_id:
            verdicts[card_id] = bool(payload.get("feasible"))
    return verdicts


def card_scope(
    state: RunState,
    *,
    cards: list[ReproductionCard] | None = None,
    limit: int = EXEC_CARD_LIMIT,
) -> CardScope:
    """The stage-4 scope: feasible and never-classified cards, capped at ``limit``.

    Cards the feasibility check ruled out are excluded and *named*. A scope that would come out
    empty (stage 3 flagged everything, or the user chose to continue against it) is relaxed to a
    best effort rather than leaving stage 4 with nothing to do.
    """
    known = cards if cards is not None else cards_of(state)
    verdicts = feasibility_verdicts(state)
    selected = [card for card in known if verdicts.get(card.card_id) is not False]
    relaxed = not selected and bool(known)
    if relaxed:
        selected, blocked = list(known), []
    else:
        blocked = [card.card_id for card in known if verdicts.get(card.card_id) is False]
    return CardScope(
        cards=selected[:limit],
        excluded_blocked=blocked,
        excluded_budget=[card.card_id for card in selected[limit:]],
        relaxed=relaxed,
    )


def truncate(text: str, limit: int, note: str = "... truncated ...") -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    return f"{text[:head]}\n{note}\n{text[-(limit - head) :]}"


def section_outline(state: RunState, limit: int = 4000) -> str:
    """Section list for the verifier, annotated with the card coverage when it is known.

    The planner's coverage ledger is the authority on what was carded, so it is rendered
    here instead of being passed around separately.  The budget is deliberately generous:
    a truncated outline hides sections from the coverage check without saying so.
    """
    sections = state.get("sections") or []
    if not sections:
        return "(no section split available)"
    coverage = {row.get("section"): row for row in state.get("coverage") or []}
    lines = [
        f"- {section.get('name', '?')} ({len(section.get('text', ''))} chars)"
        f"{_coverage_note(coverage.get(section.get('name', '?')))}"
        for section in sections
    ]
    return truncate("\n".join(lines), limit)


def _coverage_note(row: dict[str, Any] | None) -> str:
    """How one section fared in the coverage ledger, as a short suffix."""
    if row is None:
        return ""
    status = row.get("status")
    if status == "carded":
        return f" - cards {', '.join(row.get('cards') or []) or 'none'}"
    if status == "no_testable_claim":
        return " - no testable claim"
    if status == "failed":
        return " - card mining failed"
    return " - not carded"


# --------------------------------------------------------------- environment
def _package_version(name: str) -> str | None:
    try:
        if importlib.util.find_spec(name) is None:
            return None
    except (ImportError, ValueError):
        return None
    try:
        from importlib.metadata import version

        return version(name if name != "sklearn" else "scikit-learn")
    except Exception:
        return "installed"


def resolve_device(preference: str) -> str:
    if preference != "auto":
        return preference
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def connectivity_report(deps: Deps) -> ConnectivityReport:
    """Probe what this configuration needs to reach, through the real channels."""
    return deps.connectivity.run(deps.settings)


def environment_report(
    deps: Deps, *, check_network: bool = True, connectivity: ConnectivityReport | None = None
) -> str:
    """Describe the execution environment so the model reasons from facts.

    ``connectivity`` lets a caller reuse an already computed report instead of
    probing twice.
    """
    runtime = deps.settings.runtime
    device = resolve_device(runtime.device)
    packages = []
    for name in _PACKAGES:
        version = _package_version(name)
        packages.append(f"{name}{'==' + version if version else ' (missing)'}")
    lines = [
        f"python: {sys.version.split()[0]}",
        f"platform: {platform.system()} {platform.release()} ({platform.machine()})",
        f"selected device: {device}",
        f"packages: {', '.join(packages)}",
        f"wall-clock budget for the full run: {runtime.time_budget_seconds:.0f}s",
        f"synthetic data policy: {'allowed (EXPLICIT OVERRIDE)' if runtime.allow_synthetic_data else 'FORBIDDEN'}",
    ]
    rendered = [f"- {line}" for line in lines]
    if check_network:
        rendered.extend((connectivity or connectivity_report(deps)).lines())
    return "\n".join(rendered)


def collect_probe_targets(
    cards: list[ReproductionCard], extra: list[str] | None = None
) -> list[str]:
    """Everything the cards say they need: datasets first, then other requirements."""
    targets: list[str] = []
    for card in cards:
        for dataset in card.scope.dataset or []:
            targets.append(str(dataset))
        for need in card.needs:
            targets.append(str(need))
    for item in extra or []:
        targets.append(str(item))
    seen: set[str] = set()
    unique: list[str] = []
    for target in targets:
        key = target.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(target.strip())
    return unique


def probe_report(deps: Deps, targets: list[str]) -> tuple[str, list[ProbeResult]]:
    results = deps.prober.probe_many(targets)
    if not results:
        return "", []
    return "\n".join(f"- {result.describe()}" for result in results), results


def abort_state(
    state: RunState, message: str, *, extra: dict[str, Any] | None = None
) -> NodeReturn:
    """Stop the graph in a controlled way (the workspace is kept for inspection)."""
    payload: NodeReturn = {
        "status": "aborted",
        "fetch_message": message,
        "error": message,
    }
    if extra:
        payload.update(extra)
    return payload


def fail_state(state: RunState, message: str, *, extra: dict[str, Any] | None = None) -> NodeReturn:
    """Stop the graph because of an error (kept distinct from an intentional abort)."""
    payload: NodeReturn = {"status": "failed", "error": message, "fetch_message": message}
    if extra:
        payload.update(extra)
    return payload
