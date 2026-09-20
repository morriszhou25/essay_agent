"""Command-line user interface.

The UI is injected into every graph node (``deps.ui``), which keeps the nodes
testable: tests pass :class:`SilentUI` and assert on what was asked.

Human-in-the-loop touch points required by the spec:

* ``choose_candidate`` - the model is unsure which paper the user meant
* ``confirm``          - a blocker was found in the feasibility check
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from essay_agent.schemas.card import ReproductionCard, card_brief

STEP_STYLE = "bold cyan"
OK_STYLE = "bold green"
WARN_STYLE = "bold yellow"
ERR_STYLE = "bold red"

# Answers accepted by ``confirm``. Anything outside these sets re-asks instead of
# being silently read as "no" (which used to abort tasks when the user typed
# "continue" at a y/N prompt).
YES_WORDS = frozenset(
    {
        "y",
        "yes",
        "yeah",
        "yep",
        "ok",
        "okay",
        "sure",
        "continue",
        "proceed",
        "go",
        "true",
        "t",
        "1",
        "是",
        "好",
        "继续",
    }
)
NO_WORDS = frozenset(
    {
        "n",
        "no",
        "nope",
        "stop",
        "cancel",
        "abort",
        "quit",
        "exit",
        "false",
        "f",
        "0",
        "否",
        "不",
        "取消",
    }
)
CONFIRM_ATTEMPTS = 3


def literal_markup(text: str) -> str:
    """Escape ``[`` so rich renders bracketed literals such as ``[y/N]``.

    ``rich.markup.escape`` misses uppercase tags like ``[Y/n]`` (rich only
    recognises lowercase tag names), which is exactly why the y/N hint used to
    disappear from the prompt.
    """
    return text.replace("[", "\\[")


@runtime_checkable
class UI(Protocol):
    """Everything the nodes are allowed to ask of the human."""

    interactive: bool

    def info(self, message: str) -> None: ...

    def warn(self, message: str) -> None: ...

    def error(self, message: str) -> None: ...

    def success(self, message: str) -> None: ...

    def dim(self, message: str) -> None: ...

    def rule(self, title: str = "") -> None: ...

    def banner(self, version: str, model: str, provider: str) -> None: ...

    def help_text(self) -> None: ...

    def step(self, title: str, detail: str = "") -> None: ...

    def notice(self, title: str, body: str) -> None: ...

    def replan_notice(
        self, phase: str, round_no: int, max_rounds: int, issues: list[str]
    ) -> None: ...

    def verdict_notice(
        self, verdict: str, round_no: int, max_rounds: int, rationale: str, problems: list[str]
    ) -> None: ...

    def choose_candidate(self, candidates: list[Any], question: str) -> int | None: ...

    def confirm(self, question: str, default: bool = False) -> bool: ...

    def show_cards(self, cards: list[ReproductionCard]) -> None: ...

    def show_estimate(self, estimate: Any) -> None: ...

    def progress(self, label: str, total: int | None = None): ...


class ConsoleUI:
    """Rich-backed interactive UI."""

    interactive = True

    def __init__(self, console: Console | None = None, *, settings: Any = None) -> None:
        self.console = console or Console(highlight=False)
        self.settings = settings

    # ----------------------------------------------------------------- output
    def info(self, message: str) -> None:
        self.console.print(message)

    def warn(self, message: str) -> None:
        self.console.print(f"[{WARN_STYLE}]![/] {message}")

    def error(self, message: str) -> None:
        self.console.print(f"[{ERR_STYLE}]x[/] {message}")

    def success(self, message: str) -> None:
        self.console.print(f"[{OK_STYLE}]v[/] {message}")

    def step(self, title: str, detail: str = "") -> None:
        head = Text("> ", style=STEP_STYLE) + Text(title, style=STEP_STYLE)
        if detail:
            head.append("  " + detail, style="dim")
        self.console.print(head)

    def dim(self, message: str) -> None:
        self.console.print(f"[dim]{message}[/dim]")

    def rule(self, title: str = "") -> None:
        self.console.rule(title or None, style="cyan")

    def notice(self, title: str, body: str) -> None:
        self.console.print(Panel(body, title=title, border_style="cyan", expand=False))

    def banner(self, version: str, model: str, provider: str) -> None:
        body = Group(
            Text("lightweight paper-reproduction agent", style="italic"),
            Text(f"model: {provider}/{model}", style="dim"),
            Text("type /help for commands, /exit to leave", style="dim"),
        )
        self.console.print(Panel(body, title=f"essay-agent {version}", border_style="cyan"))

    def help_text(self) -> None:
        self.console.print(
            Panel(
                Group(
                    Text("/help              show this help"),
                    Text("/config            show the effective configuration"),
                    Text("/lesson            show long-term lesson memory"),
                    Text("/lesson clear      archive + reset lesson memory"),
                    Text("/exit | /quit      leave (Ctrl+C / Ctrl+D also work)"),
                    Text(""),
                    Text("Ctrl+C during a task cancels it and deletes every file it produced."),
                    Text("anything else is treated as a paper title, URL, DOI or arXiv id"),
                ),
                title="commands",
                border_style="dim",
                expand=False,
            )
        )

    # ------------------------------------------------------- phase transitions
    def replan_notice(self, phase: str, round_no: int, max_rounds: int, issues: list[str]) -> None:
        """Loud, unmissable notification that a verifier sent the work back."""
        label = "REPLAN" if phase == "plan" else "RE-EXECUTE"
        title = f"[{WARN_STYLE}]⟳ {label} - round {round_no}/{max_rounds}[/]"
        lines = [
            Text(
                f"{label.lower()} triggered by the {'card' if phase == 'plan' else 'execution'} verifier",
                style="bold",
            ),
            Text(f"{len(issues)} problem(s) reported:", style="yellow"),
        ]
        for issue in issues[:12]:
            lines.append(Text(f"  • {issue}", style="dim"))
        if len(issues) > 12:
            lines.append(Text(f"  ... {len(issues) - 12} more", style="dim"))
        lines.append(Text(""))
        lines.append(
            Text(
                "the main model will deliberate before rewriting the affected fields"
                if phase == "plan"
                else "the main model will deliberate before re-running",
                style="italic dim",
            )
        )
        self.console.print(Panel(Group(*lines), title=title, border_style="yellow", expand=False))

    def verdict_notice(
        self, verdict: str, round_no: int, max_rounds: int, rationale: str, problems: list[str]
    ) -> None:
        style = {"successful": OK_STYLE, "pending": WARN_STYLE, "unsuccessful": ERR_STYLE}.get(
            verdict, ""
        )
        lines = [Text(rationale)]
        if problems:
            lines.append(Text("problems:", style="yellow"))
            lines.extend(Text(f"  • {p}", style="dim") for p in problems[:12])
        self.console.print(
            Panel(
                Group(*lines),
                title=f"[{style}]verifier: {verdict.upper()} (round {round_no}/{max_rounds})[/]",
                border_style=style or "cyan",
                expand=False,
            )
        )

    def show_cards(self, cards: list[ReproductionCard]) -> None:
        table = Table(title=f"reproduction cards ({len(cards)})", show_lines=False)
        table.add_column("id", style="cyan", no_wrap=True)
        table.add_column("section")
        table.add_column("claim")
        table.add_column("expected outcome", overflow="fold")
        table.add_column("ok", justify="center")
        for card in cards:
            table.add_row(
                card.card_id,
                card.identity.section[:28],
                card.claim.statement[:60],
                card.expected_outcome[:60],
                "" if card.reproducible else "[yellow]?[/]",
            )
        self.console.print(table)

    def show_card_detail(self, card: ReproductionCard) -> None:
        self.console.print(
            Panel(card_brief(card, max_chars=1200), border_style="dim", expand=False)
        )

    def show_estimate(self, estimate: Any) -> None:
        describe = getattr(estimate, "describe", None)
        if callable(describe):
            self.console.print(f"[dim]{describe()}[/dim]")

    # ------------------------------------------------------------- human input
    def choose_candidate(self, candidates: list[Any], question: str) -> int | None:
        """Numbered menu. Returns a 0-based index, or ``None`` to abort the task."""
        table = Table(title=question, show_header=True, header_style="bold")
        table.add_column("#", justify="right", style="cyan", no_wrap=True)
        table.add_column("title")
        table.add_column("authors / year / venue", overflow="fold")
        table.add_column("source", style="dim")
        for position, candidate in enumerate(candidates, start=1):
            title = getattr(candidate, "title", str(candidate))
            author_bits = ", ".join(getattr(candidate, "authors", [])[:3])
            year = getattr(candidate, "year", None)
            venue = getattr(candidate, "venue", None)
            extra = " | ".join(str(b) for b in (author_bits, year, venue) if b)
            table.add_row(
                str(position), str(title)[:90], extra[:70], str(getattr(candidate, "source", ""))
            )
        self.console.print(table)
        self.console.print("[dim]0 = none of these (abort)   q = quit[/dim]")
        while True:
            try:
                raw = self.console.input("[bold cyan]select> [/bold cyan]").strip()
            except EOFError:
                return None
            if raw.lower() in {"q", "quit", "exit"}:
                return None
            if raw in {"", "0"}:
                return None
            if raw.isdigit():
                index = int(raw) - 1
                if 0 <= index < len(candidates):
                    return index
            self.console.print(
                f"[{WARN_STYLE}]enter a number between 1 and {len(candidates)}, or 0[/]"
            )

    def confirm(self, question: str, default: bool = False) -> bool:
        """Ask a yes/no question.

        Accepts y/yes/continue/ok/... and n/no/stop/...; a bare ENTER keeps the
        default, so ``[Y/n]`` means "ENTER continues". Unrecognised answers are
        re-asked up to :data:`CONFIRM_ATTEMPTS` times and then fall back to the
        default.
        """
        suffix = "[Y/n]" if default else "[y/N]"
        meaning = "ENTER = continue" if default else "ENTER = stop"
        prompt = (
            f"[{WARN_STYLE}]{literal_markup(question)}[/] "
            f"{literal_markup(suffix)} [dim]({meaning})[/dim] "
        )
        for _ in range(CONFIRM_ATTEMPTS):
            try:
                raw = self.console.input(prompt).strip().lower()
            except EOFError:
                return default
            if not raw:
                return default
            if raw in YES_WORDS:
                return True
            if raw in NO_WORDS:
                return False
            self.console.print(f"[dim]please answer y or n ({meaning})[/dim]")
        return default

    def ask(self, question: str, default: str = "") -> str:
        try:
            raw = self.console.input(f"[bold cyan]{question}[/bold cyan]").strip()
        except EOFError:
            return default
        return raw or default

    # --------------------------------------------------------------- progress
    def progress(self, label: str, total: int | None = None):
        from essay_agent.runtime.progress import build_sink

        return build_sink(self.console, label, total)

    @contextmanager
    def spinner(self, label: str) -> Iterator[None]:
        with self.console.status(f"[cyan]{label}[/cyan]", spinner="dots"):
            yield


class SilentUI:
    """Non-interactive UI: records everything, never blocks on input.

    ``answers`` scripts the human responses: ``choose_candidate`` pops from
    ``choices``, ``confirm`` pops from ``confirmations``.
    """

    interactive = False

    def __init__(
        self,
        console: Console | None = None,
        *,
        choices: list[int | None] | None = None,
        confirmations: list[bool] | None = None,
    ) -> None:
        self.console = console or Console(highlight=False, stderr=False)
        self.events: list[tuple[str, str]] = []
        self.choices = list(choices or [])
        self.confirmations = list(confirmations or [])
        self.asked: list[str] = []

    def _record(self, kind: str, message: str) -> None:
        self.events.append((kind, message))

    def info(self, message: str) -> None:
        self._record("info", message)

    def warn(self, message: str) -> None:
        self._record("warn", message)

    def error(self, message: str) -> None:
        self._record("error", message)

    def success(self, message: str) -> None:
        self._record("success", message)

    def dim(self, message: str) -> None:
        self._record("dim", message)

    def rule(self, title: str = "") -> None:
        self._record("rule", title)

    def banner(self, version: str, model: str, provider: str) -> None:
        self._record("banner", f"{version} {provider}/{model}")

    def help_text(self) -> None:
        self._record("help", "commands")

    def step(self, title: str, detail: str = "") -> None:
        self._record("step", f"{title} {detail}".strip())

    def notice(self, title: str, body: str) -> None:
        self._record("notice", f"{title}: {body}")

    def replan_notice(self, phase: str, round_no: int, max_rounds: int, issues: list[str]) -> None:
        self._record("replan", f"{phase} round {round_no}/{max_rounds}: " + "; ".join(issues))

    def verdict_notice(
        self, verdict: str, round_no: int, max_rounds: int, rationale: str, problems: list[str]
    ) -> None:
        self._record("verdict", f"{verdict} round {round_no}/{max_rounds}: {rationale}")

    def show_cards(self, cards: list[ReproductionCard]) -> None:
        self._record("cards", ", ".join(card.card_id for card in cards))

    def show_estimate(self, estimate: Any) -> None:
        self._record("estimate", str(getattr(estimate, "describe", lambda: estimate)()))

    def choose_candidate(self, candidates: list[Any], question: str) -> int | None:
        self.asked.append(question)
        if self.choices:
            return self.choices.pop(0)
        return 0 if candidates else None

    def confirm(self, question: str, default: bool = False) -> bool:
        self.asked.append(question)
        if self.confirmations:
            return self.confirmations.pop(0)
        return default

    def ask(self, question: str, default: str = "") -> str:
        self.asked.append(question)
        return default

    def progress(self, label: str, total: int | None = None):
        from essay_agent.runtime.progress import NullSink

        return NullSink()

    @contextmanager
    def spinner(self, label: str) -> Iterator[None]:
        self._record("spinner", label)
        yield

    # ------------------------------------------------------------------ query
    def rendered(self) -> str:
        return "\n".join(f"{kind}: {message}" for kind, message in self.events)

    def find(self, kind: str) -> list[str]:
        return [message for event_kind, message in self.events if event_kind == kind]


def build_ui(
    settings: Any = None, *, interactive: bool = True, console: Console | None = None
) -> UI:
    """Pick the UI implementation for this run."""
    if interactive:
        return ConsoleUI(console or Console(highlight=False), settings=settings)
    return SilentUI(console or Console(highlight=False, force_terminal=False))
