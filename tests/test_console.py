"""ConsoleUI.confirm semantics: ENTER, the accepted vocabulary and re-asking."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from essay_agent.console import CONFIRM_ATTEMPTS, ConsoleUI


class ScriptedConsole(Console):
    """A Console whose ``input`` replays a script and records the prompts."""

    def __init__(self, answers: list[object]) -> None:
        super().__init__(file=io.StringIO(), width=200, highlight=False)
        self.answers = list(answers)
        self.prompts: list[str] = []

    def input(self, prompt: str = "", **kwargs: object) -> str:  # type: ignore[override]
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return str(answer)


def ui_with(*answers: object) -> ConsoleUI:
    return ConsoleUI(ScriptedConsole(list(answers)))


@pytest.mark.parametrize("word", ["y", "Y", "yes", "Continue", "ok", "sure", "1", "是", "继续"])
def test_confirm_accepts_the_yes_vocabulary(word: str) -> None:
    assert ui_with(word).confirm("continue?", default=False) is True


@pytest.mark.parametrize("word", ["n", "N", "no", "stop", "cancel", "0", "否"])
def test_confirm_accepts_the_no_vocabulary(word: str) -> None:
    assert ui_with(word).confirm("continue?", default=True) is False


def test_confirm_enter_keeps_the_default() -> None:
    assert ui_with("").confirm("continue?", default=True) is True
    assert ui_with("").confirm("continue?", default=False) is False
    # a closed stdin must not hang or flip the answer
    assert ui_with().confirm("continue?", default=True) is True
    assert ui_with().confirm("continue?", default=False) is False


def test_confirm_prompt_says_what_enter_does() -> None:
    yes = ScriptedConsole(["y"])
    ConsoleUI(yes).confirm("continue with this document?", default=True)
    assert "ENTER = continue" in yes.prompts[0]
    # rich would swallow a bare "[Y/n]" as a markup tag, so it must be escaped
    assert "\\[Y/n]" in yes.prompts[0]

    no = ScriptedConsole(["n"])
    ConsoleUI(no).confirm("continue anyway?", default=False)
    assert "ENTER = stop" in no.prompts[0]
    assert "\\[y/N]" in no.prompts[0]


def test_confirm_reasks_on_unrecognised_input() -> None:
    console = ScriptedConsole(["maybe", "i guess so", "continue"])
    assert ConsoleUI(console).confirm("continue?", default=False) is True
    assert len(console.prompts) == 3
    assert "please answer y or n" in console.file.getvalue()


def test_confirm_gives_up_after_the_attempt_limit() -> None:
    console = ScriptedConsole(["hmm"] * 10)
    assert ConsoleUI(console).confirm("continue?", default=False) is False
    assert len(console.prompts) == CONFIRM_ATTEMPTS
