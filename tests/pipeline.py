"""Shared scaffolding for end-to-end pipeline tests (no network, no real model)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from essay_agent.config import Settings
from essay_agent.console import SilentUI
from essay_agent.memory.lesson import LessonStore
from essay_agent.nodes.base import Deps
from essay_agent.runtime.progress import NullSink
from essay_agent.runtime.runner import ScriptRunner
from essay_agent.schemas.paper import PaperSection
from tests.fakes import FakeConnectivity, FakeLLM, FakeProber, FakeSearcher, make_candidate

PAPER_TEXT = (
    "Attention-like models are strong. " * 60
    + "\n1 Introduction\nWe propose a method that improves accuracy by 2.8 points over the baseline.\n"
    + "2 Experiments\nWe train ResNet-18 on CIFAR-10 and report accuracy.\n"
)

# First page of a camera-ready PDF: the ICLR banner is the first line, the real
# title is the second (pypdf renders its small caps with extra spaces).
ICLR_PAGE_1 = (
    "Published as a conference paper at ICLR 2015\n"
    "ADAM : A M ETHOD FOR STOCHASTIC OPTIMIZATION\n"
    "Diederik P. Kingma\n"
    "University of Amsterdam\n"
    "dpkingma@openai.com\n"
    "Jimmy Lei Ba\n"
    "University of Toronto\n"
    "ABSTRACT\n"
    "We introduce Adam, an algorithm for first-order gradient-based optimization.\n"
)


def script_with_estimate(seconds: float, steps: int = 12) -> str:
    """A contract-compliant repro.py that claims a given full-run duration."""
    return f"""
import argparse, json, os, sys, time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--out", default=".")
    args = parser.parse_args()
    out = args.out
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    total = {steps}
    if args.preflight:
        print(json.dumps({{"event": "estimate", "estimated_full_seconds": {seconds},
                          "params": {{"steps": total}}}}), flush=True)
        return 0
    for step in range(1, total + 1):
        print(json.dumps({{"event": "progress", "step": step, "total": total}}), flush=True)
        time.sleep(0.001)
    print(json.dumps({{"event": "metric", "name": "accuracy", "value": 0.912}}), flush=True)
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump({{"accuracy": 0.912, "baseline_accuracy": 0.884,
                   "hf_endpoint_set": int(bool(os.environ.get("HF_ENDPOINT")))}}, handle)
    print(json.dumps({{"event": "log",
                       "message": "HF_ENDPOINT=" + os.environ.get("HF_ENDPOINT", "")}}), flush=True)
    with open(os.path.join(out, "figures", "fig1.png"), "wb") as handle:
        handle.write(b"PNG")
    print(json.dumps({{"event": "done", "metrics": {{"accuracy": 0.912}}}}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""


def crashing_script() -> str:
    """A contract-compliant script that dies after the pre-flight (stage-4 crash)."""
    return """
import argparse, json, sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--out", default=".")
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps({"event": "estimate", "estimated_full_seconds": 0.2,
                          "params": {"steps": 1}}), flush=True)
        return 0
    print("boom", file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
"""


def sections() -> list[PaperSection]:
    return [
        PaperSection(index=0, name="1 Introduction", text="We propose a method."),
        PaperSection(index=1, name="2 Experiments", text="We train ResNet-18 on CIFAR-10."),
    ]


def card_payload(card_id: str = "c01") -> dict[str, Any]:
    return {
        "card_id": card_id,
        "identity": {
            "section": "2 Experiments",
            "experiment": "Table 2",
            "page": "p.5",
            "original_text": "Our method improves accuracy by 2.8 points.",
        },
        "claim": {
            "statement": "The proposed method is more accurate than the baseline.",
            "subject": "the proposed method",
            "relation": "is more accurate than",
            "object": "the baseline",
        },
        "assumption": ["both models are trained on the same data"],
        "scope": {"dataset": ["sklearn:iris"], "setting": "single seed", "model": "ResNet-18"},
        "expected_outcome": "accuracy improves by about 2.8 points",
        "success_criteria": ["accuracy - baseline_accuracy >= 0.02"],
        "format": "table",
        "reproducible": True,
        "needs": ["sklearn:iris"],
    }


def plan_payload(
    *,
    approach: str = "train a small model on the iris dataset",
    params: dict[str, Any] | None = None,
    metrics: list[str] | None = None,
    figures: list[str] | None = None,
    risks: list[str] | None = None,
    cards_covered: list[str] | None = None,
) -> dict[str, Any]:
    """The stage-4 plan the planner answers with (``ExecPlan``, small JSON)."""
    return {
        "approach": approach,
        "params": params if params is not None else {"steps": 12, "seed": 0},
        "metrics": metrics if metrics is not None else ["accuracy", "baseline_accuracy"],
        "success_signal": "accuracy beats the baseline",
        "figures": figures if figures is not None else ["fig1.png"],
        "cards_covered": cards_covered if cards_covered is not None else ["c01"],
        "risks": risks if risks is not None else ["toy dataset"],
    }


def happy_responses(
    *,
    review: dict[str, Any] | None = None,
    verdict: dict[str, Any] | None = None,
    feasibility: dict[str, Any] | None = None,
    script: str | None = None,
) -> dict[Any, Any]:
    """A complete script for the model, successful at every stage."""
    return {
        "paper_search_args": {"title": "Test Paper 0: A Study", "max_results": 4},
        "paper_match": {
            "index": 0,
            "confidence": 0.98,
            "reason": "title matches the request exactly",
            "ambiguous": False,
        },
        "section_split": {"sections": [s.model_dump() for s in sections()]},
        "cards:1 Introduction": {"cards": [card_payload("c01")], "skipped": []},
        "cards:2 Experiments": {"cards": [], "skipped": ["no further testable claim"]},
        "plan_review:1": review or {"ok": True, "issues": [], "summary": "cards are testable"},
        "card_revision:1": {
            "patches": [],
            "rejected_issues": [],
            "overall_reasoning": "nothing to do",
        },
        "feasibility": feasibility
        or {
            "checks": [
                {
                    "card_id": "c01",
                    "feasible": True,
                    "severity": "minor",
                    "findings": ["dataset reachable"],
                    "dataset": "sklearn:iris",
                    "dataset_available": True,
                }
            ],
            "blockers": [],
            "proceed": True,
            "summary": "everything needed is available",
        },
        "repro_script_plan": plan_payload(),
        "repro_script": {
            "plan": {
                "approach": "train a small model on the iris dataset",
                "params": {"steps": 12, "seed": 0},
                "metrics": ["accuracy", "baseline_accuracy"],
                "success_signal": "accuracy beats the baseline",
                "figures": ["fig1.png"],
                "cards_covered": ["c01"],
                "risks": ["toy dataset"],
            },
            "code": script or script_with_estimate(0.2),
            "params": {"steps": 12},
            "notes": "lightweight run",
        },
        "exec_verdict:1": verdict
        or {
            "verdict": "successful",
            "rationale": "accuracy 0.912 exceeds the baseline 0.884 by 2.8 points",
            "problems": [],
            "evidence": ["accuracy=0.912", "baseline_accuracy=0.884"],
            "per_card": {"c01": "supported"},
        },
    }


def build_deps(
    settings: Settings,
    llm: FakeLLM,
    *,
    searcher: FakeSearcher | None = None,
    prober: FakeProber | None = None,
    connectivity: FakeConnectivity | None = None,
    ui: SilentUI | None = None,
) -> Deps:
    paths = settings.resolved_paths().ensure()
    return Deps(
        settings=settings,
        llm=llm,
        ui=ui or SilentUI(),
        searcher=searcher or FakeSearcher(candidates=[make_candidate(0)], text=PAPER_TEXT),
        prober=prober or FakeProber(),
        runner=ScriptRunner(
            settings.runtime,
            python_executable=sys.executable,
            sink_factory=lambda label, total=None: NullSink(),
        ),
        lesson_store=LessonStore(paths.lesson_dir, settings.memory),
        connectivity=connectivity or FakeConnectivity(),
    )


def result_dirs(settings: Settings) -> list[Path]:
    paths = settings.resolved_paths()
    return sorted(p for p in paths.result_dir.glob("*") if p.is_dir())
