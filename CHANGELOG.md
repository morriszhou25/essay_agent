# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet. Known gaps are listed under "Known limitations" in the README (the stage-3 confirmation has no
non-interactive escape hatch, the card replan loop does not provably converge, and the repo still
needs a LICENSE).

## [0.1.0] - 2026-09-20

First release: a five-stage, LangGraph-orchestrated paper-reproduction agent.

### Added

- **Stage 1 - retrieval.** Tool-forced paper search (arXiv / Semantic Scholar / OpenAlex) with
  retry-once backends, PDF `%PDF-` validation, arXiv fallback links, and a numbered candidate list
  when the model is not confident. No candidates prints `未匹配到对应论文` and stops.
- **Stage 2 - cards.** Section-paragraph cards (`identity`, `claim`, `assumption`, `scope`,
  `expected_outcome`, `format`, `success_criteria`) mined per section with a per-section quota and a
  coverage ledger (`cards/coverage.json`) that accounts for every eligible section.
- **Stage 2.1 - plan verifier.** Batched review (8 cards per call, capped and de-duplicated) plus a
  bounded replan loop in which the main model argues back instead of silently accepting patches.
- **Stage 3 - feasibility.** Per-card dataset/dependency/hardware checks against probe evidence,
  with a hard ban on synthetic data and licence-aware blockers.
- **Stage 4 - execution.** Plan-then-code generation, static checks with one repair pass, a
  `--preflight` timing estimate that shrinks the experiment to fit the budget, a live progress bar
  with ETA, and a `repro.py` contract (`--out`, `metrics.json`, `figures/*.png`, JSON progress
  events).
- **Stage 4.1 - verification.** `successful` / `pending` / `unsuccessful` with a severity split:
  only actionable defects go to the re-execution loop, observations are reported but never acted on,
  and out-of-scope or unsubmitted cards are never graded.
- **Stage 5 - interpretation.** An LLM-written `report.md` with figures, deviations, limitations and
  provenance, published to `result/<slug>-<id>/` next to `repro/<slug>-<id>/`.
- **Cross-cutting.** Per-purpose connectivity probing with proxy/mirror pinning, a known-good child
  environment recorded in every run outcome, dual-stream (stdout + stderr) failure visibility,
  Ctrl+C purge of the whole staging directory, and bounded lesson memory with LLM consolidation.
- **Configuration.** pydantic-settings layering (built-in defaults < packaged `default.yaml` < user
  YAML < environment/`.env`), four provider modes plus any OpenAI-compatible endpoint.
- **Tests.** 365 offline tests, including one end-to-end drive of the real LangGraph pipeline and
  17 executable experiment scripts documenting why each rule exists.