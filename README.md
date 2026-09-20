# essay-agent

> A lightweight paper-reproduction agent. Give it a paper title, URL, DOI or arXiv id and it
> fetches the full text, splits the paper into sections, writes a reproduction card per section,
> checks whether this machine can actually run the experiments, runs one lightweight experiment,
> and reports the result with figures.

A lightweight, human-in-the-loop paper-reproduction agent orchestrated with
[LangGraph](https://github.com/langchain-ai/langgraph). No synthetic data, no silent failures, no
fabricated results.

![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![tests](https://img.shields.io/badge/tests-365%20passing%2C%20offline-brightgreen)
![lint](https://img.shields.io/badge/lint-ruff-261230)
![pipeline](https://img.shields.io/badge/pipeline-LangGraph-1c3d5a)

<!--
After pushing to GitHub, replace OWNER/REPO below and remove the comment markers to show CI status:
[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)
-->

## Table of contents

- [What it does](#what-it-does)
- [Highlights](#highlights)
- [The five stages](#the-five-stages)
- [Install](#install)
- [Quickstart](#quickstart)
- [Commands](#commands)
- [Configuration](#configuration)
- [Design guarantees](#design-guarantees)
- [Run artifacts](#run-artifacts)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## What it does

One line of input is enough to start. The agent finds and downloads the paper, splits it by
section, writes a **reproduction card** for each section (claim, assumption, scope, expected
outcome, success criteria), works out whether *this* machine can do the experiment at all
(datasets, dependencies, hardware, licences), writes a lightweight script that actually tests what
it can, judges the cards against the measured numbers, and publishes the conclusion plus figures,
deviations and limitations to `result/<slug>-<id>/report.md`.

It does not chase the paper's full scale, and it does not decide for you: what it can test, it
tests; what it cannot, it says explicitly. It never fabricates results or invents data to make a
run look green.

## Highlights

- **Every section counts.** Cards are allocated per section, and a coverage ledger in
  `cards/coverage.json` accounts for each eligible section - no more "the quota ran out, the rest of
  the paper was dropped".
- **Three bounded feedback loops.** Card replanning, budget adjustment and result re-execution each
  get at most three rounds; round 3 releases and carries the open issues into the report. Every
  round is visible on the command line (`REPLAN`, `re-execute`, verdicts).
- **Only feasible cards are reproduced.** The stage-3 verdict is the single input to stages 4 and
  4.1: excluded cards are named individually, reported as `untested`, and never counted as failures.
- **Real reachability probing.** The model API, the paper sources and the dataset sources are probed
  separately, and the verified route is pinned onto the real clients - one blocked host is never
  reported as "the network is unreachable".
- **Generated scripts run in a known-good environment.** A fixed set of defaults is injected into
  every child process, and the run record states what was injected and why (see below).
- **Failures are visible.** The verifier receives both streams, labelled `[stdout]` / `[stderr]`,
  and a failed run prints the child's last error line on the console.
- **Fully offline test suite.** 365 tests need neither network nor an API key, and the whole
  pipeline (including all three loops) can be driven end to end with fakes.

## The five stages

| Stage | Nodes | Artifacts |
| --- | --- | --- |
| 1 Retrieval | `fetch` | Full text + `paper/paper.json` (the search tool is **forced**; no answering from memory) |
| 2 Splitting | `plan` → `verify_plan` → `revise_plan` | One `##reproduction card##` per section (JSON) |
| 3 Feasibility | `realize` | Dataset/dependency reachability checks (**synthetic data is forbidden**) |
| 4 Execution | `execute_plan` → `preflight` → `adjust` → `run_full` | `repro.py` + metrics + figures + live progress bar with ETA |
| 4.1 Verification | `verify_execute` → `reexecute` | `successful` / `pending` / `unsuccessful` |
| 5 Interpretation | `interpret` → `finalize` | `result/<slug>-<id>/report.md` (with figures) |

## Install

Python >= 3.11 is required.

```bash
git clone https://github.com/morriszhou25/essay_agent.git
```

Optional provider extras:

```bash
python -m pip install -e ".[deepseek]"    # DeepSeek
python -m pip install -e ".[anthropic]"   # Anthropic
```

## Quickstart

```bash
cp .env.example .env             # then put your ESSAY_AGENT_LLM__API_KEY and choose the provider here
essay config check               # validate configuration and paths; calls no model
essay agent                      # start the interactive session (the main entry point)
```

In the interactive session, type a paper title, URL, DOI or arXiv id. `/help` lists the commands and
`/exit` quits.

One-shot runs:

```bash
essay run "Attention Is All You Need"
essay run https://arxiv.org/abs/1706.03762
essay run 1706.03762
```

## Commands

| Command | What it does |
| --- | --- |
| `essay agent` | Interactive session (the main entry point) |
| `essay run "<request>"` | One-shot run. Exit codes: `0` success / `1` unsuccessful / `2` config or runtime error / `130` cancelled with Ctrl+C |
| `essay config show` | Print the effective configuration (API keys are redacted) |
| `essay config check` | Validate configuration and paths without calling a model |
| `essay config init` | Write an `essay-agent.yaml` template into the current directory |
| `essay lesson show` | Show the size of the long-term memory and its most recent entries |
| `essay lesson consolidate` | Compress the long-term memory with the LLM (also runs at the end of every project) |
| `essay lesson clear` | Archive and empty the long-term memory |

## Configuration

Precedence, later wins:

```text
built-in defaults  <  packaged src/essay_agent/defaults/default.yaml  <  user YAML (--config or ./essay-agent.yaml)  <  environment / .env
```

Environment variables use the `ESSAY_AGENT_` prefix and a double underscore for nested keys, e.g.
`ESSAY_AGENT_LLM__MODEL=gpt-4o`.

### Common environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `ESSAY_AGENT_LLM__PROVIDER` | `openai` | `openai` / `openai_compatible` / `deepseek` / `anthropic` |
| `ESSAY_AGENT_LLM__MODEL` | `gpt-4o-mini` | Main model |
| `ESSAY_AGENT_LLM__VERIFIER_MODEL` | same as the main model | Optional dedicated verifier model |
| `ESSAY_AGENT_LLM__API_KEY` | - | API key (provider-native variables such as `OPENAI_API_KEY` also work) |
| `ESSAY_AGENT_LLM__CODE_MAX_TOKENS` | same as `MAX_TOKENS` | Output ceiling for the stage-4 `repro.py` call only; raise it when the provider allows more (e.g. `32768`) |
| `ESSAY_AGENT_HTTP_PROXY` | - | One proxy for every HTTP client this agent builds (highest precedence, overrides environment and OS settings) |
| `ESSAY_AGENT_RUNTIME__TIME_BUDGET_SECONDS` | `600` | Wall-clock budget for the full experiment |
| `ESSAY_AGENT_RUNTIME__MAX_PLAN_ROUNDS` | `3` | Card replan round limit |
| `ESSAY_AGENT_RUNTIME__MAX_EXECUTE_ROUNDS` | `3` | Result re-execution round limit |
| `ESSAY_AGENT_RUNTIME__DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `ESSAY_AGENT_SEARCH__USER_CHOICE_CONFIDENCE` | `0.6` | Below this confidence the agent asks the human to disambiguate |
| `ESSAY_AGENT_SEARCH__CONTACT_EMAIL` | - | Contact address for the OpenAlex polite pool |
| `ESSAY_AGENT_SEARCH__DATASET_MIRROR_URL` | `https://hf-mirror.com` | Dataset mirror; set it to an empty value to disable the fallback |
| `ESSAY_AGENT_PATHS__HOME` | `.` | Base directory for runs, results and lesson memory |

### Using other providers

`provider` has four values, but `openai_compatible` plus `base_url` covers most services:

| Service | Configuration |
| --- | --- |
| DeepSeek | `PROVIDER=deepseek`, `MODEL=deepseek-chat`, needs `pip install -e ".[deepseek]"` |
| Anthropic | `PROVIDER=anthropic`, `MODEL=claude-3-5-sonnet-latest`, needs `.[anthropic]` |
| Local vLLM / Ollama | `PROVIDER=openai_compatible`, `BASE_URL=http://localhost:8000/v1` (Ollama: `http://localhost:11434/v1`) |
| Qwen / DashScope | `PROVIDER=openai_compatible`, `BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1` |
| Moonshot / OpenRouter | `PROVIDER=openai_compatible` plus the service's `BASE_URL` |
| Google Gemini | `PROVIDER=openai_compatible`, `BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/` |
| Azure OpenAI | `PROVIDER=openai_compatible`, `BASE_URL=https://<resource>.openai.azure.com/openai/deployments/<deployment>/?api-version=2024-10-21` |

The model name must be one the service actually serves; a missing provider dependency fails at
startup with the exact install command to run.

## Design guarantees

Every item below is a behavioural promise backed by a test. The full list (27 invariants) lives in
[`AGENTS.md`](AGENTS.md) and the implementation notes are in
[`docs/architecture.md`](docs/architecture.md).

### Retrieval and fetching

- **No match, no task.** When the search tool returns no candidates the CLI prints
  `未匹配到对应论文` ("no matching paper found") and aborts without producing any result files.
- **A flaky backend costs one retry, never the paper.** Timeouts, connection errors and fast
  429/5xx responses are retried (3 attempts, 0.5s backoff by default); a *hang* is not retried
  (recorded, then raised immediately). Only all backends failing counts as search failure. Backends
  that stayed silent are recorded in `paper/search_report.json` and warned about on the console, so
  a partial candidate list is never mistaken for a complete one. A downloaded PDF must start with
  `%PDF-` (an HTML error page is no longer accepted as a paper), and a dead primary link falls back
  to the arXiv id/title links with an explicit `full text came from a fallback link` message.
- **When unsure, ask.** If the model marks the match `ambiguous=True` or scores it below
  `search.user_choice_confidence` (0.6 by default), the candidates are listed as a numbered menu for
  the human to choose from; choosing 0 gives up.
- **Confirmation prompts must not misfire.** A downloaded PDF that clearly does not match the
  requested title warns first and then asks. `y/yes/continue/ok/sure/1` accept, `n/no/stop/cancel/0`
  decline, **a bare ENTER keeps the default** (continue-style questions continue on ENTER), and an
  unrecognised answer is asked again instead of being silently treated as "no".

### Cards and verification

- **Every section counts.** The card quota is handed out **per section**, not as one global pool.
  Every eligible section gets one card first, a section that failed or answered with nothing is
  retried once, and leftover quota goes to the densest sections up to three cards each
  (`CARD_BUDGET` is the overall ceiling and never decides who goes first). Every section ends up
  with a ledger row in `cards/coverage.json` and `cards/coverage.md`: carded, explicitly declared
  "no testable claim", or a written reason for the failure. Duplicated adjacent headings (a figure
  caption mistaken for a heading) are merged, card ids map one-to-one onto section numbers, and
  `identity.section` uses the paper's own numbering (`3.1`, `4.2`, ...).
- **The card verifier reads in batches.** Cards are reviewed 8 at a time and merged: a repeated
  (card, field) complaint keeps only the most severe wording (coverage complaints are keyed as
  `coverage:3.1`, so two different gaps are never merged into one), each batch is capped at 12
  issues and the merge at 20, and `ok` is decided on the **untruncated** merge - truncation can
  never turn a blocker into a pass. A failed batch is retried once and then recorded as a warning
  instead of blocking.
- **Only feasible cards are reproduced.** The stage-3 verdict is the single input to stages 4 and
  4.1: a card ruled out there is never handed to the code writer and never graded (`card_scope`;
  `EXEC_CARD_LIMIT` = 10 cards is the ceiling for one script). Excluded cards are named individually
  on the console, in the prompts and in the report, marked "out of scope, report as untested, not a
  failure".
- **Nothing handed in is not graded.** The execution verifier only grades cards the run actually
  submitted. Cards that were in scope but not covered by the plan, and therefore have no measured
  evidence, are listed in a separate block and must be marked `untested`, never written into
  `problems`, and never used to fail the run. The list is recorded as `unhanded` in
  `code/verdict_roundN.json`.
- **Only serious problems are raised.** `problems` carries only defects that change a card's
  conclusion *and* that the executor can fix. Everything else - noise, a margin inside the
  measurement's own resolution, "this experiment is small" (our own budget decision), naming - goes
  to `observations`, which is recorded and shown but never drives a re-run. **Noise is not a
  contradiction**: a few percent on a millisecond-scale benchmark, or a gap smaller than the spread
  visible in the metrics, is `inconclusive` and may never make a verdict `unsuccessful`.
- **Bounded feedback loops.** Card replanning and result re-execution get at most three rounds each.
  Round 3 releases, and the remaining issues are written to the lesson memory and carried
  downstream.

### Execution and environment

- **Budget first.** A `--preflight` run estimates the cost of the full experiment; if the projection
  busts `runtime.time_budget_seconds` (600s by default) the experiment is scaled down (steps,
  learning rate, dataset size) before the full run starts. The full run shows a progress bar with a
  live ETA.
- **Stage 4 generates the script in two steps.** First only the plan (a small `ExecPlan` JSON), then
  `repro.py` as **plain text**. The script never travels inside a JSON string, so a `\d` in a regex
  cannot destroy the reply, and a truncated envelope cannot take the plan down with it. The script
  has an output budget, a failed static check is sent back once with the exact error, and only if
  both steps fail does it fall back to the single-call path - which must pass the same static checks.
- **No synthetic data.** `runtime.allow_synthetic_data` is a hard switch (default `false`). A missing
  dataset is reported, never fabricated.
- **Generated scripts run in a known-good environment.** `repro.py` is a separate process, so besides
  the inherited environment it is given a fixed set of defaults (`PYTHONUNBUFFERED`,
  `PYTHONIOENCODING=utf-8`, `MPLBACKEND=Agg`, no bytecode files, and `KMP_DUPLICATE_LIB_OK=TRUE`).
  That last one is not incidental: another library can lazily load a second copy of Intel's OpenMP
  runtime (`libiomp5md.dll`), at which point the runtime kills the whole process rather than risk
  wrong numbers (`OMP: Error #15`). Because Intel documents the flag as unsafe ("may cause crashes or
  silently produce incorrect results") it is **never silent**: `ScriptRunner.env_fixes` states what
  was injected and why, and `code/preflight.json` and `code/run_outcome.json` record the environment
  the run actually had under `env_fixes`.
- **A failed run carries its cause.** The log tail handed to the execution verifier contains both
  streams, labelled `[stdout]` / `[stderr]` and bounded, and a failed run prints the child's last
  error lines on the console. A native abort (a duplicate runtime, a missing DLL, a segfault) prints
  only on stderr, which is exactly why the verifier must receive it.
- **Proxies and mirrors are not hijacked by the environment.** The verified channel is pinned
  explicitly onto the real clients (both the download clients and the model client, with
  `trust_env=False`), so a stale `HTTP_PROXY` / `NO_PROXY`, or a proxy that only exists in the OS
  settings, can no longer produce "the probe said ok, the call failed". **The same applies to the
  generated script**: when a proxy is needed, `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` are written into
  its environment; on a direct route they are *removed* instead, so a dead proxy cannot be inherited.
  Mirrors work the same way: `HF_ENDPOINT` is set only when the mirror is the channel that answered,
  and an inherited `HF_ENDPOINT` is removed when the primary host answered. Details land in
  `logs/connectivity.json` under `dataset_env` (a `null` value means "delete this variable").
- **The probe budget is fair and bounded.** Probing is channel-major: every endpoint is asked on its
  deciding channel before any fallback channel, and the go/no-go endpoints (model API, first paper
  backend, both dataset hosts) come first. The whole probe finishes within 20 seconds by default
  (`search.connectivity_budget_seconds`); running out of budget only turns a verdict into `unknown`,
  never into a false "unreachable".

### Cancellation and cleanup

- **Ctrl+C wipes the task.** All run files live under `.essay_agent/runs/<run_id>/`; Ctrl+C deletes
  the whole directory and publishes nothing (an `atexit` guard covers abnormal exits). The search
  cache in `.essay_agent/cache/` and the history file are untouched.
- **Nothing is published before `finalize`.** Lesson memory is staged inside the run directory and
  merged into `lesson/*.txt` only on success; an unfinished run leaves just the staging directory for
  inspection.
- **Lesson memory stays bounded.** `lesson/lesson_plan.txt` and `lesson/lesson_execute.txt` are
  compressed by the LLM once they exceed `memory.compress_threshold_chars`, with the originals
  archived under `lesson/archive/`.

## Run artifacts

```text
.essay_agent/runs/<run_id>/          # staging directory (deleted on Ctrl+C or failure)
  paper/    search_args.json, candidates.json, paper.json, full_text.md, sections_index.json
  cards/    cards.json, cards.md, coverage.json, coverage.md, feasibility.json
  code/     repro.py, plan_*.json, preflight.json, run_outcome.json, verdict_round*.json,
            metrics.json, figures/*.png
  lesson/   lesson_plan.txt, lesson_execute.txt          (staged only)
  result/   report.md, cards.md, cards.json, verdicts.json, metrics.json, figures/*.png
  logs/     run.log, environment.txt, connectivity.json, llm_transcript.md (--verbose only)
```

On success the run is published to `result/<slug>-<id>/` (report, metrics, figures, cards),
`repro/<slug>-<id>/` (the `repro.py` and its run record) and `lesson/*.txt` (long-term memory). The
staging directory is kept for inspection.

## Known limitations

- **Lightweight only.** The goal is a run that fits the budget and agrees with the paper's
  direction, not a reproduction at the paper's scale.
- arXiv / Semantic Scholar / OpenAlex must be reachable. When the full text cannot be fetched, the
  user is asked whether to continue from the abstract alone (`search.require_full_text` defaults to
  `true`).
- The generated `repro.py` can only use packages already installed on this machine; missing
  dependencies are reported during the feasibility stage.
- A script that passes the static checks can still contain a runtime bug. The stage-4.1 verifier plus
  the re-execution loop is the safety net, and `repro/<run>/repro.py` is there to reproduce the
  failure by hand.
- The dataset source (HuggingFace) falls back to a mirror when the primary host is unreachable; if
  the mirror is unreachable too, the run says so and never falls back to synthetic data.
- **The stage-3 "continue with a reduced scope?" question defaults to *stop* when there is no
  terminal** (only `y` continues). In a pipe, in CI, or when called by another program, `essay run`
  can therefore abort at stage 3 with exit code 1 even though usable cards remain. Confirm
  interactively, or wait for a future explicit flag.
- **The card replan loop does not provably converge.** In live runs the same paper produced 20 issues
  in each of the three rounds with `ok` always false, and round 3 released as designed. The loop
  costs a fixed two rewrites plus three verifier calls for limited benefit.
- Generated scripts may write more figures than the plan promised (one measured run promised 8 and
  produced 23; all are published, 15 are referenced by the report).
- The summary text in `cards/feasibility.json` can disagree with the numbers in the same file's
  `checks`; **`checks` is authoritative** (both the scope and the blocker list are computed from it).
- Two runs over the same paper can produce different card sets (LLM non-determinism); keep that in
  mind when comparing runs.

## Troubleshooting

**A search backend returns 429.** Semantic Scholar and OpenAlex rate-limit their free tiers. The CLI
warns that the candidate list may be incomplete, arXiv carries the search, and this is never treated
as a search failure.

**How do I configure a proxy?** Prefer `ESSAY_AGENT_HTTP_PROXY`, which overrides both the environment
and the OS settings. To diagnose the network outside the agent:

```bash
python scripts/net_probe.py                        # per-address DNS/TCP/TLS probing
python scripts/connectivity_report.py --selftest   # offline self-check
python scripts/connectivity_report.py              # real network report
```

**`langchain-openai injected a custom httpx transport ...` can be ignored.** It is a one-off notice
that langchain's default transport disables httpx proxy auto-detection. This agent already passes the
verified channel to the model client through `http_client`, so the message only means "the proxy is
ours now".

**The script was killed and the log says nothing.** It does now: stderr reaches the verification
prompt, is printed on the console, and is published with `repro/` in `code/run_outcome.json`. If what
you hit was a duplicate OpenMP runtime (`OMP: Error #15`), the compatibility flag is already part of
the child environment - see `env_fixes` for the record.

**I do not want that OpenMP flag.** It is one of the child-environment defaults in
`runtime/runner.py` (`CHILD_ENV_DEFAULTS`); call
`ScriptRunner.run(..., env={"KMP_DUPLICATE_LIB_OK": None})` to take that single variable back.

## Development

```bash
make install-dev     # pip install -e ".[dev]"
make test            # pytest (fully offline; no network, no API key)
make lint            # ruff check src tests
make fmt             # ruff check --fix && ruff format
make cov             # coverage
```

Or directly:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src tests
python -m ruff format --check src tests
```

The tests drive the pipeline through dependency injection with `FakeLLM`, `FakeSearcher`,
`FakeProber`, `FakeRunner` and `FakeConnectivity`, so the entire graph - including all three feedback
loops - runs end to end without a network. The working agreement is **test first, then change
`src/`**: a behaviour fix starts as an executable `tests/experiment_*.py` script (deliberately not
collected by pytest) that states the measured problem and the intended rule, then the source changes,
then a collected `tests/test_*.py` locks the behaviour in.

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs lint, format checks, the test suite
and a CLI smoke test on Python 3.11, 3.12 and 3.13, and builds a wheel in a separate job to confirm
that `essay_agent/defaults/default.yaml` really ships inside the package.

## Contributing

Issues and pull requests are welcome. Before opening a PR, read
[`CONTRIBUTING.md`](CONTRIBUTING.md) (workflow, PR checks, and what a useful bug report contains) and
[`AGENTS.md`](AGENTS.md) (the 27 invariants, directory responsibilities and code style). Two hard
requirements:

- **New behaviour needs a test**, and user-visible behaviour needs a README or `docs/` update in the
  same change.
- **The test suite must never need the network or an API key.**

Version history lives in [`CHANGELOG.md`](CHANGELOG.md).

## License

This repository **does not have an open-source licence yet** (there is no `LICENSE` file), so all
rights are reserved until the author picks one. Before publishing, add a `LICENSE` file (MIT and
Apache-2.0 are the usual choices) and state it here.
