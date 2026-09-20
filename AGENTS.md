# AGENTS.md

Working agreements for anyone (human or agent) changing this repository.

## Commands

```bash
python -m pip install -e ".[dev]"     # once
python -m pytest                      # full suite, offline, ~1 min
python -m pytest tests/test_graph_loops.py -q
python -m ruff check src tests        # lint
python -m ruff format src tests       # format (line-length 100, configured in pyproject)
python -m essay_agent --help
```

`make test`, `make lint`, `make fmt`, `make cov` wrap the same commands.

**Never make the test suite require the network or an API key.** Every test goes through
`tests/fakes.py` (`FakeLLM`, `FakeSearcher`, `FakeProber`, `FakeConnectivity`) and `tests/pipeline.py`
(`happy_responses()`, `build_deps()`, `script_with_estimate()`), which drive the real
LangGraph pipeline end to end.

## Layout

| Path | Responsibility |
| --- | --- |
| `src/essay_agent/cli.py` | Click entry points: `agent`, `run`, `config …`, `lesson …` |
| `src/essay_agent/agent.py` | `run_task`, `build_session`, Ctrl+C purge, `atexit` guard |
| `src/essay_agent/graph.py` | LangGraph wiring + routing helpers for the three loops |
| `src/essay_agent/state.py` | `RunState` (JSON-friendly TypedDict) |
| `src/essay_agent/config.py` | Stage 0: pydantic-settings config layer |
| `src/essay_agent/workspace.py` | Per-run staging dir, `publish()` / `purge()` |
| `src/essay_agent/connectivity.py` | Per-purpose network probe: endpoints, channels, verdicts, `HF_ENDPOINT` fallback |
| `src/essay_agent/nodes/` | One module per stage: fetch, plan, realize, execute, verify_execute, interpret |
| `src/essay_agent/prompts/` | System prompts per stage |
| `src/essay_agent/schemas/` | Pydantic contracts for papers, cards and dialogues |
| `src/essay_agent/tools/` | Paper search tool, dataset prober |
| `src/essay_agent/memory/` | `lesson_plan.txt` / `lesson_execute.txt` + LLM consolidation |
| `src/essay_agent/runtime/` | Progress protocol, pre-flight estimator, subprocess runner |
| `tests/` | Offline end-to-end and unit tests |

## Invariants (do not regress)

These are behavioural promises from the specification and from earlier bugs. Each one
has a test - keep it green.

1. **Ctrl+C wipes the task.** All run files live under `.essay_agent/runs/<run_id>/`;
   `TaskCancelled`/`KeyboardInterrupt` purges the workspace and publishes nothing.
   `RunWorkspace.purge()` refuses paths outside the workspace root.
2. **Nothing is published before `finalize`.** Lessons are staged in the run workspace
   (`RunLessons`) and merged only on success via `LessonStore.commit(..., llm=...)`.
3. **Search is tool-forced.** `fetch` calls `deps.llm.tool_args(...)` for
   `PaperSearchQuery`; the model never answers from memory. When the match is uncertain
   (`ambiguous=True` or `confidence < search.user_choice_confidence`) the human picks
   from a numbered list, and `0`/decline aborts.
4. **Exact abort strings.** No candidates -> the CLI prints `未匹配到对应论文` and the run
   aborts; the fetch status must never be `ok` in that case.
5. **No synthetic data.** `runtime.allow_synthetic_data` stays `false`; missing datasets
   are reported, never fabricated.
6. **Bounded loops, round 3 releases.** `max_plan_rounds` / `max_execute_rounds` (3 by
   default). On the last round the verifier's verdict is recorded, the warning
   `last permitted attempt` fires, and the pipeline continues instead of looping.
   `max_adjust_rounds` bounds the budget-shrinking loop the same way.
7. **`repro.py` contract.** `--preflight` prints
   `{"event": "estimate", "estimated_full_seconds": …, "params": {…}}` and returns fast;
   `--out <dir>` prints `{"event": "progress", "step": i, "total": n}` lines and writes
   `<dir>/metrics.json` plus `<dir>/figures/*.png`. `ScriptRunner` must be called with
   `args=["--preflight"]` or `args=["--out", code_dir]`.
8. **Pre-flight before the full run.** Over-budget estimates route to `adjust`, which
   rewrites the script with smaller parameters; the run only starts when the estimate
   fits (or the adjustment budget is exhausted).
9. **Config precedence.** Built-in defaults < packaged `essay_agent/defaults/default.yaml` < user YAML <
   environment/`.env`. This is why `Settings.settings_customise_sources` reorders the
   pydantic sources - do not "simplify" it back to the default.
10. **Lesson memory stays bounded.** Files above `memory.compress_threshold_chars` are
    consolidated by the LLM with the original archived under `lesson/archive/`.
11. **Prompts are never a trap.** `ConsoleUI.confirm` accepts y/yes/continue/ok/sure/1
    and n/no/stop/cancel/0, a bare ENTER keeps the default, and unrecognised answers are
    re-asked (`CONFIRM_ATTEMPTS`) instead of silently meaning "no". Continue-style
    prompts (document mismatch, abstract-only) default to *yes*, so ENTER carries on.
12. **The document-title check must not false-alarm.** `_first_title_like_line(text,
    expected_title)` skips camera-ready/arXiv/journal furniture and picks the candidate
    line that best matches the requested title; only a genuinely different first page
    should trigger the "does not obviously match" warning.
13. **Connectivity is probed per purpose, never globally.** `connectivity.py` probes the
    model API, the paper sources and the dataset sources separately, over the channels
    the pipeline actually uses (`direct`, the OS proxy, `HF_ENDPOINT` mirror); a blocked
    host must never be reported as "the network is unreachable". Any HTTP status counts
    as reachable, the verdicts land in `logs/connectivity.json` before the checks, and a
    working mirror is surfaced as `dataset_env` (`HF_ENDPOINT`) so generated code
    receives it. `nodes/base.py` no longer holds a singleton `network_reachable()`.
14. **The verdict must reach the real clients.** `ConnectivityProbe.run()` remembers the
    channel that answered each category and `provider(category)` hands it to
    `PaperSearcher` / `DatasetProber`; `pin_session()` pins them with `trust_env=False`
    so a stale `HTTP_PROXY` / `ALL_PROXY` / `NO_PROXY` (or a proxy living only in the OS
    settings) cannot make "probe says ok, download fails" happen again. Pinning keeps a
    corporate `REQUESTS_CA_BUNDLE`, and a category with no verified route stays
    untouched. `fetch_with_requests` passes a *copy* of the proxies dict because
    `requests` merges the environment into it in place.
    The model clients are pinned the same way: `build_http_client()` turns the
    verified model route into an `httpx.Client` (`trust_env=False`), `LLMClient`
    passes it to whichever provider as `http_client`, and `nodes/base.py` wires
    `probe.provider(settings, MODEL_API)` into `build_llm`. Never drop that wiring:
    `langchain-openai` injects a transport that *disables* httpx proxy
    auto-detection, so an unpinned client cannot use an OS-level proxy at all.
15. **The probe budget is spent fairly and never overshoots.** Endpoints are probed
    channel-major - the deciding channel for *every* endpoint before any fallback
    channel - the go/no-go endpoints (`Endpoint.priority`: model API, first paper
    backend, both dataset hosts) are visited first, each attempt is bounded by the time
    left, and only a *fast* failure (`RETRY_MAX_SECONDS`) is retried, because a hang is
    not a flake. A category reads `unreachable` only when nothing in it was cut short;
    otherwise `unknown`. `search.connectivity_budget_seconds` (default 20) tunes the
    total, and `run_checks(..., clock=...)` keeps the tests deterministic - never make
    the tests sleep.
16. **A flaky backend costs a retry, never the source.** `PaperSearcher` retries timeouts,
    connection errors and *fast* 429/5xx answers (`RETRY_ATTEMPTS`,
    `RETRY_BACKOFF_SECONDS`) but raises immediately after recording a hang, because a hang
    is not a flake. Backends that did not answer are kept in `backend_failures` /
    `backend_notes` (reset per `search()`), written to `paper/search_report.json` and
    surfaced by `fetch` (`ui.warn` naming them, `search_report` in the state), so a
    candidate list that lost a source can never look complete. Downloads must start with
    `%PDF-` (`_is_pdf`), and the PDF ladder is `candidate.pdf_url` -> arXiv by id ->
    arXiv by title; the link that worked is reported as `info["pdf_source_url"]`.
17. **State keys must be declared.** LangGraph keeps only the keys named in `RunState`
    (`src/essay_agent/state.py`); a node that returns an undeclared key loses it silently
    even though the workspace file it wrote survives. Declare the key in the same change.
18. **The verified route reaches the child process too.** `ConnectivityReport.dataset_env()`
    returns the dataset route as environment variables for the generated script: concrete
    proxy variables (upper *and* lower case) when the verified channel is a proxy, the
    mirror's `HF_ENDPOINT` when *only* the mirror answered, and ``None`` values when the
    verified route is a direct connection or when the primary dataset host answered - which
    `ScriptRunner._build_env` must treat as "remove this variable", because a dead inherited
    `HTTP_PROXY` sends the script nowhere and `httpx` cannot fall back to the Windows
    registry the way `requests` does, while an inherited `HF_ENDPOINT` is a route this run
    never probed. Nothing is reported when no dataset host answered, so an unverified run
    never rewrites the child's environment on a guess.
    `tests/experiment_child_proxy.py` measures this against a real child process;
    `tests/test_dataset_route_env.py` covers it offline.
19. **A reference that is not an id is not a gated dataset.** `classify_target` reads bare
    `owner/name` text as a Hub repository only when it really is one (`_HF_ID_RE`), the
    `/datasets/` section of a Hub URL is stripped by hand (`_hf_id_from_url`) so it can
    never be mistaken for the owner, and `probe_target` reports everything else as
    `unknown` ("resolve this reference to a real id") instead of asking the Hub for a path
    that cannot exist. The Hub answers the *same* 401 for a repository that does not exist
    and for a gated one, so a bare 401 must never be presented as a verified reason. See
    `tests/test_dataset_probe.py`.
20. **Coverage is per section, never a global quota.** `nodes/plan.py` cards every eligible
    section before it deepens any of them: a coverage pass (`CARDS_PER_SECTION`), one bounded
    retry pass for the sections that failed or answered nothing, then the leftover
    `CARD_BUDGET` on the richest sections up to `MAX_CARDS_PER_SECTION`. Nothing is ever
    dropped silently: `_CardMiner` gives every section a ledger row (`carded` /
    `no_testable_claim` / `failed` / `unanswered`) that is written to `cards/coverage.json`
    and `cards/coverage.md`, and a claim that repeats inside a section is not carded twice. A
    run with a `failed` section may continue but must say so (`coverage_ok` is false). The
    card verifier sees that ledger through `section_outline()` (limit 4000 - a truncated
    outline hides sections from the coverage check), and a section with neither a card nor a
    stated reason is a `major` issue. See `tests/test_card_coverage.py`.
21. **A section's identity is its own numbering.** `section_key()` reads `3.1` out of
    `3.1 U PSTREAM SCALING`, falls back to a normalised name, and a name that repeats far
    apart gets a `#2` suffix; only *adjacent* duplicates (a figure caption the splitter read as
    a heading) merge, keeping the longer body. See `tests/test_card_coverage.py`.
22. **The card verifier never sees the whole card set in one call.** One call over 24 cards
    returned 20,892 characters of JSON - 64% of the provider's hard output ceiling - so
    `_review_cards` sends `VERIFY_BATCH_SIZE` cards per call and merges: `dedupe_issues` keeps
    one issue per (card, field) with the most severe winning (a coverage gap names its section
    in the field, `coverage:3.1`), `cap_issues` trims a batch to `VERIFY_ISSUE_LIMIT` and the
    merge to `VERIFY_ISSUE_TOTAL` while stating what it dropped, and `merge_reviews` decides
    `ok` on the *uncapped* set, so trimming can never turn a blocker into an approval. A dead
    batch is retried once and then reported as a warning instead of blocking; only when every
    batch fails are the cards released. One batch keeps the label `plan_review:<round>`, several
    use `plan_review:<round>:<n>`. See `tests/test_verify_batching.py`.
23. **Stage 4 only touches the cards stage 3 allows.** `nodes/base.card_scope(state)` keeps the
    cards the feasibility verdict did not rule out (explicitly feasible, or never classified), in
    card order, capped at `EXEC_CARD_LIMIT` (10), and *names* the rest: `excluded_blocked` (ruled
    out by the feasibility check) and `excluded_budget` (the cap). `CardScope.context()` is the
    `## Cards out of scope` section carried by the plan, code, adjust, re-execute and verification
    prompts, `CardScope.summary()` lands under `state["scope"]`, and the run says the exclusion out
    loud (`ui.dim(scope.describe())`). The execution verifier grades only the in-scope cards and
    must never raise a problem asking for an excluded card to be attempted; a scope that would come
    out empty (stage 3 flagged everything, or the user continued against it) is relaxed to a best
    effort instead of leaving stage 4 with nothing to run. `result/cards.md` still publishes every
    card. Do not go back to `cards_of(state)` in stage 4/4.1 for the prompts - only the published
    card list may use it. See `tests/test_execute_scope.py`.
24. **The verifier only grades what the run handed in.** `nodes/base.unhanded_cards(state, cards)`
    is the in-scope set minus the plan's `exec_plan.cards_covered` - cards the run produced no
    evidence for - and `unhanded_context()` renders it as `## Cards with no submitted evidence`:
    `untested`, never `problems`, never a reason to fail or re-run. The set lands in
    `verdict["unhanded"]`. A plan that does not state its coverage leaves every card gradeable, so
    this can only ever shrink the graded set: it must never widen the scope and never hide a card
    the run did attempt (that stays a real defect, see `VERIFY_SYSTEM` rules 2-3). See
    `tests/test_unhanded_cards.py`.
25. **`problems` is the action channel; only serious defects go in it.** `ExecuteVerdict` carries
    `observations` for everything the verifier noticed that is *not* a defect (run-to-run noise, a
    margin inside the measurement's own resolution, the deliberately small scale of the run,
    naming). Only `problems` reaches the re-execution prompt and the lesson memory; observations
    are recorded in `code/verdict_roundN.json`, shown with `ui.dim`, and included in the report's
    verifier history (`interpret._history_line`). A criterion that can only be measured within the
    run's own noise is inconclusive: it may make the verdict `pending`, never `unsuccessful`. See
    `tests/test_verifier_severity.py`.

26. **Every generated script runs in a known-good environment.** `runtime/runner.py`
    `CHILD_ENV_DEFAULTS` is injected into every child (unbuffered, utf-8, `MPLBACKEND=Agg`, no
    bytecode, `KMP_DUPLICATE_LIB_OK=TRUE`). The last one is not plumbing: another library can lazily
    load a second `libiomp5md.dll`, at which point Intel's OpenMP runtime aborts the whole process
    (`OMP: Error #15`) - measured on run-20260920-062827-3e6074, where all three rounds died at step
    0 with no metrics and the *same* script passed once the flag was set. Because Intel documents
    the flag as unsafe ("may cause crashes or silently produce incorrect results"), it must never be
    silent: `ScriptRunner.env_fixes` names it with the reason, and `_outcome_payload` records it in
    `code/run_outcome.json` and `code/preflight.json`. Our defaults win over an inherited shell
    value; `env={name: None}` takes one back. See `tests/test_runner_env.py`.
27. **A failed run carries its cause.** `nodes/execute.log_tail(result)` hands stage 4.1 both
    streams, labelled `[stdout]` / `[stderr]` and bounded, and the `run` node prints the child's
    last error lines with `ui.dim`. A run with no stderr keeps the previous prompt shape. Never let
    the verifier receive a log tail it cannot diagnose from: a native abort prints only on stderr,
    and all three rounds of run-20260920-062827-3e6074 were spent asking for a diagnosis the prompt
    made impossible. See `tests/test_stderr_visibility.py`.

## Style

- English docstrings, comments and identifiers; the CLI surfaces Chinese strings only
  where the specification demands them.
- `from __future__ import annotations` at the top of every module; full type hints.
- Nodes are pure with respect to the graph: they take `RunState`, return a partial
  state `dict`, and get everything else from the injected `Deps` object. Add new
  collaborators to `Deps` rather than importing globals.
- Keep the UI abstraction: nodes only ever touch `deps.ui` (`info`, `warn`, `error`,
  `success`, `step`, `dim`, `notice`, `replan_notice`, `verdict_notice`,
  `choose_candidate`, `confirm`, `show_cards`, `show_estimate`, `progress`).
  `SilentUI` must stay in sync with `ConsoleUI`; `tests/test_cli.py` and the pipeline
  tests fail loudly if a method is missing.
- New behaviour needs a test; new user-visible behaviour needs a README/`docs/`
  update in the same change.

## Conventions for generated code

The model writes `repro.py`, never us - but our prompts must keep demanding: no markdown
fences, no placeholders, deterministic seeds where possible, real datasets only, and the
JSON progress protocol above.

Stage 4 must keep the plan and the code apart: the plan is a small `ExecPlan` JSON document, the
script is asked for as **plain text** (`prompts.code_user_message`) with the output budget. The
code call is static-checked and repaired once; the single-call `ReproScript` path is a fallback
only, and a script that fails `check_code` must never be returned or written.
