# Architecture

## Graph

`src/essay_agent/graph.py` compiles one `StateGraph(RunState)`. Rectangles are nodes,
`(…)` are conditional edges, `↺` marks the bounded feedback loops.

```
START ─► fetch ─(fetch_status == ok?)─► plan ─(status healthy?)─► verify_plan
              │                             ▲                        │
              └─► abort ─► END              └── revise_plan ◄────(ok? / rounds left)
                                                                    │ released
                                                                    ▼
                          realize ─(running?)─► execute_plan ─► preflight
                             │                                   │      ▲
                             └─► abort ─► END        (fits budget?)└─► adjust ↺
                                                                    │ no
                                                                    ▼
                                     run_full ─► verify_execute ─(verdict/rounds)─► interpret
                                                        │                            │
                                                        └─► reexecute ↺ ─► preflight  ▼
                                                                                  finalize ─► END
```

Routing helpers (`route_after`, `route_verify_plan`, `route_preflight`,
`make_verdict_router`) live next to the wiring so they can be unit tested. Any node that
leaves `state["status"]` as `failed`/`aborted` is routed to `abort`, which ends the graph
without publishing.

## Stages

| Stage | Node(s) | Reads | Writes |
| --- | --- | --- | --- |
| 1 fetch | `fetch` | `query` | `search_args`, `candidates`, `search_report`, `chosen_candidate`, `paper`, `paper_text`, `sections`, `fetch_status` |
| 2 plan | `plan` | `paper_text`, `sections` | `cards`, `coverage`, `plan_round` |
| 2.1 replan | `verify_plan`, `revise_plan` | `cards` | `plan_review`, `plan_round`, `plan_carryover`, staged `lesson_plan.txt` |
| 3 realize | `realize` | `cards[].scope.dataset`, `cards[].needs` | `feasibility`, `blockers` |
| 4 execute | `execute_plan`, `preflight`, `adjust` | `cards`, `feasibility` | `scope`, `exec_plan`, `exec_code_path`, `preflight`, `adjust_round` |
| 4 run | `run_full` | `exec_code_path` | `exec_result` (`metrics`, `figures`, `stdout_tail`, …) |
| 4.1 verify | `verify_execute`, `reexecute` | `exec_result`, `cards` | `verdict`, `verdict_history`, `exec_round`, staged `lesson_execute.txt` |
| 5 interpret | `interpret`, `finalize` | everything above | `report_path`, `publish`, `lesson_report`, `status=completed` |

Every LLM call goes through `essay_agent.llm.LLM`, which exposes `text`, `json(system, user,
schema)` and `tool_args(system, user, tool, schema)`. Calls are labelled (`paper_match`,
`plan_review:2`, `plan_review:2:3` for a batch of a multi-batch review, `exec_verdict:1`,
`repro_adjust1`, `repro_reexec2`, …); the labels are what
the offline `FakeLLM` keys on, so keep them stable. A review that fits in one batch keeps the
historical `plan_review:<round>` label.

When a reply cannot be used, the client raises `LLMReplyError` rather than a bare `LLMError`:
it carries the call identity (`role`, `label`, `schema`), a `reason` (`empty` / `truncated` /
`invalid_json` / `schema_mismatch` / `call_failed`), the provider's `finish_reason`
(`length` = cut off by `max_tokens`), the reply size and a head+tail `preview`. Failures are
therefore locatable from the message alone; use `--verbose` when the full transcript is needed.

Stage 4 asks for its two artifacts in two steps: `llm.json(..., ExecPlan)` for the plan and
`llm.text(...)` for the `repro.py` source. Keeping the code out of the JSON envelope is what makes
the call reliable - no backslash escaping, and a truncated envelope can no longer take the plan
down with it. The text reply is static-checked (`check_code`) and sent back once with the exact
error; a reply that looks cut off is told to be shorter rather than repeated. Only when both steps
fail does the node fall back to the single `ReproScript` call, and that code is checked too, so a
script that does not compile never reaches the runner.

Stage 2 mines the cards one section at a time and treats coverage as a property of each
section, not of the run: a coverage pass gives every eligible section one card, a single retry
pass re-asks the sections that failed or answered nothing, and only then is the leftover budget
(`CARD_BUDGET`) spent on the richest sections, up to `MAX_CARDS_PER_SECTION`. `_CardMiner`
records the outcome for every section in `cards/coverage.json` (`carded` /
`no_testable_claim` / `failed` / `unanswered`) and a repeated claim is never carded twice.
The ledger is the input to `section_outline()`, which renders it next to the section list for
the card verifier - so a section with neither a card nor a stated reason is a visible gap, and
the outline budget (4000 chars) is sized so that view is not silently truncated.

The card verifier is called per batch too (`VERIFY_BATCH_SIZE` cards), because one call over
24 cards already returned 20,892 characters of JSON - 64% of the provider's hard output
ceiling. `_review_cards` merges the batches: `dedupe_issues` keeps one issue per (card, field)
with the most severe winning (coverage gaps name their section in the field, `coverage:3.1`),
`cap_issues` trims each batch to `VERIFY_ISSUE_LIMIT` and the merged list to
`VERIFY_ISSUE_TOTAL` while saying how many it dropped, and `merge_reviews` decides `ok` on the
*uncapped* set so trimming can never approve a blocker. A batch that fails is retried once and
then reported (`ui.warn`) without blocking the loop; only when every batch fails are the cards
released, as before.

Stages 4 and 4.1 share one scope decision. `card_scope(state)` keeps the cards the stage-3
feasibility verdict did not rule out (`feasible=true`, or never classified) in card order, caps
them at `EXEC_CARD_LIMIT` (10), and *names* the rest: `excluded_blocked` (ruled out by the
feasibility check) and `excluded_budget` (the cap). `CardScope.context()` renders them as an
explicit `## Cards out of scope` section, `CardScope.summary()` is stored under `state["scope"]`,
and the node says it out loud. The plan prompt, the code call, the adjust/re-execute prompts, the
execution verifier and the report prompt all read that one object, so the verifier can no longer
grade - or demand a re-run for - a card the feasibility check just ruled out. A scope that would
come out empty (stage 3 flagged everything, or the user chose to continue against it) is relaxed to
a best effort instead of leaving stage 4 with nothing to run, and `result/cards.md` still publishes
every card.

The verifier also only grades what the run actually handed in. `unhanded_cards(state, cards)` is
the in-scope set minus the plan's `cards_covered`, and `unhanded_context()` renders it as a
`## Cards with no submitted evidence` section: those cards are `untested`, never `problems`, never a
reason to fail the run. The set is recorded as `verdict["unhanded"]`. A plan that does not state its
coverage leaves every card gradeable, so this can only shrink the graded set - never widen the
scope, and never hide a card that the run did attempt.

Severity is the verifier's other job. `ExecuteVerdict.problems` is the action channel and holds
only defects that change a card's conclusion and that the executor can fix; everything else the
verifier noticed - run-to-run noise, a margin inside the measurement's own resolution, the
deliberately small scale of the run, naming - goes to `observations`. Observations are recorded in
`code/verdict_roundN.json`, shown with `ui.dim`, folded into the report's verifier history
(`interpret._history_line`), and deliberately kept out of the re-execution prompt and out of the
lesson memory, because a problem means "fix this" and an observation does not. A comparison whose
margin sits inside the run's own noise is inconclusive rather than a contradiction, so it can make
the verdict `pending` at most.

## Paper retrieval and the backends

Stage 1 never trusts the model's memory: `fetch` asks for a `PaperSearchQuery` through the
forced `search_papers` tool, and `PaperSearcher.search()` then asks arXiv, Semantic Scholar
and OpenAlex in turn (`search.backends`) before ranking and de-duplicating. What the backends
did is part of the result: `backend_notes` / `backend_failures` are reset on every search, and
the fetch node writes them to `paper/search_report.json`, warns about the missing ones and
keeps them in the state, so a candidate list that lost a source can never look complete. The
run only fails when *every* backend failed (`PaperSearchError`).

Transient trouble is absorbed instead of escalated: timeouts, connection errors and fast
429/500/502/503/504 answers are retried `RETRY_ATTEMPTS` times with `RETRY_BACKOFF_SECONDS`
between attempts, while a *hang* is recorded and raised immediately. A 429 is the normal state
of the free Semantic Scholar pool, so it must never be the reason a paper is missed.

Full text arrives through a ladder of PDF links - `candidate.pdf_url`, then arXiv by id, then
arXiv by title - and every download must start with `%PDF-` (`_is_pdf`), which is what stops an
HTML error page from being mistaken for a paper. The link that finally worked is reported as
`info["pdf_source_url"]` and printed as `full text came from a fallback link: …`, and
`info["pdf_attempts"]` records what was tried even on success.

## Bounded loops

All three loops share the same shape: the verifier writes its problems to the staged lesson
file, the UI prints a `replan_notice` (yellow `⟳ REPLAN` / `⟳ RE-EXECUTE` banner), the main
model answers with a *reasoned* revision (`PlanRevision` must contain
`overall_reasoning` + `rejected_issues`), and the loop stops when the verifier is happy or
the round budget is spent. On the last round the verdict is accepted, `plan_carryover` /
`verdict_history` carry the open issues forward, and the warning `last permitted attempt`
is printed.

## Connectivity and proxy pinning

`connectivity.py` probes the endpoints a run actually needs, grouped by purpose (model API /
paper sources / dataset sources), over the channels real traffic uses (`direct`, the OS proxy,
the env proxy), and answers per endpoint - a blocked host is never reported as "the machine is
offline". When the primary dataset host is blocked but a mirror answers, the group is
`degraded` and `dataset_env()` hands the run `HF_ENDPOINT=...` so the generated script still
fetches real data; when the primary host answers, the same method asks for an inherited
`HF_ENDPOINT` to be *removed*, since a mirror configured machine-wide is a route this run
never probed. The per-endpoint verdicts are written to `logs/connectivity.json`.

The same module keeps the verdict and reality in sync. `ConnectivityProbe.run()` remembers, per
category, the channel that answered, and `provider(category)` hands each HTTP client a zero-arg
callable it calls before its first request; `pin_session()` then sets `trust_env=False` plus
that channel's proxies. Without that, `requests` would fall back to the environment alone - so a
stale `HTTP_PROXY` / `ALL_PROXY` / `NO_PROXY`, or a proxy that exists only in the OS settings,
could leave the probe saying "ok" while `tools/paper_search.py` and `tools/dataset_probe.py`
cannot download anything. A corporate `REQUESTS_CA_BUNDLE` is carried onto the pinned session,
and a category with no verified route leaves its client exactly as it was.

The langchain chat models get the same treatment through `build_http_client()`, which turns the
verified route into an `httpx.Client` (`trust_env=False`, explicit proxy, same CA bundle) and is
handed to every provider as `http_client` - this is the only hook langchain offers, and it is
needed twice over: the model API route is resolved by the probe on the first call, and
`langchain-openai` injects a custom transport for its socket options, which by itself *disables*
httpx's proxy auto-detection and would otherwise ignore a working OS proxy.

The generated script gets the same answer, one process further out. `repro.py` cannot see the
verdict, and the two HTTP stacks disagree about how to guess the route: `requests` reads the
environment and then the Windows registry, `httpx` reads only the environment. `dataset_env()`
therefore spells the route out as environment variables - `route_env()` writes the verified
proxy into both cases, adds the mirror's `HF_ENDPOINT` when only the mirror answered, and
returns ``None`` for every proxy variable when the verified route is a direct connection,
which `ScriptRunner._build_env` treats as "remove it": a dead inherited `HTTP_PROXY` would
otherwise send the script nowhere, however healthy the probe found the network. Nothing is
reported when no dataset host answered, so an unverified run leaves the child's environment
alone. `HF_ENDPOINT` follows the same rule and the asymmetry is deliberate: *the mirror
answering* writes it, *the primary host answering* removes it, because an endpoint nobody
probed must not outrank the verdict the probe just reached. `tests/experiment_child_proxy.py`
starts a real child process to keep this honest, and `tests/experiment_p1p2_semantics.py`
checks both directions offline.

The probe is budgeted, and it spends that budget in a deliberate order. Endpoints are visited
**channel by channel**: the deciding channel is tried for *every* endpoint before any fallback
channel is explored, so running out of time costs detail rather than the verdict. Within a
round, the endpoints that decide a go/no-go answer (`Endpoint.priority`: the model API, the
first paper backend, and both dataset hosts) go first, which is why a hanging host can no longer
starve the mirror. Every attempt is capped by the time left, so a slow network cannot push the
probe past `search.connectivity_budget_seconds` (20s by default), and only a *fast* failure is
retried - a dropped connection is a flake, a timeout is not. Verdicts stay honest about the
cut-off: a category is `unreachable` only when nothing in it was left unfinished, otherwise
`unknown`. `run_checks(..., clock=...)` injects the clock, so the tests replay all of this
instantly instead of sleeping.

## Workspace and publication

```
.essay_agent/runs/<run_id>/
  paper/    search_args.json, candidates.json, paper.json, full_text.md, sections_index.json
  cards/    cards.json, cards.md, skipped.json
  code/     repro.py, plan_*.json, run_outcome.json, verdict_round*.json,
            figures/*.png, metrics.json
  lesson/   lesson_plan.txt, lesson_execute.txt     (staged only)
  result/   report.md, cards.md, cards.json, verdicts.json, paper.json,
            metrics.json, figures/*.png
  logs/     run.log, environment.txt, connectivity.json,
            llm_transcript.md                      (only with --verbose)
```

`RunWorkspace.publish()` copies `result/` and `code/` into `<slug>-<short_id>` folders under
`result/` and `repro/`. `LessonStore.commit()` merges the staged lesson files into
`lesson/lesson_plan.txt` / `lesson/lesson_execute.txt` and consolidates them when they grow
past the threshold. `RunWorkspace.purge()` (Ctrl+C, `atexit` guard) removes the whole run
directory after verifying it really is inside the workspace root.

## Progress protocol

`runtime/progress.py` defines the contract between the generated script and the CLI:
`estimate`, `progress`, `metric`, `figure`, `done`, plus arbitrary log lines. `EtaEstimator`
turns `progress` events into a bar with a live ETA; `runtime/runner.py` runs the child with
the right argv, streams its output, kills the whole process tree on timeout/Ctrl+C and
collects metrics from both the event stream and `metrics.json`.

## Child environments and failure visibility

`runtime/runner.py:ScriptRunner` starts every generated script with `CHILD_ENV_DEFAULTS` on top of
the inherited environment: unbuffered, utf-8, `MPLBACKEND=Agg`, no `.pyc`, and
`KMP_DUPLICATE_LIB_OK=TRUE`. The last one is a documented-but-unsafe Intel workaround for a second,
lazily loaded copy of the OpenMP runtime - the `OMP: Error #15` abort that killed all three rounds of
run-20260920-062827-3e6074 before a single metric was written. It is therefore never applied
invisibly: `ScriptRunner.env_fixes` carries the variable together with the reason it is needed, and
`execute._outcome_payload` copies it into `code/run_outcome.json` and `code/preflight.json`.
Explicit `env=` entries still win, including `None`, which removes a variable.

The other half of the same story is `execute.log_tail(result)`. Stage 4.1 used to receive only
`stdout_tail`, which is empty for exactly the failures that matter (a native abort, a missing DLL, a
segfault). The helper merges both streams under `[stdout]` / `[stderr]` labels within a bounded
budget, and the `run` node prints the child's last error lines so the human sees the cause as well.
A run without stderr produces the previous prompt shape.

## Configuration

`config.py` layers pydantic-settings models: `LLMSettings`, `SearchSettings`,
`RuntimeSettings`, `MemorySettings`, `PathSettings`. Precedence is built-in defaults <
`config/default.yaml` < user YAML (`--config` or `./essay-agent.yaml`) < environment/`.env`.
`Settings.summary()` redacts the API key for `essay config show`.

## Errors

`errors.py` defines the hierarchy used for control flow: `TaskCancelled` (purge everything),
`PaperSearchError` / `PaperNotFoundError` (stage 1), `LLMError` (verifier/planner
unavailable - degrade instead of crashing), `BudgetExceeded` (pre-flight), and
`EssayAgentError` as the CLI-visible base (exit code 2).

`LLMReplyError` extends `LLMError` for "the model answered but the answer was unusable" and
renders its own context (see above). `ToolNotUsedError` is reserved for a model that really
refused to call a forced tool; a network or provider failure during a tool call raises
`LLMError` instead, so the two are never confused.
