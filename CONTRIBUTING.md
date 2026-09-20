# Contributing

Thanks for taking a look. This is a small project with a strong opinion about how changes land:
**a behaviour fix is not finished until a test proves it.**

## Setup

```bash
git clone https://github.com/OWNER/REPO.git
cd essay-agent
python -m pip install -e ".[dev]"      # Python >= 3.11
python -m pytest                       # 365 tests, fully offline
```

Useful entry points: `make help`, `essay config check`, `essay agent`.

## The rules that matter

1. **Tests never touch the network or an API key.** Everything goes through the fakes in
   `tests/fakes.py` (`FakeLLM`, `FakeSearcher`, `FakeProber`, `FakeRunner`, `FakeConnectivity`)
   and the helpers in `tests/pipeline.py`.
2. **Test first, then change `src/`.** New behaviour starts as an executable experiment script
   named `tests/experiment_<topic>.py` (deliberately *not* collected by pytest) that states the
   measured problem, the intended rule and how to run it. Get it green against a prototype, then
   change the source, then add a collected `tests/test_<topic>.py` that locks the behaviour in.
3. **`AGENTS.md` is the contract.** It lists the 27 invariants (Ctrl+C purges the task, nothing is
   published before `finalize`, no synthetic data, bounded loops that release on round 3, exact
   abort strings, ...). Each one has a test; do not regress one to make a new test pass.
4. **User-visible behaviour needs a docs update in the same change** - README, `docs/` or
   `AGENTS.md`.
5. **Keep the shape of the code.** Nodes take `RunState`, return a partial state dict and get
   everything else from the injected `Deps`; nodes only ever touch `deps.ui`. English docstrings
   and identifiers; the CLI keeps Chinese strings only where the specification demands them.

Run the same checks CI runs before opening a PR:

```bash
python -m ruff check src tests
python -m ruff format --check src tests
python -m pytest -q
```

## Reporting a bug

The most useful report includes the run directory (`.essay_agent/runs/<run_id>/`):

- `logs/run.log` and `logs/connectivity.json` (which channels answered),
- `code/preflight.json` and `code/run_outcome.json` (exit code, metrics, `env_fixes`, stdout and
  stderr tails),
- `code/repro.py` plus `code/verdict_round*.json`, so the failure can be reproduced by hand.

Please remove anything private first - the run directory contains the downloaded paper text and
your full LLM exchanges when `--verbose` was used.