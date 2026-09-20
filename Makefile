PY ?= python

.PHONY: help install install-dev test lint fmt cov run clean

help:
	@echo "install      install the package in editable mode"
	@echo "install-dev  install with dev extras"
	@echo "test         run the test suite"
	@echo "lint         run ruff"
	@echo "fmt          auto-format with ruff"
	@echo "cov          run tests with coverage"
	@echo "run          start the interactive agent"
	@echo "clean        remove caches and build artifacts"

install:
	$(PY) -m pip install -e . --no-deps

install-dev:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

fmt:
	$(PY) -m ruff check --fix src tests

cov:
	$(PY) -m pytest --cov=essay_agent --cov-report=term-missing

run:
	$(PY) -m essay_agent agent

clean:
	$(PY) -c "import shutil,pathlib;[shutil.rmtree(p,ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PY) -c "import shutil;shutil.rmtree('.pytest_cache',ignore_errors=True);shutil.rmtree('.ruff_cache',ignore_errors=True)"

