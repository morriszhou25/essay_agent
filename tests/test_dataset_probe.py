"""Dataset/requirement probes."""

from __future__ import annotations

from pathlib import Path

import pytest

from essay_agent.tools.dataset_probe import (
    DatasetProber,
    classify_target,
    probe_target,
)


@pytest.mark.parametrize(
    ("target", "kind"),
    [
        ("sklearn:iris", "sklearn"),
        ("iris", "sklearn"),
        ("mnist", "torchvision"),
        ("torchvision:cifar10", "torchvision"),
        ("hf:imdb", "hf_dataset"),
        ("allenai/c4", "hf_dataset"),
        ("https://huggingface.co/datasets/squad", "hf_dataset"),
        ("kaggle:titanic", "kaggle"),
        ("uci:iris", "uci"),
        ("package:numpy", "python_package"),
        ("https://example.org/data.csv", "url"),
        ("something vague", "unknown"),
    ],
)
def test_classify_target(target: str, kind: str) -> None:
    assert classify_target(target)[0] == kind


def test_local_packages_are_probed_offline() -> None:
    available = probe_target("package:numpy")
    assert available.reachable is True and "installed" in available.detail
    missing = probe_target("package:definitely_not_installed_pkg")
    assert missing.reachable is False and missing.status() == "unavailable"


def test_sklearn_dataset_needs_no_network() -> None:
    result = probe_target("sklearn:iris")
    assert result.reachable is True
    assert "ships with it" in result.detail


def test_kaggle_requires_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    result = probe_target("kaggle:titanic")
    assert result.reachable is False
    assert result.requires_credentials is True
    assert result.status() == "requires credentials"

    (tmp_path / ".kaggle").mkdir()
    (tmp_path / ".kaggle" / "kaggle.json").write_text("{}", encoding="utf-8")
    assert probe_target("kaggle:titanic").reachable is True


class _Response:
    def __init__(self, status: int = 200) -> None:
        self.status_code = status


class _Session:
    def __init__(self, status: int = 200, *, raises: bool = False) -> None:
        self.status = status
        self.raises = raises
        self.calls: list[str] = []

    def head(self, url: str, **kwargs):
        self.calls.append(url)
        if self.raises:
            raise OSError("no network")
        return _Response(self.status)

    def get(self, url: str, **kwargs):
        return self.head(url, **kwargs)


def test_http_probe_success() -> None:
    result = probe_target("https://example.org/data.csv", session=_Session(200))
    assert result.reachable is True and "HTTP 200" in result.detail


def test_http_probe_requires_credentials() -> None:
    session = _Session(403)
    result = probe_target("https://example.org/private", session=session)
    assert result.reachable is False
    assert result.requires_credentials is True


def test_http_probe_reports_network_failure() -> None:
    result = probe_target("https://example.org/data.csv", session=_Session(raises=True))
    assert result.reachable is False
    assert "no network" in result.detail


def test_hf_dataset_probe_uses_the_api() -> None:
    session = _Session(200)
    result = probe_target("hf:imdb", session=session)
    assert result.reachable is True
    assert any("api/datasets/imdb" in url for url in session.calls)
    assert result.source and "huggingface.co/datasets/imdb" in result.source


def test_prober_caches_results() -> None:
    session = _Session(200)
    prober = DatasetProber(session=session)
    first = prober.probe("hf:imdb")
    second = prober.probe("hf:imdb")
    assert first is second
    assert len(session.calls) == 1


def test_probe_many_skips_blanks() -> None:
    prober = DatasetProber(session=_Session(200))
    results = prober.probe_many(["hf:a/b", "", "  "])
    assert len(results) == 1


def test_a_prose_reference_is_not_a_repository_id() -> None:
    """Verbatim from a real run: both were probed as Hub repos and read as gated."""
    prose = [
        "a sparse expert architecture with a tunable number of experts / sparsity level",
        "compute for timing/FLOP measurement",
    ]
    for reference in prose:
        assert classify_target(reference)[0] == "unknown"

        result = probe_target(reference, session=_Session(401))
        assert result.reachable is None
        assert result.requires_credentials is False
        assert result.status() == "unknown"
        assert "resolve it manually" in result.detail


def test_an_hf_prefix_cannot_smuggle_prose_past_the_id_check() -> None:
    session = _Session(401)
    result = probe_target("hf:compute for timing/FLOP measurement", session=session)

    assert result.status() == "unknown"
    assert result.requires_credentials is False
    assert session.calls == []  # the Hub was never asked about an impossible path


def test_a_hugging_face_url_keeps_the_id_and_drops_the_section() -> None:
    assert classify_target("https://huggingface.co/datasets/allenai/c4") == (
        "hf_dataset",
        "allenai/c4",
    )
    # ``datasets/squad`` is not an id: the section is a section, not the owner.
    assert classify_target("https://huggingface.co/datasets/squad") == ("hf_dataset", "squad")
    assert classify_target("https://huggingface.co/datasets/nyu-mll/glue/tree/main") == (
        "hf_dataset",
        "nyu-mll/glue",
    )


def test_a_401_on_an_existing_id_says_that_the_hub_does_not_distinguish() -> None:
    result = probe_target("hf:imdb", session=_Session(401))

    assert result.reachable is False
    assert result.requires_credentials is True  # a token is still the one thing to try
    assert "does not exist" in result.detail  # but a wrong id looks exactly like this
