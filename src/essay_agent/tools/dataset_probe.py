"""Dataset availability probes for stage 3 (``realize``).

The card verifier decides nothing here: this module only reports facts (reachable,
requires credentials, missing package) so the model can judge feasibility from
evidence instead of guessing.

Synthetic data is never a substitute - ``allow_synthetic_data`` exists only so a
user can *explicitly* override the policy, and the probe output never suggests it.

A reference that cannot be read as a real id is reported as ``unknown`` ("resolve it
first"), never as "requires credentials": the Hub answers HTTP 401 both for a gated
repository and for one that does not exist, so a bare 401 on a made-up path would turn
prose into a fabricated permission problem.
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from essay_agent.connectivity import RouteProvider, pin_session

HF_DATASET_API = "https://huggingface.co/api/datasets/{dataset_id}"
HF_DATASET_PAGE = "https://huggingface.co/datasets/{dataset_id}"

SKLEARN_DATASETS = {
    "iris",
    "digits",
    "wine",
    "breast_cancer",
    "diabetes",
    "linnerud",
    "sample_images",
    "fetch_california_housing",
    "fetch_covtype",
    "fetch_kddcup99",
    "fetch_lfw_people",
    "fetch_olivetti_faces",
    "fetch_20newsgroups",
    "load_digits",
    "load_iris",
    "load_wine",
    "load_breast_cancer",
    "load_diabetes",
}

TORCHVISION_DATASETS = {
    "mnist",
    "fashion-mnist",
    "fashionmnist",
    "cifar10",
    "cifar-100",
    "cifar100",
    "imagenet",
    "coco",
    "voc",
    "svhn",
    "emnist",
    "kitti",
    "celeba",
    "stl10",
}

_SKLEARN_PREFIXES = ("sklearn:", "sklearn-dataset:")
_UCI_PREFIXES = ("uci:",)
_KAGGLE_PREFIXES = ("kaggle:",)
_HF_PREFIXES = ("hf:", "huggingface:", "huggingface/datasets:")
_PACKAGE_PREFIXES = ("package:", "pip:", "module:")
_TORCHVISION_PREFIXES = ("torchvision:",)
_HF_URL_RE = re.compile(r"huggingface\.co/(?P<path>[\w.\-]+(?:/[\w.\-]+)*)")
# ``name`` or ``owner/name``, and nothing else: prose that happens to contain one slash
# ("compute for timing/FLOP measurement") is a reference, not a repository.
_HF_ID_RE = re.compile(r"[A-Za-z0-9][\w.\-]*(?:/[A-Za-z0-9][\w.\-]*)?")
_HF_URL_SECTIONS = ("datasets", "models", "spaces")


@dataclass
class ProbeResult:
    """One dataset/target availability check."""

    target: str
    kind: str = "unknown"
    reachable: bool | None = None
    detail: str = ""
    source: str | None = None
    requires_credentials: bool = False
    resolved_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def checkable(self) -> bool:
        return self.reachable is not None

    def status(self) -> str:
        if self.reachable is True:
            return "available"
        if self.reachable is False and self.requires_credentials:
            return "requires credentials"
        if self.reachable is False:
            return "unavailable"
        return "unknown"

    def describe(self) -> str:
        return f"{self.target} [{self.kind}] -> {self.status()}: {self.detail}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "kind": self.kind,
            "reachable": self.reachable,
            "status": self.status(),
            "detail": self.detail,
            "source": self.source,
            "requires_credentials": self.requires_credentials,
            "resolved_id": self.resolved_id,
        }


def _hf_id_from_url(path: str) -> str:
    """``datasets/owner/name/tree/main`` -> ``owner/name``.

    The section is stripped by hand rather than by an optional regex group: a group
    backtracks, and ``/datasets/squad`` would come out as the id ``datasets/squad`` -
    a repository that cannot exist, which the Hub answers with a 401.
    """
    parts = [part for part in path.split("/") if part]
    if parts and parts[0] in _HF_URL_SECTIONS:
        parts = parts[1:]
    return "/".join(parts[:2])


def classify_target(target: str) -> tuple[str, str]:
    """Return ``(kind, resolved_id)`` for a free-text dataset reference."""
    raw = (target or "").strip()
    lowered = raw.lower()

    for prefix in _HF_PREFIXES:
        if lowered.startswith(prefix):
            return "hf_dataset", raw[len(prefix) :].strip()
    match = _HF_URL_RE.search(lowered)
    if match:
        return "hf_dataset", _hf_id_from_url(match.group("path"))
    for prefix in _SKLEARN_PREFIXES:
        if lowered.startswith(prefix):
            return "sklearn", raw[len(prefix) :].strip()
    for prefix in _UCI_PREFIXES:
        if lowered.startswith(prefix):
            return "uci", raw[len(prefix) :].strip()
    for prefix in _KAGGLE_PREFIXES:
        if lowered.startswith(prefix):
            return "kaggle", raw[len(prefix) :].strip()
    for prefix in _TORCHVISION_PREFIXES:
        if lowered.startswith(prefix):
            return "torchvision", raw[len(prefix) :].strip()
    for prefix in _PACKAGE_PREFIXES:
        if lowered.startswith(prefix):
            return "python_package", raw[len(prefix) :].strip()
    if lowered.startswith(("http://", "https://")):
        return "url", raw
    if lowered.startswith("doi:") or lowered.startswith("10."):
        return "unknown", raw
    if re.fullmatch(r"[A-Za-z_][\w.\-]*", raw):
        if lowered in SKLEARN_DATASETS:
            return "sklearn", raw
        if lowered in TORCHVISION_DATASETS:
            return "torchvision", raw
        if "." in raw or raw.isidentifier():
            return "python_package", raw
    if "/" in raw and _HF_ID_RE.fullmatch(raw):
        return "hf_dataset", raw
    return "unknown", raw


def _http_probe(
    session: requests.Session, url: str, timeout: float
) -> tuple[bool, int | None, str]:
    try:
        response = session.head(url, timeout=timeout, allow_redirects=True)
        if response.status_code in {403, 405, 501} or response.status_code >= 400:
            response = session.get(url, timeout=timeout, allow_redirects=True, stream=True)
        code = response.status_code
        if code < 400:
            return True, code, f"HTTP {code}"
        return False, code, f"HTTP {code}"
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def _package_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name.split(".")[0]) is not None
    except (ImportError, ValueError):
        return False


def probe_target(
    target: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 15.0,
    route: RouteProvider | None = None,
) -> ProbeResult:
    """Check whether a dataset/target referenced by a card can actually be obtained."""
    kind, resolved = classify_target(target)
    result = ProbeResult(target=target, kind=kind, resolved_id=resolved or None)

    if kind == "python_package":
        available = _package_available(resolved)
        result.reachable = available
        result.source = "importlib"
        result.detail = (
            f"python package {resolved!r} is installed"
            if available
            else f"python package {resolved!r} is NOT installed (pip install {resolved})"
        )
        return result

    if kind == "sklearn":
        available = _package_available("sklearn")
        result.reachable = available
        result.source = "sklearn"
        result.detail = (
            f"scikit-learn is installed; {resolved!r} ships with it (no download needed)"
            if available
            else "scikit-learn is not installed"
        )
        return result

    if kind == "torchvision":
        available = _package_available("torchvision")
        result.reachable = available
        result.source = "torchvision"
        result.detail = (
            f"torchvision is installed; {resolved!r} downloads on first use"
            if available
            else "torchvision is not installed"
        )
        return result

    if kind == "kaggle":
        token = Path.home() / ".kaggle" / "kaggle.json"
        has_credentials = token.is_file()
        result.reachable = has_credentials
        result.requires_credentials = not has_credentials
        result.source = "kaggle"
        result.detail = (
            f"kaggle credentials found at {token}"
            if has_credentials
            else "no ~/.kaggle/kaggle.json; Kaggle downloads require an API token"
        )
        return result

    client = session or requests.Session()
    pin_session(client, route() if route else None)
    if kind == "hf_dataset":
        dataset_id = resolved or target
        if not _HF_ID_RE.fullmatch(dataset_id):
            result.reachable = None
            result.detail = (
                f"{dataset_id!r} is not a Hugging Face repository id (expected 'name' or "
                "'owner/name'); resolve this reference to a real id before running"
            )
            return result
        url = HF_DATASET_API.format(dataset_id=dataset_id)
        reachable, code, detail = _http_probe(client, url, timeout)
        result.reachable = reachable
        result.requires_credentials = code in {401, 403}
        result.source = HF_DATASET_PAGE.format(dataset_id=dataset_id)
        result.detail = f"{dataset_id}: {detail}"
        if code in {401, 403}:
            result.detail = (
                f"{dataset_id}: {detail} - the Hub answers the same 401 for a repository "
                "that is gated and for one that does not exist"
            )
        return result

    if kind == "uci":
        url = f"https://archive.ics.uci.edu/dataset/{resolved}"
        reachable, code, detail = _http_probe(client, url, timeout)
        result.reachable = reachable
        result.requires_credentials = code in {401, 403}
        result.source = url
        result.detail = f"UCI {resolved}: {detail}"
        return result

    if kind == "url":
        reachable, code, detail = _http_probe(client, resolved, timeout)
        result.reachable = reachable
        result.requires_credentials = code in {401, 403}
        result.source = resolved
        result.detail = detail
        return result

    result.reachable = None
    result.detail = (
        "could not classify this reference; the agent must resolve it manually before running"
    )
    return result


class DatasetProber:
    """Caching wrapper around :func:`probe_target`."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        timeout: float = 15.0,
        route: RouteProvider | None = None,
    ) -> None:
        self._session = session
        self._route = route
        self._pinned = False
        self.timeout = timeout
        self._cache: dict[str, ProbeResult] = {}

    @property
    def session(self) -> requests.Session:
        """Built lazily, then pinned to the route the probe verified for datasets."""
        if self._session is None:
            self._session = requests.Session()
        if not self._pinned:
            self._pinned = True
            pin_session(self._session, self._route() if self._route else None)
        return self._session

    def probe(self, target: str) -> ProbeResult:
        key = (target or "").strip()
        if key not in self._cache:
            self._cache[key] = probe_target(key, session=self.session, timeout=self.timeout)
        return self._cache[key]

    def probe_many(self, targets: list[str]) -> list[ProbeResult]:
        seen: list[ProbeResult] = []
        for target in targets:
            if not target or not str(target).strip():
                continue
            seen.append(self.probe(str(target)))
        return seen

    def report(self, targets: list[str]) -> list[dict[str, Any]]:
        return [result.to_dict() for result in self.probe_many(targets)]
