"""Parse, fetch and verify a planner's `weights.yaml` manifest."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import urllib.request
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NamedTuple

_MANIFEST = "weights.yaml"
_BLOCK = 8 << 20
_OK = ".sha256-ok"
_HF_KEYS = frozenset({"repo", "filename", "revision", "dest", "sha256"})
_URL_KEYS = frozenset({"url", "dest", "sha256"})
_URL_SCHEMES = ("https://", "http://", "gs://")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ManifestError(ValueError):
    """A `weights.yaml` violates the manifest schema."""


class ChecksumError(RuntimeError):
    """A fetched file's sha256 differs from the manifest."""


@dataclass(frozen=True)
class Entry:
    """One validated `files` item: an HF file when `repo` is set, else a URL."""

    dest: str
    sha256: str
    repo: str | None = None
    filename: str | None = None
    revision: str | None = None
    url: str | None = None


class Fetched(NamedTuple):
    """Outcome for one entry: `status` is `cached`, `verified` or `downloaded`."""

    dest: str
    size: int
    status: str


def manifest_path(planner_dir: Path) -> Path:
    return planner_dir / _MANIFEST


def _str(item: Mapping, key: str, where: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{where}: `{key}` must be a non-empty string")
    return value


def _entry(item: object, where: str) -> Entry:
    if not isinstance(item, dict):
        raise ManifestError(f"{where}: expected a mapping")
    if ("repo" in item) == ("url" in item):
        raise ManifestError(f"{where}: exactly one of `repo` or `url` is required")
    allowed = _HF_KEYS if "repo" in item else _URL_KEYS
    unknown = sorted(set(item) - allowed)
    if unknown:
        raise ManifestError(f"{where}: unknown keys {unknown}, allowed {sorted(allowed)}")
    sha = _str(item, "sha256", where).lower()
    if not _SHA256.fullmatch(sha):
        raise ManifestError(f"{where}: `sha256` must be 64 hex characters")
    dest = _str(item, "dest", where)
    parts = PurePosixPath(dest)
    if parts.is_absolute() or ".." in parts.parts:
        raise ManifestError(f"{where}: `dest` must be a relative path inside the planner dir")
    if "url" in item:
        url = _str(item, "url", where)
        if not url.startswith(_URL_SCHEMES):
            raise ManifestError(f"{where}: `url` must start with one of {list(_URL_SCHEMES)}")
        return Entry(dest=dest, sha256=sha, url=url)
    revision = _str(item, "revision", where) if "revision" in item else None
    return Entry(
        dest=dest,
        sha256=sha,
        repo=_str(item, "repo", where),
        filename=_str(item, "filename", where),
        revision=revision,
    )


def read(planner_dir: Path) -> list[Entry]:
    """Return the validated `files` entries of `<planner_dir>/weights.yaml`, empty if none."""
    path = manifest_path(planner_dir)
    if not path.is_file():
        return []
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ManifestError(f"{path}: expected a mapping with `files`")
    unknown = sorted(set(data) - {"files"})
    if unknown:
        raise ManifestError(f"{path}: unknown keys {unknown}, allowed ['files']")
    items = data.get("files") or []
    if not isinstance(items, list):
        raise ManifestError(f"{path}: `files` must be a list")
    entries = [_entry(item, f"{path}: files[{i}]") for i, item in enumerate(items)]
    dests = [e.dest for e in entries]
    dupes = sorted({d for d in dests if dests.count(d) > 1})
    if dupes:
        raise ManifestError(f"{path}: duplicate `dest` {dupes}")
    return entries


def missing(planner_dir: Path, cache: Path | None = None) -> list[str]:
    """Return the declared `dest` paths that are absent or hold another file than their `sha256`."""
    cache = cache_root() if cache is None else cache
    return [entry.dest for entry in read(planner_dir) if not _current(planner_dir / entry.dest, entry.sha256, cache)]


def _current(dest: Path, sha: str, cache: Path) -> bool:
    if not dest.is_file():
        return False
    try:
        _verify(dest, sha, _hf_marker(cache, dest.resolve()))
    except ChecksumError:
        return False
    return True


def http_url(url: str) -> str:
    """Map `gs://bucket/path` to its public `https://storage.googleapis.com` URL."""
    if url.startswith("gs://"):
        return "https://storage.googleapis.com/" + url.removeprefix("gs://")
    return url


def hf_kwargs(entry: Entry) -> dict[str, str]:
    """Keyword arguments for `hf_hub_download` and `try_to_load_from_cache`."""
    kwargs = {"repo_id": entry.repo, "filename": entry.filename}
    if entry.revision:
        kwargs["revision"] = entry.revision
    return kwargs


def cache_root(env: Mapping[str, str] = os.environ) -> Path:
    """`$ARENA_WEIGHTS_CACHE`, else `arena-planners-weights` next to the HF hub cache."""
    if env.get("ARENA_WEIGHTS_CACHE"):
        return Path(env["ARENA_WEIGHTS_CACHE"]).expanduser()
    if env.get("HF_HUB_CACHE"):
        hub = Path(env["HF_HUB_CACHE"])
    elif env.get("HF_HOME"):
        hub = Path(env["HF_HOME"]) / "hub"
    else:
        hub = Path(env.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "huggingface" / "hub"
    return hub.expanduser().parent / "arena-planners-weights"


def sha256_file(path: Path) -> str:
    """Hex sha256 of `path`, read in 8 MB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def _marked(marker: Path, sha: str) -> bool:
    return marker.is_file() and marker.read_text().strip() == sha


def _verify(path: Path, sha: str, marker: Path) -> str:
    """Return `cached` on a marker hit, `verified` after a matching re-hash, raise otherwise."""
    if _marked(marker, sha):
        return "cached"
    actual = sha256_file(path)
    if actual != sha:
        raise ChecksumError(f"{path}: sha256 mismatch, expected {sha}, got {actual}")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(sha + "\n")
    return "verified"


def _hf_marker(cache: Path, blob: Path) -> Path:
    return cache / "hf" / (hashlib.sha256(str(blob).encode()).hexdigest() + _OK)


def _download(url: str, blob: Path, sha: str) -> None:
    fd, part = tempfile.mkstemp(dir=blob.parent, prefix=f".{blob.name}.", suffix=".part")
    tmp = Path(part)
    try:
        digest = hashlib.sha256()
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(http_url(url), timeout=60) as resp:
            while block := resp.read(_BLOCK):
                digest.update(block)
                out.write(block)
        actual = digest.hexdigest()
        if actual != sha:
            raise ChecksumError(f"{url}: sha256 mismatch, expected {sha}, got {actual}")
        os.replace(tmp, blob)
    finally:
        tmp.unlink(missing_ok=True)


def _fetch_url(entry: Entry, cache: Path) -> tuple[Path, str]:
    cache.mkdir(parents=True, exist_ok=True)
    blob = cache / entry.sha256
    marker = cache / (entry.sha256 + _OK)
    if blob.is_file():
        try:
            return blob, _verify(blob, entry.sha256, marker)
        except ChecksumError:
            blob.unlink()
            marker.unlink(missing_ok=True)
    _download(entry.url, blob, entry.sha256)
    marker.write_text(entry.sha256 + "\n")
    return blob, "downloaded"


def _fetch_hf(entry: Entry, dest: Path, cache: Path) -> tuple[Path | None, str]:
    if dest.is_file():
        try:
            return None, _verify(dest, entry.sha256, _hf_marker(cache, dest.resolve()))
        except ChecksumError:
            pass
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    kwargs = hf_kwargs(entry)
    was_cached = isinstance(try_to_load_from_cache(**kwargs), str)
    path = Path(hf_hub_download(**kwargs))
    status = _verify(path, entry.sha256, _hf_marker(cache, path.resolve()))
    return path, status if was_cached else "downloaded"


def _link(dest: Path, target: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() and Path(os.readlink(dest)) == target:
        return
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.symlink_to(target)


def fetch(planner_dir: Path, cache: Path | None = None) -> Iterator[Fetched]:
    """Download, verify and link each declared file, yielding one `Fetched` per entry as it completes."""
    cache = cache_root() if cache is None else cache
    for entry in read(planner_dir):
        dest = planner_dir / entry.dest
        if entry.url is None:
            target, status = _fetch_hf(entry, dest, cache)
        else:
            target, status = _fetch_url(entry, cache)
        if target is not None:
            _link(dest, target)
        yield Fetched(entry.dest, dest.stat().st_size, status)
