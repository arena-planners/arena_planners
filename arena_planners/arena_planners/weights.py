"""Parse and fetch a planner's `weights.yaml` manifest."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

_MANIFEST = "weights.yaml"


def manifest_path(planner_dir: Path) -> Path:
    return planner_dir / _MANIFEST


def read(planner_dir: Path) -> list[dict]:
    """Return the `files` entries from `<planner_dir>/weights.yaml`, empty if none."""
    path = manifest_path(planner_dir)
    if not path.is_file():
        return []
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("files") or [])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _current(planner_dir: Path, entry: dict) -> bool:
    """Whether `dest` holds the file the entry declares, by sha256 or else by HF cache link target."""
    dest = planner_dir / entry["dest"]
    if not dest.is_file():
        return False
    if entry.get("sha256"):
        return _sha256(dest) == entry["sha256"]
    if not dest.is_symlink():
        return True
    target = os.readlink(dest)
    repo_dir = f"/models--{entry['repo'].replace('/', '--')}/snapshots/"
    return repo_dir in target and target.endswith(f"/{entry['filename']}")


def missing(planner_dir: Path) -> list[str]:
    """Return the declared `dest` paths that are absent or hold another file than declared."""
    return [entry["dest"] for entry in read(planner_dir) if not _current(planner_dir, entry)]


def fetch(planner_dir: Path) -> list[str]:
    """Download each declared file via huggingface_hub and symlink it to `dest`.

    Current files are skipped, stale ones replaced. Returns the `dest` paths newly linked.
    """
    stale = [entry for entry in read(planner_dir) if not _current(planner_dir, entry)]
    if not stale:
        return []
    from huggingface_hub import hf_hub_download

    fetched: list[str] = []
    for entry in stale:
        dest = planner_dir / entry["dest"]
        cached = hf_hub_download(repo_id=entry["repo"], filename=entry["filename"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        dest.symlink_to(cached)
        if entry.get("sha256") and _sha256(dest) != entry["sha256"]:
            raise ValueError(
                f"{entry['dest']}: sha256 of {entry['repo']}/{entry['filename']} does not match weights.yaml"
            )
        fetched.append(entry["dest"])
    return fetched
