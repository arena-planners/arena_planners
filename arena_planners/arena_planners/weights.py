"""Parse and fetch a planner's `weights.yaml` manifest."""

from __future__ import annotations

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


def missing(planner_dir: Path) -> list[str]:
    """Return the declared `dest` paths that are not present on disk."""
    return [entry["dest"] for entry in read(planner_dir) if not (planner_dir / entry["dest"]).is_file()]


def fetch(planner_dir: Path) -> list[str]:
    """Download each declared file via huggingface_hub and symlink it to `dest`.

    Files already present are skipped. Returns the `dest` paths newly linked.
    """
    entries = read(planner_dir)
    if not entries:
        return []
    from huggingface_hub import hf_hub_download

    fetched: list[str] = []
    for entry in entries:
        dest = planner_dir / entry["dest"]
        if dest.is_file():
            continue
        cached = hf_hub_download(repo_id=entry["repo"], filename=entry["filename"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        dest.symlink_to(cached)
        fetched.append(entry["dest"])
    return fetched
