"""Planner registry: names, paths, and status from .gitmodules `planner=` tags unioned with local planner dirs."""

from __future__ import annotations

import configparser
import subprocess
from pathlib import Path

_SDK = "arena_planners"
PLANNERS_SUBDIR = f"{_SDK}/planners"
_MARKER = "planner.py"


def workspace_root(hint: Path | None = None) -> Path:
    from arena_planners.resolver import _find_workspace_root  # noqa: PLC0415

    return _find_workspace_root(hint)


def submodule_paths(root: Path) -> dict[str, list[str]]:
    """Planner name -> submodule paths (relative to root) from arena_planners/.gitmodules planner= tags."""
    cfg = configparser.ConfigParser()
    cfg.read(root / _SDK / ".gitmodules")
    out: dict[str, list[str]] = {}
    for section in cfg.sections():
        if not section.startswith("submodule "):
            continue
        path = cfg[section].get("path", "").strip()
        if not path:
            continue
        for name in cfg[section].get("planner", "").split():
            out.setdefault(name, []).append(f"{_SDK}/{path}")
    return out


def local_planners(root: Path) -> list[str]:
    """Planner names from local dirs (a `planner.py`) under the planners subdir, excluding registered submodules."""
    pdir = root / PLANNERS_SUBDIR
    if not pdir.is_dir():
        return []
    registered = set(submodule_paths(root))
    names: list[str] = []
    for entry in sorted(pdir.iterdir()):
        if not entry.is_dir() or entry.name.startswith(".") or entry.name.endswith("_wrap"):
            continue
        if entry.name not in registered and (entry / _MARKER).is_file():
            names.append(entry.name)
    return names


def all_planners(root: Path) -> list[str]:
    """All planner names: registered submodules unioned with local planner dirs."""
    return sorted(set(submodule_paths(root)) | set(local_planners(root)))


def submodule_status(root: Path) -> dict[str, str]:
    """Submodule path -> 'init' | 'uninit' from git submodule status --recursive."""
    out = subprocess.run(
        ["git", "submodule", "status", "--recursive"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    ).stdout
    status: dict[str, str] = {}
    for line in out.splitlines():
        if not line:
            continue
        parts = line[1:].split()
        if len(parts) < 2:
            continue
        status[parts[1]] = "uninit" if line[0] == "-" else "init"
    return status


def planner_dir(name: str, root: Path) -> Path | None:
    """Directory for a planner: its submodule path, else a local planner dir. None if neither exists."""
    for path in submodule_paths(root).get(name, []):
        candidate = root / path
        if candidate.is_dir():
            return candidate
    local = root / PLANNERS_SUBDIR / name
    if (local / _MARKER).is_file():
        return local
    return None
