"""Minimal toy planner for tests. Returns a constant differential_drive action."""

from __future__ import annotations

import os
import pathlib

from arena_planners.sdk import load_manifest, main_loop

_CRASH_AFTER: int | None = None
_step_count = 0


def step(features: dict) -> list[float]:
    global _step_count
    _step_count += 1
    if _CRASH_AFTER is not None and _step_count > _CRASH_AFTER:
        raise RuntimeError(f"crash after {_CRASH_AFTER} steps")
    return [0.1, 0.0]


if __name__ == "__main__":
    raw = os.environ.get("TOY_PLANNER_CRASH_AFTER")
    if raw is not None:
        _CRASH_AFTER = int(raw)
    manifest = load_manifest(pathlib.Path(__file__).parent / "planner.yaml")
    main_loop(step, manifest=manifest)
