"""Goal inputs of language-conditioned (VLA) planners: what a manifest declares and what a run reveals."""

from __future__ import annotations

import copy

GOAL_INPUTS: tuple[str, ...] = ("pose", "instruction")

_POSE_GENERATORS = frozenset({"SubgoalGenerator"})


def declared(manifest: dict) -> frozenset[str] | None:
    """The manifest's `goal_inputs`, None for a planner that declares none (not a VLA)."""
    raw = manifest.get("goal_inputs")
    if raw is None:
        return None
    inputs = frozenset(raw)
    unknown = sorted(inputs - set(GOAL_INPUTS))
    if not inputs or unknown:
        raise ValueError(f"goal_inputs {sorted(raw)!r} invalid, expected a non-empty subset of {list(GOAL_INPUTS)}")
    return inputs


def reveal(value: str, accepted: frozenset[str]) -> frozenset[str]:
    """Parse `pose`, `instruction` or `pose+instruction`, empty = everything the planner accepts."""
    if not value:
        return accepted
    tokens = frozenset(value.split("+"))
    unknown = sorted(tokens - set(GOAL_INPUTS))
    if unknown:
        expected = "+".join(GOAL_INPUTS)
        raise ValueError(f"goal {value!r}: unknown input {unknown}, expected inputs from {expected!r} joined by '+'")
    unsupported = sorted(tokens - accepted)
    if unsupported:
        raise ValueError(f"goal {value!r}: planner does not accept {unsupported}, it accepts {sorted(accepted)}")
    return tokens


def withhold_pose(manifest: dict) -> dict:
    """Copy of `manifest` without the goal-pose datasources and the generators built on them."""
    if (manifest.get("depends") or {}).get("global_plan"):
        raise ValueError("cannot withhold the goal pose from a planner that depends on a global plan to it")
    out = copy.deepcopy(manifest)
    observations = out.get("observations") or {}
    datasources = observations.get("datasources") or {}
    dropped = {
        name
        for name, source in datasources.items()
        if source.get("type") in _POSE_GENERATORS
        or (
            source.get("type") == "geometry_msgs/PoseStamped"
            and (source.get("params") or {}).get("topic", name) == "goal_pose"
        )
    }
    for name in dropped:
        del datasources[name]
    aliases = observations.get("aliases") or {}
    for alias, target in list(aliases.items()):
        if target in dropped:
            del aliases[alias]
    return out


def initial_state(goal_pose: dict, instruction: str, revealed: frozenset[str]) -> dict:
    """Reset payload holding only the revealed goal inputs."""
    state: dict = {}
    if "pose" in revealed:
        state["goal_pose"] = goal_pose
    if "instruction" in revealed:
        state["instruction"] = instruction
    return state
