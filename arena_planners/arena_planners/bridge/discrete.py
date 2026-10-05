"""Closed-loop execution of discrete navigation primitives (forward, left, right) on 2D map-frame poses."""

from __future__ import annotations

import dataclasses
import math
import typing

COMMANDS: frozenset[str] = frozenset({"forward", "left", "right"})

_GAIN = 2.0
_FORWARD_TOLERANCE_M = 0.02
_TURN_TOLERANCE_RAD = math.radians(1.0)
_V_MIN = 0.05
_W_MIN = 0.1
_V_CAP = 0.5
_W_CAP = 1.0
_TIMEOUT_FACTOR = 3.0
_TIMEOUT_SLACK_S = 1.0


def wrap_angle(angle: float) -> float:
    """Angle wrapped into [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


@dataclasses.dataclass(frozen=True)
class Primitives:
    """Length of one forward step and angle of one turn."""

    forward_m: float = 0.25
    turn_deg: float = 15.0


def primitives_from_manifest(manifest: dict) -> Primitives:
    """Primitives of a discrete manifest, which must declare a RobotPoseTFGenerator datasource named robot_pose."""
    datasources = (manifest.get("observations") or {}).get("datasources") or {}
    source = datasources.get("robot_pose") or {}
    if source.get("type") != "RobotPoseTFGenerator":
        raise ValueError(
            "action_type discrete needs observations.datasources.robot_pose of type RobotPoseTFGenerator, "
            f"got {source.get('type')!r}"
        )
    raw = manifest.get("primitives") or {}
    known = {field.name for field in dataclasses.fields(Primitives)}
    if unknown := sorted(set(raw) - known):
        raise ValueError(f"unknown primitives keys {unknown}, accepted: {sorted(known)}")
    primitives = Primitives(**{key: float(value) for key, value in raw.items()})
    if primitives.forward_m <= 0.0 or not 0.0 < primitives.turn_deg < 180.0:
        raise ValueError(f"primitives need forward_m > 0 and 0 < turn_deg < 180, got {primitives}")
    return primitives


@dataclasses.dataclass(frozen=True)
class Limits:
    """Speed caps for executing primitives."""

    v_max: float = _V_CAP
    w_max: float = _W_CAP

    @classmethod
    def from_velocity_limits(cls, velocity_limits: dict[str, tuple[float, float]]) -> Limits:
        """Positive `linear` and `angular` upper limits, capped at 0.5 m/s and 1.0 rad/s."""

        def _cap(key: str, cap: float) -> float:
            _, hi = velocity_limits.get(key, (0.0, 0.0))
            return min(hi, cap) if hi > 0.0 else cap

        return cls(v_max=_cap("linear", _V_CAP), w_max=_cap("angular", _W_CAP))


@dataclasses.dataclass(frozen=True)
class Move:
    """One primitive in progress, anchored at the pose and sim time it started from."""

    command: str
    x0: float
    y0: float
    theta0: float
    distance_m: float
    turn_rad: float
    limits: Limits
    deadline_s: float


@dataclasses.dataclass(frozen=True)
class MoveStep:
    """Twist for one tick, `done` ends the move and `timed_out` marks a move cut off at its deadline."""

    v: float = 0.0
    omega: float = 0.0
    done: bool = False
    timed_out: bool = False


def start_move(
    command: str,
    pose: typing.Sequence[float],
    t_s: float,
    primitives: Primitives,
    limits: Limits,
) -> Move:
    """Move for `command` from `pose` = [x, y, theta] at sim time `t_s`."""
    if command not in COMMANDS:
        raise ValueError(f"discrete command {command!r} not in {sorted(COMMANDS)}")
    x, y, theta = (float(c) for c in pose[:3])
    if command == "forward":
        distance_m, turn_rad = primitives.forward_m, 0.0
        nominal_s = distance_m / limits.v_max
    else:
        distance_m = 0.0
        turn_rad = math.radians(primitives.turn_deg) * (1.0 if command == "left" else -1.0)
        nominal_s = abs(turn_rad) / limits.w_max
    return Move(
        command=command,
        x0=x,
        y0=y,
        theta0=theta,
        distance_m=distance_m,
        turn_rad=turn_rad,
        limits=limits,
        deadline_s=t_s + _TIMEOUT_FACTOR * nominal_s + _TIMEOUT_SLACK_S,
    )


def _track(move: Move, pose: typing.Sequence[float]) -> MoveStep:
    x, y, theta = (float(c) for c in pose[:3])
    if move.command == "forward":
        progress = (x - move.x0) * math.cos(move.theta0) + (y - move.y0) * math.sin(move.theta0)
        remaining = move.distance_m - progress
        if remaining <= _FORWARD_TOLERANCE_M:
            return MoveStep(done=True)
        return MoveStep(
            v=_clamp(_GAIN * remaining, _V_MIN, move.limits.v_max),
            omega=_clamp(_GAIN * wrap_angle(move.theta0 - theta), -move.limits.w_max, move.limits.w_max),
        )
    err = wrap_angle(move.theta0 + move.turn_rad - theta)
    if abs(err) <= _TURN_TOLERANCE_RAD:
        return MoveStep(done=True)
    return MoveStep(omega=math.copysign(_clamp(_GAIN * abs(err), _W_MIN, move.limits.w_max), err))


def step_move(move: Move, pose: typing.Sequence[float] | None, t_s: float) -> MoveStep:
    """Twist toward the move's target from `pose`, zero while the pose is unknown."""
    step = MoveStep() if pose is None else _track(move, pose)
    if not step.done and t_s > move.deadline_s:
        return MoveStep(done=True, timed_out=True)
    return step
