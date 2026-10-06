"""Execution of chunked actions (waypoint polylines and twist sequences) on 2D map-frame poses."""

from __future__ import annotations

import dataclasses
import math
import typing

from .discrete import Limits, MoveStep, require_robot_pose

CHUNK_TYPES: frozenset[str] = frozenset({"waypoints", "velocity_chunk"})
CHUNK_WIDTHS: dict[str, tuple[int, ...]] = {"waypoints": (2, 3), "velocity_chunk": (2,)}

_GAIN = 2.0
_GOAL_TOLERANCE_M = 0.05
_V_MIN = 0.05
_W_MIN = 0.1
_W_FALLBACK = 1.0
_TURN_IN_PLACE_RAD = 0.5 * math.pi
_TIMEOUT_FACTOR = 3.0
_TIMEOUT_SLACK_S = 1.0
_TIME_EPS_S = 1e-6


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


@dataclasses.dataclass(frozen=True)
class ChunkConfig:
    """Manifest `chunk` block: execution horizon, twist entry period, pure-pursuit lookahead and cruise speed."""

    execute_s: float = 1.0
    dt: float = 0.1
    lookahead_m: float = 0.3
    speed_mps: float = 0.5


def chunk_config_from_manifest(manifest: dict) -> ChunkConfig:
    """Chunk block of a chunked manifest, which must declare a RobotPoseTFGenerator datasource named robot_pose."""
    require_robot_pose(manifest, str(manifest.get("action_type")))
    raw = manifest.get("chunk") or {}
    known = {field.name for field in dataclasses.fields(ChunkConfig)}
    if unknown := sorted(set(raw) - known):
        raise ValueError(f"unknown chunk keys {unknown}, accepted: {sorted(known)}")
    config = ChunkConfig(**{key: float(value) for key, value in raw.items()})
    if not all(value > 0.0 for value in dataclasses.astuple(config)):
        raise ValueError(f"chunk needs positive {sorted(known)}, got {config}")
    return config


def chunk_limits(config: ChunkConfig, velocity_limits: dict[str, tuple[float, float]]) -> Limits:
    """Cruise speed capped by the robot's positive linear limit, turn rate its positive angular limit (else 1 rad/s)."""
    _, v_hi = velocity_limits.get("linear", (0.0, 0.0))
    _, w_hi = velocity_limits.get("angular", (0.0, 0.0))
    return Limits(
        v_max=min(config.speed_mps, v_hi) if v_hi > 0.0 else config.speed_mps,
        w_max=w_hi if w_hi > 0.0 else _W_FALLBACK,
    )


def check_chunk(kind: str, entries: typing.Iterable[typing.Sequence[float]]) -> list[list[float]]:
    """Entries of a `kind` chunk as float lists, raising ValueError on a wrong width or a non-finite value."""
    if kind not in CHUNK_TYPES:
        raise ValueError(f"chunk type {kind!r} not in {sorted(CHUNK_TYPES)}")
    widths = CHUNK_WIDTHS[kind]
    rows = [[float(c) for c in entry] for entry in entries]
    for row in rows:
        if len(row) not in widths:
            shape = "[x, y] or [x, y, yaw]" if kind == "waypoints" else "[v, omega]"
            raise ValueError(f"{kind} entries are {shape}, got {row}")
        if not all(math.isfinite(c) for c in row):
            raise ValueError(f"{kind} entries must be finite, got {row}")
    return rows


def robot_to_map(
    pose: typing.Sequence[float], points: typing.Iterable[typing.Sequence[float]]
) -> list[tuple[float, float]]:
    """Robot-frame points (x forward, y left) in the map frame of `pose` = [x, y, theta]."""
    x, y, theta = (float(c) for c in pose[:3])
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return [(x + px * cos_t - py * sin_t, y + px * sin_t + py * cos_t) for px, py, *_ in points]


@dataclasses.dataclass(frozen=True)
class WaypointChunk:
    """A map-frame polyline in progress, its first vertex the robot's position when it began."""

    kind: typing.ClassVar[str] = "waypoints"
    path: tuple[tuple[float, float], ...]
    lookahead_m: float
    limits: Limits
    end_s: float
    deadline_s: float


@dataclasses.dataclass(frozen=True)
class VelocityChunk:
    """A twist sequence in progress, entry i applied from `t0_s + i * dt`."""

    kind: typing.ClassVar[str] = "velocity_chunk"
    twists: tuple[tuple[float, float], ...]
    t0_s: float
    dt: float
    end_s: float


def start_chunk(
    kind: str,
    entries: typing.Iterable[typing.Sequence[float]],
    pose: typing.Sequence[float] | None,
    t_s: float,
    config: ChunkConfig,
    limits: Limits,
) -> WaypointChunk | VelocityChunk:
    """Chunk answering an observation taken at `pose` = [x, y, theta], begun at sim time `t_s`."""
    rows = check_chunk(kind, entries)
    if not rows:
        raise ValueError(f"empty {kind} chunk")
    end_s = t_s + config.execute_s
    if kind == "velocity_chunk":
        return VelocityChunk(twists=tuple((v, w) for v, w in rows), t0_s=t_s, dt=config.dt, end_s=end_s)
    if pose is None:
        raise ValueError("a waypoints chunk needs the robot pose")
    start = (float(pose[0]), float(pose[1]))
    return WaypointChunk(
        path=(start, *robot_to_map(pose, rows)),
        lookahead_m=config.lookahead_m,
        limits=limits,
        end_s=end_s,
        deadline_s=t_s + _TIMEOUT_FACTOR * config.execute_s + _TIMEOUT_SLACK_S,
    )


def _closest(path: tuple[tuple[float, float], ...], x: float, y: float) -> tuple[int, float]:
    """Segment index and fraction of the point on `path` closest to (x, y)."""
    best = (0, 0.0)
    best_d2 = math.inf
    for i in range(len(path) - 1):
        (ax, ay), (bx, by) = path[i], path[i + 1]
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        frac = _clamp(((x - ax) * dx + (y - ay) * dy) / length2, 0.0, 1.0) if length2 > 0.0 else 0.0
        d2 = (ax + frac * dx - x) ** 2 + (ay + frac * dy - y) ** 2
        if d2 < best_d2:
            best, best_d2 = (i, frac), d2
    return best


def _lookahead(
    path: tuple[tuple[float, float], ...], segment: int, frac: float, lookahead_m: float
) -> tuple[tuple[float, float], float]:
    """Point `lookahead_m` along `path` past (segment, frac), clipped to its end, and the path length left there."""
    target: tuple[float, float] | None = None
    ahead = lookahead_m
    remaining = 0.0
    for i in range(segment, len(path) - 1):
        (ax, ay), (bx, by) = path[i], path[i + 1]
        length = math.hypot(bx - ax, by - ay)
        start = frac * length if i == segment else 0.0
        left = length - start
        remaining += left
        if target is None and left >= ahead:
            s = (start + ahead) / length
            target = (ax + s * (bx - ax), ay + s * (by - ay))
        elif target is None:
            ahead -= left
    return (target if target is not None else path[-1]), remaining


def _track(chunk: WaypointChunk, pose: typing.Sequence[float]) -> MoveStep:
    x, y, theta = (float(c) for c in pose[:3])
    gx, gy = chunk.path[-1]
    to_goal = math.hypot(gx - x, gy - y)
    if to_goal <= _GOAL_TOLERANCE_M:
        return MoveStep(done=True)
    segment, frac = _closest(chunk.path, x, y)
    (tx, ty), remaining = _lookahead(chunk.path, segment, frac, chunk.lookahead_m)
    dx, dy = tx - x, ty - y
    lx = dx * math.cos(theta) + dy * math.sin(theta)
    ly = -dx * math.sin(theta) + dy * math.cos(theta)
    bearing = math.atan2(ly, lx)
    w_max = chunk.limits.w_max
    if abs(bearing) > _TURN_IN_PLACE_RAD:
        return MoveStep(omega=math.copysign(_clamp(_GAIN * abs(bearing), _W_MIN, w_max), bearing))
    v = _clamp(_GAIN * max(remaining, to_goal), _V_MIN, chunk.limits.v_max)
    omega = 2.0 * ly / max(lx * lx + ly * ly, 1e-9) * v
    if abs(omega) > w_max:
        v *= w_max / abs(omega)
        omega = math.copysign(w_max, omega)
    return MoveStep(v=v, omega=omega)


def step_chunk(chunk: WaypointChunk | VelocityChunk, pose: typing.Sequence[float] | None, t_s: float) -> MoveStep:
    """Twist for this tick of `chunk`, done at its horizon, past its last twist, or within 5 cm of its last point."""
    if isinstance(chunk, VelocityChunk):
        index = max(int((t_s - chunk.t0_s + _TIME_EPS_S) // chunk.dt), 0)
        if t_s + _TIME_EPS_S >= chunk.end_s or index >= len(chunk.twists):
            return MoveStep(done=True)
        v, omega = chunk.twists[index]
        return MoveStep(v=v, omega=omega)
    step = MoveStep() if pose is None else _track(chunk, pose)
    if step.done:
        return step
    if t_s > chunk.deadline_s:
        return MoveStep(done=True, timed_out=True)
    if t_s + _TIME_EPS_S >= chunk.end_s:
        return MoveStep(done=True)
    return step
