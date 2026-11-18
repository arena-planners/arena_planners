"""FollowPath paths for Nav2 DWB from discrete moves and waypoint chunks, and the goal in flight."""

from __future__ import annotations

import dataclasses
import math
import typing

if typing.TYPE_CHECKING:
    from rclpy.action.client import ClientGoalHandle
    from rclpy.task import Future

Pose = tuple[float, float, float]


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def densify(
    points: typing.Sequence[tuple[float, float]], start_yaw: float, spacing: float = 0.05, end_yaw: float | None = None
) -> list[Pose]:
    """Points every `spacing` m along the polyline with segment headings; a single point keeps `start_yaw`."""
    if not points:
        return []
    out: list[Pose] = []
    yaw = start_yaw
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:], strict=False):
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg < 1e-9:
            continue
        yaw = math.atan2(y1 - y0, x1 - x0)
        n = max(1, int(math.ceil(seg / spacing)))
        for k in range(n):
            f = k / n
            out.append((x0 + f * (x1 - x0), y0 + f * (y1 - y0), yaw))
    x, y = points[-1]
    out.append((x, y, yaw if end_yaw is None else end_yaw))
    return out


def move_path(
    command: str, amount: float, pose: Pose, forward_m: float, turn_deg: float, spacing: float = 0.05
) -> list[Pose]:
    """Map-frame path of one discrete move from `pose` = (x, y, yaw)."""
    x, y, yaw = pose
    if command == "forward":
        d = amount if amount > 0 else forward_m
        return densify([(x, y), (x + d * math.cos(yaw), y + d * math.sin(yaw))], yaw, spacing)
    if command in ("left", "right"):
        a = math.radians(amount if amount > 0 else turn_deg) * (1.0 if command == "left" else -1.0)
        return [(x, y, yaw), (x, y, _wrap(yaw + a))]
    raise ValueError(f"no DWB path for discrete command {command!r}")


def chunk_path(entries: typing.Sequence[typing.Sequence[float]], pose: Pose, spacing: float = 0.05) -> list[Pose]:
    """Map-frame path of a waypoint chunk given in the robot frame of `pose` (x forward, y left)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(x, y)] + [(x + c * float(e[0]) - s * float(e[1]), y + s * float(e[0]) + c * float(e[1])) for e in entries]
    return densify(pts, yaw, spacing)


def path_length(path: typing.Sequence[Pose]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path[:-1], path[1:], strict=False))


def nominal_timeout_s(path: typing.Sequence[Pose], v_max: float = 0.5, w_max: float = 1.0) -> float:
    """Three times the nominal duration plus 2 s, as for the bridge's own moves."""
    if not path:
        return 2.0
    turn = abs(_wrap(path[-1][2] - path[0][2]))
    nominal = path_length(path) / max(v_max, 1e-3) + turn / max(w_max, 1e-3)
    return 3.0 * nominal + 2.0


@dataclasses.dataclass
class DwbGoal:
    """A FollowPath goal in flight; release_at ends a chunk early while DWB keeps driving it."""

    what: str
    path: list[Pose]
    t_start: float
    timeout_s: float
    release_at: float | None = None
    send_future: Future | None = None
    goal_handle: ClientGoalHandle | None = None
    result_future: Future | None = None
    rejected: bool = False

    def step(self, t_s: float) -> tuple[bool, bool]:
        """(done, timed_out) at sim time `t_s`, polling the futures the edge node attached."""
        if self.send_future is not None and self.goal_handle is None and self.send_future.done():
            handle = self.send_future.result()
            if handle is None or not handle.accepted:
                self.rejected = True
                return True, False
            self.goal_handle = handle
            self.result_future = handle.get_result_async()
        if self.result_future is not None and self.result_future.done():
            return True, False
        if self.release_at is not None and t_s >= self.release_at:
            return True, False
        if t_s - self.t_start > self.timeout_s:
            return True, True
        return False, False


__all__ = ["DwbGoal", "chunk_path", "densify", "move_path", "nominal_timeout_s", "path_length"]
