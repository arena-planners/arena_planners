"""Planner-side geometry helpers.

Shared between planner subprocesses (drlvo, crowdnav) for path-following
subgoal extraction. Pure numpy, no ROS, no rclpy.
"""

from __future__ import annotations

import numpy as np


def lookahead_on_path(
    path: np.ndarray,
    robot_pose: np.ndarray,
    lookahead: float,
) -> tuple[float, float] | None:
    """Pure-pursuit lookahead point on a path, returned in world frame.

    Args:
        path: (N, >=2) array of [x, y, ...] world-frame waypoints.
        robot_pose: (>=2,) array [x, y, ...] in the same frame as path.
        lookahead: target distance from robot along the path, in meters.

    Returns:
        (x, y) lookahead point in world frame, or None if path is empty.
        If the path is shorter than lookahead, returns the last path point.
    """
    if path.shape[0] == 0:
        return None

    rx, ry = float(robot_pose[0]), float(robot_pose[1])
    pts = np.asarray(path[:, :2], dtype=np.float32)
    if pts.shape[0] == 1:
        return float(pts[0, 0]), float(pts[0, 1])

    head = pts[:-1]
    delta = pts[1:] - head
    seg_len2 = (delta * delta).sum(axis=1)
    robot_xy = np.array((rx, ry), dtype=np.float32)
    offset = robot_xy - head
    along = np.divide((offset * delta).sum(axis=1), seg_len2, out=np.zeros_like(seg_len2), where=seg_len2 > 0)
    proj = head + np.clip(along, 0.0, 1.0)[:, None] * delta
    closest = int(np.argmin(((proj - robot_xy) ** 2).sum(axis=1)))

    px, py = float(proj[closest, 0]), float(proj[closest, 1])
    seg = float(np.hypot(float(pts[closest + 1, 0]) - px, float(pts[closest + 1, 1]) - py))
    if seg >= lookahead:
        s = lookahead / seg if seg > 0 else 0.0
        return px + s * (float(pts[closest + 1, 0]) - px), py + s * (float(pts[closest + 1, 1]) - py)

    acc = seg
    for i in range(closest + 1, pts.shape[0] - 1):
        seg = float(np.hypot(pts[i + 1, 0] - pts[i, 0], pts[i + 1, 1] - pts[i, 1]))
        if acc + seg >= lookahead:
            s = (lookahead - acc) / seg if seg > 0 else 0.0
            nx = float(pts[i, 0] + s * (pts[i + 1, 0] - pts[i, 0]))
            ny = float(pts[i, 1] + s * (pts[i + 1, 1] - pts[i, 1]))
            return nx, ny
        acc += seg

    return float(pts[-1, 0]), float(pts[-1, 1])


def world_to_robot_frame(
    point: tuple[float, float],
    robot_pose: np.ndarray,
) -> tuple[float, float]:
    """Transform a world-frame (x, y) into the robot's local frame."""
    rx, ry, rtheta = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
    cos_t = float(np.cos(rtheta))
    sin_t = float(np.sin(rtheta))
    dx = point[0] - rx
    dy = point[1] - ry
    return cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy
