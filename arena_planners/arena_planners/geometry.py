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

    dx = pts[:, 0] - rx
    dy = pts[:, 1] - ry
    dist2 = dx * dx + dy * dy
    closest = int(np.argmin(dist2))

    target_x: float = float(pts[-1, 0])
    target_y: float = float(pts[-1, 1])
    acc = 0.0
    for i in range(closest, pts.shape[0] - 1):
        seg = float(np.hypot(pts[i + 1, 0] - pts[i, 0], pts[i + 1, 1] - pts[i, 1]))
        if acc + seg >= lookahead:
            t = (lookahead - acc) / seg if seg > 0 else 0.0
            target_x = float(pts[i, 0] + t * (pts[i + 1, 0] - pts[i, 0]))
            target_y = float(pts[i, 1] + t * (pts[i + 1, 1] - pts[i, 1]))
            break
        acc += seg

    return target_x, target_y


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
