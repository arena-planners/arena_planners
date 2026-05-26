"""Pose conversion helpers used by collectors."""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import Pose2D

from .types import Pose2DType  # noqa: F401  (re-exported for collectors that import from here)


def euler_from_quaternion(quaternion):
    """Quaternion (x, y, z, w) -> (roll, pitch, yaw)."""
    x, y, z, w = quaternion

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.copysign(np.pi / 2, sinp) if abs(sinp) >= 1 else np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)


def pose3d_to_pose2d(pose3d) -> Pose2D:
    pose2d = Pose2D()
    pose2d.x = pose3d.position.x
    pose2d.y = pose3d.position.y
    q = pose3d.orientation
    _, _, yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))
    pose2d.theta = yaw
    return pose2d
