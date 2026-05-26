"""Bare type aliases for the observation bridge.

Each alias just gives a name to a wire-friendly shape (ndarray / primitive).
No DataSpec / Annotated metadata, no schema framework. The heavy validation
version lives in rosnav_rl.observations.
"""

from __future__ import annotations

import numpy as np

Pose2DType = np.float32

Pose2D = np.ndarray  # shape (3,) -> [x, y, theta]
RobotState = np.ndarray  # shape (6,) -> [x, y, vx, vy, theta, omega]
LidarRanges = np.ndarray  # shape (N,) -> range readings
RobotVelocity = np.ndarray  # shape (3,) -> [vx, vy, omega]
NavigationPath = np.ndarray  # shape (N, 2) -> XY waypoints
ImageData = np.ndarray  # shape (C, H, W) or (H, W)
SafetyStatus = bool
PedestrianDetections = np.ndarray  # shape (N, 5) -> [id, x, y, vx, vy]
ArenaPedestrianDetections = np.ndarray  # shape (N, 5) -> [id, x, y, vx, vy]
