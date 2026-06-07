"""Action-space projection between holonomic and differential-drive twists."""

from __future__ import annotations

import math


def project_holonomic_to_diff_drive(
    vx: float,
    vy: float,
    omega_in: float,
    robot_theta: float,
    step_dt_s: float,
) -> tuple[float, float]:
    if step_dt_s <= 0.0:
        raise ValueError(f"step_dt_s must be positive, got {step_dt_s}")
    v = math.hypot(vx, vy)
    if v < 1e-6:
        return 0.0, float(omega_in)
    desired_theta = math.atan2(vy, vx)
    heading_err = (desired_theta - robot_theta + math.pi) % (2.0 * math.pi) - math.pi
    omega = heading_err / step_dt_s + float(omega_in)
    return float(v), float(omega)


def unpack_omnidirectional(action: list[float]) -> tuple[float, float, float]:
    if len(action) == 2:
        return float(action[0]), float(action[1]), 0.0
    if len(action) == 3:
        return float(action[0]), float(action[1]), float(action[2])
    raise ValueError(f"omnidirectional action must have length 2 or 3, got {len(action)}: {action!r}")


def unpack_differential_drive(action: list[float]) -> tuple[float, float]:
    if len(action) == 2:
        return float(action[0]), float(action[1])
    raise ValueError(f"differential_drive action must have length 2, got {len(action)}: {action!r}")
