"""Pure-pursuit subgoal generator. Turns a global plan into a near goal_pose."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...geometry import lookahead_on_path
from ..utils.types import Pose2D, Pose2DType
from .base import Generator


class SubgoalGenerator(Generator[Pose2D]):
    """Lookahead point on `global_plan`, falling back to the raw goal when the plan is empty."""

    requires: dict = {
        "global_plan": None,
        "robot_pose_from_tf": None,
        "goal_pose": None,
    }

    def __init__(self, name: str, lookahead: float = 2.0, **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._lookahead = float(lookahead)

    def _generate(self, **kwargs: Any) -> Pose2D:
        goal_pose = kwargs["goal_pose"]
        robot_pose = kwargs["robot_pose_from_tf"]
        global_plan = kwargs["global_plan"]

        if global_plan is None or robot_pose is None or len(global_plan) == 0:
            if goal_pose is None:
                return np.zeros(3, dtype=Pose2DType)
            return np.asarray(goal_pose, dtype=Pose2DType)

        target = lookahead_on_path(np.asarray(global_plan), np.asarray(robot_pose), self._lookahead)
        if target is None:
            return np.asarray(goal_pose, dtype=Pose2DType)

        tx, ty = target
        theta = float(np.arctan2(ty - float(robot_pose[1]), tx - float(robot_pose[0])))
        return np.array((tx, ty, theta), dtype=Pose2DType)
