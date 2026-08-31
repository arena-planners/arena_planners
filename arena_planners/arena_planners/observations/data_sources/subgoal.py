"""Pure-pursuit subgoal generator. Turns a global plan into a near goal_pose."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...geometry import lookahead_clear_on_path, lookahead_on_path
from ..utils.types import Pose2D, Pose2DType
from .base import Generator


class SubgoalGenerator(Generator[Pose2D]):
    """Lookahead point on `global_plan`. Under `require_plan` the robot's own pose stands in while the plan is empty.

    Consumers steer straight at this point, so the carrot is only advanced while the
    chord to it stays out of the costmap's inscribed region. Without a costmap the
    lookahead is fixed, which cuts corners wherever the plan hugs one.
    """

    requires: dict = {
        "global_plan": None,
        "robot_pose_from_tf": None,
        "goal_pose": None,
        "costmap": None,
    }

    def __init__(self, name: str, lookahead: float = 2.0, require_plan: bool = False, **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._lookahead = float(lookahead)
        self._require_plan = bool(require_plan)

    def _generate(self, **kwargs: Any) -> Pose2D:
        goal_pose = kwargs["goal_pose"]
        robot_pose = kwargs["robot_pose_from_tf"]
        global_plan = kwargs["global_plan"]
        costmap = kwargs["costmap"]

        if global_plan is None or robot_pose is None or len(global_plan) == 0:
            if self._require_plan and robot_pose is not None:
                return np.asarray(robot_pose, dtype=Pose2DType)
            if goal_pose is None:
                return np.zeros(3, dtype=Pose2DType)
            return np.asarray(goal_pose, dtype=Pose2DType)

        plan = np.asarray(global_plan)
        pose = np.asarray(robot_pose)
        if costmap is None:
            target = lookahead_on_path(plan, pose, self._lookahead)
        else:
            target = lookahead_clear_on_path(plan, pose, self._lookahead, costmap.is_clear)
        if target is None:
            return np.asarray(goal_pose, dtype=Pose2DType)

        tx, ty = target
        theta = float(np.arctan2(ty - float(robot_pose[1]), tx - float(robot_pose[0])))
        return np.array((tx, ty, theta), dtype=Pose2DType)
