"""TF-based pose generator. Only generator the bridge ships natively."""

from __future__ import annotations

from typing import Any

import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
import tf2_ros

from ..utils.pose import euler_from_quaternion
from ..utils.types import Pose2D, Pose2DType
from .base import Generator


class RobotPoseTFGenerator(Generator[Pose2D]):
    """Robot 2D pose (x, y, theta) from the TF tree."""

    requires: dict = {}

    def __init__(
        self,
        name: str,
        node: rclpy.node.Node | None = None,
        source_frame: str = "",
        target_frame: str = "map",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        if node is None:
            raise ValueError("RobotPoseTFGenerator requires a ROS 2 node.")
        if not source_frame:
            raise ValueError("RobotPoseTFGenerator requires a non-empty source_frame.")

        self._node = node
        self._source_frame = source_frame
        self._target_frame = target_frame
        self._tf_buffer = tf2_ros.Buffer(node=node)
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, node)
        self._last_pose = np.array((0.0, 0.0, 0.0), dtype=Pose2DType)
        self._ready = False

    def _generate(self, **kwargs: Any) -> Pose2D:
        if not self._ready:
            try:
                self._tf_buffer.can_transform(
                    self._target_frame,
                    self._source_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0),
                )
                self._ready = True
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
                self._node.get_logger().warn(
                    f"Waiting for transform {self._source_frame!r} -> {self._target_frame!r}: {exc}"
                )
                return self._last_pose

        try:
            tf_stamped = self._tf_buffer.lookup_transform(
                self._target_frame,
                self._source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            t = tf_stamped.transform.translation
            r = tf_stamped.transform.rotation
            _, _, theta = euler_from_quaternion([r.x, r.y, r.z, r.w])
            self._last_pose = np.array((t.x, t.y, theta), dtype=Pose2DType)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            self._node.get_logger().warn(
                f"Could not get transform {self._source_frame!r} -> {self._target_frame!r}: {exc}"
            )
        return self._last_pose
