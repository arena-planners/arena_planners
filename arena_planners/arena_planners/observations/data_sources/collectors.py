"""Topic-subscriber collectors. Each preprocess returns a wire-friendly value."""

from __future__ import annotations

import arena_people_msgs.msg as arena_people_msgs
import geometry_msgs.msg as geometry_msgs
import nav_msgs.msg as nav_msgs
import numpy as np
import people_msgs.msg as people_msgs
import sensor_msgs.msg as sensor_msgs

from ..utils.pose import Pose2DType, pose3d_to_pose2d
from ..utils.types import (
    ArenaPedestrianDetections,
    LidarRanges,
    NavigationPath,
    PedestrianDetections,
    Pose2D,
    RobotState,
    RobotVelocity,
)
from .base import Collector


class LaserScanCollector(Collector[sensor_msgs.LaserScan, LidarRanges]):
    def _preprocess(self, msg: sensor_msgs.LaserScan) -> LidarRanges:
        if len(msg.ranges) == 0:
            return np.array([])
        laser = np.array(msg.ranges, np.float32)
        laser[np.isnan(laser)] = msg.range_max
        laser[np.isinf(laser)] = msg.range_max
        return laser


class OdometryCollector(Collector[nav_msgs.Odometry, RobotState]):
    def _preprocess(self, msg: nav_msgs.Odometry) -> RobotState:
        pose2d = pose3d_to_pose2d(msg.pose.pose)
        return np.array(
            (
                pose2d.x,
                pose2d.y,
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                pose2d.theta,
                msg.twist.twist.angular.z,
            ),
            dtype=Pose2DType,
        )


class PoseStampedCollector(Collector[geometry_msgs.PoseStamped, Pose2D]):
    def _preprocess(self, msg: geometry_msgs.PoseStamped) -> Pose2D:
        pose2d = pose3d_to_pose2d(msg.pose)
        return np.array((pose2d.x, pose2d.y, pose2d.theta), dtype=Pose2DType)


class PathCollector(Collector[nav_msgs.Path, NavigationPath]):
    """Flattens nav_msgs/Path poses to (N, 3) ndarray: [x, y, yaw]."""

    def _preprocess(self, msg: nav_msgs.Path) -> NavigationPath:
        if not msg.poses:
            return np.empty((0, 3), dtype=Pose2DType)
        out = np.empty((len(msg.poses), 3), dtype=Pose2DType)
        for i, ps in enumerate(msg.poses):
            pose2d = pose3d_to_pose2d(ps.pose)
            out[i] = (pose2d.x, pose2d.y, pose2d.theta)
        return out


class TwistCollector(Collector[geometry_msgs.Twist, RobotVelocity]):
    def _preprocess(self, msg: geometry_msgs.Twist) -> RobotVelocity:
        return np.array((msg.linear.x, msg.linear.y, msg.angular.z), dtype=np.float32)


class PeopleCollector(Collector[people_msgs.People, PedestrianDetections]):
    """Flattens people_msgs/People to (N, 5) ndarray: [id, x, y, vx, vy]."""

    def _preprocess(self, msg: people_msgs.People) -> PedestrianDetections:
        if not msg.people:
            return np.empty((0, 5), dtype=np.float32)
        out = np.empty((len(msg.people), 5), dtype=np.float32)
        for i, p in enumerate(msg.people):
            try:
                pid = float(p.tags[p.tagnames.index("id")])
            except (ValueError, IndexError):
                pid = float(i)
            out[i] = (pid, p.position.x, p.position.y, p.velocity.x, p.velocity.y)
        return out


class ArenaPedestrianCollector(Collector[arena_people_msgs.Pedestrians, ArenaPedestrianDetections]):
    """Flattens arena_people_msgs/Pedestrians to (N, 5) ndarray: [id, x, y, vx, vy]."""

    def _preprocess(self, msg: arena_people_msgs.Pedestrians) -> ArenaPedestrianDetections:
        if not msg.pedestrians:
            return np.empty((0, 5), dtype=np.float32)
        out = np.empty((len(msg.pedestrians), 5), dtype=np.float32)
        for i, ped in enumerate(msg.pedestrians):
            out[i] = (
                float(ped.id),
                ped.pose.position.x,
                ped.pose.position.y,
                ped.twist.linear.x,
                ped.twist.linear.y,
            )
        return out
