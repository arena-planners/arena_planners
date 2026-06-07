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
    ImageData,
    LidarRanges,
    NavigationPath,
    PedestrianDetections,
    Pose2D,
    RobotState,
    RobotVelocity,
)
from .base import Collector

_ENCODING_TABLE: dict[str, tuple[type, int]] = {
    "rgb8": (np.uint8, 3),
    "bgr8": (np.uint8, 3),
    "rgba8": (np.uint8, 4),
    "bgra8": (np.uint8, 4),
    "mono8": (np.uint8, 1),
    "mono16": (np.uint16, 1),
    "16UC1": (np.uint16, 1),
    "32FC1": (np.float32, 1),
}

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _apply_resize(arr: np.ndarray, output_size: tuple[int, int] | None) -> np.ndarray:
    if output_size is None:
        return arr
    import cv2

    if arr.dtype == np.uint8:
        h, w = arr.shape[:2]
        # area-downsample anti-aliases; nearest-neighbour on depth/float preserves valid ranges at edges.
        interp = cv2.INTER_AREA if (output_size[0] <= w and output_size[1] <= h) else cv2.INTER_LINEAR
    else:
        interp = cv2.INTER_NEAREST
    return cv2.resize(arr, output_size, interpolation=interp)


def _apply_normalize(arr: np.ndarray, mode: str | None) -> np.ndarray:
    if mode is None:
        return arr
    if mode == "unit":
        return arr.astype(np.float32) / 255.0
    if mode == "imagenet":
        f = arr.astype(np.float32) / 255.0
        return (f - _IMAGENET_MEAN) / _IMAGENET_STD
    if mode == "depth_mm_to_m":
        return (arr.astype(np.float32) / 1000.0).clip(0.0, 10.0)
    raise ValueError(f"unknown normalize mode {mode!r}; expected one of: unit, imagenet, depth_mm_to_m")


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


class ImageCollector(Collector[sensor_msgs.Image, ImageData]):
    """sensor_msgs/Image -> (H, W) or (H, W, C) ndarray, optionally resized + normalized."""

    def __init__(
        self,
        name: str,
        topic: str,
        encoding: str | None = None,
        output_size: list[int] | tuple[int, int] | None = None,
        normalize: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name, topic, **kwargs)
        self._expected_encoding = encoding
        self._output_size = tuple(output_size) if output_size is not None else None
        self._normalize = normalize

    def _preprocess(self, msg: sensor_msgs.Image) -> ImageData:
        if msg.height == 0 or msg.width == 0 or not msg.data:
            return np.empty((0,), dtype=np.uint8)
        if self._expected_encoding and msg.encoding != self._expected_encoding:
            raise ValueError(
                f"{self.name}: encoding mismatch, expected {self._expected_encoding!r}, got {msg.encoding!r}"
            )
        if msg.encoding not in _ENCODING_TABLE:
            raise ValueError(
                f"{self.name}: unsupported encoding {msg.encoding!r}; supported: {sorted(_ENCODING_TABLE)}"
            )
        dtype, ch = _ENCODING_TABLE[msg.encoding]
        arr = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(msg.height, msg.width, ch)
        if ch == 1:
            arr = arr.squeeze(-1)
        arr = _apply_resize(arr, self._output_size)
        arr = _apply_normalize(arr, self._normalize)
        return arr


class CompressedImageCollector(Collector[sensor_msgs.CompressedImage, ImageData]):
    """sensor_msgs/CompressedImage -> uint8 BGR ndarray via cv2.imdecode, optionally resized + normalized."""

    def __init__(
        self,
        name: str,
        topic: str,
        output_size: list[int] | tuple[int, int] | None = None,
        normalize: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name, topic, **kwargs)
        self._output_size = tuple(output_size) if output_size is not None else None
        self._normalize = normalize

    def _preprocess(self, msg: sensor_msgs.CompressedImage) -> ImageData:
        if not msg.data:
            return np.empty((0,), dtype=np.uint8)
        import cv2

        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        arr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"{self.name}: cv2.imdecode returned None (corrupt frame on {self.topic}?)")
        arr = _apply_resize(arr, self._output_size)
        arr = _apply_normalize(arr, self._normalize)
        return arr
