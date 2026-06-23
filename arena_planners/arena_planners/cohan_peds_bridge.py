"""Bridge arena_peds -> cohan_msgs/TrackedAgents for CoHAN's hateb planner."""

from __future__ import annotations

import math

import rclpy
import tf2_ros
from arena_people_msgs.msg import Pedestrians
from arena_rclpy_mixins.spin import spin_node
from cohan_msgs.msg import AgentType, TrackedAgent, TrackedAgents, TrackedSegment, TrackedSegmentType
from geometry_msgs.msg import PoseStamped, PoseWithCovariance, TwistWithCovariance
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_geometry_msgs import do_transform_pose_stamped

_MOVING_THRESHOLD: float = 0.1


def _rotate_twist_by_tf(
    twist: object,
    tf: object,
) -> TwistWithCovariance:
    q = tf.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w

    def _rotate(vx: float, vy: float, vz: float) -> tuple[float, float, float]:
        # Rodrigues: v' = v + 2w*(q x v) + 2*(q x (q x v))
        cx = y * vz - z * vy
        cy = z * vx - x * vz
        cz = x * vy - y * vx
        return (
            vx + 2.0 * (w * cx + y * cz - z * cy),
            vy + 2.0 * (w * cy + z * cx - x * cz),
            vz + 2.0 * (w * cz + x * cy - y * cx),
        )

    out = TwistWithCovariance()
    out.twist.linear.x, out.twist.linear.y, out.twist.linear.z = _rotate(twist.linear.x, twist.linear.y, twist.linear.z)
    out.twist.angular.x, out.twist.angular.y, out.twist.angular.z = _rotate(
        twist.angular.x, twist.angular.y, twist.angular.z
    )
    return out


class CohanPedsBridge(Node):
    def __init__(self) -> None:
        super().__init__("cohan_peds_bridge")

        self._target_frame: str = self.declare_parameter("target_frame", "map").value
        self._source_frame_fallback: str = self.declare_parameter("source_frame_fallback", "map").value

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._latest: Pedestrians | None = None

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Pedestrians, "arena_peds", self._on_peds, qos)
        self._pub = self.create_publisher(TrackedAgents, "tracked_agents", qos)
        self.create_timer(0.1, self._publish)

    def _on_peds(self, msg: Pedestrians) -> None:
        self._latest = msg

    def _publish(self) -> None:
        if self._latest is None:
            return

        msg = self._latest
        source_frame = msg.header.frame_id or self._source_frame_fallback
        stamp = msg.header.stamp

        try:
            tf = self._tf_buffer.lookup_transform(
                self._target_frame,
                source_frame,
                stamp,
                timeout=rclpy.duration.Duration(seconds=0.0),
            )
        except Exception as exc:
            self.get_logger().warning(
                f"TF lookup {source_frame!r} -> {self._target_frame!r} failed: {exc}",
                throttle_duration_sec=2.0,
            )
            return

        agents: list[TrackedAgent] = []
        for ped in msg.pedestrians:
            ps = PoseStamped()
            ps.header.frame_id = source_frame
            ps.header.stamp = stamp
            ps.pose = ped.pose

            transformed_ps = do_transform_pose_stamped(ps, tf)

            speed = math.hypot(ped.twist.linear.x, ped.twist.linear.y)
            state = TrackedAgent.MOVING if speed > _MOVING_THRESHOLD else TrackedAgent.STATIC

            seg = TrackedSegment()
            seg.type = TrackedSegmentType.TORSO
            seg.pose = PoseWithCovariance()
            seg.pose.pose = transformed_ps.pose
            seg.twist = _rotate_twist_by_tf(ped.twist, tf)

            agent = TrackedAgent()
            agent.track_id = ped.id
            agent.type = AgentType.HUMAN
            agent.name = ped.name
            agent.state = state
            agent.segments = [seg]
            agents.append(agent)

        out = TrackedAgents()
        out.header.stamp = stamp
        out.header.frame_id = self._target_frame
        out.agents = agents
        self._pub.publish(out)


def main() -> None:
    rclpy.init()
    node = CohanPedsBridge()
    try:
        spin_node(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
