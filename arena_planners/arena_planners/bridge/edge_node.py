"""Edge node and subprocess helper for the DRL planner bridge."""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import math
import os
import queue
import signal
import subprocess
import threading
import typing
import uuid

import geometry_msgs.msg
import numpy as np
import rclpy.qos
import yaml
from arena_rclpy_mixins import ArenaMixinNode
from arena_robots.lockstep_beat import LockstepBeat
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters

from .protocol import (
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    Action,
    Bye,
    Cancel,
    CancelAck,
    Frame,
    Heartbeat,
    Init,
    InitAck,
    Obs,
    ProtocolError,
    Reset,
    ResetAck,
    Shutdown,
    decode_frame,
    encode_frame,
)
from .transport import (
    OBS_POLICY_LATEST_ONLY,
    OBS_POLICY_LOSSLESS,
    TransportSet,
    ZmqPullTransport,
    ZmqPushTransport,
    env_from_endpoints,
    generate_transport_set,
)

if typing.TYPE_CHECKING:
    from arena_planners.observations.pipeline import Pipeline


_PR_SET_PDEATHSIG = 1
_PHYSICS_DT_TIMEOUT_S = 5.0
_PLANNER_THREAD_ENV = {
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "OMP_WAIT_POLICY": "PASSIVE",
}
_TIMING_LOG_EVERY = 100


class _PlannerDied:
    """Action-queue sentinel from the I/O thread."""


_libc = ctypes.CDLL("libc.so.6", use_errno=True)


def _die_with_parent() -> None:
    _libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)


class PlannerProcess:
    """Thin subprocess wrapper owning one planner child process."""

    def __init__(
        self,
        command: list[str],
        endpoints: TransportSet,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self._command = command
        self._endpoints = endpoints
        self._extra_env = extra_env or {}
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        env = os.environ.copy()
        for key, value in _PLANNER_THREAD_ENV.items():
            env.setdefault(key, value)
        env.update(env_from_endpoints(self._endpoints))
        env.update(self._extra_env)
        self._proc = subprocess.Popen(
            self._command,
            start_new_session=True,
            preexec_fn=_die_with_parent,
            env=env,
        )

    def is_alive(self) -> bool:
        if self._proc is None:
            return False
        return self._proc.poll() is None

    @property
    def returncode(self) -> int | None:
        if self._proc is None:
            return None
        return self._proc.poll()

    def stop(self, grace_seconds: float = 2.0) -> int:
        if self._proc is None:
            return 0
        if self._proc.poll() is not None:
            return self._proc.returncode  # type: ignore[return-value]
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
        try:
            self._proc.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            self._proc.wait()
        return self._proc.returncode  # type: ignore[return-value]

    def __enter__(self) -> PlannerProcess:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


_CMD_VEL_QOS = rclpy.qos.QoSProfile(
    depth=1,
    reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
    durability=rclpy.qos.DurabilityPolicy.VOLATILE,
)


def _load_manifest(manifest: dict | str) -> dict:
    if isinstance(manifest, dict):
        return manifest
    with open(manifest) as fh:
        return yaml.safe_load(fh) or {}


class PlannerEdgeNode(ArenaMixinNode):
    """Edge-side ROS node that owns one planner subprocess and drives the obs/action loop."""

    def __init__(
        self,
        node_name: str,
        manifest: dict | str,
        planner_command: list[str],
        cmd_vel_topic: str,
        namespace: str = "",
        source_frame: str = "",
        target_frame: str = "map",
        is_holonomic: bool = False,
        simulation_namespace: str = "",
        velocity_limits: dict[str, tuple[float, float]] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(node_name, namespace=namespace, use_global_arguments=False, **kwargs)
        self._manifest_raw = manifest
        self._planner_command = planner_command
        self._cmd_vel_topic = cmd_vel_topic
        self._source_frame = source_frame
        self._target_frame = target_frame
        self._is_holonomic = bool(is_holonomic)
        self._simulation_namespace = simulation_namespace
        self._velocity_limits = velocity_limits or {}

        self._obs_manager: Pipeline | None = None
        self._proc: PlannerProcess | None = None
        self._data_push: ZmqPushTransport | None = None
        self._data_pull: ZmqPullTransport | None = None
        self._control_push: ZmqPushTransport | None = None
        self._control_pull: ZmqPullTransport | None = None
        self._run_id: str = uuid.uuid4().hex

        self._obs_queue: queue.Queue[bytes] = queue.Queue()
        self._action_queue: asyncio.Queue[Frame | _PlannerDied] = asyncio.Queue()
        self._ack_queue: asyncio.Queue[Frame] = asyncio.Queue()
        self._io_thread: threading.Thread | None = None
        self._io_stop = threading.Event()

        self._cmd_vel_pub: rclpy.publisher.Publisher | None = None
        self._seq: int = 0
        self._last_heartbeat_ns: int = 0
        self._heartbeat_period_s: float = 0.0
        self._control_ack_timeout_s: float = 5.0

        self._rate = self.ROSParam[float]("planner_rate_hz", 10.0)
        self._action_timeout = self.ROSParam[float]("planner_action_timeout_s", 0.5)
        self._init_timeout = self.ROSParam[float]("planner_init_timeout_s", 60.0)
        self._dropped_features_logged: set[str] = set()
        self._interval: float = 1.0 / self._rate.value
        self._beat: LockstepBeat | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        manifest = _load_manifest(self._manifest_raw)
        obs_config = manifest.get("observations") or {}
        action_type = manifest.get("action_type", "differential_drive")
        self._control_ack_timeout_s = float(manifest.get("control_ack_timeout_s", 5.0))

        endpoints = generate_transport_set()

        self._cmd_vel_pub = self.create_publisher(
            geometry_msgs.msg.Twist,
            self._cmd_vel_topic,
            _CMD_VEL_QOS,
        )

        from arena_planners.observations.pipeline import Pipeline  # noqa: PLC0415

        ns = self.get_namespace()
        self._obs_manager = Pipeline.from_config(
            config=obs_config,
            node=self,
            ns=ns,
            source_frame=self._source_frame,
            target_frame=self._target_frame,
            simulation_ns=self._simulation_namespace,
        )

        self._interval = await self._snap_interval()
        self._beat = LockstepBeat(self, "planner", grace_s=None, sleep=asyncio.sleep)

        self._proc = PlannerProcess(self._planner_command, endpoints)
        self._proc.start()

        self._data_push = ZmqPushTransport(endpoints.obs.endpoint, mode="bind", obs_policy=OBS_POLICY_LOSSLESS)
        self._data_pull = ZmqPullTransport(endpoints.action.endpoint, mode="bind", obs_policy=OBS_POLICY_LOSSLESS)
        self._control_push = ZmqPushTransport(endpoints.control.endpoint, mode="bind", control=True)
        self._control_pull = ZmqPullTransport(endpoints.ctrl_ack.endpoint, mode="bind", control=True)

        init_frame = Init(
            protocol_version=PROTOCOL_VERSION,
            schema_version=SCHEMA_VERSION,
            obs_schema=obs_config,
            action_schema={"action_type": action_type},
            planner_config={},
            run_id=self._run_id,
        )
        self._control_push.send_frame(encode_frame(init_frame))

        deadline = self.event_loop.time() + self._init_timeout.value
        init_ack_buf: bytes | None = None
        while True:
            if self._control_pull.poll(timeout_ms=200):
                init_ack_buf = self._control_pull.recv_frame()
                break
            if not self._proc.is_alive():
                rc = self._proc.returncode
                raise RuntimeError(
                    f"planner subprocess exited before init_ack (returncode={rc}); command={self._planner_command!r}"
                )
            if self.event_loop.time() >= deadline:
                raise TimeoutError(
                    f"planner did not send init_ack within {self._init_timeout.value}s; "
                    f"command={self._planner_command!r}"
                )
            await asyncio.sleep(0)
        init_ack = decode_frame(init_ack_buf)
        if not isinstance(init_ack, InitAck):
            raise ProtocolError(f"expected init_ack, got {init_ack!r}")

        caps: dict = init_ack.capabilities if isinstance(init_ack.capabilities, dict) else {}
        obs_policy_str: str = caps.get("obs_policy", OBS_POLICY_LOSSLESS)
        self._heartbeat_period_s = float(caps.get("heartbeat_period_s", 0.0))

        if obs_policy_str == OBS_POLICY_LATEST_ONLY:
            self._data_push.close()
            self._data_pull.close()
            policy = OBS_POLICY_LATEST_ONLY
            self._data_push = ZmqPushTransport(endpoints.obs.endpoint, mode="bind", obs_policy=policy)
            self._data_pull = ZmqPullTransport(endpoints.action.endpoint, mode="bind", obs_policy=policy)

        self._io_stop.clear()
        self._io_thread = threading.Thread(target=self._io_loop, name="planner-io", daemon=True)
        self._io_thread.start()

    async def teardown(self) -> None:
        if self._control_push is not None:
            try:
                self._control_push.send_frame(encode_frame(Shutdown()))
                if self._control_pull is not None and self._control_pull.poll(2000):
                    bye = decode_frame(self._control_pull.recv_frame())
                    if not isinstance(bye, Bye):
                        self.get_logger().warning(f"expected bye, got {bye!r}")
            except Exception as exc:
                self.get_logger().warning(f"shutdown handshake failed: {exc}")

        self._io_stop.set()
        if self._io_thread is not None:
            self._io_thread.join(timeout=3.0)

        rc: int = 0
        if self._proc is not None:
            rc = self._proc.stop()

        for sock in (self._data_push, self._data_pull, self._control_push, self._control_pull):
            if sock is not None:
                sock.close()

        if self._obs_manager is not None:
            self._obs_manager.shutdown()

        if rc not in (0, -signal.SIGTERM, -signal.SIGKILL):
            raise RuntimeError(f"planner subprocess exited with code {rc}")

    # ------------------------------------------------------------------
    # I/O thread
    # ------------------------------------------------------------------

    def _io_loop(self) -> None:
        """Dedicated I/O thread: drains _obs_queue to data_push, routes data_pull → action_queue,
        control_pull → ack_queue. Heartbeats arrive on control_pull and update last-seen."""
        loop = self.event_loop
        while not self._io_stop.is_set():
            try:
                buf = self._obs_queue.get_nowait()
                if self._data_push is not None:
                    self._data_push.send_frame(buf)
            except queue.Empty:
                pass

            if self._data_pull is not None and self._data_pull.poll(timeout_ms=10):
                try:
                    frame = decode_frame(self._data_pull.recv_frame())
                    if isinstance(frame, Action):
                        loop.call_soon_threadsafe(self._action_queue.put_nowait, frame)
                    else:
                        self.get_logger().warning(f"unexpected frame on data channel: {frame!r}")
                except Exception as exc:
                    self.get_logger().error(f"IO thread data decode error: {exc}")

            if self._control_pull is not None and self._control_pull.poll(timeout_ms=0):
                try:
                    frame = decode_frame(self._control_pull.recv_frame())
                    if isinstance(frame, Heartbeat):
                        self._last_heartbeat_ns = frame.monotonic_ns
                    else:
                        loop.call_soon_threadsafe(self._ack_queue.put_nowait, frame)
                except Exception as exc:
                    self.get_logger().error(f"IO thread control decode error: {exc}")

            if not self._proc or not self._proc.is_alive():
                self.get_logger().error("planner process died unexpectedly")
                loop.call_soon_threadsafe(self._action_queue.put_nowait, _PlannerDied())
                break

    # ------------------------------------------------------------------
    # Per-tick main loop
    # ------------------------------------------------------------------

    async def run_loop(self) -> None:
        """Drive the obs/action cycle under the `planner/<robot>` lockstep beat,
        pulsing each tick with the obs stamp its action was computed for."""
        assert self._obs_manager is not None
        assert self._beat is not None

        self.get_logger().info(
            f"run_loop entered interval_s={self._interval:.4f} action_timeout_s={self._action_timeout.value}"
        )

        try:
            with self.sim_time_rate(1.0 / self._interval) as (done, rate_events):
                loop = self.event_loop
                timing = {"sim": 0.0, "collect": 0.0, "action": 0.0}
                pulsed_at = loop.time()
                try:
                    await self._beat.acquire(self._interval)
                    while not done.is_set():
                        await rate_events.get()
                        tick_start = loop.time()
                        timing["sim"] += tick_start - pulsed_at

                        t = self.sim_time
                        features = self._filter_wire_features(self._obs_manager.collect())
                        self._seq += 1
                        collected_at = loop.time()
                        timing["collect"] += collected_at - tick_start

                        obs_frame = Obs(
                            t_sec=t.sec,
                            t_nanosec=t.nanosec,
                            seq=self._seq,
                            features=features,
                        )
                        self._obs_queue.put_nowait(encode_frame(obs_frame))

                        frame = await self._await_action()
                        timing["action"] += loop.time() - collected_at
                        if isinstance(frame, Action):
                            if frame.seq > self._seq:
                                self.get_logger().warning(
                                    f"discarding impossible Action seq={frame.seq} > sent={self._seq}"
                                )
                            else:
                                self._publish_action(frame, features)
                        else:
                            self.get_logger().warning(
                                f"unexpected non-Action frame on action queue seq={self._seq}: "
                                f"{type(frame).__name__}={frame!r}"
                            )
                        self._beat.pulse(stamp=t.to_msg())
                        pulsed_at = loop.time()
                        if self._seq % _TIMING_LOG_EVERY == 0:
                            avg = {k: 1e3 * v / _TIMING_LOG_EVERY for k, v in timing.items()}
                            gens = " ".join(
                                f"{k}={1e3 * v / _TIMING_LOG_EVERY:.1f}"
                                for k, v in self._obs_manager.generator_seconds.items()
                            )
                            self.get_logger().info(
                                f"tick wall ms (avg of {_TIMING_LOG_EVERY}): "
                                f"sim={avg['sim']:.1f} collect={avg['collect']:.1f} action={avg['action']:.1f} "
                                f"[{gens}]"
                            )
                            timing = dict.fromkeys(timing, 0.0)
                            self._obs_manager.generator_seconds.clear()
                finally:
                    await self._beat.release()
        except BaseException as exc:
            self.get_logger().error(f"run_loop crashed: {exc!r}")
            raise

    async def _await_action(self) -> Frame:
        """Wait for this obs's action: the timeout only warns (skipping a tick would
        deadlock a lockstep run against the frozen sim), stale actions are dropped."""
        while True:
            try:
                frame = await asyncio.wait_for(self._action_queue.get(), timeout=self._action_timeout.value)
            except TimeoutError:
                self.get_logger().warning(
                    f"no action received within {self._action_timeout.value}s (seq={self._seq}), still waiting"
                )
                continue
            if isinstance(frame, _PlannerDied):
                raise RuntimeError("planner process died")
            if isinstance(frame, Action) and frame.seq < self._seq:
                self.get_logger().warning(f"dropping stale Action seq={frame.seq} < sent={self._seq}")
                continue
            return frame

    async def _snap_interval(self) -> float:
        """Planner tick snapped to whole physics steps so lockstep chunks and ticks coincide."""
        requested = 1.0 / self._rate.value
        client = self.create_client_wrapper(GetParameters, "/arena/get_parameters", timeout=_PHYSICS_DT_TIMEOUT_S)
        dt: float | None = None
        try:
            response = await client.call_timeout(GetParameters.Request(names=["physics_dt"]))
        except Exception as exc:
            self.get_logger().warning(f"physics_dt read failed ({exc}), planner tick stays {requested:.4f}s")
        else:
            values = response.values if response is not None else []
            if values and values[0].type == ParameterType.PARAMETER_DOUBLE and values[0].double_value > 0.0:
                dt = values[0].double_value
            else:
                self.get_logger().warning(f"physics_dt unavailable on /arena, planner tick stays {requested:.4f}s")
        finally:
            self.destroy_client(client.client)
        if dt is None:
            return requested
        interval = max(1, round(requested / dt)) * dt
        if abs(interval - requested) > 1e-9:
            self.get_logger().info(
                f"planner_rate_hz={self._rate.value:g} snapped to {1.0 / interval:.3f} Hz "
                f"({interval:.4f}s, physics_dt={dt:g})"
            )
        return interval

    # ------------------------------------------------------------------
    # Outbound control messages
    # ------------------------------------------------------------------

    async def request_cancel(self) -> None:
        """Send Cancel and await CancelAck."""
        assert self._control_push is not None
        self._control_push.send_frame(encode_frame(Cancel()))
        frame = await self._drain_until(CancelAck, timeout=self._control_ack_timeout_s)
        if not isinstance(frame, CancelAck):
            raise ProtocolError(f"expected cancel_ack, got {frame!r}")

    async def request_reset(
        self,
        episode_id: str,
        initial_state: dict | None = None,
    ) -> None:
        """Send Reset and await ResetAck; warn if round-trip exceeds one sim step."""
        assert self._control_push is not None
        t_before = self.sim_time
        self._control_push.send_frame(encode_frame(Reset(episode_id=episode_id, initial_state=initial_state)))
        frame = await self._drain_until(ResetAck, timeout=self._control_ack_timeout_s)
        if not isinstance(frame, ResetAck):
            raise ProtocolError(f"expected reset_ack, got {frame!r}")
        t_after = self.sim_time
        step_s = self._interval
        elapsed = (t_after - t_before).to_seconds()
        if elapsed > step_s:
            self.get_logger().warning(f"reset round-trip {elapsed:.3f}s > one sim step {step_s:.3f}s")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _filter_wire_features(self, features: dict) -> dict:
        """Drop features that aren't msgpack-serializable (raw ROS messages)."""
        out: dict = {}
        for key, value in features.items():
            if value is None or isinstance(value, (bool, int, float, str, bytes, np.ndarray, list, tuple, dict)):
                out[key] = value
                continue
            if key not in self._dropped_features_logged:
                self._dropped_features_logged.add(key)
                self.get_logger().warning(
                    f"dropping non-serializable feature {key!r}: {type(value).__module__}.{type(value).__name__} "
                    "(only emitted on first occurrence per key)"
                )
        return out

    async def _drain_until(self, target: type, timeout: float) -> Frame:
        """Wait for a frame of `target` type from the ack queue, with timeout."""
        deadline = self.event_loop.time() + timeout
        while True:
            remaining = deadline - self.event_loop.time()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for {target.__name__}")
            frame = await asyncio.wait_for(self._ack_queue.get(), timeout=remaining)
            if isinstance(frame, target):
                return frame
            self.get_logger().warning(f"discarding unexpected frame while waiting for {target.__name__}: {frame!r}")

    def _publish_action(self, action: Action, features: dict) -> None:
        if self._cmd_vel_pub is None:
            return
        from arena_planners.bridge.projection import (  # noqa: PLC0415
            project_holonomic_to_diff_drive,
            unpack_differential_drive,
            unpack_omnidirectional,
        )

        msg = geometry_msgs.msg.Twist()
        if action.action_type == "differential_drive":
            try:
                v, omega = unpack_differential_drive(action.action)
            except ValueError as exc:
                self.get_logger().warning(str(exc))
                return
            msg.linear.x = v
            msg.angular.z = omega
        elif action.action_type == "omnidirectional":
            try:
                vx, vy, omega_in = unpack_omnidirectional(action.action)
            except ValueError as exc:
                self.get_logger().warning(str(exc))
                return
            robot_pose = features.get("robot_pose") if features else None
            if robot_pose is None or len(robot_pose) < 3:
                self.get_logger().warning("cannot apply omnidirectional action: robot_pose missing or malformed")
                return
            theta = float(robot_pose[2])
            if self._is_holonomic:
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                msg.linear.x = vx * cos_t + vy * sin_t
                msg.linear.y = -vx * sin_t + vy * cos_t
                msg.angular.z = omega_in
            else:
                step_dt_s = 1.0 / float(self._rate.value)
                v, omega = project_holonomic_to_diff_drive(vx, vy, omega_in, theta, step_dt_s)
                msg.linear.x = v
                msg.angular.z = omega
        else:
            self.get_logger().warning(f"unknown action_type {action.action_type!r}")
            return
        self._clamp_to_limits(msg)
        self._cmd_vel_pub.publish(msg)

    def _clamp_to_limits(self, msg: geometry_msgs.msg.Twist) -> None:
        """Scale the whole twist into the velocity envelope; one factor, so curvature is preserved."""
        if not self._velocity_limits:
            return

        def _scale(value: float, key: str) -> float:
            lo, hi = self._velocity_limits.get(key, (-math.inf, math.inf))
            if value > hi > 0.0:
                return hi / value
            if value < lo < 0.0:
                return lo / value
            return 1.0

        scale = min(
            _scale(msg.linear.x, "linear"),
            _scale(msg.linear.y, "lateral"),
            _scale(msg.angular.z, "angular"),
        )
        msg.linear.x *= scale
        msg.linear.y *= scale
        msg.angular.z *= scale
