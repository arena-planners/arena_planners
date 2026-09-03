"""Planner-side wire client for the DRL planner bridge. Pure Python, no ROS."""

from __future__ import annotations

import logging
import pathlib
import time
import typing

import yaml
import zmq

from arena_planners.bridge.protocol import (
    PROTOCOL_VERSION,
    Action,
    Bye,
    Cancel,
    CancelAck,
    Error,
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
from arena_planners.bridge.transport import (
    ZmqPullTransport,
    ZmqPushTransport,
    endpoints_from_env,
)

_log = logging.getLogger(__name__)

KNOWN_ACTION_TYPES: frozenset[str] = frozenset({"differential_drive", "omnidirectional"})
_ACTION_DIMS: dict[str, int] = {"differential_drive": 2, "omnidirectional": 3}

_DEFAULT_HEARTBEAT_PERIOD_S: float = 1.0


def load_manifest(path: str | pathlib.Path) -> dict:
    """Load a planner.yaml manifest as a dict."""
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


class PlannerSDK:
    """Wire client used by planner subprocess.

    Opens four ZMQ transports: data (obs/action) and control (ctrl/ctrl_ack).
    The main loop polls both, processes control frames first to keep reset /
    cancel latency bounded regardless of obs backlog.
    """

    def __init__(
        self,
        manifest: dict,
        capabilities: dict | None = None,
    ) -> None:
        self._action_type: str = manifest["action_type"]
        if self._action_type not in KNOWN_ACTION_TYPES:
            raise ValueError(f"manifest action_type {self._action_type!r} not in {sorted(KNOWN_ACTION_TYPES)}")
        self._obs_policy: str = manifest.get("obs_policy", "lossless")
        self._heartbeat_period_s: float = float(manifest.get("heartbeat_period_s", _DEFAULT_HEARTBEAT_PERIOD_S))
        self._extra_capabilities = capabilities or {}

        obs_ep, action_ep, control_ep, ctrl_ack_ep = endpoints_from_env()
        self._data_pull = ZmqPullTransport(obs_ep, self._obs_policy, mode="connect")
        self._data_push = ZmqPushTransport(action_ep, self._obs_policy, mode="connect")
        self._control_pull = ZmqPullTransport(control_ep, mode="connect", control=True)
        self._control_push = ZmqPushTransport(ctrl_ack_ep, mode="connect", control=True)

        self._poller = zmq.Poller()
        self._poller.register(self._data_pull.sock, zmq.POLLIN)
        self._poller.register(self._control_pull.sock, zmq.POLLIN)

        self._heartbeat_seq: int = 0
        self._last_heartbeat_ns: int = 0
        self._active = False

    def _send_control(self, frame: Frame) -> None:
        self._control_push.send_frame(encode_frame(frame))

    def _send_data(self, frame: Frame) -> None:
        self._data_push.send_frame(encode_frame(frame))

    def _emit_heartbeat_if_due(self) -> None:
        if self._heartbeat_period_s <= 0.0:
            return
        now_ns = time.monotonic_ns()
        if self._last_heartbeat_ns == 0 or (now_ns - self._last_heartbeat_ns) >= int(self._heartbeat_period_s * 1e9):
            self._heartbeat_seq += 1
            self._send_control(Heartbeat(seq=self._heartbeat_seq, monotonic_ns=now_ns))
            self._last_heartbeat_ns = now_ns

    def _handle_obs(
        self,
        frame: Obs,
        step_fn: typing.Callable[[dict], list[float]],
    ) -> None:
        if not self._active:
            standstill = [0.0] * _ACTION_DIMS[self._action_type]
            self._send_data(
                Action(
                    t_sec=frame.t_sec,
                    t_nanosec=frame.t_nanosec,
                    seq=frame.seq,
                    action_type=self._action_type,
                    action=standstill,
                )
            )
            return
        try:
            result = step_fn(frame.features)
        except Exception as exc:
            self._send_data(Error(code="step_failed", msg=str(exc), severity="error"))
            raise
        self._send_data(
            Action(
                t_sec=frame.t_sec,
                t_nanosec=frame.t_nanosec,
                seq=frame.seq,
                action_type=self._action_type,
                action=result,
            )
        )

    def _handle_control(
        self,
        frame: Frame,
        on_reset: typing.Callable[[str, dict | None], None] | None,
        on_cancel: typing.Callable[[], None] | None,
    ) -> bool:
        """Return True to keep the main loop running, False to shut down."""
        if isinstance(frame, Reset):
            if on_reset is not None:
                on_reset(frame.episode_id, frame.initial_state)
            self._active = True
            self._send_control(ResetAck())
            return True
        if isinstance(frame, Cancel):
            self._active = False
            if on_cancel is not None:
                on_cancel()
            self._send_control(CancelAck())
            return True
        if isinstance(frame, Shutdown):
            self._send_control(Bye())
            return False
        if isinstance(frame, Error):
            _log.error("received error from edge: code=%s msg=%s severity=%s", frame.code, frame.msg, frame.severity)
            if frame.severity == "fatal":
                raise ProtocolError(f"edge fatal error: {frame.code}: {frame.msg}")
            return True
        self._send_control(Error(code="bad_op", msg=f"unexpected control op: {frame.op!r}", severity="error"))
        return True

    def run(
        self,
        step_fn: typing.Callable[[dict], list[float]],
        on_reset: typing.Callable[[str, dict | None], None] | None = None,
        on_cancel: typing.Callable[[], None] | None = None,
    ) -> None:
        """Blocking event loop. Returns when a Shutdown frame is received."""
        try:
            self._handshake()
            while True:
                self._emit_heartbeat_if_due()
                events = dict(self._poller.poll(timeout=100))
                if self._control_pull.sock in events:
                    frame = decode_frame(self._control_pull.recv_frame())
                    if not self._handle_control(frame, on_reset, on_cancel):
                        return
                if self._data_pull.sock in events:
                    frame = decode_frame(self._data_pull.recv_frame())
                    if isinstance(frame, Obs):
                        self._handle_obs(frame, step_fn)
                    else:
                        self._send_data(Error(code="bad_op", msg=f"unexpected data op: {frame.op!r}", severity="error"))
        finally:
            self._data_pull.close()
            self._data_push.close()
            self._control_pull.close()
            self._control_push.close()

    def _handshake(self) -> None:
        """Receive Init on the control channel, send InitAck back."""
        init_frame = decode_frame(self._control_pull.recv_frame())
        if not isinstance(init_frame, Init):
            raise ProtocolError(f"expected Init, got {init_frame.op!r}")
        if init_frame.protocol_version != PROTOCOL_VERSION:
            self._send_control(
                Error(
                    code="version_mismatch",
                    msg=f"expected protocol_version={PROTOCOL_VERSION}, got {init_frame.protocol_version}",
                    severity="fatal",
                )
            )
            raise ProtocolError(
                f"protocol_version mismatch: expected {PROTOCOL_VERSION}, got {init_frame.protocol_version}"
            )
        caps: dict = {
            "obs_policy": self._obs_policy,
            "heartbeat_period_s": self._heartbeat_period_s,
            "streaming_actions": False,
            "supports_hot_reload": False,
            "requires_seq_validation": True,
            **self._extra_capabilities,
        }
        self._send_control(InitAck(capabilities=caps))


def main_loop(
    step_fn: typing.Callable[[dict], list[float]],
    manifest: dict,
    on_reset: typing.Callable[[str, dict | None], None] | None = None,
    on_cancel: typing.Callable[[], None] | None = None,
    capabilities: dict | None = None,
) -> None:
    """One-liner entry point."""
    PlannerSDK(manifest=manifest, capabilities=capabilities).run(step_fn, on_reset=on_reset, on_cancel=on_cancel)
