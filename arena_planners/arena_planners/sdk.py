"""Planner-side wire client for the DRL planner bridge. Pure Python, no ROS."""

from __future__ import annotations

import logging
import pathlib
import typing

import yaml

from arena_planners.bridge.protocol import (
    PROTOCOL_VERSION,
    Action,
    Bye,
    Cancel,
    CancelAck,
    Error,
    Frame,
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


def load_manifest(path: str | pathlib.Path) -> dict:
    """Load a planner.yaml manifest as a dict."""
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


class PlannerSDK:
    """Wire client used by planner subprocess. Opens ZMQ transports from env vars.

    The manifest dict carries `action_type` and `obs_policy`. Extra capabilities
    declared by the planner are merged into the init_ack `capabilities` field.
    """

    def __init__(
        self,
        manifest: dict,
        capabilities: dict | None = None,
    ) -> None:
        self._action_type: str = manifest["action_type"]
        self._obs_policy: str = manifest.get("obs_policy", "lossless")
        self._extra_capabilities = capabilities or {}

        obs_endpoint, action_endpoint = endpoints_from_env()
        self._pull = ZmqPullTransport(obs_endpoint, self._obs_policy, mode="connect")
        self._push = ZmqPushTransport(action_endpoint, self._obs_policy, mode="connect")

    def _send(self, frame: Frame) -> None:
        self._push.send_frame(encode_frame(frame))

    def _recv(self) -> Frame:
        return decode_frame(self._pull.recv_frame())

    def _handle_frame(
        self,
        frame: Frame,
        step_fn: typing.Callable[[dict], list[float]],
        on_reset: typing.Callable[[str, dict | None], None] | None,
        on_cancel: typing.Callable[[], None] | None,
    ) -> Frame | None:
        """Dispatch one inbound frame, return response frame or None for Shutdown."""
        if isinstance(frame, Obs):
            try:
                result = step_fn(frame.features)
            except Exception as exc:
                err = Error(code="step_failed", msg=str(exc))
                self._send(err)
                raise
            return Action(
                t_sec=frame.t_sec,
                t_nanosec=frame.t_nanosec,
                seq=frame.seq,
                action_type=self._action_type,
                action=result,
            )

        if isinstance(frame, Reset):
            if on_reset is not None:
                on_reset(frame.episode_id, frame.initial_state)
            return ResetAck()

        if isinstance(frame, Cancel):
            if on_cancel is not None:
                on_cancel()
            return CancelAck()

        if isinstance(frame, Shutdown):
            return None

        if isinstance(frame, Error):
            _log.error("received error from edge: code=%s msg=%s", frame.code, frame.msg)
            raise ProtocolError(f"edge error: {frame.code}: {frame.msg}")

        return Error(code="bad_op", msg=f"unexpected op in main loop: {frame.op!r}")

    def run(
        self,
        step_fn: typing.Callable[[dict], list[float]],
        on_reset: typing.Callable[[str, dict | None], None] | None = None,
        on_cancel: typing.Callable[[], None] | None = None,
    ) -> None:
        """Blocking event loop. step_fn(features) returns action floats. Returns on Shutdown."""
        try:
            init_frame = self._recv()
            if not isinstance(init_frame, Init):
                raise ProtocolError(f"expected Init, got {init_frame.op!r}")
            if init_frame.protocol_version != PROTOCOL_VERSION:
                self._send(
                    Error(
                        code="version_mismatch",
                        msg=(f"expected protocol_version={PROTOCOL_VERSION}, got {init_frame.protocol_version}"),
                    )
                )
                raise ProtocolError(
                    f"protocol_version mismatch: expected {PROTOCOL_VERSION}, got {init_frame.protocol_version}"
                )
            caps: dict = {"obs_policy": self._obs_policy, **self._extra_capabilities}
            self._send(InitAck(capabilities=caps))

            while True:
                frame = self._recv()
                response = self._handle_frame(frame, step_fn, on_reset, on_cancel)
                if response is None:
                    self._send(Bye())
                    return
                self._send(response)
        finally:
            self._pull.close()
            self._push.close()


def main_loop(
    step_fn: typing.Callable[[dict], list[float]],
    manifest: dict,
    on_reset: typing.Callable[[str, dict | None], None] | None = None,
    on_cancel: typing.Callable[[], None] | None = None,
    capabilities: dict | None = None,
) -> None:
    """One-liner entry point. Construct PlannerSDK from manifest and run."""
    PlannerSDK(manifest=manifest, capabilities=capabilities).run(
        step_fn,
        on_reset=on_reset,
        on_cancel=on_cancel,
    )
