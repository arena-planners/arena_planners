"""ZMQ transport layer for the DRL planner bridge."""

from __future__ import annotations

import os
import platform
import typing
import uuid
from dataclasses import dataclass

import zmq

from .protocol import BridgeError

OBS_POLICY_LOSSLESS = "lossless"
OBS_POLICY_LATEST_ONLY = "latest_only"

ObsPolicy = typing.Literal["lossless", "latest_only"]

_CONTROL_HWM = 16
_SEND_TIMEOUT_MS = 1000


class Transport(typing.Protocol):
    """One directional framed pipe (send or recv side)."""

    def send_frame(self, buf: bytes) -> None: ...

    def recv_frame(self) -> bytes: ...

    def close(self) -> None: ...

    def poll(self, timeout_ms: int) -> bool: ...


@dataclass(frozen=True)
class TransportEndpoint:
    """One side of one pipe."""

    endpoint: str
    kind: typing.Literal["ipc", "tcp"]


@dataclass(frozen=True)
class TransportSet:
    """Four endpoints: data (obs/action) carry tick-rate traffic, control (ctrl/ctrl_ack)
    carry session frames. Control channel is always lossless; data channel honours
    obs_policy."""

    obs: TransportEndpoint
    action: TransportEndpoint
    control: TransportEndpoint
    ctrl_ack: TransportEndpoint


def _fresh_endpoint(kind: typing.Literal["ipc", "tcp"]) -> TransportEndpoint:
    if kind == "ipc":
        return TransportEndpoint(endpoint=f"ipc:///tmp/arena_planner_{uuid.uuid4().hex}.sock", kind="ipc")
    return TransportEndpoint(endpoint="tcp://127.0.0.1:0", kind="tcp")


def generate_transport_set(kind: typing.Literal["ipc", "tcp"] | None = None) -> TransportSet:
    """Fresh four-endpoint set. Defaults to ipc on Linux/macOS, tcp on Windows."""
    if kind is None:
        kind = "tcp" if platform.system() == "Windows" else "ipc"
    return TransportSet(
        obs=_fresh_endpoint(kind),
        action=_fresh_endpoint(kind),
        control=_fresh_endpoint(kind),
        ctrl_ack=_fresh_endpoint(kind),
    )


def _apply_data_policy(sock: zmq.Socket, obs_policy: ObsPolicy) -> None:
    if obs_policy == OBS_POLICY_LATEST_ONLY:
        sock.set(zmq.SNDHWM, 1)
        sock.set(zmq.RCVHWM, 1)
        sock.set(zmq.CONFLATE, 1)
    else:
        sock.set(zmq.SNDHWM, 1024)
        sock.set(zmq.RCVHWM, 1024)


def _apply_control_policy(sock: zmq.Socket) -> None:
    sock.set(zmq.SNDHWM, _CONTROL_HWM)
    sock.set(zmq.RCVHWM, _CONTROL_HWM)


class ZmqPushTransport:
    """PUSH socket, bind or connect role selectable per instance.

    `obs_policy` toggles data-channel CONFLATE; `control=True` overrides to
    always-lossless control semantics regardless of obs_policy.
    """

    def __init__(
        self,
        endpoint: str,
        obs_policy: ObsPolicy = OBS_POLICY_LOSSLESS,
        *,
        mode: typing.Literal["bind", "connect"] = "bind",
        control: bool = False,
        ctx: zmq.Context | None = None,
    ) -> None:
        self._ctx = ctx or zmq.Context.instance()
        self._owned_ctx = ctx is None
        self._mode = mode
        self._sock: zmq.Socket = self._ctx.socket(zmq.PUSH)
        if control:
            _apply_control_policy(self._sock)
        else:
            _apply_data_policy(self._sock, obs_policy)
        if mode == "bind":
            self._sock.bind(endpoint)
            self._bound_endpoint: str | None = self._sock.getsockopt_string(zmq.LAST_ENDPOINT)
        else:
            self._sock.connect(endpoint)
            self._bound_endpoint = None
        self._endpoint = endpoint

    @property
    def bound_endpoint(self) -> str:
        """Actual endpoint after bind (resolves port 0 on tcp). Raises if mode='connect'."""
        if self._bound_endpoint is None:
            raise RuntimeError("bound_endpoint is not available when mode='connect'")
        return self._bound_endpoint

    @property
    def sock(self) -> zmq.Socket:
        """Underlying socket, for use with zmq.Poller."""
        return self._sock

    def send_frame(self, buf: bytes) -> None:
        if not self._sock.poll(_SEND_TIMEOUT_MS, zmq.POLLOUT):
            raise BridgeError(f"no peer accepting frames on {self._endpoint} within {_SEND_TIMEOUT_MS}ms")
        try:
            self._sock.send(buf, zmq.NOBLOCK)
        except zmq.Again as exc:
            raise BridgeError(f"send on {self._endpoint} would block") from exc

    def recv_frame(self) -> bytes:
        raise NotImplementedError("PUSH socket cannot receive")

    def poll(self, timeout_ms: int) -> bool:
        return bool(self._sock.poll(timeout_ms, zmq.POLLOUT))

    def close(self) -> None:
        self._sock.close(linger=0)

    def __enter__(self) -> ZmqPushTransport:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class ZmqPullTransport:
    """PULL socket, bind or connect role selectable per instance."""

    def __init__(
        self,
        endpoint: str,
        obs_policy: ObsPolicy = OBS_POLICY_LOSSLESS,
        *,
        mode: typing.Literal["bind", "connect"] = "connect",
        control: bool = False,
        ctx: zmq.Context | None = None,
    ) -> None:
        self._ctx = ctx or zmq.Context.instance()
        self._owned_ctx = ctx is None
        self._mode = mode
        self._sock: zmq.Socket = self._ctx.socket(zmq.PULL)
        if control:
            _apply_control_policy(self._sock)
        else:
            _apply_data_policy(self._sock, obs_policy)
        if mode == "bind":
            self._sock.bind(endpoint)
            self._bound_endpoint: str | None = self._sock.getsockopt_string(zmq.LAST_ENDPOINT)
        else:
            self._sock.connect(endpoint)
            self._bound_endpoint = None

    @property
    def bound_endpoint(self) -> str:
        if self._bound_endpoint is None:
            raise RuntimeError("bound_endpoint is not available when mode='connect'")
        return self._bound_endpoint

    @property
    def sock(self) -> zmq.Socket:
        return self._sock

    def send_frame(self, buf: bytes) -> None:
        raise NotImplementedError("PULL socket cannot send")

    def recv_frame(self) -> bytes:
        return self._sock.recv()

    def poll(self, timeout_ms: int) -> bool:
        return bool(self._sock.poll(timeout_ms, zmq.POLLIN))

    def close(self) -> None:
        self._sock.close(linger=0)

    def __enter__(self) -> ZmqPullTransport:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


_ENV_VARS = (
    "ARENA_PLANNER_OBS_ENDPOINT",
    "ARENA_PLANNER_ACTION_ENDPOINT",
    "ARENA_PLANNER_CONTROL_ENDPOINT",
    "ARENA_PLANNER_CTRL_ACK_ENDPOINT",
)


def endpoints_from_env() -> tuple[str, str, str, str]:
    """Read all four endpoints (obs, action, control, ctrl_ack) from env vars."""
    values = tuple(os.environ.get(name) for name in _ENV_VARS)
    missing = [name for index, name in enumerate(_ENV_VARS) if values[index] is None]
    if missing:
        raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")
    obs, action, control, ctrl_ack = values
    return obs, action, control, ctrl_ack


def env_from_endpoints(transport_set: TransportSet) -> dict[str, str]:
    """Build the env-var dict to inject into a planner subprocess."""
    return {
        "ARENA_PLANNER_OBS_ENDPOINT": transport_set.obs.endpoint,
        "ARENA_PLANNER_ACTION_ENDPOINT": transport_set.action.endpoint,
        "ARENA_PLANNER_CONTROL_ENDPOINT": transport_set.control.endpoint,
        "ARENA_PLANNER_CTRL_ACK_ENDPOINT": transport_set.ctrl_ack.endpoint,
    }
