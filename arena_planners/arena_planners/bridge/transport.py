"""ZMQ transport layer for the DRL planner bridge."""

from __future__ import annotations

import os
import platform
import typing
import uuid
from dataclasses import dataclass

import zmq

OBS_POLICY_LOSSLESS = "lossless"
OBS_POLICY_LATEST_ONLY = "latest_only"

ObsPolicy = typing.Literal["lossless", "latest_only"]


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
class TransportPair:
    """Bidirectional pair: one pipe for obs, one for actions."""

    obs: TransportEndpoint
    action: TransportEndpoint


def generate_pair(
    kind: typing.Literal["ipc", "tcp"] | None = None,
) -> TransportPair:
    """Return a fresh TransportPair with unique endpoints.

    Defaults to ipc on Linux/macOS, tcp on Windows.
    For tcp, returns port 0; the actual port is assigned at bind time.
    """
    if kind is None:
        kind = "tcp" if platform.system() == "Windows" else "ipc"

    if kind == "ipc":
        obs_ep = f"ipc:///tmp/arena_planner_{uuid.uuid4().hex}.sock"
        action_ep = f"ipc:///tmp/arena_planner_{uuid.uuid4().hex}.sock"
        return TransportPair(
            obs=TransportEndpoint(endpoint=obs_ep, kind="ipc"),
            action=TransportEndpoint(endpoint=action_ep, kind="ipc"),
        )

    obs_ep = "tcp://127.0.0.1:0"
    action_ep = "tcp://127.0.0.1:0"
    return TransportPair(
        obs=TransportEndpoint(endpoint=obs_ep, kind="tcp"),
        action=TransportEndpoint(endpoint=action_ep, kind="tcp"),
    )


def _apply_obs_policy(sock: zmq.Socket, obs_policy: ObsPolicy) -> None:
    if obs_policy == OBS_POLICY_LATEST_ONLY:
        sock.set(zmq.SNDHWM, 1)
        sock.set(zmq.RCVHWM, 1)
        sock.set(zmq.CONFLATE, 1)
    else:
        sock.set(zmq.SNDHWM, 1024)
        sock.set(zmq.RCVHWM, 1024)


class ZmqPushTransport:
    """PUSH socket, bind or connect role selectable per instance.

    Default mode="bind" matches original edge-node usage (obs pipe going to planner).
    mode="connect" is used by the planner side (sdk.py) where the edge already bound.
    """

    def __init__(
        self,
        endpoint: str,
        obs_policy: ObsPolicy = OBS_POLICY_LOSSLESS,
        *,
        mode: typing.Literal["bind", "connect"] = "bind",
        ctx: zmq.Context | None = None,
    ) -> None:
        self._ctx = ctx or zmq.Context.instance()
        self._owned_ctx = ctx is None
        self._mode = mode
        self._sock: zmq.Socket = self._ctx.socket(zmq.PUSH)
        _apply_obs_policy(self._sock, obs_policy)
        if mode == "bind":
            self._sock.bind(endpoint)
            self._bound_endpoint: str | None = self._sock.getsockopt_string(zmq.LAST_ENDPOINT)
        else:
            self._sock.connect(endpoint)
            self._bound_endpoint = None

    @property
    def bound_endpoint(self) -> str:
        """Actual endpoint after bind (resolves port 0 on tcp). Raises if mode='connect'."""
        if self._bound_endpoint is None:
            raise RuntimeError("bound_endpoint is not available when mode='connect'")
        return self._bound_endpoint

    def send_frame(self, buf: bytes) -> None:
        self._sock.send(buf)

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
    """PULL socket, bind or connect role selectable per instance.

    Default mode="connect" matches original edge-node usage (action pipe from planner).
    mode="bind" is used when the pull side needs to be the stable anchor point.
    """

    def __init__(
        self,
        endpoint: str,
        obs_policy: ObsPolicy = OBS_POLICY_LOSSLESS,
        *,
        mode: typing.Literal["bind", "connect"] = "connect",
        ctx: zmq.Context | None = None,
    ) -> None:
        self._ctx = ctx or zmq.Context.instance()
        self._owned_ctx = ctx is None
        self._mode = mode
        self._sock: zmq.Socket = self._ctx.socket(zmq.PULL)
        _apply_obs_policy(self._sock, obs_policy)
        if mode == "bind":
            self._sock.bind(endpoint)
            self._bound_endpoint: str | None = self._sock.getsockopt_string(zmq.LAST_ENDPOINT)
        else:
            self._sock.connect(endpoint)
            self._bound_endpoint = None

    @property
    def bound_endpoint(self) -> str:
        """Actual endpoint after bind (resolves port 0 on tcp). Raises if mode='connect'."""
        if self._bound_endpoint is None:
            raise RuntimeError("bound_endpoint is not available when mode='connect'")
        return self._bound_endpoint

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


def endpoints_from_env() -> tuple[str, str]:
    """Read obs and action endpoints from env vars.

    Returns (obs_endpoint, action_endpoint).
    Raises RuntimeError if either var is missing.
    """
    obs = os.environ.get("ARENA_PLANNER_OBS_ENDPOINT")
    action = os.environ.get("ARENA_PLANNER_ACTION_ENDPOINT")
    missing = [
        name
        for name, val in (
            ("ARENA_PLANNER_OBS_ENDPOINT", obs),
            ("ARENA_PLANNER_ACTION_ENDPOINT", action),
        )
        if val is None
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")
    return typing.cast(str, obs), typing.cast(str, action)
