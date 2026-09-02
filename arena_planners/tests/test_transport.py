"""Round-trip tests for the ZMQ transport layer. No rclpy, no mocking."""

from __future__ import annotations

import time

import pytest
import zmq

from arena_planners.bridge.protocol import BridgeError
from arena_planners.bridge.transport import (
    ZmqPullTransport,
    ZmqPushTransport,
    endpoints_from_env,
    generate_transport_set,
)

PAYLOAD = b"hello-frame"
RECV_TIMEOUT_MS = 2000


# ---------------------------------------------------------------------------
# IPC round-trip
# ---------------------------------------------------------------------------


def test_ipc_round_trip(tmp_path):
    sock_path = tmp_path / "test_arena.sock"
    endpoint = f"ipc://{sock_path}"

    ctx = zmq.Context()
    try:
        push = ZmqPushTransport(endpoint, ctx=ctx)
        pull = ZmqPullTransport(push.bound_endpoint, ctx=ctx)

        push.send_frame(PAYLOAD)
        assert pull.poll(RECV_TIMEOUT_MS), "recv timed out"
        received = pull.recv_frame()

        push.close()
        pull.close()
    finally:
        ctx.term()

    assert received == PAYLOAD


# ---------------------------------------------------------------------------
# TCP round-trip with port-0 bind
# ---------------------------------------------------------------------------


def test_tcp_round_trip():
    ctx = zmq.Context()
    try:
        push = ZmqPushTransport("tcp://127.0.0.1:0", ctx=ctx)
        bound = push.bound_endpoint
        assert bound.startswith("tcp://"), f"unexpected bound endpoint: {bound}"

        port_str = bound.rsplit(":", 1)[-1]
        assert port_str.isdigit(), f"port not numeric in {bound}"
        assigned_port = int(port_str)
        assert assigned_port > 0, "port 0 was not resolved after bind"

        pull = ZmqPullTransport(bound, ctx=ctx)

        push.send_frame(PAYLOAD)
        assert pull.poll(RECV_TIMEOUT_MS), "recv timed out"
        received = pull.recv_frame()

        push.close()
        pull.close()
    finally:
        ctx.term()

    assert received == PAYLOAD


# ---------------------------------------------------------------------------
# generate_pair returns distinct UUID socket paths for ipc
# ---------------------------------------------------------------------------


def test_generate_transport_set_ipc_distinct():
    ts = generate_transport_set("ipc")
    eps = {ts.obs.endpoint, ts.action.endpoint, ts.control.endpoint, ts.ctrl_ack.endpoint}
    assert len(eps) == 4, "all four endpoints must be unique"
    for ep in eps:
        assert ep.startswith("ipc:///tmp/arena_planner_")


def test_generate_transport_set_ipc_unique_across_calls():
    a = generate_transport_set("ipc")
    b = generate_transport_set("ipc")
    eps = {
        a.obs.endpoint,
        a.action.endpoint,
        a.control.endpoint,
        a.ctrl_ack.endpoint,
        b.obs.endpoint,
        b.action.endpoint,
        b.control.endpoint,
        b.ctrl_ack.endpoint,
    }
    assert len(eps) == 8


# ---------------------------------------------------------------------------
# endpoints_from_env
# ---------------------------------------------------------------------------


_ENV_KEYS = (
    "ARENA_PLANNER_OBS_ENDPOINT",
    "ARENA_PLANNER_ACTION_ENDPOINT",
    "ARENA_PLANNER_CONTROL_ENDPOINT",
    "ARENA_PLANNER_CTRL_ACK_ENDPOINT",
)


def test_endpoints_from_env_happy(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.setenv(k, f"ipc:///tmp/{k.lower()}.sock")
    obs, action, control, ctrl_ack = endpoints_from_env()
    assert obs == "ipc:///tmp/arena_planner_obs_endpoint.sock"
    assert action == "ipc:///tmp/arena_planner_action_endpoint.sock"
    assert control == "ipc:///tmp/arena_planner_control_endpoint.sock"
    assert ctrl_ack == "ipc:///tmp/arena_planner_ctrl_ack_endpoint.sock"


@pytest.mark.parametrize("missing", _ENV_KEYS)
def test_endpoints_from_env_missing_one(monkeypatch, missing):
    for k in _ENV_KEYS:
        if k == missing:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, f"ipc:///tmp/{k}.sock")
    with pytest.raises(RuntimeError, match=missing):
        endpoints_from_env()


def test_endpoints_from_env_all_missing(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError):
        endpoints_from_env()


# ---------------------------------------------------------------------------
# send with no peer raises instead of blocking
# ---------------------------------------------------------------------------


def test_send_without_peer_raises_bridge_error(tmp_path):
    sock_path = tmp_path / "no_peer.sock"
    endpoint = f"ipc://{sock_path}"

    ctx = zmq.Context()
    try:
        push = ZmqPushTransport(endpoint, control=True, ctx=ctx)
        t0 = time.monotonic()
        with pytest.raises(BridgeError, match=endpoint):
            push.send_frame(PAYLOAD)
        elapsed = time.monotonic() - t0
        push.close()
    finally:
        ctx.term()

    assert elapsed < 3.0


# ---------------------------------------------------------------------------
# obs_policy: latest_only uses CONFLATE
# ---------------------------------------------------------------------------


def test_latest_only_policy_receives_last_frame(tmp_path):
    """With latest_only, only the most recently enqueued frame should survive."""
    sock_path = tmp_path / "conflate.sock"
    endpoint = f"ipc://{sock_path}"

    ctx = zmq.Context()
    try:
        push = ZmqPushTransport(endpoint, obs_policy="latest_only", ctx=ctx)
        pull = ZmqPullTransport(push.bound_endpoint, obs_policy="latest_only", ctx=ctx)

        # Give ZMQ time to finish the connect handshake before sending
        time.sleep(0.05)

        push.send_frame(b"frame-1")
        push.send_frame(b"frame-2")
        push.send_frame(b"frame-3")

        # Allow the frames to propagate
        time.sleep(0.05)

        assert pull.poll(RECV_TIMEOUT_MS), "recv timed out"
        received = pull.recv_frame()

        push.close()
        pull.close()
    finally:
        ctx.term()

    # With CONFLATE=1 only the last frame survives
    assert received == b"frame-3"


# ---------------------------------------------------------------------------
# mode parameter: PUSH-connect / PULL-bind (roles reversed)
# ---------------------------------------------------------------------------


def test_pull_bind_push_connect_ipc(tmp_path):
    """PULL binds, PUSH connects; receiver is the stable anchor."""
    sock_path = tmp_path / "reversed.sock"
    endpoint = f"ipc://{sock_path}"

    ctx = zmq.Context()
    try:
        pull = ZmqPullTransport(endpoint, mode="bind", ctx=ctx)
        push = ZmqPushTransport(pull.bound_endpoint, mode="connect", ctx=ctx)

        push.send_frame(PAYLOAD)
        assert pull.poll(RECV_TIMEOUT_MS), "recv timed out"
        received = pull.recv_frame()

        push.close()
        pull.close()
    finally:
        ctx.term()

    assert received == PAYLOAD


def test_pull_bind_push_connect_tcp():
    """PULL binds on tcp port 0, PUSH connects to resolved port."""
    ctx = zmq.Context()
    try:
        pull = ZmqPullTransport("tcp://127.0.0.1:0", mode="bind", ctx=ctx)
        bound = pull.bound_endpoint
        assert bound.startswith("tcp://"), f"unexpected bound endpoint: {bound}"

        port_str = bound.rsplit(":", 1)[-1]
        assert port_str.isdigit(), f"port not numeric in {bound}"
        assert int(port_str) > 0, "port 0 was not resolved after bind"

        push = ZmqPushTransport(bound, mode="connect", ctx=ctx)

        push.send_frame(PAYLOAD)
        assert pull.poll(RECV_TIMEOUT_MS), "recv timed out"
        received = pull.recv_frame()

        push.close()
        pull.close()
    finally:
        ctx.term()

    assert received == PAYLOAD


def test_bound_endpoint_raises_on_connect_mode_push(tmp_path):
    """bound_endpoint must raise RuntimeError when PUSH mode='connect'."""
    sock_path = tmp_path / "pull_bind.sock"
    endpoint = f"ipc://{sock_path}"

    ctx = zmq.Context()
    try:
        pull = ZmqPullTransport(endpoint, mode="bind", ctx=ctx)
        push = ZmqPushTransport(pull.bound_endpoint, mode="connect", ctx=ctx)
        with pytest.raises(RuntimeError, match="mode='connect'"):
            _ = push.bound_endpoint
        push.close()
        pull.close()
    finally:
        ctx.term()


def test_bound_endpoint_raises_on_connect_mode_pull(tmp_path):
    """bound_endpoint must raise RuntimeError when PULL mode='connect'."""
    sock_path = tmp_path / "push_bind.sock"
    endpoint = f"ipc://{sock_path}"

    ctx = zmq.Context()
    try:
        push = ZmqPushTransport(endpoint, mode="bind", ctx=ctx)
        pull = ZmqPullTransport(push.bound_endpoint, mode="connect", ctx=ctx)
        with pytest.raises(RuntimeError, match="mode='connect'"):
            _ = pull.bound_endpoint
        push.close()
        pull.close()
    finally:
        ctx.term()


# ---------------------------------------------------------------------------
# context manager cleanup
# ---------------------------------------------------------------------------


def test_push_context_manager(tmp_path):
    sock_path = tmp_path / "ctx_mgr.sock"
    endpoint = f"ipc://{sock_path}"
    ctx = zmq.Context()
    try:
        with ZmqPushTransport(endpoint, ctx=ctx) as push:
            assert push.bound_endpoint.startswith("ipc://")
    finally:
        ctx.term()


def test_pull_context_manager(tmp_path):
    sock_path = tmp_path / "ctx_mgr_pull.sock"
    endpoint = f"ipc://{sock_path}"
    ctx = zmq.Context()
    try:
        with ZmqPushTransport(endpoint, ctx=ctx) as push:
            with ZmqPullTransport(push.bound_endpoint, ctx=ctx) as _pull:
                pass
    finally:
        ctx.term()
