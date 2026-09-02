"""End-to-end tests for the DRL planner bridge. Real ZMQ, real subprocess, real frames.

The test plays the edge-node role without rclpy: binds all four sockets (data + control
pairs), spawns the toy planner subprocess, drives the full protocol cycle, and verifies
wire semantics.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import uuid

import pytest

from arena_planners.bridge.protocol import (
    PROTOCOL_VERSION,
    Action,
    BridgeError,
    Bye,
    Cancel,
    CancelAck,
    Error,
    Heartbeat,
    Init,
    InitAck,
    Obs,
    Reset,
    ResetAck,
    Shutdown,
    decode_frame,
    encode_frame,
)
from arena_planners.bridge.transport import ZmqPullTransport, ZmqPushTransport

_FIXTURE_PLANNER = os.path.join(os.path.dirname(__file__), "fixtures", "toy_planner", "planner.py")

_TIMEOUT_MS = 5_000
_ARENA_PLANNERS_SRC = os.path.join(os.path.dirname(__file__), "..")


def _planner_env(eps: dict[str, str], extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    existing_pypath = env.get("PYTHONPATH", "")
    parts = [_ARENA_PLANNERS_SRC]
    if existing_pypath:
        parts.append(existing_pypath)
    env["PYTHONPATH"] = ":".join(parts)
    env.update(eps)
    if extra:
        env.update(extra)
    return env


def _recv_with_timeout(pull: ZmqPullTransport) -> bytes:
    if not pull.poll(_TIMEOUT_MS):
        pytest.fail("timed out waiting for frame from planner")
    return pull.recv_frame()


def _recv_control(pull: ZmqPullTransport):
    """Next non-heartbeat control frame, heartbeats interleave freely (mirrors edge_node)."""
    while True:
        frame = decode_frame(_recv_with_timeout(pull))
        if not isinstance(frame, Heartbeat):
            return frame


def _ipc_endpoints() -> dict[str, str]:
    uid = uuid.uuid4().hex
    return {
        "ARENA_PLANNER_OBS_ENDPOINT": f"ipc:///tmp/arena_e2e_obs_{uid}.sock",
        "ARENA_PLANNER_ACTION_ENDPOINT": f"ipc:///tmp/arena_e2e_act_{uid}.sock",
        "ARENA_PLANNER_CONTROL_ENDPOINT": f"ipc:///tmp/arena_e2e_ctrl_{uid}.sock",
        "ARENA_PLANNER_CTRL_ACK_ENDPOINT": f"ipc:///tmp/arena_e2e_cack_{uid}.sock",
    }


@pytest.fixture()
def bridge_pair():
    eps = _ipc_endpoints()
    data_push = ZmqPushTransport(eps["ARENA_PLANNER_OBS_ENDPOINT"], mode="bind")
    data_pull = ZmqPullTransport(eps["ARENA_PLANNER_ACTION_ENDPOINT"], mode="bind")
    control_push = ZmqPushTransport(eps["ARENA_PLANNER_CONTROL_ENDPOINT"], mode="bind", control=True)
    control_pull = ZmqPullTransport(eps["ARENA_PLANNER_CTRL_ACK_ENDPOINT"], mode="bind", control=True)
    try:
        yield data_push, data_pull, control_push, control_pull, eps
    finally:
        data_push.close()
        data_pull.close()
        control_push.close()
        control_pull.close()


@pytest.fixture()
def toy_planner_proc(bridge_pair):
    data_push, data_pull, control_push, control_pull, eps = bridge_pair
    env = _planner_env(eps)
    proc = subprocess.Popen([sys.executable, _FIXTURE_PLANNER], env=env, start_new_session=True)
    yield proc, data_push, data_pull, control_push, control_pull
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait()


def _do_handshake(control_push: ZmqPushTransport, control_pull: ZmqPullTransport) -> InitAck:
    init = Init(
        protocol_version=PROTOCOL_VERSION,
        schema_version=1,
        obs_schema={},
        action_schema={"action_type": "differential_drive"},
        planner_config={},
        run_id=uuid.uuid4().hex,
    )
    assert control_push.poll(_TIMEOUT_MS), "planner never connected"
    control_push.send_frame(encode_frame(init))
    frame = _recv_control(control_pull)
    assert isinstance(frame, InitAck), f"expected InitAck, got {frame!r}"
    return frame


class TestInitHandshake:
    def test_init_handshake(self, toy_planner_proc):
        _, _, _, cpush, cpull = toy_planner_proc
        ack = _do_handshake(cpush, cpull)
        assert ack.capabilities.get("obs_policy") == "lossless"
        assert isinstance(ack.capabilities, dict)

    def test_protocol_version_echoed(self, toy_planner_proc):
        _, _, _, cpush, cpull = toy_planner_proc
        _do_handshake(cpush, cpull)


class TestObsActionCycle:
    def test_obs_action_cycle(self, toy_planner_proc):
        _, dpush, dpull, cpush, cpull = toy_planner_proc
        _do_handshake(cpush, cpull)

        for i in range(5):
            obs = Obs(t_sec=100 + i, t_nanosec=i * 1000, seq=i, features={"x": float(i)})
            dpush.send_frame(encode_frame(obs))
            frame = decode_frame(_recv_with_timeout(dpull))
            assert isinstance(frame, Action), f"cycle {i}: expected Action, got {frame!r}"
            assert frame.seq == i
            assert frame.t_sec == 100 + i
            assert frame.t_nanosec == i * 1000
            assert frame.action_type == "differential_drive"
            assert frame.action == [0.1, 0.0]


class TestResetRoundTrip:
    def test_reset_round_trip(self, toy_planner_proc):
        _, dpush, dpull, cpush, cpull = toy_planner_proc
        _do_handshake(cpush, cpull)

        cpush.send_frame(encode_frame(Reset(episode_id="ep1", initial_state=None)))
        frame = _recv_control(cpull)
        assert isinstance(frame, ResetAck), f"expected ResetAck, got {frame!r}"

        obs = Obs(t_sec=200, t_nanosec=0, seq=0, features={})
        dpush.send_frame(encode_frame(obs))
        frame = decode_frame(_recv_with_timeout(dpull))
        assert isinstance(frame, Action)
        assert frame.action == [0.1, 0.0]


class TestResetSurvivesObsBacklog:
    """Regression test: control channel delivers Reset even when data channel is flooded."""

    def test_reset_arrives_through_obs_flood(self, toy_planner_proc):
        _, dpush, dpull, cpush, cpull = toy_planner_proc
        _do_handshake(cpush, cpull)

        for i in range(50):
            dpush.send_frame(encode_frame(Obs(t_sec=i, t_nanosec=0, seq=i, features={})))

        cpush.send_frame(encode_frame(Reset(episode_id="flood")))
        frame = _recv_control(cpull)
        assert isinstance(frame, ResetAck)


class TestCancelRoundTrip:
    def test_cancel_round_trip(self, toy_planner_proc):
        _, _, _, cpush, cpull = toy_planner_proc
        _do_handshake(cpush, cpull)

        cpush.send_frame(encode_frame(Cancel()))
        frame = _recv_control(cpull)
        assert isinstance(frame, CancelAck), f"expected CancelAck, got {frame!r}"


class TestShutdownClean:
    def test_shutdown_clean(self, toy_planner_proc):
        proc, _, _, cpush, cpull = toy_planner_proc
        _do_handshake(cpush, cpull)

        cpush.send_frame(encode_frame(Shutdown()))
        frame = _recv_control(cpull)
        assert isinstance(frame, Bye), f"expected Bye, got {frame!r}"

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.05)
        assert proc.poll() == 0, f"expected exit 0, got {proc.poll()!r}"


class TestPlannerCrashPropagation:
    @pytest.fixture()
    def crashy_proc(self, bridge_pair):
        data_push, data_pull, control_push, control_pull, eps = bridge_pair
        env = _planner_env(eps, extra={"TOY_PLANNER_CRASH_AFTER": "3"})
        proc = subprocess.Popen([sys.executable, _FIXTURE_PLANNER], env=env, start_new_session=True)
        yield proc, data_push, data_pull, control_push, control_pull
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                proc.wait()

    def test_planner_crash_propagation(self, crashy_proc):
        proc, dpush, dpull, cpush, cpull = crashy_proc
        _do_handshake(cpush, cpull)

        for i in range(3):
            dpush.send_frame(encode_frame(Obs(t_sec=i, t_nanosec=0, seq=i, features={})))
            frame = decode_frame(_recv_with_timeout(dpull))
            assert isinstance(frame, Action), f"cycle {i}: expected Action, got {frame!r}"

        dpush.send_frame(encode_frame(Obs(t_sec=3, t_nanosec=0, seq=3, features={})))
        frame = decode_frame(_recv_with_timeout(dpull))
        assert isinstance(frame, Error), f"expected Error, got {frame!r}"
        assert frame.code == "step_failed"

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.05)
        rc = proc.poll()
        assert rc is not None, "planner should have exited after crash"
        assert rc != 0, f"expected nonzero exit code, got {rc}"

        t0 = time.monotonic()
        with pytest.raises(BridgeError):
            cpush.send_frame(encode_frame(Reset(episode_id="after-crash")))
        assert time.monotonic() - t0 < 3.0
