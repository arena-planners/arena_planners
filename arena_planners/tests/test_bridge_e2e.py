"""End-to-end tests for the DRL planner bridge. Uses real ZMQ, real subprocess, real frames.

The test plays the edge-node role without rclpy: binds both sockets, spawns the toy
planner subprocess, drives the full protocol cycle, and verifies wire semantics.
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
    Bye,
    Cancel,
    CancelAck,
    Error,
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


def _planner_env(obs_ep: str, action_ep: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    existing_pypath = env.get("PYTHONPATH", "")
    parts = [_ARENA_PLANNERS_SRC]
    if existing_pypath:
        parts.append(existing_pypath)
    env["PYTHONPATH"] = ":".join(parts)
    env["ARENA_PLANNER_OBS_ENDPOINT"] = obs_ep
    env["ARENA_PLANNER_ACTION_ENDPOINT"] = action_ep
    if extra:
        env.update(extra)
    return env


def _recv_with_timeout(pull: ZmqPullTransport) -> bytes:
    if not pull.poll(_TIMEOUT_MS):
        pytest.fail("timed out waiting for frame from planner")
    return pull.recv_frame()


def _ipc_pair() -> tuple[str, str]:
    uid = uuid.uuid4().hex
    return (
        f"ipc:///tmp/arena_e2e_obs_{uid}.sock",
        f"ipc:///tmp/arena_e2e_act_{uid}.sock",
    )


@pytest.fixture()
def bridge_pair():
    obs_ep, action_ep = _ipc_pair()
    push = ZmqPushTransport(obs_ep, mode="bind")
    pull = ZmqPullTransport(action_ep, mode="bind")
    try:
        yield push, pull, (obs_ep, action_ep)
    finally:
        push.close()
        pull.close()


@pytest.fixture()
def toy_planner_proc(bridge_pair):
    push, pull, (obs_ep, action_ep) = bridge_pair
    env = _planner_env(obs_ep, action_ep)
    proc = subprocess.Popen(
        [sys.executable, _FIXTURE_PLANNER],
        env=env,
        start_new_session=True,
    )
    yield proc, push, pull
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


def _do_handshake(push: ZmqPushTransport, pull: ZmqPullTransport) -> InitAck:
    init = Init(
        protocol_version=PROTOCOL_VERSION,
        schema_version=1,
        obs_schema={},
        action_schema={"action_type": "differential_drive"},
        planner_config={},
        run_id=uuid.uuid4().hex,
    )
    push.send_frame(encode_frame(init))
    buf = _recv_with_timeout(pull)
    frame = decode_frame(buf)
    assert isinstance(frame, InitAck), f"expected InitAck, got {frame!r}"
    return frame


class TestInitHandshake:
    def test_init_handshake(self, toy_planner_proc):
        _, push, pull = toy_planner_proc
        ack = _do_handshake(push, pull)
        assert ack.capabilities.get("obs_policy") == "lossless"
        assert isinstance(ack.capabilities, dict)

    def test_protocol_version_echoed(self, toy_planner_proc):
        _, push, pull = toy_planner_proc
        init = Init(
            protocol_version=PROTOCOL_VERSION,
            schema_version=1,
            obs_schema={},
            action_schema={},
            planner_config={},
            run_id="test-run",
        )
        push.send_frame(encode_frame(init))
        buf = _recv_with_timeout(pull)
        frame = decode_frame(buf)
        assert isinstance(frame, InitAck)
        assert frame.capabilities.get("obs_policy") == "lossless"


class TestObsActionCycle:
    def test_obs_action_cycle(self, toy_planner_proc):
        _, push, pull = toy_planner_proc
        _do_handshake(push, pull)

        for i in range(5):
            obs = Obs(t_sec=100 + i, t_nanosec=i * 1000, seq=i, features={"x": float(i)})
            push.send_frame(encode_frame(obs))
            buf = _recv_with_timeout(pull)
            frame = decode_frame(buf)
            assert isinstance(frame, Action), f"cycle {i}: expected Action, got {frame!r}"
            assert frame.seq == i, f"seq mismatch at cycle {i}"
            assert frame.t_sec == 100 + i
            assert frame.t_nanosec == i * 1000
            assert frame.action_type == "differential_drive"
            assert frame.action == [0.1, 0.0]


class TestResetRoundTrip:
    def test_reset_round_trip(self, toy_planner_proc):
        _, push, pull = toy_planner_proc
        _do_handshake(push, pull)

        push.send_frame(encode_frame(Reset(episode_id="ep1", initial_state=None)))
        buf = _recv_with_timeout(pull)
        frame = decode_frame(buf)
        assert isinstance(frame, ResetAck), f"expected ResetAck, got {frame!r}"

        obs = Obs(t_sec=200, t_nanosec=0, seq=0, features={})
        push.send_frame(encode_frame(obs))
        buf = _recv_with_timeout(pull)
        frame = decode_frame(buf)
        assert isinstance(frame, Action)
        assert frame.action == [0.1, 0.0]


class TestCancelRoundTrip:
    def test_cancel_round_trip(self, toy_planner_proc):
        _, push, pull = toy_planner_proc
        _do_handshake(push, pull)

        push.send_frame(encode_frame(Cancel()))
        buf = _recv_with_timeout(pull)
        frame = decode_frame(buf)
        assert isinstance(frame, CancelAck), f"expected CancelAck, got {frame!r}"


class TestShutdownClean:
    def test_shutdown_clean(self, toy_planner_proc):
        proc, push, pull = toy_planner_proc
        _do_handshake(push, pull)

        push.send_frame(encode_frame(Shutdown()))
        buf = _recv_with_timeout(pull)
        frame = decode_frame(buf)
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
        push, pull, (obs_ep, action_ep) = bridge_pair
        env = _planner_env(obs_ep, action_ep, extra={"TOY_PLANNER_CRASH_AFTER": "3"})
        proc = subprocess.Popen(
            [sys.executable, _FIXTURE_PLANNER],
            env=env,
            start_new_session=True,
        )
        yield proc, push, pull
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
        proc, push, pull = crashy_proc
        _do_handshake(push, pull)

        # First 3 obs cycle cleanly (crash_after=3 means crash on step > 3).
        for i in range(3):
            obs = Obs(t_sec=i, t_nanosec=0, seq=i, features={})
            push.send_frame(encode_frame(obs))
            buf = _recv_with_timeout(pull)
            frame = decode_frame(buf)
            assert isinstance(frame, Action), f"cycle {i}: expected Action, got {frame!r}"

        # 4th obs triggers the crash: planner sends Error then exits.
        obs = Obs(t_sec=3, t_nanosec=0, seq=3, features={})
        push.send_frame(encode_frame(obs))
        buf = _recv_with_timeout(pull)
        frame = decode_frame(buf)
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
