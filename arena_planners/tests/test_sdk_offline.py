"""Offline tests for arena_planners.sdk: construction and dispatch logic."""

from __future__ import annotations

import math
import os
import threading
import uuid

import numpy as np
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
    ProtocolError,
    Reset,
    ResetAck,
    Shutdown,
    decode_frame,
    encode_frame,
)
from arena_planners.bridge.transport import ZmqPullTransport, ZmqPushTransport
from arena_planners.sdk import PlannerSDK, Step


def _never_bound_ipc() -> str:
    return f"ipc:///tmp/never_bound_{uuid.uuid4().hex}.sock"


def _activate(sdk: PlannerSDK) -> None:
    """Planners answer a standstill until the first Reset."""
    sdk._control_push.send_frame = lambda buf: None
    sdk._handle_control(Reset(episode_id="e1", initial_state=None), None, None)


def _make_sdk(
    *,
    action_type: str = "differential_drive",
    heartbeat_period_s: float = 0.0,
    capabilities: dict | None = None,
) -> PlannerSDK:
    for env in (
        "ARENA_PLANNER_OBS_ENDPOINT",
        "ARENA_PLANNER_ACTION_ENDPOINT",
        "ARENA_PLANNER_CONTROL_ENDPOINT",
        "ARENA_PLANNER_CTRL_ACK_ENDPOINT",
    ):
        os.environ[env] = _never_bound_ipc()
    manifest = {"action_type": action_type, "heartbeat_period_s": heartbeat_period_s}
    return PlannerSDK(manifest=manifest, capabilities=capabilities)


def _close(sdk: PlannerSDK) -> None:
    sdk._data_pull.close()
    sdk._data_push.close()
    sdk._control_pull.close()
    sdk._control_push.close()


class TestConstruction:
    def test_init_does_not_block(self) -> None:
        sdk = _make_sdk()
        _close(sdk)

    def test_init_stores_action_type(self) -> None:
        sdk = _make_sdk(action_type="omnidirectional")
        assert sdk._action_type == "omnidirectional"
        _close(sdk)

    def test_init_merges_extra_capabilities(self) -> None:
        sdk = _make_sdk(capabilities={"model": "drlvo"})
        assert sdk._extra_capabilities == {"model": "drlvo"}
        _close(sdk)

    def test_init_accepts_discrete(self) -> None:
        sdk = _make_sdk(action_type="discrete")
        assert sdk._action_type == "discrete"
        _close(sdk)

    @pytest.mark.parametrize("action_type", ["waypoints", "velocity_chunk"])
    def test_init_accepts_chunk_types(self, action_type: str) -> None:
        sdk = _make_sdk(action_type=action_type)
        assert sdk._action_type == action_type
        _close(sdk)

    def test_init_rejects_unknown_action_type(self) -> None:
        for env in (
            "ARENA_PLANNER_OBS_ENDPOINT",
            "ARENA_PLANNER_ACTION_ENDPOINT",
            "ARENA_PLANNER_CONTROL_ENDPOINT",
            "ARENA_PLANNER_CTRL_ACK_ENDPOINT",
        ):
            os.environ[env] = _never_bound_ipc()
        with pytest.raises(ValueError, match="action_type"):
            PlannerSDK(manifest={"action_type": "tractor_beam"})


class TestHandleObs:
    def test_obs_sends_action_on_data_channel(self) -> None:
        sdk = _make_sdk(action_type="differential_drive")
        _activate(sdk)
        captured: list[bytes] = []
        sdk._data_push.send_frame = lambda buf: captured.append(buf)

        sdk._handle_obs(
            Obs(t_sec=10, t_nanosec=500, seq=3, features={"laser": [1.0, 2.0]}),
            lambda f: [0.5, 0.1],
        )
        _close(sdk)

        assert len(captured) == 1
        response = decode_frame(captured[0])
        assert isinstance(response, Action)
        assert response.t_sec == 10
        assert response.t_nanosec == 500
        assert response.seq == 3
        assert response.action_type == "differential_drive"
        assert response.action == [0.5, 0.1]

    def test_obs_step_fn_raise_emits_error_and_reraises(self) -> None:
        sdk = _make_sdk()
        _activate(sdk)
        captured: list[bytes] = []
        sdk._data_push.send_frame = lambda buf: captured.append(buf)

        with pytest.raises(ValueError, match="boom"):
            sdk._handle_obs(Obs(seq=0, features={}), lambda f: (_ for _ in ()).throw(ValueError("boom")))
        _close(sdk)

        assert len(captured) == 1
        err = decode_frame(captured[0])
        assert isinstance(err, Error)
        assert err.code == "step_failed"


class TestHandleControl:
    def test_reset_calls_on_reset_and_acks(self) -> None:
        sdk = _make_sdk()
        captured: list[bytes] = []
        sdk._control_push.send_frame = lambda buf: captured.append(buf)
        calls: list[tuple] = []

        keep_going = sdk._handle_control(
            Reset(episode_id="ep_42", initial_state={"pos": [0.0, 0.0]}),
            lambda eid, state: calls.append((eid, state)),
            None,
        )
        _close(sdk)

        assert keep_going is True
        assert calls == [("ep_42", {"pos": [0.0, 0.0]})]
        assert isinstance(decode_frame(captured[0]), ResetAck)

    def test_reset_without_callback_still_acks(self) -> None:
        sdk = _make_sdk()
        captured: list[bytes] = []
        sdk._control_push.send_frame = lambda buf: captured.append(buf)

        sdk._handle_control(Reset(episode_id="ep_0"), None, None)
        _close(sdk)

        assert isinstance(decode_frame(captured[0]), ResetAck)

    def test_cancel_calls_on_cancel_and_acks(self) -> None:
        sdk = _make_sdk()
        captured: list[bytes] = []
        sdk._control_push.send_frame = lambda buf: captured.append(buf)
        calls: list[int] = []

        sdk._handle_control(Cancel(), None, lambda: calls.append(1))
        _close(sdk)

        assert calls == [1]
        assert isinstance(decode_frame(captured[0]), CancelAck)

    def test_shutdown_returns_false_and_bye(self) -> None:
        sdk = _make_sdk()
        captured: list[bytes] = []
        sdk._control_push.send_frame = lambda buf: captured.append(buf)

        keep_going = sdk._handle_control(Shutdown(), None, None)
        _close(sdk)

        assert keep_going is False
        from arena_planners.bridge.protocol import Bye

        assert isinstance(decode_frame(captured[0]), Bye)

    def test_fatal_error_raises(self) -> None:
        sdk = _make_sdk()
        with pytest.raises(ProtocolError, match="fatal"):
            sdk._handle_control(Error(code="boom", msg="x", severity="fatal"), None, None)
        _close(sdk)

    def test_nonfatal_error_keeps_loop(self) -> None:
        sdk = _make_sdk()
        keep_going = sdk._handle_control(Error(code="boom", msg="x", severity="warn"), None, None)
        _close(sdk)
        assert keep_going is True


def _edge_sockets(tmp_path) -> dict:
    endpoints = {
        "ARENA_PLANNER_OBS_ENDPOINT": f"ipc://{tmp_path}/obs.sock",
        "ARENA_PLANNER_ACTION_ENDPOINT": f"ipc://{tmp_path}/action.sock",
        "ARENA_PLANNER_CONTROL_ENDPOINT": f"ipc://{tmp_path}/control.sock",
        "ARENA_PLANNER_CTRL_ACK_ENDPOINT": f"ipc://{tmp_path}/ctrl_ack.sock",
    }
    os.environ.update(endpoints)
    return {
        "obs": ZmqPushTransport(endpoints["ARENA_PLANNER_OBS_ENDPOINT"], mode="bind"),
        "action": ZmqPullTransport(endpoints["ARENA_PLANNER_ACTION_ENDPOINT"], mode="bind"),
        "control": ZmqPushTransport(endpoints["ARENA_PLANNER_CONTROL_ENDPOINT"], mode="bind", control=True),
        "ctrl_ack": ZmqPullTransport(endpoints["ARENA_PLANNER_CTRL_ACK_ENDPOINT"], mode="bind", control=True),
    }


def _recv(transport: ZmqPullTransport) -> object:
    assert transport.poll(2000)
    return decode_frame(transport.recv_frame())


class TestPlannerConfig:
    def test_init_planner_config_reaches_on_init(self, tmp_path) -> None:
        edge = _edge_sockets(tmp_path)
        sdk = PlannerSDK(manifest={"action_type": "differential_drive", "heartbeat_period_s": 0.0})
        received: list[dict] = []
        thread = threading.Thread(target=sdk.run, args=(lambda _f: [0.0, 0.0],), kwargs={"on_init": received.append})
        thread.start()
        config = {"device": "cpu", "max_linear": 0.3}
        try:
            edge["control"].send_frame(encode_frame(Init(protocol_version=PROTOCOL_VERSION, planner_config=config)))
            assert isinstance(_recv(edge["ctrl_ack"]), InitAck)
            edge["control"].send_frame(encode_frame(Shutdown()))
            assert isinstance(_recv(edge["ctrl_ack"]), Bye)
            thread.join(timeout=2.0)
        finally:
            for transport in edge.values():
                transport.close()
        assert not thread.is_alive()
        assert received == [config]
        assert sdk.planner_config == config


class TestSignal:
    @pytest.mark.parametrize(("result", "signal"), [(Step([0.0, 0.0], signal="arrived"), "arrived"), ([0.1, 0.0], "")])
    def test_step_result_sets_action_signal(self, tmp_path, result: object, signal: str) -> None:
        edge = _edge_sockets(tmp_path)
        sdk = PlannerSDK(manifest={"action_type": "differential_drive", "heartbeat_period_s": 0.0})
        thread = threading.Thread(target=sdk.run, args=(lambda _f: result,))
        thread.start()
        try:
            edge["control"].send_frame(encode_frame(Init(protocol_version=PROTOCOL_VERSION)))
            assert isinstance(_recv(edge["ctrl_ack"]), InitAck)
            edge["control"].send_frame(encode_frame(Reset(episode_id="e1")))
            assert isinstance(_recv(edge["ctrl_ack"]), ResetAck)
            edge["obs"].send_frame(encode_frame(Obs(seq=1, features={})))
            action = _recv(edge["action"])
            edge["control"].send_frame(encode_frame(Shutdown()))
            assert isinstance(_recv(edge["ctrl_ack"]), Bye)
            thread.join(timeout=2.0)
        finally:
            for transport in edge.values():
                transport.close()
        assert isinstance(action, Action)
        assert action.seq == 1
        assert action.signal == signal


def test_step_defaults() -> None:
    assert Step() == Step(action=[], command="", signal="", amount=0.0, chunk=[])
    assert Step([0.1, 0.2], signal="arrived") == Step(action=[0.1, 0.2], command="", signal="arrived")
    assert Step([0.1, 0.2], "forward", "arrived", 0.5, [[1.0, 0.0]]) == Step(
        action=[0.1, 0.2], command="forward", signal="arrived", amount=0.5, chunk=[[1.0, 0.0]]
    )


class TestDiscrete:
    @staticmethod
    def _connect(tmp_path, action_type: str = "discrete") -> tuple[dict, PlannerSDK]:
        edge = _edge_sockets(tmp_path)
        return edge, PlannerSDK(manifest={"action_type": action_type, "heartbeat_period_s": 0.0})

    @staticmethod
    def _close_all(edge: dict, sdk: PlannerSDK) -> None:
        _close(sdk)
        for transport in edge.values():
            transport.close()

    def _reset(self, edge: dict, sdk: PlannerSDK) -> None:
        sdk._handle_control(Reset(episode_id="e1"), None, None)
        assert isinstance(_recv(edge["ctrl_ack"]), ResetAck)

    def test_inactive_standstill_is_empty_hold(self, tmp_path) -> None:
        edge, sdk = self._connect(tmp_path)
        try:
            sdk._handle_obs(Obs(seq=4, features={}), lambda _f: Step(command="forward"))
            action = _recv(edge["action"])
        finally:
            self._close_all(edge, sdk)
        assert isinstance(action, Action)
        assert (action.seq, action.action_type, action.action, action.command) == (4, "discrete", [], "")

    @pytest.mark.parametrize(
        ("step", "command", "signal"),
        [
            (Step(command="forward"), "forward", ""),
            (Step(command="left"), "left", ""),
            (Step(command="right"), "right", ""),
            (Step(signal="arrived"), "", "arrived"),
            (Step(command="forward", amount=0.75), "forward", ""),
            (Step(command="left", amount=30), "left", ""),
        ],
    )
    def test_command_reaches_action(self, tmp_path, step: Step, command: str, signal: str) -> None:
        edge, sdk = self._connect(tmp_path)
        try:
            self._reset(edge, sdk)
            sdk._handle_obs(Obs(seq=7, features={}), lambda _f: step)
            action = _recv(edge["action"])
        finally:
            self._close_all(edge, sdk)
        assert isinstance(action, Action)
        assert (action.seq, action.action_type, action.action) == (7, "discrete", [])
        assert (action.command, action.signal) == (command, signal)
        assert action.amount == float(step.amount)
        assert isinstance(action.amount, float)
        assert action.chunk == []

    @pytest.mark.parametrize(
        ("action_type", "result", "error"),
        [
            ("discrete", Step(command="jump"), ValueError),
            ("discrete", Step(command="Forward"), ValueError),
            ("discrete", [0.0, 0.0], TypeError),
            ("discrete", Step([0.1, 0.0], command="forward"), ValueError),
            ("differential_drive", Step([0.1, 0.0], command="forward"), ValueError),
            ("discrete", Step(command="forward", amount=-0.25), ValueError),
            ("discrete", Step(command="left", amount=-30.0), ValueError),
            ("discrete", Step(command="right", amount=360.0), ValueError),
            ("discrete", Step(command="forward", amount=math.nan), ValueError),
            ("discrete", Step(amount=0.5), ValueError),
            ("discrete", Step(command="forward", chunk=[[0.5, 0.0]]), ValueError),
            ("differential_drive", Step([0.1, 0.0], amount=0.5), ValueError),
            ("differential_drive", Step([0.1, 0.0], chunk=[[0.1, 0.0]]), ValueError),
            ("omnidirectional", Step([0.1, 0.0, 0.0], chunk=[[0.1, 0.0]]), ValueError),
        ],
    )
    def test_invalid_result_is_step_failure(self, tmp_path, action_type: str, result: object, error: type) -> None:
        edge, sdk = self._connect(tmp_path, action_type)
        try:
            self._reset(edge, sdk)
            with pytest.raises(error):
                sdk._handle_obs(Obs(seq=1, features={}), lambda _f: result)
            frame = _recv(edge["action"])
        finally:
            self._close_all(edge, sdk)
        assert isinstance(frame, Error)
        assert frame.code == "step_failed"

    def test_run_loop_sends_command(self, tmp_path) -> None:
        edge, sdk = self._connect(tmp_path)
        thread = threading.Thread(target=sdk.run, args=(lambda _f: Step(command="right"),))
        thread.start()
        try:
            edge["control"].send_frame(encode_frame(Init(protocol_version=PROTOCOL_VERSION)))
            assert isinstance(_recv(edge["ctrl_ack"]), InitAck)
            edge["control"].send_frame(encode_frame(Reset(episode_id="e1")))
            assert isinstance(_recv(edge["ctrl_ack"]), ResetAck)
            edge["obs"].send_frame(encode_frame(Obs(seq=1, features={"robot_pose": [0.0, 0.0, 0.0]})))
            action = _recv(edge["action"])
            edge["control"].send_frame(encode_frame(Shutdown()))
            assert isinstance(_recv(edge["ctrl_ack"]), Bye)
            thread.join(timeout=2.0)
        finally:
            for transport in edge.values():
                transport.close()
        assert not thread.is_alive()
        assert isinstance(action, Action)
        assert (action.action_type, action.action, action.command) == ("discrete", [], "right")


class TestChunk:
    @staticmethod
    def _exchange(tmp_path, action_type: str, result: object) -> object:
        edge = _edge_sockets(tmp_path)
        sdk = PlannerSDK(manifest={"action_type": action_type, "heartbeat_period_s": 0.0})
        try:
            sdk._handle_control(Reset(episode_id="e1"), None, None)
            assert isinstance(_recv(edge["ctrl_ack"]), ResetAck)
            try:
                sdk._handle_obs(Obs(seq=5, features={}), lambda _f: result)
            except (TypeError, ValueError) as exc:
                frame = _recv(edge["action"])
                assert isinstance(frame, Error)
                assert frame.code == "step_failed"
                return exc
            return _recv(edge["action"])
        finally:
            _close(sdk)
            for transport in edge.values():
                transport.close()

    @pytest.mark.parametrize(
        ("action_type", "chunk"),
        [
            ("waypoints", [[0.5, 0.0], [1.0, 0.2, 0.1]]),
            ("waypoints", [(1, 2)]),
            ("velocity_chunk", [[0.2, 0.1], [0.3, -0.1]]),
            ("waypoints", []),
            ("velocity_chunk", []),
        ],
    )
    def test_chunk_reaches_action(self, tmp_path, action_type: str, chunk: list) -> None:
        action = self._exchange(tmp_path, action_type, Step(chunk=chunk, signal="arrived"))
        assert isinstance(action, Action)
        assert (action.seq, action.action_type, action.action, action.command) == (5, action_type, [], "")
        assert action.chunk == [[float(c) for c in entry] for entry in chunk]
        assert all(isinstance(c, float) for entry in action.chunk for c in entry)
        assert (action.amount, action.signal) == (0.0, "arrived")

    def test_numpy_chunk_arrives_as_lists(self, tmp_path) -> None:
        chunk = np.array([[0.5, 0.0], [1.0, 0.25]], dtype=np.float32)
        action = self._exchange(tmp_path, "waypoints", Step(chunk=chunk))
        assert isinstance(action, Action)
        assert action.chunk == [[0.5, 0.0], [1.0, 0.25]]

    @pytest.mark.parametrize(
        ("action_type", "result", "error", "match"),
        [
            ("waypoints", [[0.5, 0.0]], TypeError, "Step"),
            ("velocity_chunk", [0.2, 0.0], TypeError, "Step"),
            ("waypoints", Step(command="forward"), ValueError, "command"),
            ("velocity_chunk", Step(command="left", chunk=[[0.1, 0.0]]), ValueError, "command"),
            ("waypoints", Step([0.1, 0.0]), ValueError, "action vector"),
            ("waypoints", Step(chunk=[[1.0, 0.0]], amount=0.5), ValueError, "amount"),
            ("waypoints", Step(chunk=[[1.0]]), ValueError, "yaw"),
            ("waypoints", Step(chunk=[[1.0, 0.0, 0.0, 0.0]]), ValueError, "yaw"),
            ("velocity_chunk", Step(chunk=[[0.1, 0.0, 0.0]]), ValueError, "omega"),
            ("velocity_chunk", Step(chunk=[[0.1]]), ValueError, "omega"),
            ("waypoints", Step(chunk=[[math.nan, 0.0]]), ValueError, "finite"),
        ],
    )
    def test_invalid_result_is_step_failure(
        self, tmp_path, action_type: str, result: object, error: type, match: str
    ) -> None:
        exc = self._exchange(tmp_path, action_type, result)
        assert isinstance(exc, error)
        assert match in str(exc)

    @pytest.mark.parametrize("action_type", ["waypoints", "velocity_chunk"])
    def test_inactive_standstill_is_empty_chunk(self, tmp_path, action_type: str) -> None:
        edge = _edge_sockets(tmp_path)
        sdk = PlannerSDK(manifest={"action_type": action_type, "heartbeat_period_s": 0.0})
        try:
            sdk._handle_obs(Obs(seq=4, features={}), lambda _f: Step(chunk=[[1.0, 0.0]]))
            action = _recv(edge["action"])
        finally:
            _close(sdk)
            for transport in edge.values():
                transport.close()
        assert isinstance(action, Action)
        assert (action.seq, action.action_type, action.action, action.chunk) == (4, action_type, [], [])

    def test_run_loop_sends_chunk(self, tmp_path) -> None:
        edge = _edge_sockets(tmp_path)
        sdk = PlannerSDK(manifest={"action_type": "waypoints", "heartbeat_period_s": 0.0})
        thread = threading.Thread(target=sdk.run, args=(lambda _f: Step(chunk=[[0.3, 0.0], [0.6, 0.1]]),))
        thread.start()
        try:
            edge["control"].send_frame(encode_frame(Init(protocol_version=PROTOCOL_VERSION)))
            assert isinstance(_recv(edge["ctrl_ack"]), InitAck)
            edge["control"].send_frame(encode_frame(Reset(episode_id="e1")))
            assert isinstance(_recv(edge["ctrl_ack"]), ResetAck)
            edge["obs"].send_frame(encode_frame(Obs(seq=1, features={"robot_pose": [0.0, 0.0, 0.0]})))
            action = _recv(edge["action"])
            edge["control"].send_frame(encode_frame(Shutdown()))
            assert isinstance(_recv(edge["ctrl_ack"]), Bye)
            thread.join(timeout=2.0)
        finally:
            for transport in edge.values():
                transport.close()
        assert not thread.is_alive()
        assert isinstance(action, Action)
        assert (action.action_type, action.chunk) == ("waypoints", [[0.3, 0.0], [0.6, 0.1]])
