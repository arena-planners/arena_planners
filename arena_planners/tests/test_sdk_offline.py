"""Offline tests for arena_planners.sdk: construction and dispatch logic."""

from __future__ import annotations

import os
import uuid

import pytest

from arena_planners.bridge.protocol import (
    Action,
    Cancel,
    CancelAck,
    Error,
    Obs,
    ProtocolError,
    Reset,
    ResetAck,
    Shutdown,
    decode_frame,
)
from arena_planners.sdk import PlannerSDK


def _never_bound_ipc() -> str:
    return f"ipc:///tmp/never_bound_{uuid.uuid4().hex}.sock"


def _activate(sdk: PlannerSDK) -> None:
    """Planners answer a standstill until the first Reset."""
    sdk._control_push.send_frame = lambda buf: None
    sdk._handle_control(Reset(episode_id="e1", initial_state=None), None, None)


def _make_sdk(
    *,
    action_type: str = "differential_drive",
    obs_policy: str = "lossless",
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
    manifest = {"action_type": action_type, "obs_policy": obs_policy, "heartbeat_period_s": heartbeat_period_s}
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

    def test_init_stores_obs_policy(self) -> None:
        sdk = _make_sdk(obs_policy="latest_only")
        assert sdk._obs_policy == "latest_only"
        _close(sdk)

    def test_init_merges_extra_capabilities(self) -> None:
        sdk = _make_sdk(capabilities={"model": "drlvo"})
        assert sdk._extra_capabilities == {"model": "drlvo"}
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
