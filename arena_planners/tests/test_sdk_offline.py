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
)
from arena_planners.sdk import PlannerSDK


def _make_sdk(
    tmp_obs: str,
    tmp_action: str,
    *,
    action_type: str = "differential_drive",
    obs_policy: str = "lossless",
    capabilities: dict | None = None,
) -> PlannerSDK:
    os.environ["ARENA_PLANNER_OBS_ENDPOINT"] = tmp_obs
    os.environ["ARENA_PLANNER_ACTION_ENDPOINT"] = tmp_action
    return PlannerSDK(
        manifest={"action_type": action_type, "obs_policy": obs_policy},
        capabilities=capabilities,
    )


def _never_bound_ipc() -> str:
    return f"ipc:///tmp/never_bound_{uuid.uuid4().hex}.sock"


class TestConstruction:
    def test_init_does_not_block(self) -> None:
        obs_ep = _never_bound_ipc()
        action_ep = _never_bound_ipc()
        sdk = _make_sdk(obs_ep, action_ep)
        sdk._pull.close()
        sdk._push.close()

    def test_init_stores_action_type(self) -> None:
        obs_ep = _never_bound_ipc()
        action_ep = _never_bound_ipc()
        sdk = _make_sdk(obs_ep, action_ep, action_type="omnidirectional")
        assert sdk._action_type == "omnidirectional"
        sdk._pull.close()
        sdk._push.close()

    def test_init_stores_obs_policy(self) -> None:
        obs_ep = _never_bound_ipc()
        action_ep = _never_bound_ipc()
        sdk = _make_sdk(obs_ep, action_ep, obs_policy="latest_only")
        assert sdk._obs_policy == "latest_only"
        sdk._pull.close()
        sdk._push.close()

    def test_init_merges_extra_capabilities(self) -> None:
        obs_ep = _never_bound_ipc()
        action_ep = _never_bound_ipc()
        sdk = _make_sdk(obs_ep, action_ep, capabilities={"model": "drlvo"})
        assert sdk._extra_capabilities == {"model": "drlvo"}
        sdk._pull.close()
        sdk._push.close()


class TestHandleFrame:
    def _sdk(self) -> PlannerSDK:
        obs_ep = _never_bound_ipc()
        action_ep = _never_bound_ipc()
        sdk = _make_sdk(obs_ep, action_ep, action_type="differential_drive")
        return sdk

    def _step_fn(self, features: dict) -> list[float]:
        return [0.5, 0.1]

    def test_obs_returns_action(self) -> None:
        sdk = self._sdk()
        frame = Obs(t_sec=10, t_nanosec=500, seq=3, features={"laser": [1.0, 2.0]})
        response = sdk._handle_frame(frame, self._step_fn, None, None)
        sdk._pull.close()
        sdk._push.close()
        assert isinstance(response, Action)
        assert response.t_sec == 10
        assert response.t_nanosec == 500
        assert response.seq == 3
        assert response.action_type == "differential_drive"
        assert response.action == [0.5, 0.1]

    def test_obs_step_fn_receives_features(self) -> None:
        sdk = self._sdk()
        received: list[dict] = []

        def capturing_step(features: dict) -> list[float]:
            received.append(features)
            return [0.0, 0.0]

        frame = Obs(t_sec=1, t_nanosec=0, seq=0, features={"x": 42})
        sdk._handle_frame(frame, capturing_step, None, None)
        sdk._pull.close()
        sdk._push.close()
        assert received == [{"x": 42}]

    def test_obs_step_fn_raises_re_raises(self) -> None:
        sdk = self._sdk()
        sent_frames: list[bytes] = []
        sdk._push.send_frame = lambda buf: sent_frames.append(buf)

        def failing_step(features: dict) -> list[float]:
            raise ValueError("boom")

        frame = Obs(t_sec=0, t_nanosec=0, seq=0, features={})
        with pytest.raises(ValueError, match="boom"):
            sdk._handle_frame(frame, failing_step, None, None)
        sdk._pull.close()
        sdk._push.close()
        assert len(sent_frames) == 1

    def test_reset_calls_on_reset_and_returns_reset_ack(self) -> None:
        sdk = self._sdk()
        calls: list[tuple] = []

        def on_reset(episode_id: str, initial_state: dict | None) -> None:
            calls.append((episode_id, initial_state))

        frame = Reset(episode_id="ep_42", initial_state={"pos": [0.0, 0.0]})
        response = sdk._handle_frame(frame, self._step_fn, on_reset, None)
        sdk._pull.close()
        sdk._push.close()
        assert isinstance(response, ResetAck)
        assert calls == [("ep_42", {"pos": [0.0, 0.0]})]

    def test_reset_without_on_reset_returns_reset_ack(self) -> None:
        sdk = self._sdk()
        frame = Reset(episode_id="ep_0")
        response = sdk._handle_frame(frame, self._step_fn, None, None)
        sdk._pull.close()
        sdk._push.close()
        assert isinstance(response, ResetAck)

    def test_cancel_calls_on_cancel_and_returns_cancel_ack(self) -> None:
        sdk = self._sdk()
        calls: list[int] = []

        def on_cancel() -> None:
            calls.append(1)

        frame = Cancel()
        response = sdk._handle_frame(frame, self._step_fn, None, on_cancel)
        sdk._pull.close()
        sdk._push.close()
        assert isinstance(response, CancelAck)
        assert calls == [1]

    def test_cancel_without_on_cancel_returns_cancel_ack(self) -> None:
        sdk = self._sdk()
        frame = Cancel()
        response = sdk._handle_frame(frame, self._step_fn, None, None)
        sdk._pull.close()
        sdk._push.close()
        assert isinstance(response, CancelAck)

    def test_shutdown_returns_none(self) -> None:
        sdk = self._sdk()
        frame = Shutdown()
        response = sdk._handle_frame(frame, self._step_fn, None, None)
        sdk._pull.close()
        sdk._push.close()
        assert response is None

    def test_incoming_error_raises_protocol_error(self) -> None:
        sdk = self._sdk()
        frame = Error(code="some_error", msg="something went wrong")
        with pytest.raises(ProtocolError, match="some_error"):
            sdk._handle_frame(frame, self._step_fn, None, None)
        sdk._pull.close()
        sdk._push.close()

    def test_unknown_op_returns_error_frame(self) -> None:
        sdk = self._sdk()
        frame = Action(t_sec=0, t_nanosec=0, seq=0, action_type="differential_drive", action=[0.0])
        response = sdk._handle_frame(frame, self._step_fn, None, None)
        sdk._pull.close()
        sdk._push.close()
        assert isinstance(response, Error)
        assert response.code == "bad_op"
