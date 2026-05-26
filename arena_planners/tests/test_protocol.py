"""Round-trip tests for arena_planners.bridge.protocol."""

from __future__ import annotations

import msgpack
import msgpack_numpy
import numpy as np
import pytest

from arena_planners.bridge.protocol import (
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

msgpack_numpy.patch()


def _roundtrip(frame):
    return decode_frame(encode_frame(frame))


def test_init_roundtrip():
    frame = Init(
        protocol_version=1,
        schema_version=1,
        obs_schema={"lidar": "float32[360]"},
        action_schema={"type": "differential_drive"},
        planner_config={"timeout_sec": 5},
        run_id="abc-123",
    )
    result = _roundtrip(frame)
    assert result.op == "init"
    assert result.protocol_version == 1
    assert result.schema_version == 1
    assert result.obs_schema == {"lidar": "float32[360]"}
    assert result.action_schema == {"type": "differential_drive"}
    assert result.planner_config == {"timeout_sec": 5}
    assert result.run_id == "abc-123"


def test_init_ack_roundtrip():
    frame = InitAck(capabilities={"obs_policy": "lossless", "supports_reset": True})
    result = _roundtrip(frame)
    assert result.op == "init_ack"
    assert result.capabilities == {"obs_policy": "lossless", "supports_reset": True}


def test_reset_roundtrip():
    frame = Reset(episode_id="ep-42", initial_state={"x": 1.0, "y": 2.0})
    result = _roundtrip(frame)
    assert result.op == "reset"
    assert result.episode_id == "ep-42"
    assert result.initial_state == {"x": 1.0, "y": 2.0}


def test_reset_none_initial_state():
    frame = Reset(episode_id="ep-0", initial_state=None)
    result = _roundtrip(frame)
    assert result.initial_state is None


def test_reset_ack_roundtrip():
    frame = ResetAck()
    result = _roundtrip(frame)
    assert result.op == "reset_ack"


def test_obs_roundtrip_scalars():
    frame = Obs(t_sec=100, t_nanosec=500_000_000, seq=7, features={"scalar": 3.14})
    result = _roundtrip(frame)
    assert result.op == "obs"
    assert result.t_sec == 100
    assert result.t_nanosec == 500_000_000
    assert result.seq == 7


def test_obs_numpy_roundtrip():
    arr_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    arr_1d = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    frame = Obs(
        t_sec=10,
        t_nanosec=0,
        seq=1,
        features={"scan": arr_2d, "goal": arr_1d},
    )
    result = _roundtrip(frame)
    assert result.op == "obs"
    np.testing.assert_array_equal(result.features["scan"], arr_2d)
    np.testing.assert_array_equal(result.features["goal"], arr_1d)
    assert result.features["scan"].dtype == np.float32
    assert result.features["goal"].dtype == np.float64


def test_action_roundtrip():
    frame = Action(
        t_sec=10,
        t_nanosec=0,
        seq=1,
        action_type="differential_drive",
        action=[0.5, 0.0, 0.2],
    )
    result = _roundtrip(frame)
    assert result.op == "action"
    assert result.t_sec == 10
    assert result.seq == 1
    assert result.action_type == "differential_drive"
    assert result.action == [0.5, 0.0, 0.2]


def test_action_omnidirectional():
    frame = Action(
        t_sec=1,
        t_nanosec=0,
        seq=2,
        action_type="omnidirectional",
        action=[0.3, 0.1, -0.1],
    )
    result = _roundtrip(frame)
    assert result.action_type == "omnidirectional"
    assert result.action == [0.3, 0.1, -0.1]


def test_action_manipulator():
    frame = Action(
        t_sec=2,
        t_nanosec=0,
        seq=3,
        action_type="manipulator",
        action=[0.1, -0.2, 0.3, 0.4, 0.5, 0.6],
    )
    result = _roundtrip(frame)
    assert result.action_type == "manipulator"


def test_action_humanoid():
    frame = Action(
        t_sec=3,
        t_nanosec=0,
        seq=4,
        action_type="humanoid",
        action=[0.0, 0.5, -0.3],
    )
    result = _roundtrip(frame)
    assert result.action_type == "humanoid"


def test_cancel_roundtrip():
    frame = Cancel()
    result = _roundtrip(frame)
    assert result.op == "cancel"


def test_cancel_ack_roundtrip():
    frame = CancelAck()
    result = _roundtrip(frame)
    assert result.op == "cancel_ack"


def test_shutdown_roundtrip():
    frame = Shutdown()
    result = _roundtrip(frame)
    assert result.op == "shutdown"


def test_bye_roundtrip():
    frame = Bye()
    result = _roundtrip(frame)
    assert result.op == "bye"


def test_error_roundtrip():
    frame = Error(code="TIMEOUT", msg="planner did not respond within deadline")
    result = _roundtrip(frame)
    assert result.op == "error"
    assert result.code == "TIMEOUT"
    assert result.msg == "planner did not respond within deadline"


def test_unknown_op_raises_protocol_error():
    buf = msgpack.packb({"op": "nonsense", "data": 42}, use_bin_type=True)
    with pytest.raises(ProtocolError):
        decode_frame(buf)


def test_malformed_payload_raises_protocol_error():
    buf = b"\x00\x01\x02bad bytes"
    with pytest.raises(ProtocolError):
        decode_frame(buf)


def test_non_mapping_payload_raises_protocol_error():
    buf = msgpack.packb([1, 2, 3], use_bin_type=True)
    with pytest.raises(ProtocolError):
        decode_frame(buf)
