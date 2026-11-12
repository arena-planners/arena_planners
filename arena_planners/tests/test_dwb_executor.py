"""DWB executor: paths from discrete moves and waypoint chunks, goal polling (no ROS)."""

from __future__ import annotations

import math

import pytest

from arena_planners.bridge.dwb import DwbGoal, chunk_path, densify, move_path, nominal_timeout_s, path_length


def test_forward_is_straight_along_heading() -> None:
    p = move_path("forward", 0.0, (1.0, 2.0, math.pi / 2), 0.25, 15.0)
    assert p[0][:2] == pytest.approx((1.0, 2.0))
    assert p[-1][:2] == pytest.approx((1.0, 2.25))
    assert path_length(p) == pytest.approx(0.25)
    assert all(abs(q[2] - math.pi / 2) < 1e-9 for q in p)
    assert len(p) >= 5  # densified at 5 cm
    assert path_length(move_path("forward", 0.75, (0, 0, 0), 0.25, 15.0)) == pytest.approx(0.75)


def test_turns_keep_position_and_turn_the_heading() -> None:
    left = move_path("left", 0.0, (0.0, 0.0, 0.0), 0.25, 15.0)
    assert left[-1] == pytest.approx((0.0, 0.0, math.radians(15)))
    right = move_path("right", 30.0, (0.0, 0.0, math.pi - 0.1), 0.25, 15.0)
    assert right[-1][2] == pytest.approx(math.pi - 0.1 - math.radians(30))
    wrap = move_path("left", 30.0, (0.0, 0.0, math.pi - 0.1), 0.25, 15.0)
    assert -math.pi <= wrap[-1][2] <= math.pi
    with pytest.raises(ValueError):
        move_path("jump", 0.0, (0, 0, 0), 0.25, 15.0)


def test_chunk_goes_to_the_map_frame() -> None:
    # robot at (2, 3) facing +y: robot-frame (1, 0) is map (2, 4), robot-frame (1, 1) is map (1, 4)
    p = chunk_path([[1.0, 0.0], [1.0, 1.0]], (2.0, 3.0, math.pi / 2))
    assert p[0][:2] == pytest.approx((2.0, 3.0))
    assert p[-1][:2] == pytest.approx((1.0, 4.0))
    assert path_length(p) == pytest.approx(2.0)
    assert p[-1][2] == pytest.approx(math.pi)  # heading of the last segment


def test_densify_and_timeout() -> None:
    assert densify([], 0.0) == []
    assert densify([(1.0, 1.0)], 0.3) == [(1.0, 1.0, 0.3)]
    t = nominal_timeout_s(move_path("forward", 0.5, (0, 0, 0), 0.25, 15), v_max=0.5)
    assert t == pytest.approx(3 * 1.0 + 2.0)
    assert nominal_timeout_s(move_path("left", 90, (0, 0, 0), 0.25, 15), w_max=1.0) == pytest.approx(
        3 * math.pi / 2 + 2.0
    )


def test_goal_times_out_without_an_answer() -> None:
    task = pytest.importorskip("rclpy.task")
    g = DwbGoal("x", [], 0.0, 1.0, send_future=task.Future())
    assert g.step(0.5) == (False, False)
    assert g.step(1.5) == (True, True)


def test_goal_rejected_by_the_server() -> None:
    task = pytest.importorskip("rclpy.task")
    fut = task.Future()
    fut.set_result(None)
    g = DwbGoal("x", [], 0.0, 1.0, send_future=fut)
    assert g.step(0.1) == (True, False) and g.rejected


def test_chunk_released_after_execute_s() -> None:
    task = pytest.importorskip("rclpy.task")
    g = DwbGoal("chunk", [], 0.0, 9.0, release_at=0.4, send_future=task.Future())
    assert g.step(0.3) == (False, False)
    assert g.step(0.41) == (True, False)
