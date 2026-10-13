from __future__ import annotations

import math

import numpy as np
import pytest

from arena_planners.geometry import lookahead_clear_on_path, lookahead_on_path, world_to_robot_frame

L_PATH = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 2.0), (3.0, 2.0)])


def _path(*points: tuple[float, ...]) -> np.ndarray:
    return np.array(points, dtype=np.float64)


def test_lookahead_on_empty_path_is_none():
    assert lookahead_on_path(np.zeros((0, 2)), np.array([0.0, 0.0]), 1.0) is None


def test_lookahead_on_single_point_path_is_that_point():
    assert lookahead_on_path(_path((4.0, -2.0)), np.array([0.0, 0.0]), 1.0) == (4.0, -2.0)


def test_lookahead_measures_from_the_projection_onto_the_path():
    assert lookahead_on_path(_path((0.0, 0.0), (10.0, 0.0)), np.array([2.0, 1.0]), 3.0) == pytest.approx((5.0, 0.0))


def test_lookahead_clamps_a_robot_behind_the_path_start_to_the_start():
    assert lookahead_on_path(_path((0.0, 0.0), (10.0, 0.0)), np.array([-5.0, 0.0, 0.7]), 2.0) == pytest.approx(
        (2.0, 0.0)
    )


def test_lookahead_walks_around_a_corner():
    assert lookahead_on_path(L_PATH, np.array([0.0, 0.0]), 2.0) == pytest.approx((1.0, 1.0))
    assert lookahead_on_path(L_PATH, np.array([0.0, 0.0]), 3.5) == pytest.approx((1.5, 2.0))


def test_lookahead_beyond_the_path_returns_the_last_point():
    assert lookahead_on_path(L_PATH, np.array([0.0, 0.0]), 100.0) == pytest.approx((3.0, 2.0))


def test_lookahead_starts_on_the_segment_nearest_the_robot():
    assert lookahead_on_path(L_PATH, np.array([1.2, 1.0]), 0.5) == pytest.approx((1.0, 1.5))


def test_lookahead_zero_at_a_segment_end_returns_that_end():
    assert lookahead_on_path(_path((0.0, 0.0), (1.0, 0.0)), np.array([1.0, 0.0]), 0.0) == pytest.approx((1.0, 0.0))


def test_lookahead_skips_a_zero_length_segment():
    assert lookahead_on_path(_path((0.0, 0.0), (0.0, 0.0), (4.0, 0.0)), np.array([1.0, 1.0]), 1.0) == pytest.approx(
        (2.0, 0.0)
    )


def test_lookahead_ignores_columns_beyond_xy():
    path = np.array([(0.0, 0.0, 9.0, 9.0), (0.0, 5.0, -9.0, 9.0)])
    assert lookahead_on_path(path, np.array([0.0, 0.0]), 1.0) == pytest.approx((0.0, 1.0))


def test_lookahead_returns_python_floats():
    x, y = lookahead_on_path(L_PATH, np.array([0.0, 0.0]), 2.0)
    assert type(x) is float and type(y) is float


def test_clear_lookahead_on_empty_path_is_none():
    assert lookahead_clear_on_path(np.zeros((0, 2)), np.array([0.0, 0.0]), 1.0, lambda x, y: True) is None


def test_clear_lookahead_takes_the_last_candidate_within_lookahead_when_free():
    carrot = lookahead_clear_on_path(_path((0.0, 0.0), (10.0, 0.0)), np.array([0.0, 0.0]), 2.0, lambda x, y: True)
    assert carrot == pytest.approx((1.8, 0.0))


def test_clear_lookahead_stops_before_a_blocked_chord():
    carrot = lookahead_clear_on_path(_path((0.0, 0.0), (10.0, 0.0)), np.array([0.0, 0.0]), 2.0, lambda x, y: x < 1.0)
    assert carrot == pytest.approx((0.8, 0.0))


def test_clear_lookahead_falls_back_to_the_minimum_carrot_when_nothing_is_clear():
    carrot = lookahead_clear_on_path(
        _path((0.0, 0.0), (10.0, 0.0)), np.array([0.0, 0.0]), 2.0, lambda x, y: False, minimum=0.5
    )
    assert carrot == pytest.approx((0.5, 0.0))


def test_clear_lookahead_below_minimum_returns_the_minimum_carrot_unchecked():
    calls: list[tuple[float, float]] = []

    def is_clear(x: float, y: float) -> bool:
        calls.append((x, y))
        return True

    carrot = lookahead_clear_on_path(_path((0.0, 0.0), (10.0, 0.0)), np.array([0.0, 0.0]), 0.2, is_clear)
    assert carrot == pytest.approx((0.3, 0.0))
    assert calls == []


def _clear_of_pillar(x: float, y: float) -> bool:
    return not (0.9 < x < 1.1 and y < 0.05)


def test_clear_lookahead_keeps_scanning_past_a_blocked_candidate():
    path = _path((0.0, 0.0), (1.0, 0.0), (1.0, 3.0))
    assert lookahead_on_path(path, np.array([0.0, 0.0]), 1.05) == pytest.approx((1.0, 0.05))
    carrot = lookahead_clear_on_path(path, np.array([0.0, 0.0]), 1.4, _clear_of_pillar)
    assert carrot == pytest.approx((1.0, 0.3))


def test_clear_lookahead_samples_the_chord_at_five_centimeters():
    calls: list[tuple[float, float]] = []

    def is_clear(x: float, y: float) -> bool:
        calls.append((x, y))
        return True

    lookahead_clear_on_path(_path((0.0, 0.0), (10.0, 0.0)), np.array([0.0, 0.0]), 1.0, is_clear, minimum=1.0)
    assert len(calls) == 20
    assert calls[0] == pytest.approx((0.0, 0.0))
    assert calls[-1] == pytest.approx((1.0, 0.0))


def test_world_to_robot_frame_identity_pose_is_a_translation():
    assert world_to_robot_frame((3.0, 4.0), np.array([1.0, 1.0, 0.0])) == pytest.approx((2.0, 3.0))


def test_world_to_robot_frame_rotates_into_the_heading():
    pose = np.array([1.0, 2.0, math.pi / 2])
    assert world_to_robot_frame((1.0, 3.0), pose) == pytest.approx((1.0, 0.0))
    assert world_to_robot_frame((2.0, 2.0), pose) == pytest.approx((0.0, -1.0))


def test_world_to_robot_frame_point_behind_a_reversed_robot_is_ahead():
    assert world_to_robot_frame((-1.0, 0.0), np.array([0.0, 0.0, math.pi])) == pytest.approx((1.0, 0.0))
