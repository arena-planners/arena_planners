from __future__ import annotations

import math

import pytest

from arena_planners.bridge.projection import (
    project_holonomic_to_diff_drive,
    unpack_differential_drive,
    unpack_omnidirectional,
)


def test_zero_velocity_returns_zero_v_keeps_omega_in():
    v, omega = project_holonomic_to_diff_drive(0.0, 0.0, 0.3, 0.0, 0.1)
    assert v == 0.0
    assert omega == pytest.approx(0.3)


def test_aligned_velocity_pure_forward():
    v, omega = project_holonomic_to_diff_drive(0.5, 0.0, 0.0, 0.0, 0.1)
    assert v == pytest.approx(0.5)
    assert omega == pytest.approx(0.0)


def test_perpendicular_velocity_pure_turn():
    v, omega = project_holonomic_to_diff_drive(0.0, 0.5, 0.0, 0.0, 0.1)
    assert v == pytest.approx(0.5)
    assert omega == pytest.approx(math.pi / 2.0 / 0.1)


def test_negative_x_wraps_to_minus_pi_not_plus_pi():
    v, omega = project_holonomic_to_diff_drive(-0.5, 0.0, 0.0, 0.0, 0.1)
    assert v == pytest.approx(0.5)
    assert abs(abs(omega) - math.pi / 0.1) < 1e-9


def test_robot_already_aligned_no_omega():
    v, omega = project_holonomic_to_diff_drive(0.0, 0.5, 0.0, math.pi / 2.0, 0.1)
    assert v == pytest.approx(0.5)
    assert omega == pytest.approx(0.0)


def test_omega_in_added_on_top_of_heading_term():
    v, omega = project_holonomic_to_diff_drive(0.5, 0.0, 0.4, 0.0, 0.1)
    assert v == pytest.approx(0.5)
    assert omega == pytest.approx(0.4)


def test_invalid_step_dt_raises():
    with pytest.raises(ValueError):
        project_holonomic_to_diff_drive(0.5, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        project_holonomic_to_diff_drive(0.5, 0.0, 0.0, 0.0, -0.1)


def test_unpack_omnidirectional_length_2_defaults_omega():
    assert unpack_omnidirectional([0.3, 0.4]) == (0.3, 0.4, 0.0)


def test_unpack_omnidirectional_length_3():
    assert unpack_omnidirectional([0.3, 0.4, 0.5]) == (0.3, 0.4, 0.5)


def test_unpack_omnidirectional_rejects_other_lengths():
    with pytest.raises(ValueError):
        unpack_omnidirectional([0.3])
    with pytest.raises(ValueError):
        unpack_omnidirectional([0.3, 0.4, 0.5, 0.6])


def test_unpack_differential_drive():
    assert unpack_differential_drive([0.5, 0.7]) == (0.5, 0.7)


def test_unpack_differential_drive_rejects_other_lengths():
    with pytest.raises(ValueError):
        unpack_differential_drive([0.5])
    with pytest.raises(ValueError):
        unpack_differential_drive([0.5, 0.7, 0.9])
