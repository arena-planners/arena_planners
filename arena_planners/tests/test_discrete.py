"""Tests for arena_planners.bridge.discrete: primitive execution on a kinematic unicycle."""

from __future__ import annotations

import math

import pytest

from arena_planners.bridge.discrete import (
    Limits,
    Move,
    MoveStep,
    Primitives,
    check_amount,
    primitives_from_manifest,
    start_move,
    step_move,
    wrap_angle,
)

_DT = 0.1
_DEFAULT_PRIMITIVES = Primitives()
_DEFAULT_LIMITS = Limits()
_POSE_SOURCE = {"observations": {"datasources": {"robot_pose": {"type": "RobotPoseTFGenerator", "params": {}}}}}


def _drive(
    command: str,
    pose: tuple[float, float, float],
    *,
    primitives: Primitives = _DEFAULT_PRIMITIVES,
    limits: Limits = _DEFAULT_LIMITS,
    yaw_bias: float = 0.0,
    amount: float = 0.0,
    max_steps: int = 1000,
) -> tuple[Move, tuple[float, float, float], list[MoveStep]]:
    x, y, theta = pose
    t = 0.0
    move = start_move(command, [x, y, theta], t, primitives, limits, amount)
    steps: list[MoveStep] = []
    for _ in range(max_steps):
        step = step_move(move, [x, y, theta], t)
        steps.append(step)
        if step.done:
            return move, (x, y, theta), steps
        x += step.v * math.cos(theta) * _DT
        y += step.v * math.sin(theta) * _DT
        theta = wrap_angle(theta + (step.omega + yaw_bias) * _DT)
        t += _DT
    raise AssertionError(f"{command} did not finish in {max_steps} steps")


@pytest.mark.parametrize(
    ("angle", "wrapped"),
    [(0.0, 0.0), (math.pi + 0.1, -math.pi + 0.1), (-math.pi - 0.1, math.pi - 0.1), (4 * math.pi + 0.2, 0.2)],
)
def test_wrap_angle(angle: float, wrapped: float) -> None:
    assert wrap_angle(angle) == pytest.approx(wrapped)


class TestManifest:
    def test_defaults(self) -> None:
        assert primitives_from_manifest({"action_type": "discrete", **_POSE_SOURCE}) == Primitives(0.25, 15.0)

    def test_primitives_block(self) -> None:
        manifest = {**_POSE_SOURCE, "primitives": {"forward_m": 0.5, "turn_deg": 30}}
        assert primitives_from_manifest(manifest) == Primitives(0.5, 30.0)

    @pytest.mark.parametrize(
        "observations",
        [
            {},
            {"datasources": {}},
            {"datasources": {"robot_pose": {"type": "geometry_msgs/PoseStamped"}}},
            {"datasources": {"pose": {"type": "RobotPoseTFGenerator"}}},
        ],
    )
    def test_requires_robot_pose_tf_datasource(self, observations: dict) -> None:
        with pytest.raises(ValueError, match="robot_pose"):
            primitives_from_manifest({"action_type": "discrete", "observations": observations})

    @pytest.mark.parametrize(
        ("primitives", "match"),
        [
            ({"forward_m": 0.0}, "forward_m > 0"),
            ({"turn_deg": 180.0}, "turn_deg < 180"),
            ({"turn_deg": -15.0}, "turn_deg < 180"),
            ({"stride": 1.0}, "unknown primitives keys"),
        ],
    )
    def test_rejects_bad_primitives(self, primitives: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            primitives_from_manifest({**_POSE_SOURCE, "primitives": primitives})


class TestLimits:
    def test_defaults_without_robot_limits(self) -> None:
        assert Limits.from_velocity_limits({}) == Limits(0.5, 1.0)

    def test_robot_limits_below_caps(self) -> None:
        limits = Limits.from_velocity_limits({"linear": (-0.1, 0.3), "angular": (-0.6, 0.6), "lateral": (0.0, 0.0)})
        assert limits == Limits(0.3, 0.6)

    def test_robot_limits_capped(self) -> None:
        assert Limits.from_velocity_limits({"linear": (-2.0, 2.0), "angular": (-4.0, 4.0)}) == Limits(0.5, 1.0)

    def test_non_positive_upper_limit_falls_back_to_cap(self) -> None:
        assert Limits.from_velocity_limits({"linear": (-0.5, 0.0), "angular": (0.0, 0.0)}) == Limits(0.5, 1.0)


class TestForward:
    @pytest.mark.parametrize("theta", [0.0, 2.0, -math.pi + 0.01, math.pi - 0.01])
    def test_reaches_step_length_along_start_heading(self, theta: float) -> None:
        move, (x, y, final_theta), steps = _drive("forward", (1.0, -2.0, theta))
        progress = (x - 1.0) * math.cos(theta) + (y + 2.0) * math.sin(theta)
        lateral = -(x - 1.0) * math.sin(theta) + (y + 2.0) * math.cos(theta)
        assert 0.23 <= progress <= 0.26
        assert abs(lateral) < 1e-9
        assert wrap_angle(final_theta - theta) == pytest.approx(0.0, abs=1e-9)
        assert steps[-1] == MoveStep(done=True)
        assert move.turn_rad == 0.0

    def test_custom_step_length(self) -> None:
        _, (x, _, _), _ = _drive("forward", (0.0, 0.0, 0.0), primitives=Primitives(forward_m=1.0))
        assert 0.98 <= x <= 1.01

    def test_speed_tapers_and_respects_limits(self) -> None:
        limits = Limits(v_max=0.2, w_max=0.4)
        _, _, steps = _drive("forward", (0.0, 0.0, 0.0), limits=limits, yaw_bias=0.3)
        moving = steps[:-1]
        assert all(0.05 <= s.v <= 0.2 for s in moving)
        assert all(abs(s.omega) <= 0.4 for s in moving)
        assert moving[0].v == pytest.approx(0.2)
        assert moving[-1].v < 0.1

    def test_heading_hold_counters_yaw_drift(self) -> None:
        _, (x, y, theta), steps = _drive("forward", (0.0, 0.0, 0.0), yaw_bias=0.2)
        assert 0.23 <= x <= 0.26
        assert abs(theta) < math.radians(10.0)
        assert all(s.omega <= 0.0 for s in steps[1:-1])


class TestTurn:
    @pytest.mark.parametrize(
        ("command", "theta0"),
        [
            ("left", 0.0),
            ("right", 0.0),
            ("left", math.pi - 0.1),
            ("right", -math.pi + 0.1),
            ("left", -math.pi + 0.05),
            ("right", math.pi - 0.05),
        ],
    )
    def test_turns_fifteen_degrees_across_wraparound(self, command: str, theta0: float) -> None:
        sign = 1.0 if command == "left" else -1.0
        move, (x, y, theta), steps = _drive(command, (3.0, 4.0, theta0))
        target = theta0 + sign * math.radians(15.0)
        assert abs(wrap_angle(theta - target)) <= math.radians(1.0)
        assert (x, y) == (3.0, 4.0)
        assert move.turn_rad == pytest.approx(sign * math.radians(15.0))
        assert all(s.v == 0.0 for s in steps)
        assert all(math.copysign(1.0, s.omega) == sign for s in steps[:-1])

    def test_turn_rate_respects_limits(self) -> None:
        limits = Limits(v_max=0.5, w_max=0.3)
        _, _, steps = _drive("left", (0.0, 0.0, 0.0), primitives=Primitives(turn_deg=90.0), limits=limits)
        moving = steps[:-1]
        assert moving[0].omega == pytest.approx(0.3)
        assert all(0.1 <= s.omega <= 0.3 for s in moving)

    def test_overshoot_turns_back(self) -> None:
        move = start_move("left", [0.0, 0.0, 0.0], 0.0, Primitives(), Limits())
        step = step_move(move, [0.0, 0.0, math.radians(20.0)], 0.1)
        assert not step.done
        assert step.omega < 0.0


class TestTimeout:
    def test_blocked_forward_ends_at_deadline(self) -> None:
        move = start_move("forward", [0.0, 0.0, 0.0], 10.0, Primitives(), Limits())
        assert move.deadline_s == pytest.approx(10.0 + 3.0 * 0.25 / 0.5 + 1.0)
        assert not step_move(move, [0.0, 0.0, 0.0], move.deadline_s).done
        assert step_move(move, [0.0, 0.0, 0.0], move.deadline_s + 0.01) == MoveStep(done=True, timed_out=True)

    def test_blocked_turn_ends_at_deadline(self) -> None:
        limits = Limits(v_max=0.5, w_max=0.5)
        move = start_move("right", [0.0, 0.0, 1.0], 0.0, Primitives(), limits)
        assert move.deadline_s == pytest.approx(3.0 * math.radians(15.0) / 0.5 + 1.0)
        assert step_move(move, [0.0, 0.0, 1.0], move.deadline_s + 0.01).timed_out

    def test_missing_pose_holds_still_until_deadline(self) -> None:
        move = start_move("forward", [0.0, 0.0, 0.0], 0.0, Primitives(), Limits())
        assert step_move(move, None, 1.0) == MoveStep()
        assert step_move(move, None, move.deadline_s + 0.01) == MoveStep(done=True, timed_out=True)

    def test_reaching_the_target_late_is_not_a_timeout(self) -> None:
        move = start_move("forward", [0.0, 0.0, 0.0], 0.0, Primitives(), Limits())
        assert step_move(move, [0.25, 0.0, 0.0], move.deadline_s + 5.0) == MoveStep(done=True)


class TestAmount:
    @pytest.mark.parametrize("theta", [0.0, 2.0, -math.pi + 0.01])
    def test_forward_amount_overrides_the_step_length(self, theta: float) -> None:
        move, (x, y, _), _ = _drive("forward", (1.0, -2.0, theta), amount=0.75)
        progress = (x - 1.0) * math.cos(theta) + (y + 2.0) * math.sin(theta)
        assert move.distance_m == 0.75
        assert 0.73 <= progress <= 0.76

    @pytest.mark.parametrize(
        ("command", "degrees", "theta0"),
        [
            ("left", 30.0, 0.0),
            ("right", 30.0, math.pi - 0.1),
            ("left", 90.0, math.pi - 0.2),
            ("right", 90.0, -math.pi + 0.2),
            ("left", 180.0, 0.5),
            ("right", 180.0, -2.5),
            ("left", 270.0, 3.0),
        ],
    )
    def test_turn_amount_in_degrees(self, command: str, degrees: float, theta0: float) -> None:
        sign = 1.0 if command == "left" else -1.0
        move, (x, y, theta), steps = _drive(command, (3.0, 4.0, theta0), amount=degrees)
        assert move.turn_rad == pytest.approx(sign * math.radians(degrees))
        assert abs(wrap_angle(theta - theta0 - sign * math.radians(degrees))) <= math.radians(1.0)
        assert (x, y) == (3.0, 4.0)
        assert all(math.copysign(1.0, s.omega) == sign for s in steps[:-1])

    def test_zero_amount_is_the_primitive(self) -> None:
        primitives = Primitives(forward_m=0.4, turn_deg=20.0)
        forward = start_move("forward", [0.0, 0.0, 0.0], 0.0, primitives, Limits(), 0.0)
        turn = start_move("right", [0.0, 0.0, 0.0], 0.0, primitives, Limits(), 0.0)
        assert forward.distance_m == 0.4
        assert turn.turn_rad == pytest.approx(-math.radians(20.0))

    def test_timeout_scales_with_the_amount(self) -> None:
        forward = start_move("forward", [0.0, 0.0, 0.0], 2.0, Primitives(), Limits(), 0.75)
        turn = start_move("left", [0.0, 0.0, 0.0], 2.0, Primitives(), Limits(v_max=0.5, w_max=0.5), 90.0)
        assert forward.deadline_s == pytest.approx(2.0 + 3.0 * 0.75 / 0.5 + 1.0)
        assert turn.deadline_s == pytest.approx(2.0 + 3.0 * (0.5 * math.pi) / 0.5 + 1.0)

    @pytest.mark.parametrize(
        ("command", "amount", "match"),
        [
            ("forward", -0.1, ">= 0"),
            ("left", -30.0, ">= 0"),
            ("forward", math.nan, "finite"),
            ("right", math.inf, "finite"),
            ("left", 360.0, "below 360"),
        ],
    )
    def test_rejects_bad_amount(self, command: str, amount: float, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            check_amount(command, amount)
        with pytest.raises(ValueError, match=match):
            start_move(command, [0.0, 0.0, 0.0], 0.0, Primitives(), Limits(), amount)

    def test_long_forward_accepted(self) -> None:
        check_amount("forward", 400.0)


def test_start_move_rejects_unknown_command() -> None:
    with pytest.raises(ValueError, match="jump"):
        start_move("jump", [0.0, 0.0, 0.0], 0.0, Primitives(), Limits())
