"""Tests for arena_planners.bridge.chunk: waypoint and twist chunks on a kinematic unicycle."""

from __future__ import annotations

import math

import pytest

from arena_planners.bridge.chunk import (
    ChunkConfig,
    VelocityChunk,
    WaypointChunk,
    check_chunk,
    chunk_config_from_manifest,
    chunk_limits,
    robot_to_map,
    start_chunk,
    step_chunk,
)
from arena_planners.bridge.discrete import Limits, MoveStep, wrap_angle

_DT = 0.1
_POSE_SOURCE = {"observations": {"datasources": {"robot_pose": {"type": "RobotPoseTFGenerator", "params": {}}}}}
_LONG = ChunkConfig(execute_s=30.0)
_LIMITS = Limits(v_max=0.5, w_max=1.0)


def _drive(
    chunk: WaypointChunk | VelocityChunk,
    pose: tuple[float, float, float],
    *,
    t0: float = 0.0,
    yaw_bias: float = 0.0,
    max_steps: int = 1000,
) -> tuple[tuple[float, float, float], list[MoveStep], list[tuple[float, float, float]], float]:
    x, y, theta = pose
    t = t0
    steps: list[MoveStep] = []
    trace = [(x, y, theta)]
    for _ in range(max_steps):
        step = step_chunk(chunk, [x, y, theta], t)
        steps.append(step)
        if step.done:
            return (x, y, theta), steps, trace, t
        x += step.v * math.cos(theta) * _DT
        y += step.v * math.sin(theta) * _DT
        theta = wrap_angle(theta + (step.omega + yaw_bias) * _DT)
        t += _DT
        trace.append((x, y, theta))
    raise AssertionError(f"chunk did not finish in {max_steps} steps")


def _local(pose: tuple[float, float, float], point: tuple[float, float]) -> tuple[float, float]:
    """`point` in the frame of `pose`."""
    dx, dy = point[0] - pose[0], point[1] - pose[1]
    return (dx * math.cos(pose[2]) + dy * math.sin(pose[2]), -dx * math.sin(pose[2]) + dy * math.cos(pose[2]))


def _distance_to_polyline(point: tuple[float, float], path: tuple[tuple[float, float], ...]) -> float:
    best = math.inf
    for (ax, ay), (bx, by) in zip(path[:-1], path[1:], strict=True):
        dx, dy = bx - ax, by - ay
        frac = min(max(((point[0] - ax) * dx + (point[1] - ay) * dy) / (dx * dx + dy * dy), 0.0), 1.0)
        best = min(best, math.hypot(ax + frac * dx - point[0], ay + frac * dy - point[1]))
    return best


def _waypoints(entries: list[list[float]], pose: tuple[float, float, float], **config: float) -> WaypointChunk:
    chunk = start_chunk("waypoints", entries, pose, 0.0, ChunkConfig(**{"execute_s": 30.0, **config}), _LIMITS)
    assert isinstance(chunk, WaypointChunk)
    return chunk


class TestManifest:
    def test_defaults(self) -> None:
        config = chunk_config_from_manifest({"action_type": "waypoints", **_POSE_SOURCE})
        assert config == ChunkConfig(execute_s=1.0, dt=0.1, lookahead_m=0.3, speed_mps=0.5)

    def test_chunk_block(self) -> None:
        manifest = {"action_type": "velocity_chunk", **_POSE_SOURCE, "chunk": {"execute_s": 2, "dt": 0.2}}
        assert chunk_config_from_manifest(manifest) == ChunkConfig(execute_s=2.0, dt=0.2)

    def test_requires_robot_pose_tf_datasource(self) -> None:
        with pytest.raises(ValueError, match="action_type waypoints needs observations.datasources.robot_pose"):
            chunk_config_from_manifest({"action_type": "waypoints", "observations": {}})

    @pytest.mark.parametrize(
        ("chunk", "match"),
        [
            ({"horizon_s": 1.0}, "unknown chunk keys"),
            ({"execute_s": 0.0}, "positive"),
            ({"dt": -0.1}, "positive"),
            ({"lookahead_m": 0.0}, "positive"),
            ({"speed_mps": 0.0}, "positive"),
        ],
    )
    def test_rejects_bad_block(self, chunk: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            chunk_config_from_manifest({"action_type": "waypoints", **_POSE_SOURCE, "chunk": chunk})


class TestLimits:
    def test_defaults_without_robot_limits(self) -> None:
        assert chunk_limits(ChunkConfig(), {}) == Limits(0.5, 1.0)

    def test_robot_limits_cap_cruise_speed_and_turn_rate(self) -> None:
        limits = chunk_limits(ChunkConfig(), {"linear": (-0.1, 0.3), "angular": (-0.6, 0.6)})
        assert limits == Limits(0.3, 0.6)

    def test_cruise_speed_above_half_a_meter_per_second(self) -> None:
        limits = chunk_limits(ChunkConfig(speed_mps=1.2), {"linear": (-2.0, 2.0), "angular": (-2.5, 2.5)})
        assert limits == Limits(1.2, 2.5)

    def test_non_positive_upper_limit_is_ignored(self) -> None:
        assert chunk_limits(ChunkConfig(), {"linear": (-0.5, 0.0), "angular": (0.0, 0.0)}) == Limits(0.5, 1.0)


class TestRobotToMap:
    @pytest.mark.parametrize(
        "theta", [0.0, 0.5 * math.pi, -0.5 * math.pi, 2.5, math.pi - 1e-6, -math.pi + 1e-6, -math.pi]
    )
    def test_round_trips_through_the_pose_frame(self, theta: float) -> None:
        pose = (2.0, -1.0, theta)
        points = [[1.0, 0.0], [0.0, 1.0], [-0.5, 0.25, 1.3]]
        mapped = robot_to_map(pose, points)
        for (px, py, *_), point in zip(points, mapped, strict=True):
            assert _local(pose, point) == pytest.approx((px, py), abs=1e-12)

    @pytest.mark.parametrize(
        ("theta", "expected"),
        [
            (0.0, (3.0, -1.0)),
            (0.5 * math.pi, (2.0, 0.0)),
            (math.pi - 1e-9, (1.0, -1.0)),
            (-math.pi + 1e-9, (1.0, -1.0)),
            (-0.5 * math.pi, (2.0, -2.0)),
        ],
    )
    def test_forward_point_lands_along_the_heading(self, theta: float, expected: tuple[float, float]) -> None:
        assert robot_to_map((2.0, -1.0, theta), [[1.0, 0.0]])[0] == pytest.approx(expected, abs=1e-6)

    def test_yaw_is_ignored(self) -> None:
        assert robot_to_map((0.0, 0.0, 0.3), [[1.0, 2.0, 3.0]]) == robot_to_map((0.0, 0.0, 0.3), [[1.0, 2.0]])


class TestCheckChunk:
    def test_entries_become_float_lists(self) -> None:
        assert check_chunk("waypoints", [(1, 2), [3, 4, 5]]) == [[1.0, 2.0], [3.0, 4.0, 5.0]]

    @pytest.mark.parametrize(
        ("kind", "entries", "match"),
        [
            ("waypoints", [[1.0]], r"\[x, y\] or \[x, y, yaw\]"),
            ("waypoints", [[1.0, 2.0, 3.0, 4.0]], r"\[x, y\] or \[x, y, yaw\]"),
            ("velocity_chunk", [[0.1, 0.0, 0.0]], r"\[v, omega\]"),
            ("waypoints", [[math.nan, 0.0]], "finite"),
            ("velocity_chunk", [[math.inf, 0.0]], "finite"),
            ("spline", [[0.0, 0.0]], "chunk type"),
        ],
    )
    def test_rejects(self, kind: str, entries: list, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            check_chunk(kind, entries)


class TestStart:
    def test_empty_chunk_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            start_chunk("velocity_chunk", [], None, 0.0, ChunkConfig(), _LIMITS)

    def test_waypoints_need_a_pose(self) -> None:
        with pytest.raises(ValueError, match="robot pose"):
            start_chunk("waypoints", [[1.0, 0.0]], None, 0.0, ChunkConfig(), _LIMITS)

    def test_velocity_chunk_needs_no_pose(self) -> None:
        chunk = start_chunk("velocity_chunk", [[0.1, 0.0]], None, 4.0, ChunkConfig(), _LIMITS)
        assert chunk == VelocityChunk(twists=((0.1, 0.0),), t0_s=4.0, dt=0.1, end_s=5.0)

    def test_waypoint_path_starts_at_the_robot_in_the_map_frame(self) -> None:
        chunk = start_chunk("waypoints", [[1.0, 0.0, 0.2]], (1.0, 2.0, 0.5 * math.pi), 10.0, ChunkConfig(), _LIMITS)
        assert isinstance(chunk, WaypointChunk)
        assert chunk.path[0] == (1.0, 2.0)
        assert chunk.path[1] == pytest.approx((1.0, 3.0))
        assert chunk.end_s == pytest.approx(11.0)
        assert chunk.deadline_s == pytest.approx(10.0 + 3.0 * 1.0 + 1.0)


class TestWaypoints:
    @pytest.mark.parametrize("theta", [0.0, 1.0, math.pi - 0.01, -math.pi + 0.01])
    @pytest.mark.parametrize("yaw_bias", [0.0, 0.15, -0.15])
    def test_straight_meter_within_five_centimeters(self, theta: float, yaw_bias: float) -> None:
        start = (1.0, -2.0, theta)
        chunk = _waypoints([[0.25, 0.0], [0.5, 0.0], [0.75, 0.0], [1.0, 0.0]], start)
        final, steps, trace, _ = _drive(chunk, start, yaw_bias=yaw_bias)
        assert steps[-1] == MoveStep(done=True)
        assert math.dist(final[:2], chunk.path[-1]) <= 0.05
        assert max(abs(_local(start, pose[:2])[1]) for pose in trace) < 0.05
        assert all(s.v > 0.0 for s in steps[:-1])

    def test_l_shape_turns_the_corner(self) -> None:
        start = (0.0, 0.0, 0.0)
        chunk = _waypoints([[0.5, 0.0], [1.0, 0.0], [1.0, 0.5], [1.0, 1.0]], start)
        final, steps, trace, _ = _drive(chunk, start)
        assert not steps[-1].timed_out
        assert math.dist(final[:2], (1.0, 1.0)) <= 0.05
        assert final[2] == pytest.approx(0.5 * math.pi, abs=math.radians(15.0))
        assert max(_distance_to_polyline(pose[:2], chunk.path) for pose in trace) < 0.15
        assert max(pose[0] for pose in trace) < 1.1
        assert any(s.omega > 0.5 for s in steps)

    @pytest.mark.parametrize("theta", [0.0, 2.0, -math.pi + 0.01])
    def test_points_behind_turn_around_in_place_first(self, theta: float) -> None:
        start = (3.0, 4.0, theta)
        chunk = _waypoints([[-0.5, 0.0], [-1.0, 0.0]], start)
        final, steps, _, _ = _drive(chunk, start)
        assert not steps[-1].timed_out
        assert math.dist(final[:2], chunk.path[-1]) <= 0.05
        assert steps[0].v == 0.0
        assert abs(steps[0].omega) > 0.0
        turning = [s for s in steps if s.v == 0.0 and not s.done]
        assert len(turning) >= 10
        assert abs(wrap_angle(final[2] - theta)) > math.radians(150.0)

    def test_point_beside_the_robot_is_reached(self) -> None:
        start = (0.0, 0.0, 0.0)
        chunk = _waypoints([[0.0, 0.6]], start)
        final, steps, _, _ = _drive(chunk, start)
        assert not steps[-1].timed_out
        assert math.dist(final[:2], (0.0, 0.6)) <= 0.05

    def test_speed_tapers_near_the_final_point(self) -> None:
        chunk = _waypoints([[1.0, 0.0]], (0.0, 0.0, 0.0))
        _, steps, _, _ = _drive(chunk, (0.0, 0.0, 0.0))
        moving = steps[:-1]
        assert moving[0].v == pytest.approx(0.5)
        assert moving[-1].v < 0.15

    def test_limits_respected(self) -> None:
        limits = Limits(v_max=0.2, w_max=0.4)
        entries = [[0.4, 0.3], [0.8, -0.3], [1.2, 0.3], [-0.5, 0.0]]
        chunk = start_chunk("waypoints", entries, (0.0, 0.0, 0.0), 0.0, _LONG, limits)
        _, steps, _, _ = _drive(chunk, (0.0, 0.0, 0.0))
        assert all(0.0 <= s.v <= 0.2 + 1e-12 for s in steps)
        assert all(abs(s.omega) <= 0.4 + 1e-12 for s in steps)
        assert any(s.v == pytest.approx(0.2) for s in steps)
        assert any(abs(s.omega) == pytest.approx(0.4) for s in steps)

    def test_execute_s_cuts_execution(self) -> None:
        chunk = start_chunk("waypoints", [[2.0, 0.0]], (0.0, 0.0, 0.0), 5.0, ChunkConfig(), _LIMITS)
        final, steps, _, t = _drive(chunk, (0.0, 0.0, 0.0), t0=5.0)
        assert steps[-1] == MoveStep(done=True)
        assert t == pytest.approx(6.0)
        assert len(steps) == 11
        assert 0.4 <= final[0] <= 0.5

    def test_reaching_the_final_point_ends_before_execute_s(self) -> None:
        chunk = _waypoints([[0.3, 0.0]], (0.0, 0.0, 0.0))
        _, steps, _, t = _drive(chunk, (0.0, 0.0, 0.0))
        assert steps[-1] == MoveStep(done=True)
        assert t < 2.0


class TestTimeout:
    def test_tick_past_the_deadline_times_out(self) -> None:
        chunk = start_chunk("waypoints", [[1.0, 0.0]], (0.0, 0.0, 0.0), 0.0, ChunkConfig(), _LIMITS)
        assert chunk.deadline_s == pytest.approx(4.0)
        assert step_chunk(chunk, [0.0, 0.0, 0.0], 4.01) == MoveStep(done=True, timed_out=True)

    def test_reaching_the_final_point_late_is_not_a_timeout(self) -> None:
        chunk = start_chunk("waypoints", [[1.0, 0.0]], (0.0, 0.0, 0.0), 0.0, ChunkConfig(), _LIMITS)
        assert step_chunk(chunk, [1.0, 0.0, 0.0], 10.0) == MoveStep(done=True)

    def test_missing_pose_holds_still_until_execute_s(self) -> None:
        chunk = start_chunk("waypoints", [[1.0, 0.0]], (0.0, 0.0, 0.0), 0.0, ChunkConfig(), _LIMITS)
        assert step_chunk(chunk, None, 0.5) == MoveStep()
        assert step_chunk(chunk, None, 1.0) == MoveStep(done=True)


class TestVelocityChunk:
    @staticmethod
    def _replay(chunk: VelocityChunk, t0: float) -> tuple[list[MoveStep], float]:
        t = t0
        steps: list[MoveStep] = []
        while True:
            step = step_chunk(chunk, None, t)
            steps.append(step)
            if step.done:
                return steps, t
            t += _DT

    def test_each_entry_applied_for_one_tick(self) -> None:
        entries = [[0.1 * i, 0.05 * i] for i in range(1, 6)]
        chunk = start_chunk("velocity_chunk", entries, None, 2.0, ChunkConfig(), _LIMITS)
        steps, t = self._replay(chunk, 2.0)
        assert [[s.v, s.omega] for s in steps[:-1]] == entries
        assert t == pytest.approx(2.5)

    def test_entries_held_for_dt(self) -> None:
        entries = [[0.1, 0.0], [0.2, 0.1], [0.3, -0.1]]
        chunk = start_chunk("velocity_chunk", entries, None, 0.0, ChunkConfig(dt=0.2), _LIMITS)
        steps, t = self._replay(chunk, 0.0)
        assert [[s.v, s.omega] for s in steps[:-1]] == [e for e in entries for _ in range(2)]
        assert t == pytest.approx(0.6)

    def test_stops_at_execute_s(self) -> None:
        entries = [[0.01 * i, 0.0] for i in range(25)]
        chunk = start_chunk("velocity_chunk", entries, None, 0.0, ChunkConfig(), _LIMITS)
        steps, t = self._replay(chunk, 0.0)
        assert [[s.v, s.omega] for s in steps[:-1]] == entries[:10]
        assert t == pytest.approx(1.0)
        assert steps[-1] == MoveStep(done=True)

    def test_twists_pass_through_unclamped(self) -> None:
        chunk = start_chunk("velocity_chunk", [[3.0, -4.0]], None, 0.0, ChunkConfig(), _LIMITS)
        assert step_chunk(chunk, None, 0.0) == MoveStep(v=3.0, omega=-4.0)
