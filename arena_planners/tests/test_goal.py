from __future__ import annotations

import pytest

from arena_planners import goal

_DRLVO_LIKE = {
    "action_type": "differential_drive",
    "observations": {
        "aliases": {"goal_pose": "subgoal_from_plan", "robot_pose": "robot_pose_from_tf"},
        "datasources": {
            "robot_pose_from_tf": {"type": "RobotPoseTFGenerator", "params": {}},
            "goal_pose": {"type": "geometry_msgs/PoseStamped"},
            "goal_explicit": {"type": "geometry_msgs/PoseStamped", "params": {"topic": "goal_pose"}},
            "target": {"type": "geometry_msgs/PoseStamped"},
            "subgoal_from_plan": {"type": "SubgoalGenerator", "params": {"lookahead": 2.0}},
            "camera": {"type": "sensor_msgs/Image", "params": {"sensor": "image"}},
        },
    },
}


def test_declared_none_without_goal_inputs():
    assert goal.declared({"action_type": "differential_drive"}) is None


def test_declared_reads_goal_inputs():
    assert goal.declared({"goal_inputs": ["pose", "instruction"]}) == {"pose", "instruction"}


@pytest.mark.parametrize("raw", [[], ["pose", "smell"]])
def test_declared_rejects_empty_or_unknown(raw):
    with pytest.raises(ValueError, match="pose"):
        goal.declared({"goal_inputs": raw})


def test_reveal_empty_means_everything_accepted():
    assert goal.reveal("", frozenset({"pose", "instruction"})) == {"pose", "instruction"}


def test_reveal_parses_joined_inputs():
    assert goal.reveal("instruction", frozenset({"pose", "instruction"})) == {"instruction"}
    assert goal.reveal("pose+instruction", frozenset({"pose", "instruction"})) == {"pose", "instruction"}


def test_reveal_unknown_input_lists_choices():
    with pytest.raises(ValueError, match="instruction"):
        goal.reveal("pose+image", frozenset({"pose", "instruction"}))


def test_reveal_refuses_input_the_planner_lacks():
    with pytest.raises(ValueError, match="accepts \\['pose'\\]"):
        goal.reveal("instruction", frozenset({"pose"}))


def test_withhold_pose_drops_goal_sources_and_their_aliases():
    out = goal.withhold_pose(_DRLVO_LIKE)
    assert sorted(out["observations"]["datasources"]) == ["camera", "robot_pose_from_tf", "target"]
    assert out["observations"]["aliases"] == {"robot_pose": "robot_pose_from_tf"}
    assert "goal_pose" in _DRLVO_LIKE["observations"]["datasources"]


def test_withhold_pose_refuses_global_plan_planner():
    with pytest.raises(ValueError, match="global plan"):
        goal.withhold_pose({**_DRLVO_LIKE, "depends": {"global_plan": True}})


def test_initial_state_carries_only_revealed_inputs():
    pose = {"x": 1.0, "y": 2.0, "theta": 0.0}
    assert goal.initial_state(pose, "go to the door", frozenset({"instruction"})) == {"instruction": "go to the door"}
    assert goal.initial_state(pose, "go to the door", frozenset({"pose"})) == {"goal_pose": pose}
