"""Decision trace: one jsonl per episode with answers, logs, poses and frames."""

from __future__ import annotations

import json
import logging
import pathlib

import numpy as np

from arena_planners.sdk import Step
from arena_planners.trace import Tracer


def _records(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_tracer_without_root_writes_nothing(tmp_path: pathlib.Path) -> None:
    tr = Tracer(None, "navila")
    tr.reset("ep1", {"instruction": "go"})
    tr.step({}, 1, 0.0, 0.01, Step(command="forward"))
    tr.cancel()
    assert not tr.enabled
    assert list(tmp_path.iterdir()) == []


def test_trace_records_answer_pose_and_frame(tmp_path: pathlib.Path) -> None:
    log = logging.getLogger("navila")
    tr = Tracer(tmp_path, "navila")
    tr.reset("ep/1", {"instruction": "Go to the laboratory.", "signal": "arrived"})
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    pose = np.array([1.0, 2.0, 0.5])
    log.info("navila: %r after %d frames in %.2f s", "The next action is move forward 75 cm.", 8, 0.25)
    tr.step({"rgb": rgb, "robot_pose": pose}, 7, 12.5, 0.26, Step(command="forward", amount=0.0))
    tr.step({"rgb": rgb, "robot_pose": pose}, 8, 12.6, 0.001, Step(command="forward"))
    tr.step({"rgb": rgb}, 9, 12.7, 0.001, Step(signal="arrived"))
    tr.cancel()

    ep = tmp_path / "navila" / "ep_1"
    recs = _records(ep / "trace.jsonl")
    assert [r["ev"] for r in recs] == ["reset", "step", "step", "step", "cancel"]
    assert recs[0]["initial_state"]["instruction"] == "Go to the laboratory."
    first = recs[1]
    assert first["command"] == "forward" and first["seq"] == 7 and first["pose"] == [1.0, 2.0, 0.5]
    assert any("move forward 75 cm" in m for m in first["logs"])
    assert (ep / first["frame"]).is_file()
    assert "logs" not in recs[2] and "frame" not in recs[2]
    assert recs[3]["signal"] == "arrived"


def test_trace_without_frames_keeps_logs(tmp_path: pathlib.Path) -> None:
    log = logging.getLogger("cma")
    tr = Tracer(tmp_path, "cma", frames=False)
    tr.reset("e", None)
    log.warning("cma: something")
    tr.step({"rgb": np.zeros((4, 4, 3), np.uint8)}, 1, 0.0, 0.0, Step(command="left"))
    tr.cancel()
    recs = _records(tmp_path / "cma" / "e" / "trace.jsonl")
    assert "frame" not in recs[1] and recs[1]["logs"] == ["cma: cma: something"]
