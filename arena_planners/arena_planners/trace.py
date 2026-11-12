"""Opt-in per-episode decision trace of a bridge planner: answers, logs, poses and model input frames."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pathlib
import re
import time
import typing

import numpy as np

if typing.TYPE_CHECKING:
    from arena_planners.sdk import Step

ENV_DIR = "ARENA_PLANNER_TRACE_DIR"
ENV_FRAMES = "ARENA_PLANNER_TRACE_FRAMES"
ENV_NAME = "ARENA_PLANNER_TRACE_NAME"

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")

Json = bool | int | float | str | None | list["Json"] | dict[str, "Json"]


class _Capture(logging.Handler):
    """Buffers INFO+ messages of every logger except the bridge transport."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.buf: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("arena_planners.bridge"):
            return
        self.buf.append(f"{record.name}: {record.getMessage()}")


def _jsonable(x: object) -> Json:
    if isinstance(x, np.ndarray):
        return np.round(x.astype(np.float64), 4).tolist() if x.size <= 64 else f"<array {x.shape}>"
    if isinstance(x, np.floating | np.integer):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, bool | int | float | str) or x is None:
        return x
    return repr(x)


@dataclasses.dataclass
class Tracer:
    """Writes <root>/<planner>/<episode_id>/trace.jsonl (+ frames/); a no-op when root is None."""

    root: pathlib.Path | None
    planner: str = "planner"
    frames: bool = True
    _fh: typing.TextIO | None = dataclasses.field(default=None, init=False)
    _dir: pathlib.Path | None = dataclasses.field(default=None, init=False)
    _capture: _Capture = dataclasses.field(default_factory=_Capture, init=False)

    def __post_init__(self) -> None:
        self.planner = _SAFE.sub("_", self.planner or "planner")

    @classmethod
    def from_env(cls, planner: str) -> Tracer:
        root = os.environ.get(ENV_DIR, "").strip()
        return cls(
            root=pathlib.Path(root) if root else None,
            planner=os.environ.get(ENV_NAME, "") or planner,
            frames=os.environ.get(ENV_FRAMES, "1").strip() not in ("0", "false", "off"),
        )

    @property
    def enabled(self) -> bool:
        return self.root is not None

    def _attach(self) -> None:
        # after the planner's basicConfig and imports, so loggers created at import time are covered
        root = logging.getLogger()
        if self._capture not in root.handlers:
            root.addHandler(self._capture)
        if root.level > logging.INFO:
            root.setLevel(logging.INFO)
        for name, lg in list(logging.root.manager.loggerDict.items()):
            if (
                isinstance(lg, logging.Logger)
                and not lg.propagate
                and not name.startswith("arena_planners.bridge")
                and self._capture not in lg.handlers
            ):
                lg.addHandler(self._capture)
                if lg.level > logging.INFO:
                    lg.setLevel(logging.INFO)

    def _write(self, rec: dict[str, object]) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(_jsonable(rec), separators=(",", ":")) + "\n")
        self._fh.flush()

    def reset(self, episode_id: str, initial_state: dict[str, object] | None) -> None:
        if self.root is None:
            return
        self.close()
        self._attach()
        self._dir = self.root / self.planner / _SAFE.sub("_", str(episode_id) or "episode")
        (self._dir / "frames").mkdir(parents=True, exist_ok=True)
        self._fh = (self._dir / "trace.jsonl").open("a")
        self._capture.buf.clear()
        self._write(
            {"ev": "reset", "wall": time.time(), "episode_id": str(episode_id), "initial_state": initial_state or {}}
        )

    def step(self, features: dict[str, object], seq: int, stamp: float, latency_s: float, step: Step) -> None:
        if self._fh is None:
            return
        logs, self._capture.buf = self._capture.buf, []
        rec: dict[str, object] = {
            "ev": "step",
            "wall": time.time(),
            "stamp": stamp,
            "seq": seq,
            "latency_s": round(latency_s, 4),
            "command": step.command,
            "amount": step.amount,
            "signal": step.signal,
            "action": list(step.action),
            "chunk": [list(map(float, c)) for c in step.chunk],
        }
        pose = features.get("robot_pose")
        if isinstance(pose, np.ndarray):
            rec["pose"] = pose.astype(np.float64).ravel()[:3]
        if logs:
            rec["logs"] = logs
            rgb = features.get("rgb")
            if self.frames and isinstance(rgb, np.ndarray) and rgb.ndim == 3 and self._dir is not None:
                rec["frame"] = self._save_frame(rgb, seq)
        self._write(rec)

    def _save_frame(self, rgb: np.ndarray, seq: int) -> str:
        import cv2

        assert self._dir is not None
        name = f"frames/{seq:07d}.jpg"
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(self._dir / name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return name

    def cancel(self) -> None:
        if self._fh is not None:
            logs, self._capture.buf = self._capture.buf, []
            self._write({"ev": "cancel", "wall": time.time(), "logs": logs})
        self.close()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


__all__ = ["ENV_DIR", "ENV_FRAMES", "ENV_NAME", "Tracer"]
