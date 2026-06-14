"""Wire-format frame dataclasses and msgpack encode/decode for the DRL planner bridge."""

from __future__ import annotations

import dataclasses
import typing

import msgpack
import msgpack_numpy

msgpack_numpy.patch()

PROTOCOL_VERSION: int = 2
SCHEMA_VERSION: int = 1


class ProtocolError(Exception):
    """Raised on unknown op or malformed payload."""


@dataclasses.dataclass(frozen=False)
class Init:
    """Edge to planner: open the session."""

    op: str = dataclasses.field(default="init", init=False, repr=True)
    protocol_version: int = 0
    schema_version: int = 0
    obs_schema: dict = dataclasses.field(default_factory=dict)
    action_schema: dict = dataclasses.field(default_factory=dict)
    planner_config: dict = dataclasses.field(default_factory=dict)
    run_id: str = ""


@dataclasses.dataclass(frozen=False)
class InitAck:
    """Planner to edge: session accepted."""

    op: str = dataclasses.field(default="init_ack", init=False, repr=True)
    capabilities: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=False)
class Reset:
    """Edge to planner: begin a new episode."""

    op: str = dataclasses.field(default="reset", init=False, repr=True)
    episode_id: str = ""
    initial_state: dict | None = None


@dataclasses.dataclass(frozen=False)
class ResetAck:
    """Planner to edge: ready for observations."""

    op: str = dataclasses.field(default="reset_ack", init=False, repr=True)
    applied_state: dict | None = None


@dataclasses.dataclass(frozen=False)
class Obs:
    """Edge to planner: one observation tick."""

    op: str = dataclasses.field(default="obs", init=False, repr=True)
    t_sec: int = 0
    t_nanosec: int = 0
    seq: int = 0
    features: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=False)
class Action:
    """Planner to edge: velocity command for the current tick."""

    op: str = dataclasses.field(default="action", init=False, repr=True)
    t_sec: int = 0
    t_nanosec: int = 0
    seq: int = 0
    action_type: str = ""
    action: list[float] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=False)
class Cancel:
    """Edge to planner: preempt the current goal."""

    op: str = dataclasses.field(default="cancel", init=False, repr=True)


@dataclasses.dataclass(frozen=False)
class CancelAck:
    """Planner to edge: cancel acknowledged."""

    op: str = dataclasses.field(default="cancel_ack", init=False, repr=True)
    cancelled_seq: int | None = None


@dataclasses.dataclass(frozen=False)
class Shutdown:
    """Edge to planner: terminate cleanly."""

    op: str = dataclasses.field(default="shutdown", init=False, repr=True)


@dataclasses.dataclass(frozen=False)
class Bye:
    """Planner to edge: goodbye before exit."""

    op: str = dataclasses.field(default="bye", init=False, repr=True)


@dataclasses.dataclass(frozen=False)
class Error:
    """Either direction: structured error."""

    op: str = dataclasses.field(default="error", init=False, repr=True)
    code: str = ""
    msg: str = ""
    severity: str = "error"


@dataclasses.dataclass(frozen=False)
class Heartbeat:
    """Planner to edge: periodic liveness signal."""

    op: str = dataclasses.field(default="heartbeat", init=False, repr=True)
    seq: int = 0
    monotonic_ns: int = 0


Frame = typing.Union[  # noqa: UP007 - runtime alias, must import on Python < 3.10
    Init,
    InitAck,
    Reset,
    ResetAck,
    Obs,
    Action,
    Cancel,
    CancelAck,
    Shutdown,
    Bye,
    Error,
    Heartbeat,
]

_OP_TO_CLASS: dict[str, type] = {
    "init": Init,
    "init_ack": InitAck,
    "reset": Reset,
    "reset_ack": ResetAck,
    "obs": Obs,
    "action": Action,
    "cancel": Cancel,
    "cancel_ack": CancelAck,
    "shutdown": Shutdown,
    "bye": Bye,
    "error": Error,
    "heartbeat": Heartbeat,
}

_SLOTS: dict[type, frozenset[str]] = {
    cls: frozenset(f.name for f in dataclasses.fields(cls) if f.name != "op") for cls in _OP_TO_CLASS.values()
}


def encode_frame(frame: Frame) -> bytes:
    """Serialize a frame to msgpack bytes."""
    payload: dict = {"op": frame.op}
    for field in dataclasses.fields(frame):
        if field.name == "op":
            continue
        payload[field.name] = getattr(frame, field.name)
    return msgpack.packb(payload, use_bin_type=True)


def decode_frame(buf: bytes) -> Frame:
    """Deserialize msgpack bytes to a Frame, raising ProtocolError on bad input."""
    try:
        raw: dict = msgpack.unpackb(buf, raw=False)
    except Exception as exc:
        raise ProtocolError(f"msgpack decode failed: {exc}") from exc

    if not isinstance(raw, dict):
        raise ProtocolError("frame payload is not a mapping")

    op = raw.get("op")
    if not isinstance(op, str) or op not in _OP_TO_CLASS:
        raise ProtocolError(f"unknown or missing op: {op!r}")

    cls = _OP_TO_CLASS[op]
    known = _SLOTS[cls]
    kwargs = {k: v for k, v in raw.items() if k in known}

    try:
        return cls(**kwargs)
    except Exception as exc:
        raise ProtocolError(f"failed to construct {cls.__name__}: {exc}") from exc
