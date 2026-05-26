"""Lifecycle tests for PlannerProcess. No rclpy, no mocking."""

from __future__ import annotations

import signal
import sys
import time

import pytest

from arena_planners.bridge.edge_node import PlannerProcess
from arena_planners.bridge.transport import generate_pair


def _make_process(command: list[str]) -> PlannerProcess:
    endpoints = generate_pair("ipc")
    return PlannerProcess(command=command, endpoints=endpoints)


def _sleep_command(seconds: float = 30.0) -> list[str]:
    return [sys.executable, "-c", f"import time; time.sleep({seconds})"]


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------


def test_is_alive_after_start():
    proc = _make_process(_sleep_command())
    proc.start()
    try:
        assert proc.is_alive()
        assert proc.returncode is None
    finally:
        proc.stop(grace_seconds=0.5)


def test_stop_returns_within_grace():
    proc = _make_process(_sleep_command(30.0))
    proc.start()
    assert proc.is_alive()

    t0 = time.monotonic()
    rc = proc.stop(grace_seconds=0.5)
    elapsed = time.monotonic() - t0

    # SIGTERM delivers before grace expires; SIGKILL is the fallback.
    assert rc in (-signal.SIGTERM, -signal.SIGKILL)
    # Should finish well under 5 s even with SIGKILL path.
    assert elapsed < 5.0


def test_stop_returns_zero_for_clean_exit():
    proc = _make_process([sys.executable, "-c", "raise SystemExit(0)"])
    proc.start()
    # Wait for the process to exit naturally.
    for _ in range(50):
        if not proc.is_alive():
            break
        time.sleep(0.05)
    rc = proc.stop(grace_seconds=0.5)
    assert rc == 0


def test_stop_idempotent_after_exit():
    proc = _make_process([sys.executable, "-c", "raise SystemExit(0)"])
    proc.start()
    for _ in range(50):
        if not proc.is_alive():
            break
        time.sleep(0.05)
    rc1 = proc.stop(grace_seconds=0.5)
    rc2 = proc.stop(grace_seconds=0.5)
    assert rc1 == rc2
    assert rc2 == 0


def test_stop_before_start_is_safe():
    proc = _make_process(_sleep_command())
    rc = proc.stop(grace_seconds=0.5)
    assert rc == 0


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager_starts_and_stops():
    proc = _make_process(_sleep_command())
    with proc:
        assert proc.is_alive()
    assert not proc.is_alive()
    assert proc.returncode in (-signal.SIGTERM, -signal.SIGKILL)


def test_context_manager_stops_on_exception():
    with pytest.raises(RuntimeError, match="test"):
        with _make_process(_sleep_command()) as proc:
            assert proc.is_alive()
            raise RuntimeError("test")
    assert not proc.is_alive()


# ---------------------------------------------------------------------------
# Non-zero exit code surfacing
# ---------------------------------------------------------------------------


def test_nonzero_exit_code_surfaces():
    proc = _make_process([sys.executable, "-c", "import sys; sys.exit(42)"])
    proc.start()
    for _ in range(50):
        if not proc.is_alive():
            break
        time.sleep(0.05)
    rc = proc.stop(grace_seconds=0.5)
    assert rc == 42


# ---------------------------------------------------------------------------
# Extra env forwarded
# ---------------------------------------------------------------------------


def test_extra_env_forwarded():
    sentinel = "ARENA_PLANNER_TEST_SENTINEL"
    check_script = f"import os, sys; sys.exit(0 if os.environ.get('{sentinel}') == '1' else 1)"
    endpoints = generate_pair("ipc")
    proc = PlannerProcess(
        command=[sys.executable, "-c", check_script],
        endpoints=endpoints,
        extra_env={sentinel: "1"},
    )
    proc.start()
    for _ in range(50):
        if not proc.is_alive():
            break
        time.sleep(0.05)
    rc = proc.stop(grace_seconds=0.5)
    assert rc == 0


def test_endpoint_env_vars_forwarded():
    check_script = (
        "import os, sys; "
        "assert os.environ.get('ARENA_PLANNER_OBS_ENDPOINT'), 'obs missing'; "
        "assert os.environ.get('ARENA_PLANNER_ACTION_ENDPOINT'), 'action missing'; "
        "sys.exit(0)"
    )
    proc = _make_process([sys.executable, "-c", check_script])
    proc.start()
    for _ in range(50):
        if not proc.is_alive():
            break
        time.sleep(0.05)
    rc = proc.stop(grace_seconds=0.5)
    assert rc == 0
