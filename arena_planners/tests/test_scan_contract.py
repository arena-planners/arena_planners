"""canonical_scan yields one layout from every simulator's LaserScan convention."""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np

_SCAN_PY = pathlib.Path(__file__).parents[1] / "arena_planners" / "observations" / "utils" / "scan.py"
_spec = importlib.util.spec_from_file_location("scan", _SCAN_PY)
assert _spec is not None and _spec.loader is not None
_scan_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scan_mod)
canonical_scan = _scan_mod.canonical_scan

RANGE_MIN, RANGE_MAX, BEAMS = 0.2, 12.0, 720


def _room(bearing: np.ndarray) -> np.ndarray:
    """Square room, 4 m half-width, robot at center: range along each bearing."""
    return 4.0 / np.maximum(np.abs(np.cos(bearing)), np.abs(np.sin(bearing)))


def _scan(n: int, angle_min: float, increment: float, miss: float, miss_from: float = 5.0) -> np.ndarray:
    bearing = angle_min + increment * np.arange(n)
    r = _room(bearing)
    r[r > miss_from] = miss
    return r.astype(np.float32)


def _canon(n: int, angle_min: float, increment: float, miss: float) -> np.ndarray:
    return canonical_scan(_scan(n, angle_min, increment, miss), angle_min, increment, RANGE_MIN, RANGE_MAX, BEAMS)


def _away_from_edges(*scans: np.ndarray) -> np.ndarray:
    """Mask out rays within two beams of a hit/miss discontinuity, where interpolation is placement-dependent."""
    keep = np.ones(BEAMS, dtype=bool)
    for a in scans:
        edge = np.abs(np.diff(a, append=a[0])) > 1.0
        for k in range(-2, 3):
            keep &= ~np.roll(edge, k)
    return keep


def test_gz_and_isaac_conventions_agree():
    gz = _canon(640, -np.pi, 2 * np.pi / 639, np.inf)
    isaac = _canon(900, 0.0, 2 * np.pi / 900, -1.0)
    clockwise = _canon(720, np.pi, -2 * np.pi / 720, np.nan)
    assert gz.shape == isaac.shape == clockwise.shape == (BEAMS,)
    keep = _away_from_edges(gz, isaac, clockwise)
    assert keep.sum() > BEAMS * 0.9
    np.testing.assert_allclose(gz[keep], isaac[keep], atol=0.05)
    np.testing.assert_allclose(gz[keep], clockwise[keep], atol=0.05)


def test_layout_is_heading_first_ccw():
    out = _canon(640, -np.pi, 2 * np.pi / 639, np.inf)
    phi = 2 * np.pi * np.arange(BEAMS) / BEAMS
    expected = np.minimum(_room(phi), RANGE_MAX)
    expected[expected > 5.0] = RANGE_MAX
    keep = _away_from_edges(out, expected)
    np.testing.assert_allclose(out[keep], expected[keep], atol=0.05)


def test_misses_and_floor():
    out = _canon(900, 0.0, 2 * np.pi / 900, -1.0)
    assert np.all(out >= RANGE_MIN)
    assert np.all(np.isfinite(out))
    assert np.any(out == RANGE_MAX)


def test_sector_pads_uncovered_bearings():
    n, inc = 180, np.pi / 179
    front = canonical_scan(_scan(n, -np.pi / 2, inc, np.inf), -np.pi / 2, inc, RANGE_MIN, RANGE_MAX, BEAMS)
    rear = canonical_scan(_scan(n, np.pi / 2, inc, np.inf), np.pi / 2, inc, RANGE_MIN, RANGE_MAX, BEAMS)
    q, m = BEAMS // 4, 4
    assert np.all(front[q + m : 3 * q - m] == RANGE_MAX)
    assert np.any(front[:q] < RANGE_MAX)
    assert np.all(rear[m : q - m] == RANGE_MAX) and np.all(rear[3 * q + m : -m] == RANGE_MAX)
    assert np.any(rear[q + m : 3 * q - m] < RANGE_MAX)


def test_degenerate_inputs():
    assert canonical_scan(np.array([]), 0.0, 0.1, RANGE_MIN, RANGE_MAX, BEAMS).size == 0
    out = canonical_scan(np.array([1.0]), 0.0, 0.0, RANGE_MIN, RANGE_MAX, BEAMS)
    assert out.shape == (BEAMS,) and np.all(out == RANGE_MAX)
