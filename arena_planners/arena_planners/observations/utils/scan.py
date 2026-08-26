"""Canonical 2D scan: `beams` rays over a full circle, ray 0 along the robot heading, CCW."""

from __future__ import annotations

import numpy as np

_TAU = 2.0 * np.pi


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % _TAU - np.pi


def canonical_scan(
    ranges: np.ndarray,
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    beams: int,
) -> np.ndarray:
    """Resample a LaserScan to the canonical layout from its own angle_min/increment.

    Non-returns (nan, inf, any value <= 0 or below range_min) read as range_max, values
    clip to [range_min, range_max], bearings no beam covers fill with range_max.
    """
    r = np.array(ranges, dtype=np.float32)
    n = r.size
    if n == 0:
        return r
    r[~np.isfinite(r) | (r <= 0.0) | (r < range_min)] = range_max
    np.clip(r, range_min, range_max, out=r)
    if n < 2 or angle_increment == 0.0:
        return np.full(beams, range_max, dtype=np.float32)

    theta = _wrap(angle_min + angle_increment * np.arange(n, dtype=np.float64))
    order = np.argsort(theta, kind="stable")
    theta = theta[order]
    r = r[order]
    phi = _wrap((_TAU / beams) * np.arange(beams, dtype=np.float64))
    out = np.interp(phi, theta, r, period=_TAU)

    idx = np.searchsorted(theta, phi)
    gap = np.minimum(
        np.abs(_wrap(phi - theta[(idx - 1) % n])),
        np.abs(_wrap(phi - theta[idx % n])),
    )
    out[gap > 1.5 * abs(angle_increment)] = range_max
    return out.astype(np.float32)
