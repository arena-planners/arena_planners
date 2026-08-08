"""Costmap grid wrapper for clearance queries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# nav2 publishes its costmap as an OccupancyGrid rescaled to 0-100, where the
# inscribed (253) and lethal (254) costs land on 99 and 100. A robot centre at or
# above the inscribed cost is in collision by definition, which makes 99 the
# threshold regardless of the robot's footprint.
_BLOCKED = 99


@dataclass(frozen=True)
class CostmapGrid:
    """Occupancy costmap with world->cell lookup. Not wire-serialisable by design."""

    data: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float

    def is_clear(self, x: float, y: float) -> bool:
        """True if a robot centre at (x, y) is outside the inscribed region; unknown is unobserved, not blocked."""
        if self.resolution <= 0.0 or self.data.size == 0:
            return True
        col = int((x - self.origin_x) / self.resolution)
        row = int((y - self.origin_y) / self.resolution)
        if not (0 <= row < self.data.shape[0] and 0 <= col < self.data.shape[1]):
            return True
        return int(self.data[row, col]) < _BLOCKED
