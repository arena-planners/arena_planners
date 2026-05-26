"""KISS observation bridge: collectors + Pipeline + RobotPoseTFGenerator.

For sensor fusion / derived signals later: subclass Generator (from
.data_sources.base) and register the class in pipeline._TYPE_REGISTRY.
"""

from .data_sources.base import Collector, DataSource, Generator
from .pipeline import Pipeline

__all__ = ["Collector", "DataSource", "Generator", "Pipeline"]
