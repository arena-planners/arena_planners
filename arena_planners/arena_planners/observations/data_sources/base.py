"""Base classes for observation data sources.

DataSource → Collector (topic-driven) | Generator (derived from other sources).

These are intentionally small. Heavyweight orchestration (dependency resolution,
schema validation, error aggregation, sync/wait_for_obs) lives in rosnav_rl.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from rclpy.clock import Clock, ClockType
from rclpy.qos import QoSProfile
from rclpy.time import Time

RosMessageType = TypeVar("RosMessageType")
ProcessedDataType = TypeVar("ProcessedDataType")


class DataSource(ABC):
    def __init__(self, name: str, **_: Any):
        self.name = name

    @abstractmethod
    def get_observation(self, obs_dict: dict[str, Any], **kwargs: Any) -> Any: ...

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class Collector(Generic[RosMessageType, ProcessedDataType], DataSource):  # noqa: UP046
    """Topic-subscribing data source. Subclasses provide `_preprocess`."""

    message_type: type[RosMessageType]
    data_class: type[ProcessedDataType]

    timeout: float = 0.1
    fallback_value: ProcessedDataType | None = None
    up_to_date_required: bool = False
    simulation_scoped: bool = False  # resolve topic at the simulation namespace, not the robot one

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for base in getattr(cls, "__orig_bases__", ()):
            origin = getattr(base, "__origin__", None)
            args = getattr(base, "__args__", ())
            if origin is Collector and len(args) == 2:
                cls.message_type = args[0]
                cls.data_class = args[1]
                break

    def __init__(
        self,
        name: str,
        topic: str,
        node: Any = None,
        up_to_date_required: bool = False,
        **kwargs: Any,
    ):
        super().__init__(name, **kwargs)
        self.topic = topic
        self._node = node
        self.up_to_date_required = up_to_date_required

        self._clock = Clock(clock_type=ClockType.ROS_TIME) if node else None
        self._value: ProcessedDataType | None = self._preprocess(self.message_type())
        self._stale: bool = True
        self._timestamp: Time | None = None
        self._qos_profile: QoSProfile | None = None
        self._latest_msg: RosMessageType | None = self.message_type()

        self.update_count: int = 0
        self.error_count: int = 0

    @abstractmethod
    def _preprocess(self, msg: RosMessageType) -> ProcessedDataType: ...

    def update(self, msg: RosMessageType) -> None:
        self._latest_msg = msg
        self._value = self._preprocess(msg)
        self._stale = False
        self._timestamp = self._clock.now() if self._clock else None
        self.update_count += 1

    @property
    def stale(self) -> bool:
        return self._stale

    @stale.setter
    def stale(self, value: bool) -> None:
        self._stale = bool(value)

    @property
    def age(self) -> float:
        if not self._timestamp or not self._clock:
            return float("inf")
        return (self._clock.now() - self._timestamp).nanoseconds / 1e9

    @property
    def timestamp(self) -> Time | None:
        return self._timestamp

    def set_qos_profile(self, profile: QoSProfile) -> None:
        self._qos_profile = profile

    @property
    def qos_profile(self) -> QoSProfile | None:
        return self._qos_profile

    def get_observation(self, *_: Any, **__: Any) -> ProcessedDataType | None:
        if self._stale:
            return self.fallback_value
        return self._value


class Generator(Generic[ProcessedDataType], DataSource):  # noqa: UP046
    """Derived data source. Declares deps in `requires`; computes via `_generate`."""

    requires: dict[str, Any] = {}
    data_class: type[ProcessedDataType]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for base in getattr(cls, "__orig_bases__", ()):
            origin = getattr(base, "__origin__", None)
            args = getattr(base, "__args__", ())
            if origin is Generator and len(args) == 1:
                cls.data_class = args[0]
                break

    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, **kwargs)
        self.required_keys = list(self.requires.keys())

    @abstractmethod
    def _generate(self, **kwargs: Any) -> ProcessedDataType: ...

    def reset(self) -> None:
        return None

    def get_observation(self, obs_dict: dict[str, Any], **kwargs: Any) -> ProcessedDataType | None:
        deps = {k: obs_dict[k] for k in self.required_keys if k in obs_dict}
        missing = [k for k in self.required_keys if k not in obs_dict]
        if missing:
            raise KeyError(f"Generator {self.name!r}: missing deps {missing}; have {sorted(obs_dict)}")
        return self._generate(**deps, **kwargs)
