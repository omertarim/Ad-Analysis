from __future__ import annotations

from abc import ABC, abstractmethod

from task1.domain.trend import TrendSourceMetadata


class TrendCollector(ABC):
    """Source-independent interface for trend collectors."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable source identifier (e.g., platform name)."""

    @abstractmethod
    def collect(self, limit: int) -> list[TrendSourceMetadata]:
        """Collect trend items from a source up to the given limit."""

    @staticmethod
    def _validate_limit(limit: int) -> None:
        """Validate pagination/collection limit."""
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("limit must be an integer greater than 0")
        if limit <= 0:
            raise ValueError("limit must be greater than 0")
