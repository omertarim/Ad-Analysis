from datetime import datetime, timezone

import pytest

from task1.collectors.base import TrendCollector
from task1.domain.trend import TrendSourceMetadata


class FakeCollector(TrendCollector):
    source_name = "fake-source"

    def collect(self, limit: int) -> list[TrendSourceMetadata]:
        self._validate_limit(limit)
        return [
            TrendSourceMetadata(
                source=self.source_name,
                source_url=f"https://example.com/{i}",
                collected_at=datetime.now(timezone.utc),
            )
            for i in range(limit)
        ]


def test_abstract_collector_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        TrendCollector()


def test_fake_collector_collect_works() -> None:
    collector = FakeCollector()
    items = collector.collect(limit=3)
    assert len(items) == 3


def test_collect_returns_trend_metadata_list() -> None:
    collector = FakeCollector()
    items = collector.collect(limit=2)
    assert isinstance(items, list)
    assert all(isinstance(item, TrendSourceMetadata) for item in items)


@pytest.mark.parametrize("invalid_limit", [0, -1, -10])
def test_invalid_limit_is_rejected(invalid_limit: int) -> None:
    collector = FakeCollector()
    with pytest.raises(ValueError):
        collector.collect(limit=invalid_limit)
