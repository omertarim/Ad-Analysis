from datetime import datetime, timezone

from task1.domain.trend import TrendSourceMetadata, generate_trend_id


def test_model_builds_with_minimum_required_fields() -> None:
    item = TrendSourceMetadata(
        source="tiktok",
        source_url="https://example.com/video/123",
        collected_at=datetime.now(timezone.utc),
    )
    assert item.source == "tiktok"
    assert item.source_url == "https://example.com/video/123"
    assert item.trend_id


def test_optional_fields_can_be_empty() -> None:
    item = TrendSourceMetadata(
        source="instagram",
        source_url="https://example.com/reel/abc",
        collected_at=datetime.now(timezone.utc),
    )
    assert item.caption is None
    assert item.author is None
    assert item.published_at is None
    assert item.views is None
    assert item.likes is None
    assert item.comments is None
    assert item.shares is None
    assert item.duration_seconds is None
    assert item.thumbnail_url is None
    assert item.video_url is None
    assert item.audio_name is None
    assert item.audio_id is None
    assert item.language is None
    assert item.country is None
    assert item.saves is None
    assert item.engagement_rate is None
    assert item.author_followers is None
    assert item.author_verified is None
    assert item.trend_rank is None


def test_same_source_and_url_generate_same_id() -> None:
    source = "tiktok"
    source_url = "https://example.com/video/same"
    id1 = generate_trend_id(source, source_url)
    id2 = generate_trend_id(source, source_url)
    assert id1 == id2


def test_different_url_generates_different_id() -> None:
    source = "tiktok"
    id1 = generate_trend_id(source, "https://example.com/video/1")
    id2 = generate_trend_id(source, "https://example.com/video/2")
    assert id1 != id2


def test_hashtags_default_is_empty_list() -> None:
    item = TrendSourceMetadata(
        source="youtube-shorts",
        source_url="https://example.com/shorts/1",
        collected_at=datetime.now(timezone.utc),
    )
    assert item.hashtags == []


def test_raw_source_metadata_default_is_empty_dict() -> None:
    item = TrendSourceMetadata(
        source="youtube-shorts",
        source_url="https://example.com/shorts/2",
        collected_at=datetime.now(timezone.utc),
    )
    assert item.raw_source_metadata == {}
