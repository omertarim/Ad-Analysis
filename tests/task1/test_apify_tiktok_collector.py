import json
from datetime import datetime, timezone
from pathlib import Path

from task1.collectors.apify_tiktok import ApifyTikTokTrendCollector


def _load_fixture() -> list[dict]:
    fixture_path = Path(__file__).parent / "fixtures" / "apify_trending_raw.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def test_normalization_mapping_from_raw_fixture() -> None:
    collector = ApifyTikTokTrendCollector(region="TR")
    raw = _load_fixture()[0]
    item = collector._normalize_item(raw, collected_at=datetime.now(timezone.utc))

    assert item.source == "tiktok_trending_apify"
    assert item.source_url == raw["videoUrl"]
    assert item.video_url == raw["videoPlayUrl"]
    assert item.caption == raw["description"]
    assert item.author == raw["authorHandle"]
    assert item.published_at is not None
    assert item.views == raw["playCount"]
    assert item.likes == raw["likeCount"]
    assert item.comments == raw["commentCount"]
    assert item.shares == raw["shareCount"]
    assert item.duration_seconds == float(raw["durationSec"])
    assert item.thumbnail_url == raw["coverUrl"]
    assert item.audio_name == raw["musicTitle"]
    assert item.audio_id == raw["musicId"]
    assert item.country == raw["region"]
    assert item.language is None
    assert item.raw_source_metadata["rank"] == raw["rank"]
    assert item.raw_source_metadata["saveCount"] == raw["saveCount"]
    assert item.raw_source_metadata["engagementRate"] == raw["engagementRate"]
    assert item.hashtags == ["#iphone15", "#keşfet", "#öneçıkar", "#keşfetteyizzz"]
    assert item.saves == raw["saveCount"]
    assert item.engagement_rate == raw["engagementRate"]
    assert item.author_followers == raw["authorFollowers"]
    assert item.author_verified == raw["authorVerified"]
    assert item.trend_rank == raw["rank"]


def test_description_hashtag_fallback_when_raw_hashtags_empty() -> None:
    collector = ApifyTikTokTrendCollector(region="TR")
    raw = _load_fixture()[1].copy()
    raw["description"] = "Yeni trend #kesfet #fyp"
    raw["hashtags"] = []
    item = collector._normalize_item(raw, collected_at=datetime.now(timezone.utc))
    assert item.hashtags == ["#kesfet", "#fyp"]


def test_collect_skips_ads_and_duplicates(monkeypatch) -> None:
    collector = ApifyTikTokTrendCollector(region="TR")
    fixture = _load_fixture()

    ad_item = fixture[0].copy()
    ad_item["videoId"] = "ad-video"
    ad_item["isAd"] = True

    duplicate_item = fixture[0].copy()
    duplicate_item["rank"] = 999

    payload = [ad_item, fixture[0], duplicate_item, fixture[1], fixture[2]]

    monkeypatch.setattr(collector, "apify_token", "fake-token")
    monkeypatch.setattr(collector, "_run_actor", lambda _: {"id": "run1", "defaultDatasetId": "ds1"})
    monkeypatch.setattr(collector, "_wait_for_run", lambda _: None)
    monkeypatch.setattr(collector, "_fetch_dataset_items", lambda _: payload)

    items = collector.collect(limit=10)
    assert len(items) == 3
    assert all(item.raw_source_metadata.get("isAd") is False for item in items)
    assert len({item.source_url for item in items}) == 3
