from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from task1.collectors.apify_tiktok import ApifyTikTokTrendCollector
from task1.domain.trend import TrendSourceMetadata


def discover_trends(region: str, limit: int) -> list[TrendSourceMetadata]:
    collector = ApifyTikTokTrendCollector(region=region, feed="explore")
    return collector.collect(limit=limit)


def persist_trend_metadata(trend: TrendSourceMetadata, raw_root: Path) -> Path:
    trend_dir = raw_root / trend.trend_id
    trend_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = trend_dir / "trend_metadata.json"
    if metadata_path.exists():
        return metadata_path

    payload = {
        "trend_id": trend.trend_id,
        "source": trend.source,
        "source_url": trend.source_url,
        "caption": trend.caption,
        "hashtags": trend.hashtags,
        "author": trend.author,
        "published_at": trend.published_at.isoformat() if trend.published_at else None,
        "collected_at": trend.collected_at.isoformat(),
        "views": trend.views,
        "likes": trend.likes,
        "comments": trend.comments,
        "shares": trend.shares,
        "saves": trend.saves,
        "engagement_rate": trend.engagement_rate,
        "trend_rank": trend.trend_rank,
        "country": trend.country,
        "raw_source_metadata": trend.raw_source_metadata,
        "author_followers": trend.author_followers,
        "author_verified": trend.author_verified,
        "duration_seconds": trend.duration_seconds,
        "thumbnail_url": trend.thumbnail_url,
        "video_url": trend.video_url,
        "audio_name": trend.audio_name,
        "audio_id": trend.audio_id,
        "language": trend.language,
    }
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata_path


def fetch_kv_urls_for_trends(trends: list[TrendSourceMetadata], apify_token: str) -> dict[str, str]:
    if not trends:
        return {}
    run = _run_apify_actor(
        actor_id="khadinakbar/tiktok-no-watermark-downloader",
        actor_input={
            "videoUrls": [trend.source_url for trend in trends],
            "extractAudio": False,
            "includeWatermarked": False,
            "includeCover": True,
            "providerOrder": "scrapecreators-first",
            "maxVideos": len(trends),
        },
        token=apify_token,
    )
    rows = _dataset_items(run["defaultDatasetId"], apify_token)
    kv_by_video_id: dict[str, str] = {}
    kv_by_video_url: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        kv = row.get("kvNoWatermarkUrl")
        if not isinstance(kv, str):
            continue
        video_id = row.get("videoId")
        video_url = row.get("videoUrl")
        if isinstance(video_id, str):
            kv_by_video_id[video_id] = kv
        if isinstance(video_url, str):
            kv_by_video_url[video_url] = kv

    resolved: dict[str, str] = {}
    for trend in trends:
        video_id = trend.raw_source_metadata.get("videoId")
        if isinstance(video_id, str) and video_id in kv_by_video_id:
            resolved[trend.trend_id] = kv_by_video_id[video_id]
            continue
        if trend.source_url in kv_by_video_url:
            resolved[trend.trend_id] = kv_by_video_url[trend.source_url]
    return resolved


def _run_apify_actor(actor_id: str, actor_input: dict[str, Any], token: str) -> dict[str, Any]:
    run_url = (
        f"https://api.apify.com/v2/acts/{quote(actor_id, safe='')}/runs?"
        + urlencode({"token": token})
    )
    request = Request(
        run_url,
        data=json.dumps(actor_input).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        run_data = json.loads(response.read().decode("utf-8"))["data"]
    run_id = run_data["id"]
    start = time.time()
    while True:
        status_url = f"https://api.apify.com/v2/actor-runs/{run_id}?" + urlencode({"token": token})
        with urlopen(Request(status_url, headers={"Accept": "application/json"}), timeout=60) as response:
            status_data = json.loads(response.read().decode("utf-8"))["data"]
        status = status_data.get("status")
        if status == "SUCCEEDED":
            return status_data
        if status in {"FAILED", "ABORTED", "TIMED-OUT"}:
            raise RuntimeError(f"{actor_id} ended with status={status}")
        if time.time() - start > 420:
            raise TimeoutError(f"{actor_id} timeout")
        time.sleep(3)


def _dataset_items(dataset_id: str, token: str) -> list[dict[str, Any]]:
    url = (
        f"https://api.apify.com/v2/datasets/{dataset_id}/items?"
        + urlencode({"token": token, "format": "json", "clean": "true"})
    )
    with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]
