from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from task1.collectors.base import TrendCollector
from task1.domain.trend import TrendSourceMetadata

logger = logging.getLogger(__name__)

HASHTAG_REGEX = re.compile(r"#\w+")


class ApifyTikTokTrendCollector(TrendCollector):
    """Collect TikTok trending videos from Apify."""

    source_name = "tiktok_trending_apify"
    default_actor_id = "khadinakbar/tiktok-trending-videos-scraper"
    default_feed = "explore"

    def __init__(
        self,
        actor_id: str | None = None,
        region: str = "TR",
        feed: str = "explore",
    ) -> None:
        self.apify_token = os.getenv("APIFY_API_TOKEN")
        self.actor_id = actor_id or self.default_actor_id
        self.region = region.upper()
        self.feed = feed

    def collect(self, limit: int) -> list[TrendSourceMetadata]:
        self._validate_limit(limit)
        if not self.apify_token:
            raise ValueError("APIFY_API_TOKEN is missing. Add it to .env before running.")

        actor_input = {
            "feed": self.feed,
            "maxResults": limit,
            "proxyConfiguration": {
                "useApifyProxy": True,
                "apifyProxyGroups": ["RESIDENTIAL"],
                "apifyProxyCountry": self.region,
            },
            "region": self.region,
        }
        logger.info(
            "[APIFY] discovery_input actor_id=%s feed=%s region=%s maxResults=%s proxy_group=%s proxy_country=%s",
            self.actor_id,
            actor_input["feed"],
            actor_input["region"],
            actor_input["maxResults"],
            actor_input["proxyConfiguration"]["apifyProxyGroups"],
            actor_input["proxyConfiguration"]["apifyProxyCountry"],
        )

        run = self._run_actor(actor_input)
        run_id = run.get("id")
        dataset_id = run.get("defaultDatasetId")
        if not run_id or not dataset_id:
            raise RuntimeError("Apify run did not return run id or dataset id")

        self._wait_for_run(run_id)
        rows = self._fetch_dataset_items(dataset_id)

        collected_at = datetime.now(UTC)
        normalized: list[TrendSourceMetadata] = []
        seen_video_urls: set[str] = set()
        seen_video_ids: set[str] = set()

        for row in rows:
            if not isinstance(row, dict):
                logger.warning("Skipping non-dict row: %r", row)
                continue

            if row.get("isAd") is True:
                logger.warning("Skipping ad row videoId=%s", row.get("videoId"))
                continue

            video_url = row.get("videoUrl")
            video_id = row.get("videoId")
            if isinstance(video_url, str) and video_url in seen_video_urls:
                continue
            if isinstance(video_id, str) and video_id in seen_video_ids:
                continue

            try:
                item = self._normalize_item(row=row, collected_at=collected_at)
                normalized.append(item)

                if isinstance(video_url, str):
                    seen_video_urls.add(video_url)
                if isinstance(video_id, str):
                    seen_video_ids.add(video_id)

                if len(normalized) >= limit:
                    break
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping malformed row videoId=%s videoUrl=%s: %s",
                    row.get("videoId"),
                    row.get("videoUrl"),
                    exc,
                )
                continue

        return normalized

    def _run_actor(self, actor_input: dict[str, Any]) -> dict[str, Any]:
        actor_id_encoded = quote(self.actor_id, safe="")
        url = (
            f"https://api.apify.com/v2/acts/{actor_id_encoded}/runs"
            f"?{urlencode({'token': self.apify_token})}"
        )
        payload = json.dumps(actor_input).encode("utf-8")
        request = Request(
            url=url,
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        response = self._request_json(request)
        data = response.get("data")
        if not isinstance(data, dict):
            raise RuntimeError("Unexpected Apify run response format")
        logger.info(
            "[APIFY] run_started run_id=%s actor_id=%s started_at=%s status=%s",
            data.get("id"),
            self.actor_id,
            data.get("startedAt"),
            data.get("status"),
        )
        return data

    def _wait_for_run(self, run_id: str) -> None:
        start = time.time()
        last_log_at = 0.0
        while True:
            run_data = self._get_run(run_id)
            status = run_data.get("status")
            now = time.time()
            elapsed_seconds = int(now - start)
            if elapsed_seconds == 0 or (now - last_log_at) >= 10:
                logger.info(
                    "[APIFY] run_id=%s status=%s elapsed=%ss statusMessage=%s startedAt=%s finishedAt=%s computeUnits=%s defaultDatasetId=%s defaultKeyValueStoreId=%s",
                    run_id,
                    status,
                    elapsed_seconds,
                    run_data.get("statusMessage"),
                    run_data.get("startedAt"),
                    run_data.get("finishedAt"),
                    run_data.get("usage", {}).get("ACTOR_COMPUTE_UNITS")
                    if isinstance(run_data.get("usage"), dict)
                    else None,
                    run_data.get("defaultDatasetId"),
                    run_data.get("defaultKeyValueStoreId"),
                )
                last_log_at = now
            if status == "SUCCEEDED":
                logger.info(
                    "[APIFY] run_id=%s final_status=%s elapsed=%ss statusMessage=%s startedAt=%s finishedAt=%s computeUnits=%s defaultDatasetId=%s defaultKeyValueStoreId=%s",
                    run_id,
                    status,
                    elapsed_seconds,
                    run_data.get("statusMessage"),
                    run_data.get("startedAt"),
                    run_data.get("finishedAt"),
                    run_data.get("usage", {}).get("ACTOR_COMPUTE_UNITS")
                    if isinstance(run_data.get("usage"), dict)
                    else None,
                    run_data.get("defaultDatasetId"),
                    run_data.get("defaultKeyValueStoreId"),
                )
                return
            if status in {"FAILED", "TIMED-OUT", "ABORTED"}:
                raise RuntimeError(
                    "Apify actor run failed "
                    f"status={status} statusMessage={run_data.get('statusMessage')} "
                    f"run_id={run_id} startedAt={run_data.get('startedAt')} finishedAt={run_data.get('finishedAt')} "
                    f"defaultDatasetId={run_data.get('defaultDatasetId')} defaultKeyValueStoreId={run_data.get('defaultKeyValueStoreId')}"
                )
            time.sleep(2)

    def _get_run(self, run_id: str) -> dict[str, Any]:
        url = (
            f"https://api.apify.com/v2/actor-runs/{run_id}"
            f"?{urlencode({'token': self.apify_token})}"
        )
        request = Request(url=url, headers={"Accept": "application/json"}, method="GET")
        response = self._request_json(request)
        data = response.get("data")
        if not isinstance(data, dict):
            raise RuntimeError("Unexpected Apify run status response")
        return data

    def _fetch_dataset_items(self, dataset_id: str) -> list[dict[str, Any]]:
        url = (
            f"https://api.apify.com/v2/datasets/{dataset_id}/items"
            f"?{urlencode({'token': self.apify_token, 'format': 'json', 'clean': 'true'})}"
        )
        request = Request(url=url, headers={"Accept": "application/json"}, method="GET")
        data = self._request_json(request)
        if not isinstance(data, list):
            raise RuntimeError("Unexpected Apify dataset response format")
        return [row for row in data if isinstance(row, dict)]

    def _request_json(self, request: Request) -> Any:
        try:
            with urlopen(request, timeout=40) as response:
                body = response.read().decode("utf-8")
                return json.loads(body)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {exc.code}: {body[:400]}") from exc
        except URLError as exc:
            raise RuntimeError(f"Network error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("Request timed out") from exc

    def _normalize_item(self, row: dict[str, Any], collected_at: datetime) -> TrendSourceMetadata:
        source_url = row.get("videoUrl")
        if not self._is_valid_tiktok_post_url(source_url):
            raise ValueError("missing valid TikTok post URL in videoUrl")

        hashtags = self._normalize_hashtags(row.get("hashtags"))
        if not hashtags:
            hashtags = self._extract_hashtags_from_description(row.get("description"))
        description = row.get("description")
        caption = description if isinstance(description, str) else None

        return TrendSourceMetadata(
            source=self.source_name,
            source_url=source_url,
            video_url=self._as_http_str(row.get("videoPlayUrl")),
            caption=caption,
            hashtags=hashtags,
            author=self._as_str(row.get("authorHandle")),
            published_at=self._parse_datetime(row.get("createTime")),
            collected_at=collected_at,
            views=self._as_int(row.get("playCount")),
            likes=self._as_int(row.get("likeCount")),
            comments=self._as_int(row.get("commentCount")),
            shares=self._as_int(row.get("shareCount")),
            duration_seconds=self._as_float(row.get("durationSec")),
            thumbnail_url=self._as_http_str(row.get("coverUrl")),
            audio_name=self._as_str(row.get("musicTitle")),
            audio_id=self._as_str(row.get("musicId")),
            language=None,
            country=self._as_str(row.get("region")),
            saves=self._as_int(row.get("saveCount")),
            engagement_rate=self._as_float(row.get("engagementRate")),
            author_followers=self._as_int(row.get("authorFollowers")),
            author_verified=self._as_bool(row.get("authorVerified")),
            trend_rank=self._as_int(row.get("rank")),
            raw_source_metadata=row,
        )

    @staticmethod
    def _is_valid_tiktok_post_url(value: Any) -> bool:
        return (
            isinstance(value, str)
            and value.startswith("https://www.tiktok.com/@")
            and "/video/" in value
        )

    @staticmethod
    def _normalize_hashtags(raw_hashtags: Any) -> list[str]:
        if not isinstance(raw_hashtags, list):
            return []
        normalized: list[str] = []
        for tag in raw_hashtags:
            if not isinstance(tag, str):
                continue
            clean = tag.strip()
            if not clean:
                continue
            normalized.append(clean if clean.startswith("#") else f"#{clean}")
        return normalized

    @staticmethod
    def _extract_hashtags_from_description(description: Any) -> list[str]:
        if not isinstance(description, str):
            return []
        return HASHTAG_REGEX.findall(description)

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None

    @staticmethod
    def _as_int(value: Any) -> int | None:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            compact = value.replace(",", "").strip()
            if compact.isdigit():
                return int(compact)
        return None

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return None
        return None

    @staticmethod
    def _as_str(value: Any) -> str | None:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    @staticmethod
    def _as_http_str(value: Any) -> str | None:
        if isinstance(value, str) and value.startswith("http"):
            return value
        return None

    @staticmethod
    def _as_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        return None
