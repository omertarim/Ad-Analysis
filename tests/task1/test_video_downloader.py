from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError

import pytest

from task1.domain.trend import TrendSourceMetadata
from task1.services.video_acquisition import VideoAcquisitionService


class FakeResponse:
    def __init__(self, body: bytes, content_type: str = "video/mp4", status: int = 200) -> None:
        self._body = body
        self._offset = 0
        self.status = status
        self.headers = {"Content-Type": content_type}

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._body) - self._offset
        chunk = self._body[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[no-untyped-def]
        return False


def _make_trend(video_url: str | None) -> TrendSourceMetadata:
    return TrendSourceMetadata(
        source="tiktok_trending_apify",
        source_url="https://www.tiktok.com/@creator/video/123",
        video_url=video_url,
        collected_at=datetime.now(timezone.utc),
    )


def test_successful_mocked_download(tmp_path: Path) -> None:
    body = b"fake-mp4-bytes"

    def opener(_request, timeout=0):  # type: ignore[no-untyped-def]
        return FakeResponse(body=body, content_type="video/mp4", status=200)

    downloader = VideoAcquisitionService(opener=opener)
    trend = _make_trend("https://cdn.example.com/video.mp4")
    result = downloader.acquire(trend=trend, output_dir=tmp_path)

    assert result.success is True
    assert result.local_path is not None
    assert result.bytes_downloaded == len(body)
    assert Path(result.local_path).exists()
    assert Path(result.local_path).name == "source.mp4"


def test_missing_video_url_returns_failure(tmp_path: Path) -> None:
    downloader = VideoAcquisitionService()
    trend = _make_trend(None)
    result = downloader.acquire(trend=trend, output_dir=tmp_path)

    assert result.success is False
    assert result.error == "video_url is missing"
    assert result.local_path is None


def test_http_failure_returns_failure(tmp_path: Path) -> None:
    def opener(_request, timeout=0):  # type: ignore[no-untyped-def]
        raise HTTPError(
            url="https://cdn.example.com/video.mp4",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=None,
        )

    downloader = VideoAcquisitionService(opener=opener)
    trend = _make_trend("https://cdn.example.com/video.mp4")
    result = downloader.acquire(trend=trend, output_dir=tmp_path)

    assert result.success is False
    assert result.local_path is None
    assert result.bytes_downloaded == 0
    assert "500" in (result.error or "")


def test_empty_response_returns_failure(tmp_path: Path) -> None:
    def opener(_request, timeout=0):  # type: ignore[no-untyped-def]
        return FakeResponse(body=b"", content_type="video/mp4", status=200)

    downloader = VideoAcquisitionService(opener=opener)
    trend = _make_trend("https://cdn.example.com/video.mp4")
    result = downloader.acquire(trend=trend, output_dir=tmp_path)

    assert result.success is False
    assert result.error == "empty_download"
    assert result.local_path is None


def test_apify_kv_url_download_uses_authenticated_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = b"mp4-bytes"
    seen = {}

    def opener(request, timeout=0):  # type: ignore[no-untyped-def]
        seen["url"] = request.full_url
        seen["auth"] = request.get_header("Authorization")
        return FakeResponse(body=body, content_type="video/mp4", status=200)

    monkeypatch.setenv("APIFY_API_TOKEN", "test_token")
    downloader = VideoAcquisitionService(opener=opener)
    trend = _make_trend(None)
    trend.raw_source_metadata["kvNoWatermarkUrl"] = (
        "https://api.apify.com/v2/key-value-stores/store123/records/video__abc.mp4"
    )
    result = downloader.acquire(trend=trend, output_dir=tmp_path)

    assert result.success is True
    assert "token=test_token" in (seen.get("url") or "")
    assert seen.get("auth") == "Bearer test_token"


def test_apify_kv_url_requires_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    downloader = VideoAcquisitionService()
    trend = _make_trend(None)
    trend.raw_source_metadata["kvNoWatermarkUrl"] = (
        "https://api.apify.com/v2/key-value-stores/store123/records/video__abc.mp4"
    )
    result = downloader.acquire(trend=trend, output_dir=tmp_path)
    assert result.success is False
    assert result.error == "APIFY_API_TOKEN is required for kvNoWatermarkUrl download"
