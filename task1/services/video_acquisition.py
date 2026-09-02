from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

from task1.domain.trend import TrendSourceMetadata


@dataclass
class VideoAcquisitionResult:
    trend_id: str
    success: bool
    local_path: str | None
    bytes_downloaded: int
    error: str | None


class VideoAcquisitionService:
    def __init__(
        self,
        timeout_seconds: int = 30,
        opener: Callable[..., BinaryIO] | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self._opener = opener or urlopen
        self.apify_api_token = os.getenv("APIFY_API_TOKEN")

    def acquire(self, trend: TrendSourceMetadata, output_dir: str | Path) -> VideoAcquisitionResult:
        resolved = self._resolve_download_url(trend)
        if resolved.error:
            return VideoAcquisitionResult(
                trend_id=trend.trend_id,
                success=False,
                local_path=None,
                bytes_downloaded=0,
                error=resolved.error,
            )
        assert resolved.url is not None

        output_root = Path(output_dir)
        target_dir = output_root / trend.trend_id
        target_path = target_dir / "source.mp4"
        temp_path = target_dir / "source.mp4.part"

        target_dir.mkdir(parents=True, exist_ok=True)
        if temp_path.exists():
            temp_path.unlink()

        request = Request(
            resolved.url,
            headers={
                "User-Agent": "dream-games-case-study/0.1",
                "Accept": "*/*",
            },
        )
        if resolved.is_apify_kv and self.apify_api_token:
            request.add_header("Authorization", f"Bearer {self.apify_api_token}")

        bytes_downloaded = 0
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                status = getattr(response, "status", 200)
                if status < 200 or status >= 300:
                    return VideoAcquisitionResult(
                        trend_id=trend.trend_id,
                        success=False,
                        local_path=None,
                        bytes_downloaded=0,
                        error=f"http_status={status}",
                    )

                content_type = (response.headers.get("Content-Type") or "").lower()
                if not self._is_valid_video_content_type(content_type):
                    return VideoAcquisitionResult(
                        trend_id=trend.trend_id,
                        success=False,
                        local_path=None,
                        bytes_downloaded=0,
                        error=f"invalid_content_type={content_type}",
                    )

                with temp_path.open("wb") as out_file:
                    bytes_downloaded = self._stream_to_file(response=response, out_file=out_file)

            if not temp_path.exists():
                return VideoAcquisitionResult(
                    trend_id=trend.trend_id,
                    success=False,
                    local_path=None,
                    bytes_downloaded=0,
                    error="file_not_created",
                )
            if bytes_downloaded <= 0:
                temp_path.unlink(missing_ok=True)
                return VideoAcquisitionResult(
                    trend_id=trend.trend_id,
                    success=False,
                    local_path=None,
                    bytes_downloaded=0,
                    error="empty_download",
                )

            temp_path.replace(target_path)
            final_size = target_path.stat().st_size if target_path.exists() else 0
            if final_size <= 0:
                target_path.unlink(missing_ok=True)
                return VideoAcquisitionResult(
                    trend_id=trend.trend_id,
                    success=False,
                    local_path=None,
                    bytes_downloaded=0,
                    error="zero_byte_file",
                )

            return VideoAcquisitionResult(
                trend_id=trend.trend_id,
                success=True,
                local_path=str(target_path),
                bytes_downloaded=final_size,
                error=None,
            )
        except (HTTPError, URLError, TimeoutError) as exc:
            temp_path.unlink(missing_ok=True)
            return VideoAcquisitionResult(
                trend_id=trend.trend_id,
                success=False,
                local_path=None,
                bytes_downloaded=0,
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            temp_path.unlink(missing_ok=True)
            return VideoAcquisitionResult(
                trend_id=trend.trend_id,
                success=False,
                local_path=None,
                bytes_downloaded=0,
                error=f"unexpected_error={exc}",
            )

    def acquire_many(
        self,
        trends: list[TrendSourceMetadata],
        output_dir: str | Path,
    ) -> list[VideoAcquisitionResult]:
        return [self.acquire(trend=trend, output_dir=output_dir) for trend in trends]

    @staticmethod
    def _stream_to_file(response: BinaryIO, out_file: BinaryIO) -> int:
        total = 0
        while True:
            chunk = response.read(1024 * 512)
            if not chunk:
                break
            out_file.write(chunk)
            total += len(chunk)
        return total

    def _resolve_download_url(self, trend: TrendSourceMetadata) -> "_ResolvedDownload":
        kv_url = trend.raw_source_metadata.get("kvNoWatermarkUrl")
        if isinstance(kv_url, str) and kv_url:
            if not self.apify_api_token:
                return _ResolvedDownload(
                    url=None,
                    is_apify_kv=True,
                    error="APIFY_API_TOKEN is required for kvNoWatermarkUrl download",
                )
            parsed = self._parse_apify_kv_url(kv_url)
            if not parsed:
                return _ResolvedDownload(
                    url=None,
                    is_apify_kv=True,
                    error="invalid_kvNoWatermarkUrl",
                )
            auth_url = self._build_authenticated_apify_url(kv_url, self.apify_api_token)
            return _ResolvedDownload(url=auth_url, is_apify_kv=True, error=None)

        if trend.video_url:
            return _ResolvedDownload(url=trend.video_url, is_apify_kv=False, error=None)

        return _ResolvedDownload(url=None, is_apify_kv=False, error="video_url is missing")

    @staticmethod
    def _parse_apify_kv_url(url: str) -> tuple[str, str] | None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return None
        if parsed.netloc != "api.apify.com":
            return None
        segments = [s for s in parsed.path.split("/") if s]
        if len(segments) < 5:
            return None
        if segments[0] != "v2" or segments[1] != "key-value-stores" or segments[3] != "records":
            return None
        store_id = segments[2]
        record_key = "/".join(segments[4:])
        if not store_id or not record_key:
            return None
        return store_id, record_key

    @staticmethod
    def _build_authenticated_apify_url(url: str, token: str) -> str:
        parsed = urlparse(url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query["token"] = token
        new_query = urlencode(query)
        return urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                new_query,
                parsed.fragment,
            )
        )

    @staticmethod
    def _is_valid_video_content_type(content_type: str) -> bool:
        if not content_type:
            return False
        normalized = content_type.split(";")[0].strip()
        if normalized.startswith("video/"):
            return True
        if normalized in {"application/octet-stream", "binary/octet-stream"}:
            return True
        return False


@dataclass
class _ResolvedDownload:
    url: str | None
    is_apify_kv: bool
    error: str | None
