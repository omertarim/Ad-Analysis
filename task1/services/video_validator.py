from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class VideoValidationResult:
    trend_id: str
    valid: bool
    duration_seconds: float | None
    width: int | None
    height: int | None
    codec: str | None
    fps: float | None
    has_audio: bool | None
    file_size_bytes: int | None
    error: str | None


class VideoValidator:
    def __init__(
        self,
        ffprobe_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self._ffprobe_runner = ffprobe_runner or self._default_ffprobe_runner

    def validate(self, trend_id: str, video_path: str | Path) -> VideoValidationResult:
        path = Path(video_path)
        if not path.exists():
            return self._fail(trend_id=trend_id, file_size_bytes=None, error="file_not_found")
        file_size = path.stat().st_size
        if file_size <= 0:
            return self._fail(trend_id=trend_id, file_size_bytes=file_size, error="empty_file")

        command = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ]
        try:
            completed = self._ffprobe_runner(command)
        except FileNotFoundError:
            return self._fail(
                trend_id=trend_id,
                file_size_bytes=file_size,
                error="ffprobe_not_found",
            )
        except Exception as exc:  # noqa: BLE001
            return self._fail(
                trend_id=trend_id,
                file_size_bytes=file_size,
                error=f"ffprobe_exec_error={exc}",
            )

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            return self._fail(
                trend_id=trend_id,
                file_size_bytes=file_size,
                error=f"ffprobe_failed={stderr or 'unknown'}",
            )

        try:
            payload = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError:
            return self._fail(
                trend_id=trend_id,
                file_size_bytes=file_size,
                error="ffprobe_invalid_json",
            )

        streams = payload.get("streams", [])
        format_info = payload.get("format", {})
        video_stream = self._first_video_stream(streams)
        if not video_stream:
            return self._fail(
                trend_id=trend_id,
                file_size_bytes=file_size,
                error="no_video_stream",
            )

        width = self._as_int(video_stream.get("width"))
        height = self._as_int(video_stream.get("height"))
        if not width or not height:
            return self._fail(
                trend_id=trend_id,
                file_size_bytes=file_size,
                error="invalid_resolution",
            )

        duration_seconds = self._pick_duration_seconds(video_stream, format_info)
        if duration_seconds is None or duration_seconds <= 0:
            return self._fail(
                trend_id=trend_id,
                file_size_bytes=file_size,
                error="invalid_duration",
            )

        codec = video_stream.get("codec_name")
        fps = self._parse_fps(video_stream.get("avg_frame_rate"))
        has_audio = any(
            isinstance(stream, dict) and stream.get("codec_type") == "audio"
            for stream in streams
        )

        return VideoValidationResult(
            trend_id=trend_id,
            valid=True,
            duration_seconds=duration_seconds,
            width=width,
            height=height,
            codec=codec if isinstance(codec, str) else None,
            fps=fps,
            has_audio=has_audio,
            file_size_bytes=file_size,
            error=None,
        )

    @staticmethod
    def _default_ffprobe_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, capture_output=True, text=True, check=False)

    @staticmethod
    def _first_video_stream(streams: object) -> dict | None:
        if not isinstance(streams, list):
            return None
        for stream in streams:
            if isinstance(stream, dict) and stream.get("codec_type") == "video":
                return stream
        return None

    @staticmethod
    def _pick_duration_seconds(video_stream: dict, format_info: dict) -> float | None:
        stream_duration = VideoValidator._as_float(video_stream.get("duration"))
        if stream_duration and stream_duration > 0:
            return stream_duration
        format_duration = VideoValidator._as_float(format_info.get("duration"))
        if format_duration and format_duration > 0:
            return format_duration
        return None

    @staticmethod
    def _parse_fps(value: object) -> float | None:
        if not isinstance(value, str) or not value.strip():
            return None
        text = value.strip()
        if "/" in text:
            left, right = text.split("/", 1)
            numerator = VideoValidator._as_float(left)
            denominator = VideoValidator._as_float(right)
            if numerator is None or denominator in (None, 0):
                return None
            return numerator / denominator
        return VideoValidator._as_float(text)

    @staticmethod
    def _as_float(value: object) -> float | None:
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
    def _as_int(value: object) -> int | None:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if text.isdigit():
                return int(text)
        return None

    @staticmethod
    def _fail(trend_id: str, file_size_bytes: int | None, error: str) -> VideoValidationResult:
        return VideoValidationResult(
            trend_id=trend_id,
            valid=False,
            duration_seconds=None,
            width=None,
            height=None,
            codec=None,
            fps=None,
            has_audio=None,
            file_size_bytes=file_size_bytes,
            error=error,
        )
