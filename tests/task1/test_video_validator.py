from __future__ import annotations

import json
import subprocess
from pathlib import Path

from task1.services.video_validator import VideoValidator


def _completed(stdout: dict, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ffprobe"],
        returncode=returncode,
        stdout=json.dumps(stdout),
        stderr=stderr,
    )


def test_valid_mocked_ffprobe_result(tmp_path: Path) -> None:
    video_file = tmp_path / "ok.mp4"
    video_file.write_bytes(b"123456")

    payload = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1080,
                "height": 1920,
                "avg_frame_rate": "30/1",
                "duration": "8.2",
            },
            {"codec_type": "audio"},
        ],
        "format": {"duration": "8.2"},
    }

    validator = VideoValidator(ffprobe_runner=lambda _cmd: _completed(payload))
    result = validator.validate(trend_id="trend_1", video_path=video_file)

    assert result.valid is True
    assert result.duration_seconds == 8.2
    assert result.width == 1080
    assert result.height == 1920
    assert result.codec == "h264"
    assert result.fps == 30.0
    assert result.has_audio is True
    assert result.error is None


def test_missing_file(tmp_path: Path) -> None:
    validator = VideoValidator(ffprobe_runner=lambda _cmd: _completed({}))
    result = validator.validate(trend_id="trend_2", video_path=tmp_path / "missing.mp4")
    assert result.valid is False
    assert result.error == "file_not_found"


def test_invalid_corrupt_file_ffprobe_failure(tmp_path: Path) -> None:
    video_file = tmp_path / "corrupt.mp4"
    video_file.write_bytes(b"invalid")
    validator = VideoValidator(
        ffprobe_runner=lambda _cmd: _completed({}, returncode=1, stderr="Invalid data found")
    )
    result = validator.validate(trend_id="trend_3", video_path=video_file)
    assert result.valid is False
    assert result.error is not None
    assert "ffprobe_failed" in result.error


def test_no_video_stream(tmp_path: Path) -> None:
    video_file = tmp_path / "audio_only.mp4"
    video_file.write_bytes(b"audio")
    payload = {
        "streams": [{"codec_type": "audio"}],
        "format": {"duration": "12.0"},
    }
    validator = VideoValidator(ffprobe_runner=lambda _cmd: _completed(payload))
    result = validator.validate(trend_id="trend_4", video_path=video_file)
    assert result.valid is False
    assert result.error == "no_video_stream"
