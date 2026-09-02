from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from task1.services.multimodal_analyzer import MultimodalAnalyzer


class _FakeFilesApi:
    def __init__(self, upload_error: Exception | None = None) -> None:
        self.upload_error = upload_error

    def upload(self, file: str):  # type: ignore[no-untyped-def]
        if self.upload_error:
            raise self.upload_error
        return SimpleNamespace(name="file-1", state=SimpleNamespace(name="ACTIVE"))

    def get(self, name: str):  # type: ignore[no-untyped-def]
        return SimpleNamespace(name=name, state=SimpleNamespace(name="ACTIVE"))


class _FakeClient:
    def __init__(self, upload_error: Exception | None = None) -> None:
        self.files = _FakeFilesApi(upload_error=upload_error)
        self.models = SimpleNamespace()


def _valid_structured_payload(trend_id: str) -> dict:
    return {
        "trend_id": trend_id,
        "hook_type": "curiosity_gap",
        "hook_description": "Unexpected setup creates curiosity.",
        "first_2_seconds_description": "Immediate close-up with movement and text.",
        "visual_format": "selfie_ugc",
        "ugc_style": True,
        "pacing": "fast",
        "primary_emotions": ["curiosity"],
        "scene_summary": "Creator teases an outcome and reveals quickly.",
        "characters_count": 1,
        "text_overlay_present": True,
        "text_overlay_summary": "Short caption appears at top.",
        "spoken_dialogue_present": False,
        "spoken_dialogue_summary": None,
        "reaction_present": True,
        "reveal_present": True,
        "challenge_present": False,
        "before_after_present": False,
        "gameplay_present": False,
        "trend_mechanic": "setup then quick reveal",
        "audio_role": "music_driven",
        "visual_dependency": "high",
        "audio_dependency": "medium",
        "analysis_confidence": 0.77,
    }


def test_analyze_video_success_writes_artifacts(tmp_path: Path) -> None:
    video_file = tmp_path / "source.mp4"
    prompt_file = tmp_path / "video_analysis.txt"
    video_file.write_bytes(b"video-bytes")
    prompt_file.write_text("analyze", encoding="utf-8")

    analyzer = MultimodalAnalyzer(api_key="test-key", client=_FakeClient())
    analyzer._request_structured_analysis = lambda **kwargs: _valid_structured_payload("trend_1")  # type: ignore[method-assign]

    result = analyzer.analyze_video(
        trend_id="trend_1",
        source_video_path=video_file,
        prompt_file_path=prompt_file,
        output_root=tmp_path / "analysis_ready",
    )

    assert result.success is True
    assert result.upload_succeeded is True
    assert result.analysis_path is not None and Path(result.analysis_path).exists()
    assert result.metadata_path is not None and Path(result.metadata_path).exists()


def test_analyze_video_missing_api_key_returns_error(tmp_path: Path) -> None:
    video_file = tmp_path / "source.mp4"
    prompt_file = tmp_path / "video_analysis.txt"
    video_file.write_bytes(b"video-bytes")
    prompt_file.write_text("analyze", encoding="utf-8")

    analyzer = MultimodalAnalyzer(api_key=None, client=_FakeClient())
    analyzer.api_key = None

    result = analyzer.analyze_video(
        trend_id="trend_2",
        source_video_path=video_file,
        prompt_file_path=prompt_file,
        output_root=tmp_path / "analysis_ready",
    )
    assert result.success is False
    assert result.error == "api_key_missing"


def test_analyze_video_missing_video_returns_error(tmp_path: Path) -> None:
    prompt_file = tmp_path / "video_analysis.txt"
    prompt_file.write_text("analyze", encoding="utf-8")

    analyzer = MultimodalAnalyzer(api_key="test-key", client=_FakeClient())
    result = analyzer.analyze_video(
        trend_id="trend_3",
        source_video_path=tmp_path / "missing.mp4",
        prompt_file_path=prompt_file,
        output_root=tmp_path / "analysis_ready",
    )
    assert result.success is False
    assert result.error == "video_missing"


def test_analyze_video_upload_failure_returns_error(tmp_path: Path) -> None:
    video_file = tmp_path / "source.mp4"
    prompt_file = tmp_path / "video_analysis.txt"
    video_file.write_bytes(b"video-bytes")
    prompt_file.write_text("analyze", encoding="utf-8")

    analyzer = MultimodalAnalyzer(
        api_key="test-key",
        client=_FakeClient(upload_error=RuntimeError("upload boom")),
    )
    result = analyzer.analyze_video(
        trend_id="trend_4",
        source_video_path=video_file,
        prompt_file_path=prompt_file,
        output_root=tmp_path / "analysis_ready",
    )
    assert result.success is False
    assert result.error is not None
    assert result.error.startswith("upload_failure=")


def test_analyze_video_invalid_structured_response_returns_error(tmp_path: Path) -> None:
    video_file = tmp_path / "source.mp4"
    prompt_file = tmp_path / "video_analysis.txt"
    video_file.write_bytes(b"video-bytes")
    prompt_file.write_text("analyze", encoding="utf-8")

    analyzer = MultimodalAnalyzer(api_key="test-key", client=_FakeClient())
    bad_payload = _valid_structured_payload("trend_5")
    bad_payload["analysis_confidence"] = 1.2
    analyzer._request_structured_analysis = lambda **kwargs: bad_payload  # type: ignore[method-assign]

    result = analyzer.analyze_video(
        trend_id="trend_5",
        source_video_path=video_file,
        prompt_file_path=prompt_file,
        output_root=tmp_path / "analysis_ready",
    )
    assert result.success is False
    assert result.error is not None
    assert result.error.startswith("invalid_structured_response=")


def test_pipeline_trend_id_overrides_model_trend_id(tmp_path: Path) -> None:
    video_file = tmp_path / "source.mp4"
    prompt_file = tmp_path / "video_analysis.txt"
    video_file.write_bytes(b"video-bytes")
    prompt_file.write_text("analyze", encoding="utf-8")

    analyzer = MultimodalAnalyzer(api_key="test-key", client=_FakeClient())
    hacked_payload = _valid_structured_payload("made_up_id")
    analyzer._request_structured_analysis = lambda **kwargs: hacked_payload  # type: ignore[method-assign]

    result = analyzer.analyze_video(
        trend_id="trend_583af55c4eedd1bc",
        source_video_path=video_file,
        prompt_file_path=prompt_file,
        output_root=tmp_path / "analysis_ready",
    )

    assert result.success is True
    assert result.analysis is not None
    assert result.analysis.trend_id == "trend_583af55c4eedd1bc"
