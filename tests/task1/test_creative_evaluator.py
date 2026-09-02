from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from task1.domain.trend import TrendSourceMetadata
from task1.domain.video_analysis import VideoContentAnalysis
from task1.services.creative_evaluator import CreativeEvaluator


class _FakeClient:
    def __init__(self) -> None:
        self.models = SimpleNamespace()


def _trend(trend_id: str = "trend_input") -> TrendSourceMetadata:
    return TrendSourceMetadata(
        trend_id=trend_id,
        source="tiktok_trending_apify",
        source_url="https://www.tiktok.com/@demo/video/1234567890",
        collected_at=datetime.now(UTC),
    )


def _analysis(trend_id: str = "trend_input") -> VideoContentAnalysis:
    return VideoContentAnalysis(
        trend_id=trend_id,
        hook_type="curiosity_gap",
        hook_description="Anomaly hook",
        first_2_seconds_description="Unexpected visual appears",
        visual_format="selfie_ugc",
        ugc_style=True,
        pacing="fast",
        primary_emotions=["curiosity"],
        scene_summary="Short sequence",
        characters_count=1,
        text_overlay_present=True,
        text_overlay_summary="Help him now",
        spoken_dialogue_present=False,
        spoken_dialogue_summary=None,
        reaction_present=True,
        reveal_present=True,
        challenge_present=False,
        before_after_present=False,
        gameplay_present=False,
        trend_mechanic="quick anomaly and reveal",
        audio_role="music_driven",
        visual_dependency="high",
        audio_dependency="medium",
        analysis_confidence=0.8,
    )


def _payload() -> dict:
    return {
        "trend_id": "made_up_id",
        "hook_strength_score": 8.0,
        "first_seconds_strength_score": 8.0,
        "royal_match_world_transferability_score": 7.0,
        "player_intervention_score": 8.0,
        "gameplay_bridge_score": 7.0,
        "visual_clarity_score": 8.0,
        "production_feasibility_score": 7.0,
        "tonal_fit_score": 7.0,
        "trend_transferability_score": 7.0,
        "brand_safety_score": 8.0,
        "context_dependency_risk": 2.0,
        "audio_dependency_risk": 3.0,
        "ip_or_identity_risk": 2.0,
        "strengths": ["clear fast hook"],
        "concerns": [],
        "rejection_reasons": [],
        "adaptation_idea": "King Robert is in danger and player picks rescue action.",
        "evaluation_confidence": 0.85,
        "overall_score": 1.0,
        "adaptable": False,
    }


def test_deterministic_overall_score_and_trend_override(tmp_path: Path) -> None:
    prompt = tmp_path / "creative_evaluation.txt"
    prompt.write_text("evaluate", encoding="utf-8")
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"video")
    analysis_file = tmp_path / "analysis.json"
    analysis_file.write_text("{}", encoding="utf-8")

    evaluator = CreativeEvaluator(api_key="x", client=_FakeClient())
    evaluator._request_structured_evaluation = lambda **_: _payload()  # type: ignore[method-assign]

    result = evaluator.evaluate(
        trend=_trend("trend_real"),
        analysis=_analysis("trend_real"),
        prompt_file_path=prompt,
        output_root=tmp_path / "evaluations",
        source_video_path=source_video,
        analysis_artifact_path=analysis_file,
    )

    assert result.success is True
    assert result.evaluation is not None
    assert result.evaluation.trend_id == "trend_real"
    assert result.evaluation.overall_score == 7.17
    assert result.evaluation.adaptable is True


def test_high_ip_risk_rejects_even_if_other_scores_good(tmp_path: Path) -> None:
    prompt = tmp_path / "creative_evaluation.txt"
    prompt.write_text("evaluate", encoding="utf-8")
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"video")
    analysis_file = tmp_path / "analysis.json"
    analysis_file.write_text("{}", encoding="utf-8")

    payload = _payload()
    payload["ip_or_identity_risk"] = 9.0

    evaluator = CreativeEvaluator(api_key="x", client=_FakeClient())
    evaluator._request_structured_evaluation = lambda **_: payload  # type: ignore[method-assign]
    result = evaluator.evaluate(
        trend=_trend(),
        analysis=_analysis(),
        prompt_file_path=prompt,
        output_root=tmp_path / "evaluations",
        source_video_path=source_video,
        analysis_artifact_path=analysis_file,
    )
    assert result.success is True
    assert result.evaluation is not None
    assert result.evaluation.adaptable is False


def test_low_gameplay_bridge_rejects(tmp_path: Path) -> None:
    prompt = tmp_path / "creative_evaluation.txt"
    prompt.write_text("evaluate", encoding="utf-8")
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"video")
    analysis_file = tmp_path / "analysis.json"
    analysis_file.write_text("{}", encoding="utf-8")

    payload = _payload()
    payload["gameplay_bridge_score"] = 4.0

    evaluator = CreativeEvaluator(api_key="x", client=_FakeClient())
    evaluator._request_structured_evaluation = lambda **_: payload  # type: ignore[method-assign]
    result = evaluator.evaluate(
        trend=_trend(),
        analysis=_analysis(),
        prompt_file_path=prompt,
        output_root=tmp_path / "evaluations",
        source_video_path=source_video,
        analysis_artifact_path=analysis_file,
    )
    assert result.success is True
    assert result.evaluation is not None
    assert result.evaluation.adaptable is False


def test_low_overall_score_rejects(tmp_path: Path) -> None:
    prompt = tmp_path / "creative_evaluation.txt"
    prompt.write_text("evaluate", encoding="utf-8")
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"video")
    analysis_file = tmp_path / "analysis.json"
    analysis_file.write_text("{}", encoding="utf-8")

    payload = _payload()
    payload["hook_strength_score"] = 2.0
    payload["first_seconds_strength_score"] = 2.0
    payload["royal_match_world_transferability_score"] = 2.0
    payload["player_intervention_score"] = 2.0
    payload["gameplay_bridge_score"] = 2.0
    payload["trend_transferability_score"] = 2.0

    evaluator = CreativeEvaluator(api_key="x", client=_FakeClient())
    evaluator._request_structured_evaluation = lambda **_: payload  # type: ignore[method-assign]
    result = evaluator.evaluate(
        trend=_trend(),
        analysis=_analysis(),
        prompt_file_path=prompt,
        output_root=tmp_path / "evaluations",
        source_video_path=source_video,
        analysis_artifact_path=analysis_file,
    )
    assert result.success is True
    assert result.evaluation is not None
    assert result.evaluation.overall_score < 6.0
    assert result.evaluation.adaptable is False


def test_evaluation_confidence_bounds_rejected(tmp_path: Path) -> None:
    prompt = tmp_path / "creative_evaluation.txt"
    prompt.write_text("evaluate", encoding="utf-8")
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"video")
    analysis_file = tmp_path / "analysis.json"
    analysis_file.write_text("{}", encoding="utf-8")

    payload = _payload()
    payload["evaluation_confidence"] = 1.4

    evaluator = CreativeEvaluator(api_key="x", client=_FakeClient())
    evaluator._request_structured_evaluation = lambda **_: payload  # type: ignore[method-assign]
    result = evaluator.evaluate(
        trend=_trend(),
        analysis=_analysis(),
        prompt_file_path=prompt,
        output_root=tmp_path / "evaluations",
        source_video_path=source_video,
        analysis_artifact_path=analysis_file,
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.startswith("invalid_structured_response=")


def test_adaptable_true_forces_empty_rejection_and_preserves_concerns(tmp_path: Path) -> None:
    prompt = tmp_path / "creative_evaluation.txt"
    prompt.write_text("evaluate", encoding="utf-8")
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"video")
    analysis_file = tmp_path / "analysis.json"
    analysis_file.write_text("{}", encoding="utf-8")

    payload = _payload()
    payload["rejection_reasons"] = ["celebrity recognition dependency", "forced gameplay bridge"]
    payload["concerns"] = ["localized context dependency"]

    evaluator = CreativeEvaluator(api_key="x", client=_FakeClient())
    evaluator._request_structured_evaluation = lambda **_: payload  # type: ignore[method-assign]
    result = evaluator.evaluate(
        trend=_trend("trend_adaptable"),
        analysis=_analysis("trend_adaptable"),
        prompt_file_path=prompt,
        output_root=tmp_path / "evaluations",
        source_video_path=source_video,
        analysis_artifact_path=analysis_file,
    )

    assert result.success is True
    assert result.evaluation is not None
    assert result.evaluation.adaptable is True
    assert result.evaluation.rejection_reasons == []
    assert "celebrity recognition dependency" in result.evaluation.concerns
    assert "forced gameplay bridge" in result.evaluation.concerns
    assert "localized context dependency" in result.evaluation.concerns


def test_adaptable_false_keeps_rejection_reasons(tmp_path: Path) -> None:
    prompt = tmp_path / "creative_evaluation.txt"
    prompt.write_text("evaluate", encoding="utf-8")
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"video")
    analysis_file = tmp_path / "analysis.json"
    analysis_file.write_text("{}", encoding="utf-8")

    payload = _payload()
    payload["gameplay_bridge_score"] = 3.0
    payload["rejection_reasons"] = ["bridge to gameplay is forced"]
    payload["concerns"] = ["audio trend may age quickly"]

    evaluator = CreativeEvaluator(api_key="x", client=_FakeClient())
    evaluator._request_structured_evaluation = lambda **_: payload  # type: ignore[method-assign]
    result = evaluator.evaluate(
        trend=_trend("trend_reject"),
        analysis=_analysis("trend_reject"),
        prompt_file_path=prompt,
        output_root=tmp_path / "evaluations",
        source_video_path=source_video,
        analysis_artifact_path=analysis_file,
    )

    assert result.success is True
    assert result.evaluation is not None
    assert result.evaluation.adaptable is False
    assert result.evaluation.rejection_reasons == ["bridge to gameplay is forced"]
    assert result.evaluation.concerns == ["audio trend may age quickly"]
