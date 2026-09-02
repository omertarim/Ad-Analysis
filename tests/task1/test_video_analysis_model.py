import pytest
from pydantic import ValidationError

from task1.domain.video_analysis import VideoContentAnalysis


def _valid_payload() -> dict:
    return {
        "trend_id": "trend_123",
        "hook_type": "curiosity_gap",
        "hook_description": "A surprising setup creates immediate curiosity.",
        "first_2_seconds_description": "Close-up reaction and bold on-screen movement.",
        "visual_format": "selfie_ugc",
        "ugc_style": True,
        "pacing": "fast",
        "primary_emotions": ["curiosity", "surprise"],
        "scene_summary": "Creator introduces a challenge, then reveals result quickly.",
        "characters_count": 1,
        "text_overlay_present": True,
        "text_overlay_summary": "Short caption at top center.",
        "spoken_dialogue_present": True,
        "spoken_dialogue_summary": "Creator narrates the setup.",
        "reaction_present": True,
        "reveal_present": True,
        "challenge_present": True,
        "before_after_present": False,
        "gameplay_present": False,
        "trend_mechanic": "quick reveal after tension build-up",
        "audio_role": "mixed",
        "visual_dependency": "high",
        "audio_dependency": "medium",
        "analysis_confidence": 0.82,
    }


def test_valid_model() -> None:
    model = VideoContentAnalysis(**_valid_payload())
    assert model.trend_id == "trend_123"
    assert model.analysis_confidence == 0.82


def test_confidence_less_than_zero_rejected() -> None:
    payload = _valid_payload()
    payload["analysis_confidence"] = -0.01
    with pytest.raises(ValidationError):
        VideoContentAnalysis(**payload)


def test_confidence_greater_than_one_rejected() -> None:
    payload = _valid_payload()
    payload["analysis_confidence"] = 1.01
    with pytest.raises(ValidationError):
        VideoContentAnalysis(**payload)


def test_optional_text_dialogue_fields_can_be_none() -> None:
    payload = _valid_payload()
    payload["text_overlay_summary"] = None
    payload["spoken_dialogue_summary"] = None
    model = VideoContentAnalysis(**payload)
    assert model.text_overlay_summary is None
    assert model.spoken_dialogue_summary is None
