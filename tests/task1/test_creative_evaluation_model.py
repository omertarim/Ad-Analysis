from __future__ import annotations

import pytest
from pydantic import ValidationError

from task1.domain.creative_evaluation import CreativeAdaptabilityEvaluation


def _valid_payload() -> dict:
    return {
        "trend_id": "trend_1",
        "hook_strength_score": 8.0,
        "first_seconds_strength_score": 7.0,
        "royal_match_world_transferability_score": 7.5,
        "player_intervention_score": 8.0,
        "gameplay_bridge_score": 7.0,
        "visual_clarity_score": 8.0,
        "production_feasibility_score": 7.5,
        "tonal_fit_score": 7.0,
        "trend_transferability_score": 7.5,
        "brand_safety_score": 8.0,
        "context_dependency_risk": 2.0,
        "audio_dependency_risk": 3.0,
        "ip_or_identity_risk": 2.0,
        "strengths": ["clear hook", "easy to transfer"],
        "concerns": ["audio trend may age fast"],
        "rejection_reasons": [],
        "adaptation_idea": "King Robert faces a trap and player chooses the right move.",
        "evaluation_confidence": 0.8,
        "overall_score": 7.2,
        "adaptable": True,
    }


def test_valid_model() -> None:
    evaluation = CreativeAdaptabilityEvaluation(**_valid_payload())
    assert evaluation.trend_id == "trend_1"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("hook_strength_score", -0.1),
        ("hook_strength_score", 10.1),
        ("context_dependency_risk", -0.2),
        ("context_dependency_risk", 11.0),
        ("overall_score", -1.0),
        ("overall_score", 11.0),
        ("evaluation_confidence", -0.01),
        ("evaluation_confidence", 1.01),
    ],
)
def test_bounds_validation(field_name: str, value: float) -> None:
    payload = _valid_payload()
    payload[field_name] = value
    with pytest.raises(ValidationError):
        CreativeAdaptabilityEvaluation(**payload)
