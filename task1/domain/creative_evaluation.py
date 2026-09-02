from __future__ import annotations

from pydantic import BaseModel, Field


class CreativeAdaptabilityEvaluation(BaseModel):
    trend_id: str

    hook_strength_score: float = Field(ge=0.0, le=10.0)
    first_seconds_strength_score: float = Field(ge=0.0, le=10.0)
    royal_match_world_transferability_score: float = Field(ge=0.0, le=10.0)
    player_intervention_score: float = Field(ge=0.0, le=10.0)
    gameplay_bridge_score: float = Field(ge=0.0, le=10.0)
    visual_clarity_score: float = Field(ge=0.0, le=10.0)
    production_feasibility_score: float = Field(ge=0.0, le=10.0)
    tonal_fit_score: float = Field(ge=0.0, le=10.0)
    trend_transferability_score: float = Field(ge=0.0, le=10.0)
    brand_safety_score: float = Field(ge=0.0, le=10.0)

    context_dependency_risk: float = Field(ge=0.0, le=10.0)
    audio_dependency_risk: float = Field(ge=0.0, le=10.0)
    ip_or_identity_risk: float = Field(ge=0.0, le=10.0)

    strengths: list[str]
    concerns: list[str] = Field(default_factory=list)
    rejection_reasons: list[str]
    adaptation_idea: str | None = None
    evaluation_confidence: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=10.0)
    adaptable: bool
