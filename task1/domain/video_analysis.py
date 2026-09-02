from __future__ import annotations

from pydantic import BaseModel, Field


class VideoContentAnalysis(BaseModel):
    trend_id: str
    hook_type: str
    hook_description: str
    first_2_seconds_description: str
    visual_format: str
    ugc_style: bool
    pacing: str
    primary_emotions: list[str]
    scene_summary: str
    characters_count: int | None = None
    text_overlay_present: bool
    text_overlay_summary: str | None = None
    spoken_dialogue_present: bool
    spoken_dialogue_summary: str | None = None
    reaction_present: bool
    reveal_present: bool
    challenge_present: bool
    before_after_present: bool
    gameplay_present: bool
    trend_mechanic: str
    audio_role: str
    visual_dependency: str
    audio_dependency: str
    analysis_confidence: float = Field(ge=0.0, le=1.0)
