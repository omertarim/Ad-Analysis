from __future__ import annotations

from pydantic import BaseModel, Field


class CandidateRankingResult(BaseModel):
    trend_id: str
    candidate_probability: float = Field(ge=0.0, le=1.0)
    candidate_rank: int = Field(ge=1)
    selected_for_multimodal: bool
    model_version: str
