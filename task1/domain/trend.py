from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator


def generate_trend_id(source: str, source_url: str, length: int = 16) -> str:
    """Generate a deterministic trend id from source and source_url."""
    normalized_source = source.strip().lower()
    normalized_url = source_url.strip()
    raw = f"{normalized_source}|{normalized_url}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return f"trend_{digest[:length]}"


class TrendSourceMetadata(BaseModel):
    trend_id: str = ""
    source: str
    source_url: str
    caption: Optional[str] = None
    hashtags: list[str] = Field(default_factory=list)
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    collected_at: datetime
    views: Optional[int] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    shares: Optional[int] = None
    duration_seconds: Optional[float] = None
    thumbnail_url: Optional[str] = None
    video_url: Optional[str] = None
    audio_name: Optional[str] = None
    audio_id: Optional[str] = None
    language: Optional[str] = None
    country: Optional[str] = None
    saves: Optional[int] = None
    engagement_rate: Optional[float] = None
    author_followers: Optional[int] = None
    author_verified: Optional[bool] = None
    trend_rank: Optional[int] = None
    raw_source_metadata: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def ensure_trend_id(cls, values: dict) -> dict:
        if not isinstance(values, dict):
            return values

        trend_id = values.get("trend_id")
        source = values.get("source")
        source_url = values.get("source_url")

        if not trend_id and source and source_url:
            values["trend_id"] = generate_trend_id(source=source, source_url=source_url)

        return values
