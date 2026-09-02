"""Domain models for Task 1."""

from .candidate_ranking import CandidateRankingResult
from .creative_evaluation import CreativeAdaptabilityEvaluation
from .video_analysis import VideoContentAnalysis
from .trend import TrendSourceMetadata, generate_trend_id

__all__ = [
    "TrendSourceMetadata",
    "generate_trend_id",
    "VideoContentAnalysis",
    "CreativeAdaptabilityEvaluation",
    "CandidateRankingResult",
]
