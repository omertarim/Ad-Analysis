"""Collector abstractions for Task 1."""

from .apify_tiktok import ApifyTikTokTrendCollector
from .base import TrendCollector

__all__ = [
    "TrendCollector",
    "ApifyTikTokTrendCollector",
]
