from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from task1.domain.candidate_ranking import CandidateRankingResult
from task1.domain.trend import TrendSourceMetadata


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


class CandidateRanker:
    def __init__(
        self,
        model_path: str | Path = "task1/artifacts/models/candidate_ranker.joblib",
        selection_mode: str = "top_percent",
        top_percent: float = 0.50,
        minimum_candidates: int = 5,
        maximum_candidates: int | None = None,
        threshold: float | None = None,
    ) -> None:
        payload = joblib.load(model_path)
        self.model_path = str(model_path)
        self.pipeline = payload["pipeline"]
        self.feature_names = payload["feature_names"]
        self.inference_threshold = float(payload["inference_threshold"])
        self.selected_model = str(payload["selected_model"])
        trained_at = str(payload["trained_at"])
        self.model_version = f"{self.selected_model}:{trained_at}"
        self.reference_time = self._parse_datetime(trained_at) or datetime.now(UTC)
        self.selection_mode = selection_mode
        self.top_percent = top_percent
        self.minimum_candidates = minimum_candidates
        self.maximum_candidates = maximum_candidates
        self.threshold = float(threshold) if threshold is not None else self.inference_threshold
        self._validate_selection_config()

    def rank(self, trend: TrendSourceMetadata) -> CandidateRankingResult:
        feature_vector = np.array([self._to_feature_row(trend)], dtype=float)
        probability = float(self.pipeline.predict_proba(feature_vector)[:, 1][0])
        selected = probability >= self.threshold
        return CandidateRankingResult(
            trend_id=trend.trend_id,
            candidate_probability=round(max(0.0, min(1.0, probability)), 6),
            candidate_rank=1,
            selected_for_multimodal=bool(selected),
            model_version=self.model_version,
        )

    def rank_batch(self, trends: list[TrendSourceMetadata]) -> list[CandidateRankingResult]:
        if not trends:
            return []
        feature_matrix = np.array([self._to_feature_row(trend) for trend in trends], dtype=float)
        probabilities = self.pipeline.predict_proba(feature_matrix)[:, 1]

        pairs = list(zip(trends, probabilities, strict=False))
        sorted_pairs = sorted(
            pairs,
            key=lambda item: (float(item[1]), item[0].trend_id),
            reverse=True,
        )
        selected_count = self._selected_count(len(sorted_pairs))
        threshold_cutoff = self.threshold
        selected_ids: set[str] = set()
        for index, (trend, probability) in enumerate(sorted_pairs, start=1):
            if self.selection_mode == "threshold":
                if float(probability) >= threshold_cutoff:
                    selected_ids.add(trend.trend_id)
            else:
                if index <= selected_count:
                    selected_ids.add(trend.trend_id)

        results: list[CandidateRankingResult] = []
        for rank, (trend, probability) in enumerate(sorted_pairs, start=1):
            results.append(
                CandidateRankingResult(
                    trend_id=trend.trend_id,
                    candidate_probability=round(max(0.0, min(1.0, float(probability))), 6),
                    candidate_rank=rank,
                    selected_for_multimodal=trend.trend_id in selected_ids,
                    model_version=self.model_version,
                )
            )
        return results

    def inference_features(self, trend: TrendSourceMetadata) -> dict[str, float | None]:
        row = self._to_feature_row(trend)
        features: dict[str, float | None] = {}
        for name, value in zip(self.feature_names, row, strict=False):
            if isinstance(value, float) and math.isnan(value):
                features[name] = None
            else:
                features[name] = round(float(value), 6)
        return features

    def _selected_count(self, total: int) -> int:
        if self.selection_mode == "threshold":
            return total
        count = math.ceil(total * self.top_percent)
        count = max(0, min(total, count))
        count = max(count, self.minimum_candidates)
        if self.maximum_candidates is not None:
            count = min(count, self.maximum_candidates)
        return max(0, min(total, count))

    def _validate_selection_config(self) -> None:
        if self.selection_mode not in {"threshold", "top_percent"}:
            raise ValueError("selection_mode must be 'threshold' or 'top_percent'")
        if not (0.0 < self.top_percent <= 1.0):
            raise ValueError("top_percent must be in (0, 1]")
        if self.minimum_candidates < 0:
            raise ValueError("minimum_candidates must be >= 0")
        if self.maximum_candidates is not None and self.maximum_candidates < 0:
            raise ValueError("maximum_candidates must be >= 0")
        if self.maximum_candidates is not None and self.maximum_candidates < self.minimum_candidates:
            raise ValueError("maximum_candidates cannot be less than minimum_candidates")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be between 0 and 1")

    def _to_feature_row(self, trend: TrendSourceMetadata) -> list[float]:
        counts = [
            _safe_float(trend.views),
            _safe_float(trend.likes),
            _safe_float(trend.comments),
            _safe_float(trend.shares),
            _safe_float(trend.saves),
            _safe_float(trend.author_followers),
        ]
        published_at = trend.published_at
        if published_at is None:
            age_days = math.nan
        else:
            if published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=UTC)
            delta = (self.reference_time - published_at.astimezone(UTC)).total_seconds() / 86400.0
            age_days = max(0.0, delta)

        return [value if value is not None else math.nan for value in counts] + [age_days]

    @staticmethod
    def _parse_datetime(value: str) -> datetime | None:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
        except Exception:  # noqa: BLE001
            return None
