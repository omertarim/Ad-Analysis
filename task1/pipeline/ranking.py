from __future__ import annotations

import json
from pathlib import Path

from task1.domain.candidate_ranking import CandidateRankingResult
from task1.domain.trend import TrendSourceMetadata
from task1.services.candidate_ranker import CandidateRanker


def rank_and_persist(
    trends: list[TrendSourceMetadata],
    rankings_root: Path,
    ranker: CandidateRanker,
) -> list[CandidateRankingResult]:
    results = ranker.rank_batch(trends)
    for result in results:
        trend = next(item for item in trends if item.trend_id == result.trend_id)
        payload = {
            "trend_id": result.trend_id,
            "candidate_probability": result.candidate_probability,
            "candidate_rank": result.candidate_rank,
            "selected_for_multimodal": result.selected_for_multimodal,
            "model_version": result.model_version,
            "model_path": ranker.model_path,
            "inference_threshold": ranker.threshold,
            "selection_mode": ranker.selection_mode,
            "source_url": trend.source_url,
            "inference_features": ranker.inference_features(trend),
        }
        path = rankings_root / result.trend_id / "ranking.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return results
