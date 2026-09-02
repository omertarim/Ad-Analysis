from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from task1.domain.trend import TrendSourceMetadata
from task1.ml.candidate_ranker_training import train_candidate_ranker
from task1.pipeline.ranking import rank_and_persist
from task1.services.candidate_ranker import CandidateRanker


def _write_dataset(path: Path) -> None:
    rows = []
    for i in range(40):
        created_at = datetime(2026, 6, 1, tzinfo=UTC).isoformat().replace("+00:00", "Z")
        rows.append(
            {
                "niche": "General",
                "keyword": "trend",
                "video_id": str(3000 + i),
                "author": f"author_{i}",
                "followers": 1500 + i * 100,
                "views": 9000 + i * 2000,
                "likes": 400 + i * 60,
                "comments": 20 + i * 3,
                "shares": 10 + i * 2,
                "saves": 5 + i * 2,
                "engagement_rate": 0.24 if i % 3 == 0 else 0.05,
                "created_at": created_at,
            }
        )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_ranking_artifact_persists_inference_features(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    _write_dataset(dataset)
    model = tmp_path / "candidate_ranker.joblib"
    summary = tmp_path / "training_summary.json"
    train_candidate_ranker(dataset_path=dataset, model_output_path=model, summary_output_path=summary)

    ranker = CandidateRanker(model_path=model)
    trend = TrendSourceMetadata(
        trend_id="trend_snapshot",
        source="tiktok_trending_apify",
        source_url="https://www.tiktok.com/@demo/video/111",
        collected_at=datetime.now(UTC),
        views=1000,
        likes=100,
        comments=10,
        shares=5,
        saves=2,
        author_followers=999,
    )
    rankings_root = tmp_path / "rankings"
    rank_and_persist([trend], rankings_root=rankings_root, ranker=ranker)
    payload = json.loads((rankings_root / "trend_snapshot" / "ranking.json").read_text(encoding="utf-8"))
    assert payload["inference_features"]["views"] == 1000.0
    assert "video_age_days" in payload["inference_features"]
    assert payload["model_path"] == str(model)
