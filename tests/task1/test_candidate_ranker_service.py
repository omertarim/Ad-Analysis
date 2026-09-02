from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from task1.domain.trend import TrendSourceMetadata
from task1.ml.candidate_ranker_training import train_candidate_ranker
from task1.services.candidate_ranker import CandidateRanker


def _write_dataset(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _dataset_row(i: int, engagement_rate: float) -> dict:
    created_at = datetime(2026, 6, 1, tzinfo=UTC) + timedelta(days=i)
    return {
        "niche": "Finance",
        "keyword": "personal finance",
        "video_id": str(2000 + i),
        "author": f"author_{i}",
        "followers": 1200 + i * 120,
        "views": 8000 + i * 2200,
        "likes": 450 + i * 70,
        "comments": 15 + i * 5,
        "shares": 9 + i * 3,
        "saves": 7 + i * 2,
        "engagement_rate": engagement_rate,
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
    }


def test_candidate_ranker_probability_range_and_trend_id(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.jsonl"
    rows = [_dataset_row(i, 0.34 if i % 3 == 0 else 0.06) for i in range(36)]
    _write_dataset(dataset_path, rows)

    model_path = tmp_path / "candidate_ranker.joblib"
    summary_path = tmp_path / "training_summary.json"
    train_candidate_ranker(dataset_path=dataset_path, model_output_path=model_path, summary_output_path=summary_path)

    ranker = CandidateRanker(model_path=model_path, selection_mode="threshold")
    trend = TrendSourceMetadata(
        trend_id="trend_test_123",
        source="tiktok_trending_apify",
        source_url="https://www.tiktok.com/@demo/video/1234567",
        collected_at=datetime.now(UTC),
        views=145000,
        likes=9800,
        comments=240,
        shares=180,
        saves=350,
        author_followers=56000,
        published_at=datetime(2026, 6, 8, tzinfo=UTC),
    )
    result = ranker.rank(trend)
    assert result.trend_id == "trend_test_123"
    assert 0.0 <= result.candidate_probability <= 1.0
    assert result.candidate_rank == 1
    assert isinstance(result.selected_for_multimodal, bool)
    assert result.model_version


def test_candidate_ranker_handles_missing_metadata(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.jsonl"
    rows = [_dataset_row(i, 0.29 if i % 2 == 0 else 0.05) for i in range(30)]
    _write_dataset(dataset_path, rows)

    model_path = tmp_path / "candidate_ranker.joblib"
    summary_path = tmp_path / "training_summary.json"
    train_candidate_ranker(dataset_path=dataset_path, model_output_path=model_path, summary_output_path=summary_path)

    ranker = CandidateRanker(model_path=model_path)
    trend = TrendSourceMetadata(
        trend_id="trend_missing_meta",
        source="tiktok_trending_apify",
        source_url="https://www.tiktok.com/@demo/video/7654321",
        collected_at=datetime.now(UTC),
        views=None,
        likes=None,
        comments=None,
        shares=None,
        saves=None,
        author_followers=None,
        published_at=None,
    )
    result = ranker.rank(trend)
    assert 0.0 <= result.candidate_probability <= 1.0


def _trend(i: int) -> TrendSourceMetadata:
    return TrendSourceMetadata(
        trend_id=f"trend_batch_{i:02d}",
        source="tiktok_trending_apify",
        source_url=f"https://www.tiktok.com/@demo/video/{7000000 + i}",
        collected_at=datetime.now(UTC),
        views=100000 + i * 25000,
        likes=5000 + i * 700,
        comments=100 + i * 12,
        shares=50 + i * 8,
        saves=40 + i * 10,
        author_followers=5000 + i * 300,
        published_at=datetime(2026, 6, 1, tzinfo=UTC) + timedelta(days=i),
    )


def test_rank_batch_top_percent_selects_half_of_20(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.jsonl"
    rows = [_dataset_row(i, 0.29 if i % 2 == 0 else 0.06) for i in range(60)]
    _write_dataset(dataset_path, rows)
    model_path = tmp_path / "candidate_ranker.joblib"
    summary_path = tmp_path / "training_summary.json"
    train_candidate_ranker(dataset_path=dataset_path, model_output_path=model_path, summary_output_path=summary_path)

    ranker = CandidateRanker(
        model_path=model_path,
        selection_mode="top_percent",
        top_percent=0.5,
        minimum_candidates=5,
    )
    trends = [_trend(i) for i in range(20)]
    results = ranker.rank_batch(trends)

    assert len(results) == 20
    selected = [item for item in results if item.selected_for_multimodal]
    assert len(selected) == 10
    probs = [item.candidate_probability for item in results]
    assert probs == sorted(probs, reverse=True)
    ranks = [item.candidate_rank for item in results]
    assert ranks == list(range(1, 21))


def test_rank_batch_minimum_candidates_applies(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.jsonl"
    rows = [_dataset_row(i, 0.01 + ((i % 10) * 0.02)) for i in range(50)]
    _write_dataset(dataset_path, rows)
    model_path = tmp_path / "candidate_ranker.joblib"
    summary_path = tmp_path / "training_summary.json"
    train_candidate_ranker(dataset_path=dataset_path, model_output_path=model_path, summary_output_path=summary_path)

    ranker = CandidateRanker(
        model_path=model_path,
        selection_mode="top_percent",
        top_percent=0.1,
        minimum_candidates=5,
    )
    results = ranker.rank_batch([_trend(i) for i in range(20)])
    assert sum(1 for item in results if item.selected_for_multimodal) == 5


def test_rank_batch_threshold_mode_backward_compatible(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.jsonl"
    rows = [_dataset_row(i, 0.26 if i % 3 == 0 else 0.05) for i in range(55)]
    _write_dataset(dataset_path, rows)
    model_path = tmp_path / "candidate_ranker.joblib"
    summary_path = tmp_path / "training_summary.json"
    train_candidate_ranker(dataset_path=dataset_path, model_output_path=model_path, summary_output_path=summary_path)

    ranker = CandidateRanker(model_path=model_path, selection_mode="threshold")
    trend = _trend(3)
    single = ranker.rank(trend)
    batch = ranker.rank_batch([trend])[0]
    assert single.selected_for_multimodal == batch.selected_for_multimodal
    assert single.candidate_probability == batch.candidate_probability


def test_rank_batch_probabilities_match_individual_predictions(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.jsonl"
    rows = [_dataset_row(i, 0.015 + ((i % 12) * 0.017)) for i in range(64)]
    _write_dataset(dataset_path, rows)
    model_path = tmp_path / "candidate_ranker.joblib"
    summary_path = tmp_path / "training_summary.json"
    train_candidate_ranker(dataset_path=dataset_path, model_output_path=model_path, summary_output_path=summary_path)

    ranker = CandidateRanker(model_path=model_path, selection_mode="top_percent")
    trends = [_trend(i) for i in range(12)]
    batch = ranker.rank_batch(trends)
    batch_map = {item.trend_id: item.candidate_probability for item in batch}
    for trend in trends:
        single = ranker.rank(trend)
        assert batch_map[trend.trend_id] == single.candidate_probability


def test_same_model_same_feature_vector_same_probability(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.jsonl"
    rows = [_dataset_row(i, 0.02 + ((i % 8) * 0.03)) for i in range(72)]
    _write_dataset(dataset_path, rows)
    model_path = tmp_path / "candidate_ranker.joblib"
    summary_path = tmp_path / "training_summary.json"
    train_candidate_ranker(dataset_path=dataset_path, model_output_path=model_path, summary_output_path=summary_path)

    ranker = CandidateRanker(model_path=model_path, selection_mode="threshold")
    trend = TrendSourceMetadata(
        trend_id="trend_fixed_features",
        source="tiktok_trending_apify",
        source_url="https://www.tiktok.com/@demo/video/999",
        collected_at=datetime.now(UTC),
        views=123456,
        likes=9876,
        comments=321,
        shares=210,
        saves=654,
        author_followers=45678,
        published_at=datetime(2026, 6, 10, tzinfo=UTC),
    )
    first = ranker.rank(trend)
    second = ranker.rank(trend)
    assert first.candidate_probability == second.candidate_probability
