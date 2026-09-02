from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from task1.ml.candidate_ranker_training import FEATURE_NAMES, build_training_dataset, train_candidate_ranker


def _write_dataset(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _row(i: int, engagement_rate: float, include_created_at: bool = True) -> dict:
    created_at = datetime(2026, 6, 1, tzinfo=UTC) + timedelta(days=i)
    row = {
        "niche": "Finance",
        "keyword": "personal finance",
        "video_id": str(1000 + i),
        "author": f"author_{i}",
        "followers": 1000 + i * 100,
        "views": 10000 + i * 2000,
        "likes": 500 + i * 50,
        "comments": 20 + i * 3,
        "shares": 10 + i * 2,
        "saves": 8 + i * 2,
        "engagement_rate": engagement_rate,
    }
    if include_created_at:
        row["created_at"] = created_at.isoformat().replace("+00:00", "Z")
    return row


def test_build_training_dataset_schema_mapping_and_counts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.jsonl"
    rows = [_row(i=i, engagement_rate=0.05 + i * 0.01) for i in range(12)]
    rows.append({**_row(i=99, engagement_rate=0.2), "engagement_rate": None})
    _write_dataset(dataset_path, rows)

    dataset = build_training_dataset(dataset_path)
    assert dataset.total_rows == 13
    assert dataset.usable_rows == 12
    assert dataset.class_distribution["positive"] > 0
    assert dataset.class_distribution["negative"] > 0
    assert "views" in dataset.schema_fields
    assert "engagement_rate" in dataset.schema_fields
    assert FEATURE_NAMES == [
        "views",
        "likes",
        "comments",
        "shares",
        "saves",
        "followers",
        "video_age_days",
    ]


def test_train_candidate_ranker_summary_and_no_leakage_feature(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.jsonl"
    rows: list[dict] = []
    for i in range(40):
        engagement = 0.35 if i % 4 == 0 else 0.04 + ((i % 7) * 0.01)
        rows.append(_row(i=i, engagement_rate=engagement))
    _write_dataset(dataset_path, rows)

    model_path = tmp_path / "candidate_ranker.joblib"
    summary_path = tmp_path / "training_summary.json"
    summary = train_candidate_ranker(
        dataset_path=dataset_path,
        model_output_path=model_path,
        summary_output_path=summary_path,
    )

    assert model_path.exists()
    assert summary_path.exists()
    assert summary["usable_row_count"] == 40
    assert "engagement_rate" not in summary["features_used"]
    assert summary["selected_model"] in {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    }
    assert 0.0 <= summary["inference_threshold"] <= 1.0
