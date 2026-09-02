from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_DATASET_FIELDS = [
    "views",
    "likes",
    "comments",
    "shares",
    "saves",
    "followers",
    "created_at",
    "engagement_rate",
]

TARGET_PERCENTILE = 70


@dataclass
class TrainingDataset:
    X: np.ndarray
    y: np.ndarray
    total_rows: int
    usable_rows: int
    dropped_rows: int
    dropped_reasons: dict[str, int]
    unavailable_fields: list[str]
    engagement_threshold: float
    class_distribution: dict[str, int]
    schema_fields: list[str]


def safe_float(value: Any) -> float | None:
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


def parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def extract_schema_fields(dataset_path: Path) -> list[str]:
    fields: set[str] = set()
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                fields.update(payload.keys())
    return sorted(fields)


def build_training_dataset(dataset_path: str | Path) -> TrainingDataset:
    path = Path(dataset_path)
    raw_rows: list[dict[str, Any]] = []
    created_times: list[datetime] = []
    engagement_values: list[float] = []
    dropped_reasons: dict[str, int] = {}

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                dropped_reasons["invalid_json_object"] = dropped_reasons.get("invalid_json_object", 0) + 1
                continue

            engagement = safe_float(payload.get("engagement_rate"))
            if engagement is None:
                dropped_reasons["missing_engagement_rate"] = (
                    dropped_reasons.get("missing_engagement_rate", 0) + 1
                )
                continue

            created_at = parse_datetime(payload.get("created_at"))
            if created_at is not None:
                created_times.append(created_at)

            raw_rows.append(payload)
            engagement_values.append(engagement)

    total_rows = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    base_usable_rows = len(raw_rows)
    base_dropped_rows = total_rows - base_usable_rows

    if not engagement_values:
        raise ValueError("No usable rows found in training dataset")

    reference_time = max(created_times) if created_times else datetime.now(UTC)
    engagement_threshold = float(np.percentile(np.array(engagement_values, dtype=float), TARGET_PERCENTILE))

    X_rows: list[list[float]] = []
    y_rows: list[int] = []
    for payload, engagement in zip(raw_rows, engagement_values):
        numeric_values = [
            safe_float(payload.get("views")),
            safe_float(payload.get("likes")),
            safe_float(payload.get("comments")),
            safe_float(payload.get("shares")),
            safe_float(payload.get("saves")),
            safe_float(payload.get("followers")),
        ]
        created_at = parse_datetime(payload.get("created_at"))
        if created_at is None:
            video_age_days = math.nan
        else:
            delta_days = (reference_time - created_at).total_seconds() / 86400.0
            video_age_days = max(0.0, delta_days)

        feature_row = [value if value is not None else math.nan for value in numeric_values] + [video_age_days]
        if all(math.isnan(v) for v in feature_row):
            dropped_reasons["all_features_missing"] = dropped_reasons.get("all_features_missing", 0) + 1
            continue

        X_rows.append(feature_row)
        y_rows.append(1 if engagement >= engagement_threshold else 0)

    if not X_rows:
        raise ValueError("No rows left after feature mapping")

    y_np = np.array(y_rows, dtype=int)
    schema_fields = extract_schema_fields(path)
    unavailable_fields = [field for field in EXPECTED_DATASET_FIELDS if field not in schema_fields]

    return TrainingDataset(
        X=np.array(X_rows, dtype=float),
        y=y_np,
        total_rows=total_rows,
        usable_rows=len(X_rows),
        dropped_rows=base_dropped_rows + int(dropped_reasons.get("all_features_missing", 0)),
        dropped_reasons=dropped_reasons,
        unavailable_fields=unavailable_fields,
        engagement_threshold=engagement_threshold,
        class_distribution={
            "positive": int((y_np == 1).sum()),
            "negative": int((y_np == 0).sum()),
        },
        schema_fields=schema_fields,
    )
