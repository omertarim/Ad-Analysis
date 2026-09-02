from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from task1.ml.dataset_loader import TrainingDataset, build_training_dataset

FEATURE_NAMES = [
    "views",
    "likes",
    "comments",
    "shares",
    "saves",
    "followers",
    "video_age_days",
]

COUNT_FEATURE_INDICES = [0, 1, 2, 3, 4, 5]
TARGET_PERCENTILE = 70
TARGET_MIN_RECALL = 0.85
TRAINING_RANDOM_STATE = 42


class Log1pCountTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p to bounded count columns, preserving NaNs."""

    def __init__(self, count_indices: list[int]) -> None:
        self.count_indices = count_indices

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "Log1pCountTransformer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        transformed = X.astype(float, copy=True)
        for idx in self.count_indices:
            column = transformed[:, idx]
            valid_mask = np.isfinite(column)
            clipped = np.clip(column[valid_mask], a_min=0, a_max=None)
            column[valid_mask] = np.log1p(clipped)
            transformed[:, idx] = column
        return transformed


def _build_pipeline(model_name: str) -> Pipeline:
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=2000, random_state=TRAINING_RANDOM_STATE)
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=TRAINING_RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
    elif model_name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(random_state=TRAINING_RANDOM_STATE)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return Pipeline(
        steps=[
            ("log1p_counts", Log1pCountTransformer(count_indices=COUNT_FEATURE_INDICES)),
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def _select_threshold_for_recall(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    met_target = False

    for threshold in candidates:
        preds = (probabilities >= threshold).astype(int)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        if recall >= TARGET_MIN_RECALL:
            if (not met_target) or (precision > best_metrics["precision"]) or (
                precision == best_metrics["precision"] and f1 > best_metrics["f1"]
            ):
                met_target = True
                best_threshold = float(threshold)
                best_metrics = {"precision": precision, "recall": recall, "f1": f1}
        elif not met_target and recall > best_metrics["recall"]:
            best_threshold = float(threshold)
            best_metrics = {"precision": precision, "recall": recall, "f1": f1}

    return best_threshold, best_metrics


def _model_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probabilities >= threshold).astype(int)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, probabilities)
    except ValueError:
        roc_auc = 0.0
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
    }


def train_candidate_ranker(
    dataset_path: str | Path,
    model_output_path: str | Path = "task1/artifacts/models/candidate_ranker.joblib",
    summary_output_path: str | Path = "task1/artifacts/models/training_summary.json",
) -> dict[str, Any]:
    dataset = build_training_dataset(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=0.2,
        random_state=TRAINING_RANDOM_STATE,
        stratify=dataset.y,
    )
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=TRAINING_RANDOM_STATE,
        stratify=y_train,
    )

    model_names = ["logistic_regression", "random_forest", "hist_gradient_boosting"]
    metrics_by_model: dict[str, dict[str, Any]] = {}
    thresholds: dict[str, float] = {}

    for model_name in model_names:
        pipeline = _build_pipeline(model_name)
        pipeline.fit(X_subtrain, y_subtrain)
        val_prob = pipeline.predict_proba(X_val)[:, 1]
        selected_threshold, _ = _select_threshold_for_recall(y_val, val_prob)
        thresholds[model_name] = selected_threshold

        test_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = _model_metrics(y_test, test_prob, selected_threshold)
        metrics["threshold"] = round(selected_threshold, 4)
        metrics_by_model[model_name] = metrics

    sorted_candidates = sorted(
        metrics_by_model.items(),
        key=lambda kv: (kv[1]["recall"], kv[1]["f1"]),
        reverse=True,
    )
    best_recall = sorted_candidates[0][1]["recall"]
    close_recall = [item for item in sorted_candidates if abs(item[1]["recall"] - best_recall) <= 0.01]
    selected_model_name, selected_metrics = sorted(close_recall, key=lambda kv: kv[1]["f1"], reverse=True)[0]

    # Refit selected model on full train split for stronger generalization.
    selected_pipeline = _build_pipeline(selected_model_name)
    selected_pipeline.fit(X_train, y_train)
    inference_threshold = thresholds[selected_model_name]

    trained_at = datetime.now(UTC).isoformat()
    model_payload = {
        "pipeline": selected_pipeline,
        "feature_names": FEATURE_NAMES,
        "selected_model": selected_model_name,
        "inference_threshold": inference_threshold,
        "trained_at": trained_at,
        "target_engagement_threshold": dataset.engagement_threshold,
        "target_percentile": TARGET_PERCENTILE,
    }
    model_path = Path(model_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_payload, model_path)

    selected_reason = (
        f"Selected by highest recall ({selected_metrics['recall']}). "
        f"F1 ({selected_metrics['f1']}) used as tie-breaker for close recall scores."
    )
    if selected_metrics["recall"] < TARGET_MIN_RECALL:
        selected_reason += f" Target recall {TARGET_MIN_RECALL} not met on test split."

    summary = {
        "dataset_path": str(Path(dataset_path)),
        "total_row_count": dataset.total_rows,
        "usable_row_count": dataset.usable_rows,
        "unavailable_dropped_fields": dataset.unavailable_fields,
        "dropped_rows": dataset.dropped_rows,
        "dropped_reasons": dataset.dropped_reasons,
        "schema_fields": dataset.schema_fields,
        "features_used": FEATURE_NAMES,
        "target_percentile": TARGET_PERCENTILE,
        "target_engagement_threshold": round(dataset.engagement_threshold, 6),
        "class_distribution": dataset.class_distribution,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "model_metrics": metrics_by_model,
        "selected_model": selected_model_name,
        "selection_reason": selected_reason,
        "inference_threshold": round(float(inference_threshold), 4),
        "recall_target": TARGET_MIN_RECALL,
        "trained_at": trained_at,
    }
    summary_path = Path(summary_output_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
