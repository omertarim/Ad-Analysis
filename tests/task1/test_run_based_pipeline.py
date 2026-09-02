from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from task1.domain.candidate_ranking import CandidateRankingResult
from task1.domain.trend import TrendSourceMetadata
from task1.pipeline.processing import process_selected_candidates
from task1.pipeline.summaries import build_current_run_artifacts


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _trend(trend_id: str) -> TrendSourceMetadata:
    return TrendSourceMetadata(
        trend_id=trend_id,
        source="tiktok_trending_apify",
        source_url=f"https://www.tiktok.com/@demo/video/{trend_id[-4:]}",
        collected_at=datetime.now(UTC),
    )


def test_historical_adaptable_not_counted_in_current_run(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    run_root = artifacts_root / "runs" / "run_20260831_181500"
    raw_root = artifacts_root / "raw"
    ranking_root = artifacts_root / "rankings"
    analysis_root = artifacts_root / "analysis"
    eval_root = artifacts_root / "evaluations"

    current_id = "trend_current"
    historical_id = "trend_old"
    _write_json(raw_root / current_id / "trend_metadata.json", {"trend_id": current_id, "source_url": "https://www.tiktok.com/@a/video/1"})
    _write_json(raw_root / historical_id / "trend_metadata.json", {"trend_id": historical_id, "source_url": "https://www.tiktok.com/@b/video/2"})
    _write_json(ranking_root / current_id / "ranking.json", {"trend_id": current_id, "candidate_rank": 1, "candidate_probability": 0.8, "selected_for_multimodal": True})
    _write_json(ranking_root / historical_id / "ranking.json", {"trend_id": historical_id, "candidate_rank": 1, "candidate_probability": 0.9, "selected_for_multimodal": True})
    _write_json(analysis_root / current_id / "analysis.json", {"hook_type": "x", "trend_mechanic": "y"})
    _write_json(analysis_root / historical_id / "analysis.json", {"hook_type": "x", "trend_mechanic": "y"})
    _write_json(eval_root / current_id / "evaluation.json", {"trend_id": current_id, "overall_score": 4.0, "adaptable": False, "rejection_reasons": ["bad bridge"]})
    _write_json(eval_root / historical_id / "evaluation.json", {"trend_id": historical_id, "overall_score": 8.5, "adaptable": True, "rejection_reasons": []})

    result = build_current_run_artifacts(
        run_root=run_root,
        artifacts_root=artifacts_root,
        run_id="run_20260831_181500",
        started_at=datetime.now(UTC),
        region="TR",
        discovery_rounds=1,
        current_trend_ids={current_id},
        processing_stats=type("S", (), {"cache_hits": 0, "failures": {}})(),
    )
    assert result["run_summary"]["total_adaptable"] == 0
    assert result["run_summary"]["total_current_unique_discovered"] == 1


def test_cached_trend_reused_when_rediscovered(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    rankings_root = tmp_path / "rankings"
    analysis_root = tmp_path / "analysis"
    evaluations_root = tmp_path / "evaluations"
    trend_id = "trend_cached"
    trend = _trend(trend_id)

    (raw_root / trend_id).mkdir(parents=True, exist_ok=True)
    (raw_root / trend_id / "source.mp4").write_bytes(b"video")
    _write_json(
        analysis_root / trend_id / "analysis.json",
        {
            "trend_id": trend_id,
            "hook_type": "curiosity_gap",
            "hook_description": "x",
            "first_2_seconds_description": "x",
            "visual_format": "selfie_ugc",
            "ugc_style": True,
            "pacing": "fast",
            "primary_emotions": ["curiosity"],
            "scene_summary": "x",
            "characters_count": 1,
            "text_overlay_present": True,
            "text_overlay_summary": "x",
            "spoken_dialogue_present": False,
            "spoken_dialogue_summary": None,
            "reaction_present": True,
            "reveal_present": True,
            "challenge_present": False,
            "before_after_present": False,
            "gameplay_present": False,
            "trend_mechanic": "x",
            "audio_role": "music_driven",
            "visual_dependency": "high",
            "audio_dependency": "medium",
            "analysis_confidence": 0.8,
        },
    )
    _write_json(
        evaluations_root / trend_id / "evaluation.json",
        {
            "trend_id": trend_id,
            "overall_score": 7.0,
            "adaptable": True,
            "rejection_reasons": [],
        },
    )

    ranking = CandidateRankingResult(
        trend_id=trend_id,
        candidate_probability=0.9,
        candidate_rank=1,
        selected_for_multimodal=True,
        model_version="m1",
    )
    stats = process_selected_candidates(
        trends=[trend],
        rankings=[ranking],
        raw_root=raw_root,
        rankings_root=rankings_root,
        analysis_root=analysis_root,
        evaluations_root=evaluations_root,
        apify_token="x",
        current_run_trend_ids={trend_id},
    )
    assert stats.cache_hits >= 1
    assert stats.evaluated == 0
    assert stats.analyzed == 0


def test_current_run_adaptable_count_and_run_artifact_path(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    run_root = artifacts_root / "runs" / "run_20260831_190000"
    raw_root = artifacts_root / "raw"
    ranking_root = artifacts_root / "rankings"
    analysis_root = artifacts_root / "analysis"
    eval_root = artifacts_root / "evaluations"

    ids = ["trend_a", "trend_b", "trend_c"]
    for i, trend_id in enumerate(ids):
        _write_json(raw_root / trend_id / "trend_metadata.json", {"trend_id": trend_id, "source_url": f"https://www.tiktok.com/@x/video/{i}"})
        _write_json(ranking_root / trend_id / "ranking.json", {"trend_id": trend_id, "candidate_rank": i + 1, "candidate_probability": 0.9 - i * 0.1, "selected_for_multimodal": True})
        _write_json(analysis_root / trend_id / "analysis.json", {"hook_type": "h", "trend_mechanic": "m"})
    _write_json(eval_root / "trend_a" / "evaluation.json", {"trend_id": "trend_a", "overall_score": 7.5, "adaptable": True, "rejection_reasons": [], "adaptation_idea": "a"})
    _write_json(eval_root / "trend_b" / "evaluation.json", {"trend_id": "trend_b", "overall_score": 6.5, "adaptable": True, "rejection_reasons": [], "adaptation_idea": "b"})
    _write_json(eval_root / "trend_c" / "evaluation.json", {"trend_id": "trend_c", "overall_score": 3.0, "adaptable": False, "rejection_reasons": ["no bridge"], "adaptation_idea": None})

    result = build_current_run_artifacts(
        run_root=run_root,
        artifacts_root=artifacts_root,
        run_id="run_20260831_190000",
        started_at=datetime.now(UTC),
        region="TR",
        discovery_rounds=2,
        current_trend_ids=set(ids),
        processing_stats=type("S", (), {"cache_hits": 2, "failures": {}})(),
    )
    assert result["run_summary"]["total_adaptable"] == 2
    assert (run_root / "run_summary.json").exists()
    assert (run_root / "adaptable_hooks.json").exists()
    assert (run_root / "rejected_examples.json").exists()
