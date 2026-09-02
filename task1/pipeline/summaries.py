from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from task1.pipeline.processing import ProcessingStats


def build_current_run_artifacts(
    *,
    run_root: Path,
    artifacts_root: Path,
    run_id: str,
    started_at: datetime,
    region: str,
    discovery_rounds: int,
    current_trend_ids: set[str],
    processing_stats: ProcessingStats,
) -> dict[str, Any]:
    raw_root = artifacts_root / "raw"
    rankings_root = artifacts_root / "rankings"
    analysis_root = artifacts_root / "analysis"
    evaluations_root = artifacts_root / "evaluations"

    run_rows: list[dict[str, Any]] = []
    discovered_payload: list[dict[str, Any]] = []
    rankings_payload: list[dict[str, Any]] = []

    for trend_id in sorted(current_trend_ids):
        meta_path = raw_root / trend_id / "trend_metadata.json"
        ranking_path = rankings_root / trend_id / "ranking.json"
        source_path = raw_root / trend_id / "source.mp4"
        analysis_path = analysis_root / trend_id / "analysis.json"
        legacy_analysis_path = artifacts_root / "analysis_ready" / trend_id / "analysis.json"
        evaluation_path = evaluations_root / trend_id / "evaluation.json"

        meta_payload = _read_json(meta_path) if meta_path.exists() else {}
        if meta_payload:
            discovered_payload.append(meta_payload)

        ranking_payload = _read_json(ranking_path) if ranking_path.exists() else {}
        if ranking_payload:
            rankings_payload.append(ranking_payload)

        if not analysis_path.exists() and legacy_analysis_path.exists():
            _write_json(analysis_path, _read_json(legacy_analysis_path))

        if not evaluation_path.exists():
            continue
        eval_payload = _read_json(evaluation_path)
        analysis_payload = _read_json(analysis_path) if analysis_path.exists() else {}

        run_rows.append(
            {
                "trend_id": trend_id,
                "source_url": meta_payload.get("source_url"),
                "candidate_probability": ranking_payload.get("candidate_probability"),
                "candidate_rank": ranking_payload.get("candidate_rank"),
                "selected_for_multimodal": ranking_payload.get("selected_for_multimodal"),
                "overall_score": eval_payload.get("overall_score"),
                "adaptable": eval_payload.get("adaptable"),
                "hook_type": analysis_payload.get("hook_type"),
                "trend_mechanic": analysis_payload.get("trend_mechanic"),
                "adaptation_idea": eval_payload.get("adaptation_idea"),
                "rejection_reasons": eval_payload.get("rejection_reasons", []),
                "source_video_path": str(source_path),
                "analysis_path": str(analysis_path),
                "evaluation_path": str(evaluation_path),
            }
        )

    adaptable_rows = [row for row in run_rows if row.get("adaptable") is True]
    adaptable_rows = sorted(adaptable_rows, key=lambda r: float(r.get("overall_score", 0.0)), reverse=True)
    for rank, row in enumerate(adaptable_rows, start=1):
        row["rank"] = rank

    rejected_rows = [row for row in run_rows if row.get("adaptable") is False]
    rejected_rows = sorted(rejected_rows, key=lambda r: float(r.get("overall_score", 0.0)))

    adaptable_hooks = [
        {
            "rank": row["rank"],
            "trend_id": row["trend_id"],
            "source_url": row["source_url"],
            "candidate_probability": row["candidate_probability"],
            "overall_score": row["overall_score"],
            "hook_type": row["hook_type"],
            "trend_mechanic": row["trend_mechanic"],
            "adaptation_idea": row["adaptation_idea"],
            "source_video_path": row["source_video_path"],
            "analysis_path": row["analysis_path"],
            "evaluation_path": row["evaluation_path"],
        }
        for row in adaptable_rows
    ]

    rejected_examples = [
        {
            "trend_id": row["trend_id"],
            "source_url": row["source_url"],
            "overall_score": row["overall_score"],
            "rejection_reasons": row["rejection_reasons"],
        }
        for row in rejected_rows[:3]
    ]

    selected_count = sum(1 for row in rankings_payload if row.get("selected_for_multimodal") is True)
    total_analyzed = 0
    total_evaluated = len(run_rows)
    for trend_id in current_trend_ids:
        analysis_path = analysis_root / trend_id / "analysis.json"
        legacy_analysis_path = artifacts_root / "analysis_ready" / trend_id / "analysis.json"
        if analysis_path.exists() or legacy_analysis_path.exists():
            total_analyzed += 1

    run_summary = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "region": region,
        "discovery_rounds": discovery_rounds,
        "total_current_unique_discovered": len(current_trend_ids),
        "total_ml_selected": selected_count,
        "total_analyzed": total_analyzed,
        "total_evaluated": total_evaluated,
        "total_adaptable": len(adaptable_rows),
        "total_rejected": len(rejected_rows),
        "cache_hits": processing_stats.cache_hits,
        "failures": processing_stats.failures,
        "runtime_seconds": round((datetime.now(UTC) - started_at).total_seconds(), 4),
    }

    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_root / "discovered.json", discovered_payload)
    _write_json(run_root / "rankings.json", sorted(rankings_payload, key=lambda r: int(r.get("candidate_rank", 10**9))))
    _write_json(run_root / "adaptable_hooks.json", adaptable_hooks)
    _write_json(run_root / "rejected_examples.json", rejected_examples)
    _write_json(run_root / "run_summary.json", run_summary)
    return {
        "run_summary": run_summary,
        "adaptable_hooks": adaptable_hooks,
        "rejected_examples": rejected_examples,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
