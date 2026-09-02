from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from task1.pipeline.discovery import discover_trends, persist_trend_metadata
from task1.pipeline.generation import generate_from_completed_run
from task1.pipeline.processing import ProcessingStats, process_selected_candidates
from task1.pipeline.ranking import rank_and_persist
from task1.pipeline.summaries import build_current_run_artifacts
from task1.services.candidate_ranker import CandidateRanker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task 1 production process pipeline")
    parser.add_argument("--region", type=str, default="TR")
    parser.add_argument("--discovery-limit", type=int, default=20)
    parser.add_argument("--target-adaptable", type=int, default=10)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--model", type=str, default="gemini-3.6-flash")
    parser.add_argument("--selection-mode", choices=["top_percent", "threshold"], default="top_percent")
    parser.add_argument("--top-percent", type=float, default=0.50)
    parser.add_argument("--minimum-candidates", type=int, default=5)
    parser.add_argument("--maximum-candidates", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--full-pipeline", action="store_true")
    parser.add_argument("--generate-from-run", type=str, default=None)
    parser.add_argument("--generate-top", type=int, default=3)
    parser.add_argument(
        "--king-reference",
        type=str,
        default="task1/assets/king_robert_reference/king_robert.png",
    )
    parser.add_argument("--apify-api-token", type=str, default=None)
    parser.add_argument("--gemini-api-key", type=str, default=None)
    return parser


def _configure_and_validate_credentials(args: argparse.Namespace) -> tuple[str | None, str]:
    if args.apify_api_token:
        os.environ["APIFY_API_TOKEN"] = args.apify_api_token
    if args.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = args.gemini_api_key

    apify_token = args.apify_api_token or os.getenv("APIFY_API_TOKEN")
    gemini_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")
    requires_apify = args.generate_from_run is None
    requires_gemini = True

    if requires_apify and not apify_token:
        raise RuntimeError("Missing Apify API token. Provide --apify-api-token or set APIFY_API_TOKEN.")
    if requires_gemini and not gemini_key:
        raise RuntimeError("Missing Gemini API key. Provide --gemini-api-key or set GEMINI_API_KEY.")
    return apify_token, gemini_key


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.full_pipeline and args.generate_from_run:
        parser.error("--full-pipeline cannot be used with --generate-from-run")

    apify_token, _gemini_key = _configure_and_validate_credentials(args)

    artifacts_root = Path("task1/artifacts")
    if args.generate_from_run:
        generation_summary = generate_from_completed_run(
            artifacts_root=artifacts_root,
            run_id=args.generate_from_run,
            top_n=args.generate_top,
            preferred_model="veo-3.1-lite-generate-preview",
            fallback_model=None,
            king_reference_path=args.king_reference,
        )
        print("generation_summary")
        print(generation_summary)
        return

    started_perf = time.perf_counter()
    started_at = datetime.now(UTC)
    run_id = f"run_{started_at.strftime('%Y%m%d_%H%M%S')}"
    run_root = artifacts_root / "runs" / run_id
    raw_root = artifacts_root / "raw"
    rankings_root = artifacts_root / "rankings"
    analysis_root = artifacts_root / "analysis"
    evaluations_root = artifacts_root / "evaluations"
    for root in (raw_root, rankings_root, analysis_root, evaluations_root, artifacts_root / "summaries"):
        root.mkdir(parents=True, exist_ok=True)

    cumulative_stats = ProcessingStats()
    current_run_trend_ids: set[str] = set()
    discovery_rounds = 0

    for _round_idx in range(1, args.max_rounds + 1):
        discovery_rounds = _round_idx
        print("[DISCOVERY] start", flush=True)
        trends = discover_trends(region=args.region, limit=args.discovery_limit)
        print(f"[DISCOVERY] end count={len(trends)}", flush=True)
        unique_trends = []
        for trend in trends:
            persist_trend_metadata(trend, raw_root)
            if trend.trend_id in current_run_trend_ids:
                cumulative_stats.skipped_existing += 1
                print(f"[SKIP] {trend.trend_id} reason=already_seen_in_current_run", flush=True)
                continue
            current_run_trend_ids.add(trend.trend_id)
            unique_trends.append(trend)
        if not unique_trends:
            continue

        ranker = CandidateRanker(
            selection_mode=args.selection_mode,
            top_percent=args.top_percent,
            minimum_candidates=args.minimum_candidates,
            maximum_candidates=args.maximum_candidates,
            threshold=args.threshold,
        )
        print("[RANKING] start", flush=True)
        rankings = rank_and_persist(unique_trends, rankings_root=rankings_root, ranker=ranker)
        print(f"[RANKING] end count={len(rankings)}", flush=True)
        round_stats = process_selected_candidates(
            trends=unique_trends,
            rankings=rankings,
            raw_root=raw_root,
            rankings_root=rankings_root,
            analysis_root=analysis_root,
            evaluations_root=evaluations_root,
            apify_token=apify_token,
            model=args.model,
            target_adaptable=args.target_adaptable,
            analysis_timeout_seconds=120,
            evaluation_timeout_seconds=90,
            current_run_trend_ids=current_run_trend_ids,
        )
        cumulative_stats.ml_selected += round_stats.ml_selected
        cumulative_stats.downloaded += round_stats.downloaded
        cumulative_stats.validated += round_stats.validated
        cumulative_stats.analyzed += round_stats.analyzed
        cumulative_stats.evaluated += round_stats.evaluated
        cumulative_stats.skipped_existing += round_stats.skipped_existing
        cumulative_stats.cache_hits += round_stats.cache_hits
        cumulative_stats.failures.update(round_stats.failures)

        summaries = build_current_run_artifacts(
            run_root=run_root,
            artifacts_root=artifacts_root,
            run_id=run_id,
            started_at=started_at,
            region=args.region,
            discovery_rounds=discovery_rounds,
            current_trend_ids=current_run_trend_ids,
            processing_stats=cumulative_stats,
        )
        if summaries["run_summary"]["total_adaptable"] >= args.target_adaptable:
            break

    final_summaries = build_current_run_artifacts(
        run_root=run_root,
        artifacts_root=artifacts_root,
        run_id=run_id,
        started_at=started_at,
        region=args.region,
        discovery_rounds=discovery_rounds,
        current_trend_ids=current_run_trend_ids,
        processing_stats=cumulative_stats,
    )
    final_summaries["run_summary"]["runtime_seconds"] = round(time.perf_counter() - started_perf, 4)
    (run_root / "run_summary.json").write_text(
        json.dumps(final_summaries["run_summary"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"run_id={run_id}")
    print("run_summary")
    print(final_summaries["run_summary"])
    print("adaptable_count")
    print(len(final_summaries["adaptable_hooks"]))

    if args.full_pipeline:
        generation_summary = generate_from_completed_run(
            artifacts_root=artifacts_root,
            run_id=run_id,
            top_n=args.generate_top,
            preferred_model="veo-3.1-lite-generate-preview",
            fallback_model=None,
            king_reference_path=args.king_reference,
        )
        print("generation_summary")
        print(generation_summary)


if __name__ == "__main__":
    main()
