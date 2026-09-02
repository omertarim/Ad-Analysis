from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import json
from dataclasses import dataclass, field
from pathlib import Path

from task1.domain.candidate_ranking import CandidateRankingResult
from task1.domain.trend import TrendSourceMetadata
from task1.domain.video_analysis import VideoContentAnalysis
from task1.pipeline.discovery import fetch_kv_urls_for_trends
from task1.services.creative_evaluator import CreativeEvaluator
from task1.services.multimodal_analyzer import MultimodalAnalyzer
from task1.services.video_acquisition import VideoAcquisitionService
from task1.services.video_validator import VideoValidator


@dataclass
class ProcessingStats:
    ml_selected: int = 0
    downloaded: int = 0
    validated: int = 0
    analyzed: int = 0
    evaluated: int = 0
    skipped_existing: int = 0
    cache_hits: int = 0
    failures: dict[str, str] = field(default_factory=dict)


def process_selected_candidates(
    trends: list[TrendSourceMetadata],
    rankings: list[CandidateRankingResult],
    raw_root: Path,
    rankings_root: Path,
    analysis_root: Path,
    evaluations_root: Path,
    apify_token: str,
    model: str = "gemini-3.6-flash",
    target_adaptable: int = 10,
    analysis_timeout_seconds: int = 120,
    evaluation_timeout_seconds: int = 90,
    current_run_trend_ids: set[str] | None = None,
) -> ProcessingStats:
    stats = ProcessingStats()
    selected_rankings = [r for r in rankings if r.selected_for_multimodal]
    stats.ml_selected = len(selected_rankings)
    trend_map = {trend.trend_id: trend for trend in trends}
    selected_trends = [trend_map[r.trend_id] for r in selected_rankings if r.trend_id in trend_map]

    acquisition = VideoAcquisitionService(timeout_seconds=120)
    validator = VideoValidator()
    analyzer = MultimodalAnalyzer(model=model)
    evaluator = CreativeEvaluator(model=model)

    need_download = [trend for trend in selected_trends if not (raw_root / trend.trend_id / "source.mp4").exists()]
    selected_with_cache: set[str] = {
        trend.trend_id for trend in selected_trends if (raw_root / trend.trend_id / "source.mp4").exists()
    }
    if need_download:
        try:
            kv_by_trend = fetch_kv_urls_for_trends(need_download, apify_token)
            for trend in need_download:
                print(f"[ACQUIRE] {trend.trend_id} start", flush=True)
                kv = kv_by_trend.get(trend.trend_id)
                if kv:
                    trend.raw_source_metadata["kvNoWatermarkUrl"] = kv
                result = acquisition.acquire(trend=trend, output_dir=raw_root)
                if result.success:
                    stats.downloaded += 1
                    print(f"[ACQUIRE] {trend.trend_id} end", flush=True)
                else:
                    stats.failures[trend.trend_id] = result.error or "download_failed"
                    print(
                        f"[FAIL] {trend.trend_id} reason={stats.failures[trend.trend_id]}",
                        flush=True,
                    )
        except Exception as exc:  # noqa: BLE001
            stats.failures["downloader_actor"] = str(exc)
            print(f"[FAIL] downloader_actor reason={exc}", flush=True)

    for trend in selected_trends:
        trend_id = trend.trend_id
        source_path = raw_root / trend_id / "source.mp4"
        ranking_path = rankings_root / trend_id / "ranking.json"
        eval_path = evaluations_root / trend_id / "evaluation.json"
        analysis_path = analysis_root / trend_id / "analysis.json"
        analysis_meta_path = analysis_root / trend_id / "analysis_metadata.json"

        if eval_path.exists() and analysis_path.exists():
            stats.skipped_existing += 1
            selected_with_cache.add(trend_id)
            print(f"[SKIP] {trend_id} reason=analysis_and_evaluation_already_exist", flush=True)
            print(
                f"adaptable_count={_count_adaptable(evaluations_root, current_run_trend_ids)}/{target_adaptable}",
                flush=True,
            )
            continue
        if not source_path.exists():
            stats.failures[trend_id] = "source_video_missing"
            print(f"[FAIL] {trend_id} reason=source_video_missing", flush=True)
            continue

        print(f"[VALIDATE] {trend_id} start", flush=True)
        validation = validator.validate(trend_id=trend_id, video_path=source_path)
        if not validation.valid:
            stats.failures[trend_id] = validation.error or "validation_failed"
            print(f"[FAIL] {trend_id} reason={stats.failures[trend_id]}", flush=True)
            continue
        print(f"[VALIDATE] {trend_id} end", flush=True)
        stats.validated += 1

        if not analysis_path.exists():
            legacy_analysis_path = Path("task1/artifacts/analysis_ready") / trend_id / "analysis.json"
            legacy_meta_path = Path("task1/artifacts/analysis_ready") / trend_id / "analysis_metadata.json"
            if legacy_analysis_path.exists():
                _copy_text_json(legacy_analysis_path, analysis_path)
                if legacy_meta_path.exists() and not analysis_meta_path.exists():
                    _copy_text_json(legacy_meta_path, analysis_meta_path)
                selected_with_cache.add(trend_id)
                print(f"[SKIP] {trend_id} reason=analysis_reused_from_legacy_artifact", flush=True)
            else:
                print(f"[ANALYSIS] {trend_id} start", flush=True)
                analysis_started = time.perf_counter()
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            analyzer.analyze_video,
                            trend_id,
                            source_path,
                            "task1/prompts/video_analysis.txt",
                            analysis_root,
                        )
                        result = future.result(timeout=analysis_timeout_seconds)
                except FuturesTimeoutError:
                    stats.failures[trend_id] = f"analysis_timeout_{analysis_timeout_seconds}s"
                    print(f"[FAIL] {trend_id} reason={stats.failures[trend_id]}", flush=True)
                    continue
                if not result.success:
                    stats.failures[trend_id] = result.error or "analysis_failed"
                    print(f"[FAIL] {trend_id} reason={stats.failures[trend_id]}", flush=True)
                    continue
                stats.analyzed += 1
                print(
                    f"[ANALYSIS] {trend_id} end elapsed_seconds={round(time.perf_counter() - analysis_started, 4)}",
                    flush=True,
                )
        else:
            selected_with_cache.add(trend_id)
            print(f"[SKIP] {trend_id} reason=analysis_already_exists", flush=True)

        if not eval_path.exists():
            try:
                analysis_payload = json.loads(analysis_path.read_text(encoding="utf-8"))
                analysis = VideoContentAnalysis.model_validate(analysis_payload)
                print(f"[EVALUATION] {trend_id} start", flush=True)
                evaluation_started = time.perf_counter()
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            evaluator.evaluate,
                            trend,
                            analysis,
                            "task1/prompts/creative_evaluation.txt",
                            evaluations_root,
                            source_path,
                            analysis_path,
                        )
                        result = future.result(timeout=evaluation_timeout_seconds)
                except FuturesTimeoutError:
                    stats.failures[trend_id] = f"evaluation_timeout_{evaluation_timeout_seconds}s"
                    print(f"[FAIL] {trend_id} reason={stats.failures[trend_id]}", flush=True)
                    continue
                if not result.success:
                    stats.failures[trend_id] = result.error or "evaluation_failed"
                    print(f"[FAIL] {trend_id} reason={stats.failures[trend_id]}", flush=True)
                    continue
                stats.evaluated += 1
                print(
                    f"[EVALUATION] {trend_id} end elapsed_seconds={round(time.perf_counter() - evaluation_started, 4)}",
                    flush=True,
                )
                print(
                    f"adaptable_count={_count_adaptable(evaluations_root, current_run_trend_ids)}/{target_adaptable}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                stats.failures[trend_id] = f"evaluation_runtime_error={exc}"
                print(f"[FAIL] {trend_id} reason={stats.failures[trend_id]}", flush=True)
        else:
            selected_with_cache.add(trend_id)
            print(f"[SKIP] {trend_id} reason=evaluation_already_exists", flush=True)
            print(
                f"adaptable_count={_count_adaptable(evaluations_root, current_run_trend_ids)}/{target_adaptable}",
                flush=True,
            )

    stats.cache_hits += len(selected_with_cache)
    return stats


def _copy_text_json(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _count_adaptable(evaluations_root: Path, trend_ids: set[str] | None = None) -> int:
    count = 0
    paths = list(evaluations_root.glob("trend_*/evaluation.json"))
    for path in paths:
        trend_id = path.parent.name
        if trend_ids is not None and trend_id not in trend_ids:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("adaptable") is True:
                count += 1
        except Exception:  # noqa: BLE001
            continue
    return count
