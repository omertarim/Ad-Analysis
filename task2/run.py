"""Task 2 orchestrator: plan → generate → patch → QA."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from task2.pipeline.generate_assets import generate_theme_assets
from task2.pipeline.patch_html import OUTPUT_DIR, patch_theme_html
from task2.pipeline.qa import print_qa_summary, qa_theme
from task2.pipeline.theme_plan import THEMES_ROOT, plan_theme, theme_slug


def run_pipeline(
    theme: str,
    *,
    api_key: str | None = None,
    force: bool = False,
) -> dict:
    slug = theme_slug(theme)
    artifacts = THEMES_ROOT / slug
    started = time.perf_counter()

    print(f"[1/4] theme plan: {theme}", flush=True)
    plan_path = plan_theme(theme, api_key=api_key)
    print(f"      {plan_path}", flush=True)

    print("[2/4] asset generation", flush=True)
    manifest_path = generate_theme_assets(theme, api_key=api_key, force=force)
    print(f"      {manifest_path}", flush=True)

    print("[3/4] HTML patch", flush=True)
    html_path = patch_theme_html(theme)
    print(f"      {html_path}", flush=True)

    print("[4/4] QA", flush=True)
    qa_path = qa_theme(theme, api_key=api_key)
    qa_report = json.loads(qa_path.read_text(encoding="utf-8"))
    print_qa_summary(qa_report, prefix="      ")
    print(f"      report: {qa_path}", flush=True)
    if not qa_report.get("overall_pass"):
        raise RuntimeError(
            "QA failed "
            f"(playability={qa_report.get('playability_pass')}, "
            f"visual={qa_report.get('visual_pass')})"
        )

    elapsed = time.perf_counter() - started
    return {
        "theme": theme,
        "slug": slug,
        "output_html": str(html_path),
        "qa_result": "PASS" if qa_report.get("overall_pass") else "FAIL",
        "playability_pass": qa_report.get("playability_pass"),
        "visual_pass": qa_report.get("visual_pass"),
        "overall_pass": qa_report.get("overall_pass"),
        "artifacts": str(artifacts),
        "runtime_seconds": round(elapsed, 1),
        "plan": str(plan_path),
        "manifest": str(manifest_path),
        "qa_report": str(qa_path),
        "output_dir": str(OUTPUT_DIR),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task 2 theme pipeline")
    parser.add_argument("--theme", required=True)
    parser.add_argument("--api-key", dest="api_key", default=None)
    parser.add_argument("--gemini-api-key", dest="gemini_api_key", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate assets even if files already exist",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    api_key = args.api_key or args.gemini_api_key
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    try:
        summary = run_pipeline(args.theme, api_key=api_key, force=args.force)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print()
    print(f"output HTML: {summary['output_html']}")
    print(f"artifacts: {summary['artifacts']}")
    print(f"runtime: {summary['runtime_seconds']}s")
    print()
    qa_report = json.loads(Path(summary["qa_report"]).read_text(encoding="utf-8"))
    print_qa_summary(qa_report)
    print(f"qa report: {summary['qa_report']}")


if __name__ == "__main__":
    main()
