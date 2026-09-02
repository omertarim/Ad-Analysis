from __future__ import annotations

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from task1.services.video_generator import VeoVideoGenerator


def generate_from_completed_run(
    *,
    artifacts_root: Path,
    run_id: str,
    top_n: int,
    preferred_model: str = "veo-3.1-lite-generate-preview",
    fallback_model: str | None = None,
    king_reference_path: str | Path | None = None,
) -> dict[str, Any]:
    run_root = artifacts_root / "runs" / run_id
    adaptable_path = run_root / "adaptable_hooks.json"
    if not adaptable_path.exists():
        raise FileNotFoundError(f"run adaptable hooks not found: {adaptable_path}")

    adaptable = json.loads(adaptable_path.read_text(encoding="utf-8"))
    if not isinstance(adaptable, list) or not adaptable:
        raise RuntimeError("no_adaptable_hooks_in_run")
    selected = adaptable[:top_n]

    generator = VeoVideoGenerator(preferred_model=preferred_model, fallback_model=fallback_model)
    started = time.perf_counter()
    generation_root = artifacts_root / "generation"
    generation_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    model_used_values: list[str] = []
    for item in selected:
        trend_id = item["trend_id"]
        trend_dir = generation_root / trend_id
        trend_dir.mkdir(parents=True, exist_ok=True)

        brief = _build_creative_brief(artifacts_root=artifacts_root, adaptable_item=item)
        storyboard = _build_storyboard(brief=brief)
        prompt_text = _build_generation_prompt(brief=brief, storyboard=storyboard)

        creative_brief_path = trend_dir / "creative_brief.json"
        storyboard_path = trend_dir / "storyboard.json"
        generation_prompt_path = trend_dir / "generation_prompt.txt"
        generation_meta_path = trend_dir / "generation_metadata.json"
        qa_path = trend_dir / "qa.json"
        final_video_path = trend_dir / "final.mp4"
        legacy_text_only_path = trend_dir / "final_text_only.mp4"
        legacy_with_text_path = trend_dir / "final_with_text.mp4"

        _write_json(creative_brief_path, brief)
        _write_json(storyboard_path, storyboard)
        generation_prompt_path.write_text(prompt_text, encoding="utf-8")
        if final_video_path.exists():
            if not legacy_with_text_path.exists():
                final_video_path.replace(legacy_with_text_path)
            elif not legacy_text_only_path.exists():
                final_video_path.replace(legacy_text_only_path)

        gen_result = generator.generate(
            trend_id=trend_id,
            prompt_text=prompt_text,
            output_dir=trend_dir,
            duration_seconds=int(brief["target_duration_seconds"]),
            aspect_ratio=str(brief["aspect_ratio"]),
        )
        if gen_result.model_used:
            model_used_values.append(gen_result.model_used)

        meta_payload = {
            "trend_id": trend_id,
            "model": gen_result.model_used,
            "operation_name": gen_result.operation_name,
            "generation_success": gen_result.success,
            "error": gen_result.error,
            "generated_at": datetime.now(UTC).isoformat(),
            "creative_brief_path": str(creative_brief_path),
            "storyboard_path": str(storyboard_path),
            "generation_prompt_path": str(generation_prompt_path),
            "final_video_path": gen_result.final_video_path,
            "inference_mode": "veo_generation",
            "character_reference_used": False,
            "generation_mode": "text_conditioned",
            "model_used": gen_result.model_used,
        }
        _write_json(generation_meta_path, meta_payload)

        qa_result = _qa_generated_video(trend_id=trend_id, video_path=Path(gen_result.final_video_path))
        _write_json(qa_path, qa_result)

        result_item = {
            "rank": item["rank"],
            "trend_id": trend_id,
            "source_url": item.get("source_url"),
            "overall_score": item.get("overall_score"),
            "creative_brief_path": str(creative_brief_path),
            "storyboard_path": str(storyboard_path),
            "generation_prompt_path": str(generation_prompt_path),
            "final_video_path": gen_result.final_video_path,
            "qa_path": str(qa_path),
            "generation_success": gen_result.success,
            "generation_error": gen_result.error,
            "model_used": gen_result.model_used,
            "generation_elapsed_seconds": gen_result.elapsed_seconds,
            "qa_passed": qa_result["passed"],
            "qa_duration_seconds": qa_result["duration_seconds"],
            "qa_width": qa_result["width"],
            "qa_height": qa_result["height"],
        }
        results.append(result_item)

        if gen_result.error and _is_access_blocker(gen_result.error):
            raise RuntimeError(gen_result.error)

    final_creatives = run_root / "final_creatives.json"
    _write_json(final_creatives, results)
    return {
        "run_id": run_id,
        "model_used": sorted(set(model_used_values)),
        "results": results,
        "total_generation_time_seconds": round(time.perf_counter() - started, 4),
        "final_creatives_path": str(final_creatives),
    }


def _build_creative_brief(*, artifacts_root: Path, adaptable_item: dict[str, Any]) -> dict[str, Any]:
    trend_id = adaptable_item["trend_id"]
    trend_meta = _read_json(artifacts_root / "raw" / trend_id / "trend_metadata.json")
    analysis = _read_json(artifacts_root / "analysis" / trend_id / "analysis.json")
    evaluation = _read_json(artifacts_root / "evaluations" / trend_id / "evaluation.json")
    original_hook = str(analysis.get("hook_description") or analysis.get("hook_type") or "trend hook")
    opening_hook = str(analysis.get("first_2_seconds_description") or original_hook)
    royal_adaptation = str(evaluation.get("adaptation_idea") or adaptable_item.get("adaptation_idea") or "")

    return {
        "trend_id": trend_id,
        "source_url": adaptable_item.get("source_url"),
        "overall_score": adaptable_item.get("overall_score"),
        "original_hook": original_hook,
        "original_trend_mechanic": adaptable_item.get("trend_mechanic"),
        "adaptation_idea": royal_adaptation,
        "royal_match_adaptation": royal_adaptation,
        "opening_hook": opening_hook,
        "problem": "A sudden castle danger threatens King Robert.",
        "player_intervention": "Player solves a rapid match-3 move to rescue King Robert.",
        "payoff": "Danger is resolved and King Robert safely celebrates.",
        "king_robert_role": "Main character in danger, then rescued by player intervention.",
        "visual_style": "Bright polished stylized 3D mobile game ad aesthetic.",
        "pacing": "fast",
        "target_duration_seconds": 8,
        "aspect_ratio": "9:16",
        "metadata_caption": trend_meta.get("caption"),
    }


def _build_storyboard(*, brief: dict[str, Any]) -> dict[str, Any]:
    opening_hook = _visual_only_text(brief["opening_hook"])
    adaptation = _visual_only_text(brief["royal_match_adaptation"])
    shots = [
        {
            "shot_number": 1,
            "start_time": 0.0,
            "end_time": 2.0,
            "visual": opening_hook,
            "action": "Immediate visual hook with King Robert and visible danger.",
            "text_overlay": "none",
        },
        {
            "shot_number": 2,
            "start_time": 2.0,
            "end_time": 4.0,
            "visual": "Danger escalates in the royal castle environment.",
            "action": "Problem worsens fast and stakes become clear.",
            "text_overlay": "none",
        },
        {
            "shot_number": 3,
            "start_time": 4.0,
            "end_time": 6.0,
            "visual": "Match-3 board appears as player intervention moment.",
            "action": "Player makes decisive move to save King Robert.",
            "text_overlay": "none",
        },
        {
            "shot_number": 4,
            "start_time": 6.0,
            "end_time": 8.0,
            "visual": "Successful rescue payoff with bright celebratory finish.",
            "action": adaptation,
            "text_overlay": "none",
        },
    ]
    return {"trend_id": brief["trend_id"], "total_duration_seconds": 8.0, "shots": shots}


def _build_generation_prompt(*, brief: dict[str, Any], storyboard: dict[str, Any]) -> str:
    shot_lines = []
    for shot in storyboard["shots"]:
        overlay = shot.get("text_overlay") or "none"
        shot_lines.append(
            f"Shot {shot['shot_number']} ({shot['start_time']}-{shot['end_time']}s): {shot['visual']} | {shot['action']} | overlay_hint: {overlay}"
        )
    return (
        "Generate a single finished mobile game ad video.\n"
        "Format: vertical 9:16, exactly 8 seconds, fast pacing.\n"
        "Character: King Robert must be the main character.\n"
        "World: Royal Match-inspired bright stylized 3D castle environment, polished mobile-game ad quality.\n"
        "Audio: natural scene sound and music only; NO dialogue, NO voice-over, NO spoken words.\n"
        "Narrative: immediate hook in first 2 seconds, clear visual danger/problem, player intervention, satisfying payoff.\n"
        "Keep the adaptation very close to the provided trend mechanic and adaptation idea beat-by-beat while remaining Royal Match themed.\n"
        f"Preserve source trend mechanic: {brief['original_trend_mechanic']}.\n"
        f"Opening hook reference: {_visual_only_text(brief['opening_hook'])}.\n"
        f"Adaptation idea reference: {_visual_only_text(brief['royal_match_adaptation'])}.\n"
        "King Robert is a cheerful, stocky, middle-aged cartoon king with a large rounded face, warm expressive eyes, "
        "a prominent rounded nose, thick white eyebrows, and a full neatly groomed white beard and moustache.\n"
        "He wears a large polished golden crown with a prominent red jewel, a royal blue tunic with gold trim and details, "
        "a red royal cape, and a bright polished mobile-game royal costume.\n"
        "His proportions are stylized and friendly: large expressive head, compact body, rounded silhouette, short sturdy legs, "
        "and exaggerated readable facial expressions.\n"
        "His visual style should feel like a premium, bright, polished, family-friendly 3D mobile puzzle game character.\n"
        "Keep the SAME King Robert appearance consistently throughout the entire video.\n"
        "Do not redesign his facial features, crown, beard, costume colors, body proportions, or age between shots.\n"
        "ABSOLUTELY NO VISIBLE TEXT OR TYPOGRAPHY OF ANY KIND.\n"
        "Do NOT render subtitles, captions, titles, character names, 'King Robert', '9:16', numbers, labels, speech bubbles, "
        "UI text, logos, signs, banners containing words, fake letters, gibberish, pseudo-writing, or watermark-like text.\n"
        "Do not write '9:16' on screen.\n"
        "Do not generate fake subtitles or unreadable text.\n"
        "The frame should be driven by visual action, facial expression, and gameplay moments.\n"
        "Any communication must be conveyed through facial expression, acting, environment, camera movement and sound, "
        "never through spoken language.\n"
        "Do not copy creator or celebrity likeness. Do not include TikTok logos. Do not include copyrighted unrelated characters.\n"
        "Clean composition, readable motion, game-ad polish.\n"
        "Storyboard:\n"
        + "\n".join(shot_lines)
    )


def _qa_generated_video(*, trend_id: str, video_path: Path) -> dict[str, Any]:
    failure_reasons: list[str] = []
    if not video_path.exists():
        return {
            "trend_id": trend_id,
            "passed": False,
            "duration_seconds": None,
            "width": None,
            "height": None,
            "aspect_ratio": None,
            "codec": None,
            "character_consistency_review": "manual_required",
            "on_screen_text_review": "manual_required",
            "failure_reasons": ["file_not_found"],
        }

    probe = _ffprobe(video_path)
    if probe.get("error"):
        return {
            "trend_id": trend_id,
            "passed": False,
            "duration_seconds": None,
            "width": None,
            "height": None,
            "aspect_ratio": None,
            "codec": None,
            "character_consistency_review": "manual_required",
            "on_screen_text_review": "manual_required",
            "failure_reasons": [probe["error"]],
        }

    duration = float(probe["duration_seconds"])
    width = int(probe["width"])
    height = int(probe["height"])
    codec = probe["codec"]
    if duration > 15.0:
        trimmed = video_path.with_name("final_trimmed.mp4")
        trim_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-t",
            "15",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(trimmed),
        ]
        run_trim = subprocess.run(trim_cmd, capture_output=True, text=True, check=False)
        if run_trim.returncode == 0 and trimmed.exists():
            trimmed.replace(video_path)
            probe = _ffprobe(video_path)
            if not probe.get("error"):
                duration = float(probe["duration_seconds"])
                width = int(probe["width"])
                height = int(probe["height"])
                codec = probe["codec"]
        else:
            failure_reasons.append("ffmpeg_trim_failed")

    # normalize to vertical 9:16 when needed
    ratio = (width / height) if height else 0.0
    target_ratio = 9 / 16
    if abs(ratio - target_ratio) > 0.02:
        normalized = video_path.with_name("final_normalized.mp4")
        vf_expr = (
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920"
        )
        norm_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            vf_expr,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(normalized),
        ]
        run_norm = subprocess.run(norm_cmd, capture_output=True, text=True, check=False)
        if run_norm.returncode == 0 and normalized.exists():
            normalized.replace(video_path)
            probe = _ffprobe(video_path)
            if not probe.get("error"):
                duration = float(probe["duration_seconds"])
                width = int(probe["width"])
                height = int(probe["height"])
                codec = probe["codec"]
                ratio = (width / height) if height else 0.0
        else:
            failure_reasons.append("ffmpeg_aspect_normalization_failed")

    if duration < 8.0 or duration > 15.0:
        failure_reasons.append("duration_out_of_range_8_15")
    if width <= 0 or height <= 0:
        failure_reasons.append("invalid_resolution")
    if height <= width:
        failure_reasons.append("not_vertical")
    if abs((width / height) - target_ratio) > 0.02:
        failure_reasons.append("aspect_ratio_not_9_16")
    if not codec:
        failure_reasons.append("codec_missing")

    return {
        "trend_id": trend_id,
        "passed": len(failure_reasons) == 0,
        "duration_seconds": round(duration, 4),
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 6) if height else None,
        "codec": codec,
        "character_consistency_review": "manual_required",
        "on_screen_text_review": "manual_required",
        "failure_reasons": failure_reasons,
    }


def _visual_only_text(value: str) -> str:
    text = " ".join(value.split())
    text = text.replace('"', "").replace("'", "")
    lowered = text.lower()
    if "text overlay" in lowered or "caption" in lowered or "subtitle" in lowered:
        return (
            "Convey the same trend mechanic visually through character action and scene contrast only, "
            "without any written text."
        )
    return text


def _ffprobe(video_path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return {"error": "ffprobe_not_found"}
    if result.returncode != 0:
        return {"error": f"ffprobe_failed={result.stderr.strip() or 'unknown'}"}
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return {"error": "ffprobe_invalid_json"}
    streams = payload.get("streams") or []
    video_stream = None
    for stream in streams:
        if isinstance(stream, dict) and stream.get("codec_type") == "video":
            video_stream = stream
            break
    if not video_stream:
        return {"error": "no_video_stream"}
    width = video_stream.get("width")
    height = video_stream.get("height")
    duration = video_stream.get("duration") or (payload.get("format") or {}).get("duration")
    codec = video_stream.get("codec_name")
    try:
        return {
            "duration_seconds": float(duration),
            "width": int(width),
            "height": int(height),
            "codec": codec if isinstance(codec, str) else None,
        }
    except Exception:  # noqa: BLE001
        return {"error": "ffprobe_parse_error"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _is_access_blocker(error_text: str) -> bool:
    lowered = error_text.lower()
    return any(
        marker in lowered
        for marker in [
            "permission_denied",
            "insufficient",
            "billing",
            "quota",
            "403",
            "not enabled",
            "access",
        ]
    )
