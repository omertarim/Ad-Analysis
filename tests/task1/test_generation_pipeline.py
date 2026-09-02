from __future__ import annotations

import json
from pathlib import Path

from task1.pipeline import generation
from task1.services.video_generator import VeoVideoGenerator


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _seed_artifacts(root: Path, trend_id: str) -> None:
    _write_json(root / "raw" / trend_id / "trend_metadata.json", {"trend_id": trend_id, "caption": "x"})
    _write_json(
        root / "analysis" / trend_id / "analysis.json",
        {
            "trend_id": trend_id,
            "hook_type": "hook",
            "hook_description": "Hook scene",
            "first_2_seconds_description": "Fast opening",
            "trend_mechanic": "mechanic",
        },
    )
    _write_json(
        root / "evaluations" / trend_id / "evaluation.json",
        {
            "trend_id": trend_id,
            "adaptation_idea": "King Robert solves it",
            "overall_score": 7.5,
            "adaptable": True,
        },
    )


def test_load_top3_from_completed_run_and_output_paths(tmp_path: Path, monkeypatch) -> None:
    artifacts_root = tmp_path / "artifacts"
    run_root = artifacts_root / "runs" / "run_x"
    init_calls: list[dict[str, object]] = []
    items = []
    for idx in range(1, 6):
        trend_id = f"trend_{idx}"
        items.append(
            {
                "rank": idx,
                "trend_id": trend_id,
                "source_url": f"https://tiktok.com/{idx}",
                "overall_score": 8 - idx * 0.1,
                "trend_mechanic": "mechanic",
                "adaptation_idea": "idea",
            }
        )
        _seed_artifacts(artifacts_root, trend_id)
    _write_json(run_root / "adaptable_hooks.json", items)

    class FakeGenerator:
        def __init__(self, *args, **kwargs) -> None:
            init_calls.append(kwargs)

        def generate(self, *, trend_id, prompt_text, output_dir, duration_seconds, aspect_ratio):
            final = Path(output_dir) / "final.mp4"
            final.write_bytes(b"ok")
            return type(
                "R",
                (),
                {
                    "trend_id": trend_id,
                    "success": True,
                    "model_used": "veo-3.1-lite-generate-preview",
                    "final_video_path": str(final),
                    "metadata_path": str(Path(output_dir) / "generation_metadata.json"),
                    "operation_name": "op123",
                    "error": None,
                    "elapsed_seconds": 1.0,
                },
            )()

    monkeypatch.setattr(generation, "VeoVideoGenerator", FakeGenerator)
    monkeypatch.setattr(
        generation,
        "_qa_generated_video",
        lambda trend_id, video_path: {
            "trend_id": trend_id,
            "passed": True,
            "duration_seconds": 12.0,
            "width": 1080,
            "height": 1920,
            "aspect_ratio": 0.5625,
            "codec": "h264",
            "character_consistency_review": "manual_required",
            "on_screen_text_review": "manual_required",
            "failure_reasons": [],
        },
    )

    result = generation.generate_from_completed_run(
        artifacts_root=artifacts_root,
        run_id="run_x",
        top_n=3,
        king_reference_path=None,
    )
    assert len(result["results"]) == 3
    assert init_calls and init_calls[0]["preferred_model"] == "veo-3.1-lite-generate-preview"
    assert init_calls[0]["fallback_model"] is None
    assert [x["trend_id"] for x in result["results"]] == ["trend_1", "trend_2", "trend_3"]
    assert (run_root / "final_creatives.json").exists()
    for row in result["results"]:
        assert row["creative_brief_path"].endswith("creative_brief.json")
        assert row["storyboard_path"].endswith("storyboard.json")
        assert row["generation_prompt_path"].endswith("generation_prompt.txt")
        assert row["final_video_path"].endswith("final.mp4")


def test_prompt_contains_king_robert_and_storyboard_range(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    trend_id = "trend_1"
    _seed_artifacts(artifacts_root, trend_id)
    adaptable_item = {
        "trend_id": trend_id,
        "source_url": "https://tiktok.com/1",
        "overall_score": 8.2,
        "trend_mechanic": "pov fail",
        "adaptation_idea": "King Robert rescue",
    }
    brief = generation._build_creative_brief(artifacts_root=artifacts_root, adaptable_item=adaptable_item)
    storyboard = generation._build_storyboard(brief=brief)
    prompt = generation._build_generation_prompt(brief=brief, storyboard=storyboard)

    assert "King Robert" in prompt
    assert "King Robert is a cheerful, stocky, middle-aged cartoon king" in prompt
    assert "Keep the SAME King Robert appearance consistently throughout the entire video." in prompt
    assert "NO dialogue" in prompt
    assert "ABSOLUTELY NO VISIBLE TEXT OR TYPOGRAPHY OF ANY KIND." in prompt
    assert "Do NOT render subtitles, captions, titles, character names" in prompt
    assert "Do not write '9:16' on screen." in prompt
    assert all(shot.get("text_overlay") == "none" for shot in storyboard["shots"])
    assert 8 <= storyboard["total_duration_seconds"] <= 15


def test_qa_rejects_invalid_duration_and_aspect_ratio(tmp_path: Path, monkeypatch) -> None:
    video_path = tmp_path / "final.mp4"
    video_path.write_bytes(b"bytes")
    monkeypatch.setattr(
        generation,
        "_ffprobe",
        lambda _p: {"duration_seconds": 20.0, "width": 1920, "height": 1080, "codec": "h264"},
    )
    monkeypatch.setattr(
        generation.subprocess,
        "run",
        lambda *args, **kwargs: type("P", (), {"returncode": 1, "stderr": "ffmpeg failed"})(),
    )
    qa = generation._qa_generated_video(trend_id="trend_x", video_path=video_path)
    assert qa["passed"] is False
    assert qa["character_consistency_review"] == "manual_required"
    assert qa["on_screen_text_review"] == "manual_required"
    assert "duration_out_of_range_8_15" in qa["failure_reasons"]
    assert "not_vertical" in qa["failure_reasons"]


def test_missing_gemini_api_key_clear_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    generator = VeoVideoGenerator(api_key=None)
    result = generator.generate(trend_id="trend_x", prompt_text="abc", output_dir=tmp_path)
    assert result.success is False
    assert result.error == "api_key_missing"


def test_generation_metadata_is_text_conditioned(tmp_path: Path, monkeypatch) -> None:
    artifacts_root = tmp_path / "artifacts"
    run_root = artifacts_root / "runs" / "run_x"
    _write_json(
        run_root / "adaptable_hooks.json",
        [{"rank": 1, "trend_id": "trend_1", "source_url": "https://tiktok.com/1", "overall_score": 8.1}],
    )
    _seed_artifacts(artifacts_root, "trend_1")

    class FakeGenerator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def generate(self, *, trend_id, prompt_text, output_dir, duration_seconds, aspect_ratio):
            final = Path(output_dir) / "final.mp4"
            final.write_bytes(b"ok")
            return type(
                "R",
                (),
                {
                    "trend_id": trend_id,
                    "success": True,
                    "model_used": "veo-3.1-lite-generate-preview",
                    "final_video_path": str(final),
                    "metadata_path": str(Path(output_dir) / "generation_metadata.json"),
                    "operation_name": "op123",
                    "error": None,
                    "elapsed_seconds": 1.0,
                },
            )()

    monkeypatch.setattr(generation, "VeoVideoGenerator", FakeGenerator)
    monkeypatch.setattr(
        generation,
        "_qa_generated_video",
        lambda trend_id, video_path: {
            "trend_id": trend_id,
            "passed": True,
            "duration_seconds": 8.0,
            "width": 720,
            "height": 1280,
            "aspect_ratio": 0.5625,
            "codec": "h264",
            "character_consistency_review": "manual_required",
            "on_screen_text_review": "manual_required",
            "failure_reasons": [],
        },
    )
    generation.generate_from_completed_run(
        artifacts_root=artifacts_root,
        run_id="run_x",
        top_n=1,
        king_reference_path=tmp_path / "missing.png",
    )
    payload = json.loads(
        (artifacts_root / "generation" / "trend_1" / "generation_metadata.json").read_text(encoding="utf-8")
    )
    assert payload["character_reference_used"] is False
    assert payload["generation_mode"] == "text_conditioned"
    assert payload["model_used"] == "veo-3.1-lite-generate-preview"
