from __future__ import annotations

from pathlib import Path

from task1.services.video_generator import VeoVideoGenerator, VideoGenerationResult


def test_default_model_is_lite_and_no_expensive_fallback(tmp_path: Path, monkeypatch) -> None:
    generator = VeoVideoGenerator(api_key="test-key")
    assert generator.preferred_model == "veo-3.1-lite-generate-preview"
    assert generator.fallback_model is None

    called_models: list[str] = []

    def _fake_generate_with_model(**kwargs):
        called_models.append(kwargs["model_name"])
        return VideoGenerationResult(
            trend_id="trend_x",
            success=False,
            model_used=kwargs["model_name"],
            final_video_path=str(tmp_path / "final.mp4"),
            metadata_path=str(tmp_path / "generation_metadata.json"),
            operation_name="op",
            error="operation_succeeded_but_no_generated_video",
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(generator, "_generate_with_model", _fake_generate_with_model)
    result = generator.generate(
        trend_id="trend_x",
        prompt_text="prompt",
        output_dir=tmp_path,
        duration_seconds=8,
        aspect_ratio="9:16",
    )
    assert result.success is False
    assert set(called_models) == {"veo-3.1-lite-generate-preview"}
