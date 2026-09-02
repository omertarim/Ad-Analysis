from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from task1.domain.creative_evaluation import CreativeAdaptabilityEvaluation
from task1.domain.trend import TrendSourceMetadata
from task1.domain.video_analysis import VideoContentAnalysis


@dataclass
class CreativeEvaluationRunResult:
    trend_id: str
    success: bool
    evaluation_path: str | None
    metadata_path: str | None
    model: str
    model_duration_seconds: float | None
    error: str | None
    evaluation: CreativeAdaptabilityEvaluation | None


class CreativeEvaluator:
    def __init__(
        self,
        model: str = "gemini-3.6-flash",
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.api_key = api_key or env_key
        self.model = model
        self._client = client

    def evaluate(
        self,
        trend: TrendSourceMetadata,
        analysis: VideoContentAnalysis,
        prompt_file_path: str | Path,
        output_root: str | Path,
        source_video_path: str | Path,
        analysis_artifact_path: str | Path,
    ) -> CreativeEvaluationRunResult:
        prompt_path = Path(prompt_file_path)
        output_dir = Path(output_root) / trend.trend_id
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluation_path = output_dir / "evaluation.json"
        metadata_path = output_dir / "evaluation_metadata.json"

        if not self.api_key:
            return self._fail(trend.trend_id, evaluation_path, metadata_path, "api_key_missing")
        if not prompt_path.exists():
            return self._fail(trend.trend_id, evaluation_path, metadata_path, "prompt_missing")

        prompt_text = prompt_path.read_text(encoding="utf-8")
        model_duration_seconds: float | None = None
        try:
            client = self._get_client()
            model_started = time.perf_counter()
            structured_data = self._request_structured_evaluation(
                client=client,
                prompt_text=prompt_text,
                trend=trend,
                analysis=analysis,
            )
            model_duration_seconds = round(time.perf_counter() - model_started, 4)

            # trend id is pipeline-owned.
            structured_data["trend_id"] = trend.trend_id
            overall_score = self._compute_overall_score(structured_data)
            structured_data["overall_score"] = overall_score
            adaptable = self._compute_adaptable(structured_data, overall_score)
            structured_data["adaptable"] = adaptable
            self._normalize_reason_fields(structured_data, adaptable=adaptable)

            evaluation = CreativeAdaptabilityEvaluation.model_validate(structured_data)
            evaluation_path.write_text(
                json.dumps(evaluation.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            metadata = {
                "trend_id": trend.trend_id,
                "source_url": trend.source_url,
                "source_video_path": str(source_video_path),
                "analysis_artifact_path": str(analysis_artifact_path),
                "evaluation_artifact_path": str(evaluation_path),
                "model": self.model,
                "prompt_path": str(prompt_path),
                "evaluated_at": datetime.now(UTC).isoformat(),
            }
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            return CreativeEvaluationRunResult(
                trend_id=trend.trend_id,
                success=True,
                evaluation_path=str(evaluation_path),
                metadata_path=str(metadata_path),
                model=self.model,
                model_duration_seconds=model_duration_seconds,
                error=None,
                evaluation=evaluation,
            )
        except ValidationError as exc:
            return self._fail(
                trend.trend_id,
                evaluation_path,
                metadata_path,
                f"invalid_structured_response={exc}",
                model_duration_seconds=model_duration_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            return self._fail(
                trend.trend_id,
                evaluation_path,
                metadata_path,
                f"model_failure={exc}",
                model_duration_seconds=model_duration_seconds,
            )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        from google import genai

        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _request_structured_evaluation(
        self,
        client: Any,
        prompt_text: str,
        trend: TrendSourceMetadata,
        analysis: VideoContentAnalysis,
    ) -> dict[str, Any]:
        from google.genai import types

        input_payload = {
            "trend_source_metadata": trend.model_dump(mode="json"),
            "video_content_analysis": analysis.model_dump(mode="json"),
        }
        response = client.models.generate_content(
            model=self.model,
            contents=[prompt_text, json.dumps(input_payload, ensure_ascii=False)],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CreativeAdaptabilityEvaluation,
            ),
        )

        if hasattr(response, "parsed") and response.parsed is not None:
            parsed = response.parsed
            if isinstance(parsed, CreativeAdaptabilityEvaluation):
                return parsed.model_dump(mode="json")
            if isinstance(parsed, dict):
                return parsed
            raise RuntimeError("invalid_parsed_payload_type")

        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return json.loads(text)
        raise RuntimeError("empty_model_response")

    @staticmethod
    def _compute_overall_score(data: dict[str, Any]) -> float:
        positive_score = (
            float(data["hook_strength_score"]) * 0.15
            + float(data["first_seconds_strength_score"]) * 0.10
            + float(data["royal_match_world_transferability_score"]) * 0.15
            + float(data["player_intervention_score"]) * 0.15
            + float(data["gameplay_bridge_score"]) * 0.15
            + float(data["visual_clarity_score"]) * 0.05
            + float(data["production_feasibility_score"]) * 0.05
            + float(data["tonal_fit_score"]) * 0.05
            + float(data["trend_transferability_score"]) * 0.10
            + float(data["brand_safety_score"]) * 0.05
        )
        risk_penalty = (
            float(data["context_dependency_risk"]) * 0.05
            + float(data["audio_dependency_risk"]) * 0.03
            + float(data["ip_or_identity_risk"]) * 0.07
        )
        raw_score = positive_score - risk_penalty
        return max(0.0, min(10.0, round(raw_score, 4)))

    @staticmethod
    def _compute_adaptable(data: dict[str, Any], overall_score: float) -> bool:
        return bool(
            overall_score >= 6.0
            and float(data["brand_safety_score"]) >= 6.0
            and float(data["gameplay_bridge_score"]) >= 5.0
            and float(data["royal_match_world_transferability_score"]) >= 5.0
            and float(data["ip_or_identity_risk"]) < 8.0
        )

    @staticmethod
    def _normalize_reason_fields(data: dict[str, Any], adaptable: bool) -> None:
        concerns = CreativeEvaluator._clean_reason_list(data.get("concerns"))
        rejection_reasons = CreativeEvaluator._clean_reason_list(data.get("rejection_reasons"))
        if adaptable:
            # If final decision is adaptable, reject-level reasons are downgraded to concerns.
            merged_concerns = concerns + rejection_reasons
            data["concerns"] = CreativeEvaluator._dedupe_preserve_order(merged_concerns)
            data["rejection_reasons"] = []
            return
        data["concerns"] = CreativeEvaluator._dedupe_preserve_order(concerns)
        data["rejection_reasons"] = CreativeEvaluator._dedupe_preserve_order(rejection_reasons)

    @staticmethod
    def _clean_reason_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if text:
                cleaned.append(text)
        return cleaned

    @staticmethod
    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            output.append(item)
        return output

    def _fail(
        self,
        trend_id: str,
        evaluation_path: Path,
        metadata_path: Path,
        error: str,
        model_duration_seconds: float | None = None,
    ) -> CreativeEvaluationRunResult:
        return CreativeEvaluationRunResult(
            trend_id=trend_id,
            success=False,
            evaluation_path=str(evaluation_path),
            metadata_path=str(metadata_path),
            model=self.model,
            model_duration_seconds=model_duration_seconds,
            error=error,
            evaluation=None,
        )
