from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from task1.domain.video_analysis import VideoContentAnalysis


@dataclass
class AnalysisRunResult:
    trend_id: str
    success: bool
    analysis_path: str | None
    metadata_path: str | None
    model: str
    upload_succeeded: bool
    upload_duration_seconds: float | None
    model_duration_seconds: float | None
    error: str | None
    analysis: VideoContentAnalysis | None


class MultimodalAnalyzer:
    def __init__(
        self,
        model: str = "gemini-3.6-flash",
        api_key: str | None = None,
        client: Any | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.api_key = api_key or env_key
        self.model = model
        self._client = client
        self._sleep_fn = sleep_fn or time.sleep

    def analyze_video(
        self,
        trend_id: str,
        source_video_path: str | Path,
        prompt_file_path: str | Path,
        output_root: str | Path,
    ) -> AnalysisRunResult:
        video_path = Path(source_video_path)
        prompt_path = Path(prompt_file_path)
        output_dir = Path(output_root) / trend_id
        output_dir.mkdir(parents=True, exist_ok=True)

        analysis_path = output_dir / "analysis.json"
        metadata_path = output_dir / "analysis_metadata.json"

        if not self.api_key:
            return self._fail(trend_id, analysis_path, metadata_path, "api_key_missing")
        if not video_path.exists():
            return self._fail(trend_id, analysis_path, metadata_path, "video_missing")
        if not prompt_path.exists():
            return self._fail(trend_id, analysis_path, metadata_path, "prompt_missing")

        prompt_text = prompt_path.read_text(encoding="utf-8")
        upload_succeeded = False
        upload_duration_seconds: float | None = None
        model_duration_seconds: float | None = None
        try:
            client = self._get_client()
            upload_started = time.perf_counter()
            uploaded_file = client.files.upload(file=str(video_path))
            uploaded_file = self._wait_until_file_ready(client, uploaded_file)
            upload_duration_seconds = round(time.perf_counter() - upload_started, 4)
            upload_succeeded = True

            model_started = time.perf_counter()
            structured_data = self._request_structured_analysis(
                client=client,
                uploaded_file=uploaded_file,
                prompt_text=prompt_text,
            )
            model_duration_seconds = round(time.perf_counter() - model_started, 4)
            # trend_id is pipeline-owned; never trust model-provided identifier.
            structured_data["trend_id"] = trend_id
            analysis = VideoContentAnalysis.model_validate(structured_data)

            analysis_path.write_text(
                json.dumps(analysis.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            metadata = {
                "trend_id": trend_id,
                "model": self.model,
                "prompt_file": str(prompt_path),
                "source_video": str(video_path),
                "analysis_file": str(analysis_path),
                "analyzed_at": datetime.now(UTC).isoformat(),
                "provider": "google_gemini",
            }
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            return AnalysisRunResult(
                trend_id=trend_id,
                success=True,
                analysis_path=str(analysis_path),
                metadata_path=str(metadata_path),
                model=self.model,
                upload_succeeded=upload_succeeded,
                upload_duration_seconds=upload_duration_seconds,
                model_duration_seconds=model_duration_seconds,
                error=None,
                analysis=analysis,
            )
        except ValidationError as exc:
            return self._fail(
                trend_id,
                analysis_path,
                metadata_path,
                f"invalid_structured_response={exc}",
                upload_succeeded=upload_succeeded,
                upload_duration_seconds=upload_duration_seconds,
                model_duration_seconds=model_duration_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            if upload_succeeded:
                return self._fail(
                    trend_id,
                    analysis_path,
                    metadata_path,
                    f"model_failure={error}",
                    upload_succeeded=True,
                    upload_duration_seconds=upload_duration_seconds,
                    model_duration_seconds=model_duration_seconds,
                )
            return self._fail(
                trend_id,
                analysis_path,
                metadata_path,
                f"upload_failure={error}",
                upload_succeeded=False,
                upload_duration_seconds=upload_duration_seconds,
                model_duration_seconds=model_duration_seconds,
            )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        from google import genai

        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _wait_until_file_ready(self, client: Any, uploaded_file: Any) -> Any:
        state_name = self._extract_state_name(uploaded_file)
        while state_name == "PROCESSING":
            self._sleep_fn(2)
            uploaded_file = client.files.get(name=uploaded_file.name)
            state_name = self._extract_state_name(uploaded_file)
        if state_name == "FAILED":
            raise RuntimeError("file_processing_failed")
        return uploaded_file

    def _request_structured_analysis(
        self,
        client: Any,
        uploaded_file: Any,
        prompt_text: str,
    ) -> dict[str, Any]:
        from google.genai import types

        response = client.models.generate_content(
            model=self.model,
            contents=[prompt_text, uploaded_file],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VideoContentAnalysis,
            ),
        )

        if hasattr(response, "parsed") and response.parsed is not None:
            parsed = response.parsed
            if isinstance(parsed, VideoContentAnalysis):
                return parsed.model_dump()
            if isinstance(parsed, dict):
                return parsed
            raise RuntimeError("invalid_parsed_payload_type")

        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return json.loads(text)
        raise RuntimeError("empty_model_response")

    @staticmethod
    def _extract_state_name(uploaded_file: Any) -> str | None:
        state = getattr(uploaded_file, "state", None)
        if state is None:
            return None
        name = getattr(state, "name", None)
        if isinstance(name, str):
            return name
        if isinstance(state, str):
            return state
        return None

    def _fail(
        self,
        trend_id: str,
        analysis_path: Path,
        metadata_path: Path,
        error: str,
        upload_succeeded: bool = False,
        upload_duration_seconds: float | None = None,
        model_duration_seconds: float | None = None,
    ) -> AnalysisRunResult:
        return AnalysisRunResult(
            trend_id=trend_id,
            success=False,
            analysis_path=str(analysis_path),
            metadata_path=str(metadata_path),
            model=self.model,
            upload_succeeded=upload_succeeded,
            upload_duration_seconds=upload_duration_seconds,
            model_duration_seconds=model_duration_seconds,
            error=error,
            analysis=None,
        )
