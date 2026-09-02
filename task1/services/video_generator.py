from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


@dataclass
class VideoGenerationResult:
    trend_id: str
    success: bool
    model_used: str | None
    final_video_path: str
    metadata_path: str
    operation_name: str | None
    error: str | None
    elapsed_seconds: float


class VeoVideoGenerator:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        preferred_model: str = "veo-3.1-lite-generate-preview",
        fallback_model: str | None = None,
        poll_interval_seconds: float = 5.0,
        max_wait_seconds: int = 900,
        per_model_attempts: int = 2,
        retry_backoff_seconds: float = 20.0,
        client: Any | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.api_key = api_key or env_key
        self.preferred_model = preferred_model
        self.fallback_model = fallback_model
        self.poll_interval_seconds = poll_interval_seconds
        self.max_wait_seconds = max_wait_seconds
        self.per_model_attempts = max(1, int(per_model_attempts))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self._client = client
        self._sleep_fn = sleep_fn or time.sleep

    def generate(
        self,
        *,
        trend_id: str,
        prompt_text: str,
        output_dir: str | Path,
        duration_seconds: int = 12,
        aspect_ratio: str = "9:16",
    ) -> VideoGenerationResult:
        destination_dir = Path(output_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        final_video_path = destination_dir / "final.mp4"
        metadata_path = destination_dir / "generation_metadata.json"
        started = time.perf_counter()

        if not self.api_key:
            return self._fail(
                trend_id=trend_id,
                final_video_path=final_video_path,
                metadata_path=metadata_path,
                error="api_key_missing",
                elapsed_seconds=time.perf_counter() - started,
            )

        models_to_try = [self.preferred_model]
        if self.fallback_model and self.fallback_model != self.preferred_model:
            models_to_try.append(self.fallback_model)

        last_error: str | None = None
        last_result: VideoGenerationResult | None = None
        for model_name in models_to_try:
            for attempt in range(self.per_model_attempts):
                try:
                    result = self._generate_with_model(
                        trend_id=trend_id,
                        prompt_text=prompt_text,
                        output_dir=destination_dir,
                        model_name=model_name,
                        duration_seconds=duration_seconds,
                        aspect_ratio=aspect_ratio,
                        started_perf=started,
                    )
                    last_result = result
                    if result.success:
                        return result
                    last_error = result.error
                    if last_error and self._is_access_blocker(last_error):
                        return result
                    retryable = _is_retryable_generation_error(last_error)
                    if not retryable:
                        break
                    if attempt < self.per_model_attempts - 1:
                        self._sleep_fn(self.retry_backoff_seconds * (attempt + 1))
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    if self._is_access_blocker(last_error):
                        break
                    retryable = _is_retryable_generation_error(last_error)
                    if not retryable:
                        break
                    if attempt < self.per_model_attempts - 1:
                        self._sleep_fn(self.retry_backoff_seconds * (attempt + 1))

        if last_result is not None:
            return last_result
        return self._fail(
            trend_id=trend_id,
            final_video_path=final_video_path,
            metadata_path=metadata_path,
            error=last_error or "generation_failed",
            elapsed_seconds=time.perf_counter() - started,
        )

    def _generate_with_model(
        self,
        *,
        trend_id: str,
        prompt_text: str,
        output_dir: Path,
        model_name: str,
        duration_seconds: int,
        aspect_ratio: str,
        started_perf: float,
    ) -> VideoGenerationResult:
        from google.genai import types

        final_video_path = output_dir / "final.mp4"
        metadata_path = output_dir / "generation_metadata.json"
        client = self._get_client()

        operation = client.models.generate_videos(
            model=model_name,
            prompt=prompt_text,
            config=types.GenerateVideosConfig(
                number_of_videos=1,
                duration_seconds=duration_seconds,
                aspect_ratio=aspect_ratio,
            ),
        )
        operation = self._wait_until_done(client, operation)

        if bool(getattr(operation, "done", False)) and getattr(operation, "error", None):
            error_payload = getattr(operation.error, "model_dump", None)
            if callable(error_payload):
                error_text = json.dumps(operation.error.model_dump(), ensure_ascii=False)
            else:
                error_text = str(operation.error)
            return self._fail(
                trend_id=trend_id,
                final_video_path=final_video_path,
                metadata_path=metadata_path,
                error=f"operation_failed={error_text}",
                elapsed_seconds=time.perf_counter() - started_perf,
                model_used=model_name,
                operation_name=getattr(operation, "name", None),
            )

        generated_videos = (
            getattr(operation, "generated_videos", None)
            or getattr(getattr(operation, "metadata", None), "generated_videos", None)
            or getattr(getattr(operation, "response", None), "generated_videos", None)
            or getattr(getattr(operation, "result", None), "generated_videos", None)
            or []
        )
        if not generated_videos:
            debug_hint = _operation_debug_hint(operation)
            error_text = "operation_succeeded_but_no_generated_video"
            if debug_hint:
                error_text = f"{error_text}; {debug_hint}"
            return self._fail(
                trend_id=trend_id,
                final_video_path=final_video_path,
                metadata_path=metadata_path,
                error=error_text,
                elapsed_seconds=time.perf_counter() - started_perf,
                model_used=model_name,
                operation_name=getattr(operation, "name", None),
            )

        generated_video = generated_videos[0]
        video_handle = getattr(generated_video, "video", None)
        if video_handle is None:
            return self._fail(
                trend_id=trend_id,
                final_video_path=final_video_path,
                metadata_path=metadata_path,
                error="generated_video_missing_payload",
                elapsed_seconds=time.perf_counter() - started_perf,
                model_used=model_name,
                operation_name=getattr(operation, "name", None),
            )

        payload_bytes = client.files.download(file=video_handle)
        if not payload_bytes:
            return self._fail(
                trend_id=trend_id,
                final_video_path=final_video_path,
                metadata_path=metadata_path,
                error="download_empty_payload",
                elapsed_seconds=time.perf_counter() - started_perf,
                model_used=model_name,
                operation_name=getattr(operation, "name", None),
            )
        temp_path = output_dir / "final.tmp.mp4"
        temp_path.write_bytes(payload_bytes)
        temp_path.replace(final_video_path)

        metadata = {
            "trend_id": trend_id,
            "model": model_name,
            "operation_name": getattr(operation, "name", None),
            "generated_at": datetime.now(UTC).isoformat(),
            "provider": "google_gemini_veo",
            "duration_seconds_requested": duration_seconds,
            "aspect_ratio_requested": aspect_ratio,
            "final_video_path": str(final_video_path),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        return VideoGenerationResult(
            trend_id=trend_id,
            success=True,
            model_used=model_name,
            final_video_path=str(final_video_path),
            metadata_path=str(metadata_path),
            operation_name=getattr(operation, "name", None),
            error=None,
            elapsed_seconds=round(time.perf_counter() - started_perf, 4),
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        from google import genai

        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _wait_until_done(self, client: Any, operation: Any) -> Any:
        started = time.perf_counter()
        while not bool(getattr(operation, "done", False)):
            if time.perf_counter() - started > self.max_wait_seconds:
                raise TimeoutError(f"veo_operation_timeout_{self.max_wait_seconds}s")
            self._sleep_fn(self.poll_interval_seconds)
            operation = client.operations.get(operation)
        return operation

    @staticmethod
    def _is_access_blocker(error_text: str) -> bool:
        lowered = error_text.lower()
        blocker_markers = [
            "permission_denied",
            "insufficient permissions",
            "billing",
            "access",
            "not enabled",
            "403",
            "api key not valid",
            "quota",
        ]
        return any(marker in lowered for marker in blocker_markers)

    def _fail(
        self,
        *,
        trend_id: str,
        final_video_path: Path,
        metadata_path: Path,
        error: str,
        elapsed_seconds: float,
        model_used: str | None = None,
        operation_name: str | None = None,
    ) -> VideoGenerationResult:
        return VideoGenerationResult(
            trend_id=trend_id,
            success=False,
            model_used=model_used,
            final_video_path=str(final_video_path),
            metadata_path=str(metadata_path),
            operation_name=operation_name,
            error=error,
            elapsed_seconds=round(elapsed_seconds, 4),
        )



def _is_retryable_generation_error(error_text: str | None) -> bool:
    if not error_text:
        return False
    lowered = error_text.lower()
    return any(
        marker in lowered
        for marker in [
            "operation_succeeded_but_no_generated_video",
            "download_empty_payload",
            "generated_video_missing_payload",
            "timeout",
            "temporar",
            "internal",
            "unavailable",
        ]
    )


def _operation_debug_hint(operation: Any) -> str | None:
    candidates = [
        getattr(operation, "response", None),
        getattr(operation, "result", None),
        getattr(operation, "metadata", None),
    ]
    texts: list[str] = []
    for item in candidates:
        if item is None:
            continue
        dump = getattr(item, "model_dump", None)
        if callable(dump):
            try:
                payload = dump()
                texts.append(json.dumps(payload, ensure_ascii=False))
                continue
            except Exception:  # noqa: BLE001
                pass
        texts.append(str(item))
    merged = " | ".join(texts).lower()
    if not merged:
        return None
    if "safety" in merged or "policy" in merged or "blocked" in merged:
        return "provider_filtered_or_policy_blocked"
    if "quota" in merged or "billing" in merged or "insufficient" in merged:
        return "quota_or_billing_issue"
    return None
