"""Map an arbitrary theme onto playable visual roles. No HTML writes."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

TASK2_DIR = Path(__file__).resolve().parents[1]
ROLES_PATH = TASK2_DIR / "inventory" / "roles.json"
PROMPT_PATH = TASK2_DIR / "prompts" / "theme_plan.txt"
THEMES_ROOT = TASK2_DIR / "artifacts" / "themes"

DEFAULT_MODEL = "gemini-3.6-flash"
FALLBACK_MODEL = "gemini-2.5-flash"
PLAN_ATTEMPTS = 4
_SLUG_RE = re.compile(r"[^a-z0-9]+")


class BannerPlan(BaseModel):
    text: str = Field(min_length=1, max_length=48)
    text_color: str
    background_color: str

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("banner text is empty")
        return cleaned

    @field_validator("text_color", "background_color")
    @classmethod
    def hex_color(cls, value: str) -> str:
        return normalize_hex(value)


class OverridePlan(BaseModel):
    soil_color: str
    separator_wall_color: str
    rock_colors: list[str] = Field(min_length=4, max_length=4)

    @field_validator("soil_color", "separator_wall_color")
    @classmethod
    def hex_color(cls, value: str) -> str:
        return normalize_hex(value)

    @field_validator("rock_colors")
    @classmethod
    def hex_list(cls, values: list[str]) -> list[str]:
        return [normalize_hex(value) for value in values]


class RoleAssetPlan(BaseModel):
    asset_id: str
    concept: str = Field(min_length=1)
    image_direction: str = Field(min_length=1)


class RolePlan(BaseModel):
    role_id: str
    concept: str = Field(min_length=1)
    image_direction: str = Field(min_length=1)
    assets: list[RoleAssetPlan] = Field(default_factory=list)


class ThemePlan(BaseModel):
    theme: str
    visual_thesis: str = Field(min_length=1)
    art_style: str = Field(min_length=1)
    palette: list[str] = Field(min_length=3, max_length=8)
    banner: BannerPlan
    overrides: OverridePlan
    roles: list[RolePlan] = Field(min_length=1)

    @field_validator("palette")
    @classmethod
    def hex_palette(cls, values: list[str]) -> list[str]:
        return [normalize_hex(value) for value in values]


def normalize_hex(value: str) -> str:
    match = re.search(r"[0-9A-Fa-f]{6}", value.strip())
    if not match:
        raise ValueError(f"expected #RRGGBB color, got {value!r}")
    return f"#{match.group(0).upper()}"


def theme_slug(theme: str) -> str:
    slug = _SLUG_RE.sub("-", theme.strip().lower()).strip("-")
    if not slug:
        raise ValueError("theme is empty")
    return slug


def load_roles() -> dict[str, Any]:
    return json.loads(ROLES_PATH.read_text(encoding="utf-8"))


def plannable_roles(roles: dict[str, Any]) -> list[dict[str, Any]]:
    selected = []
    for role in roles["asset_roles"]:
        if not role.get("replaceable"):
            continue
        if not role.get("generate_image", True):
            continue
        if role.get("risk") == "high":
            continue
        selected.append(role)
    return selected


def build_planner_catalog(roles: dict[str, Any]) -> dict[str, Any]:
    compact_roles = []
    for role in plannable_roles(roles):
        assets = role["assets"]
        entry: dict[str, Any] = {
            "id": role["id"],
            "label": role["label"],
            "risk": role["risk"],
            "preserve_aspect": role["preserve_aspect"],
            "preserve_silhouette": role["preserve_silhouette"],
            "transparent_background": bool(role.get("transparent_background", True)),
            "tileable": bool(role.get("tileable", False)),
            "notes": role.get("notes"),
        }
        if len(assets) <= 8:
            entry["assets"] = [
                {
                    "id": asset["id"],
                    "label": asset.get("label"),
                    "width": asset["width"],
                    "height": asset["height"],
                    "kind": asset["kind"],
                }
                for asset in assets
            ]
        else:
            first = assets[0]
            entry["asset_count"] = len(assets)
            entry["frame_size"] = {
                "width": first["width"],
                "height": first["height"],
                "kind": first["kind"],
            }
            entry["homogeneous"] = True
        compact_roles.append(entry)

    return {
        "roles": compact_roles,
        "safe_overrides": [
            {
                "field": item["field"],
                "type": item["type"],
                "notes": item.get("notes"),
            }
            for item in roles["safe_overrides"]
            if item["field"] != "fontSize"
        ],
        "do_not_touch": [
            {"id": item["id"], "reason": item["reason"]}
            for item in roles.get("do_not_touch", [])
        ],
    }


def required_asset_ids(role: dict[str, Any]) -> list[str] | None:
    assets = role.get("assets")
    if not assets:
        return None
    return [str(asset["id"]) for asset in assets]


def validate_against_catalog(plan: ThemePlan, catalog: dict[str, Any], theme: str) -> ThemePlan:
    allowed = {role["id"]: role for role in catalog["roles"]}
    planned = {role.role_id: role for role in plan.roles}
    missing = sorted(set(allowed) - set(planned))
    extra = sorted(set(planned) - set(allowed))
    if missing:
        raise ValueError(f"theme plan missing roles: {missing}")
    if extra:
        raise ValueError(f"theme plan included forbidden roles: {extra}")

    checked_roles = []
    for role_id, catalog_role in allowed.items():
        role_plan = planned[role_id]
        expected_ids = required_asset_ids(catalog_role)
        if expected_ids is None:
            role_plan = role_plan.model_copy(update={"assets": []})
        else:
            got = {asset.asset_id: asset for asset in role_plan.assets}
            filled = []
            for asset_id in expected_ids:
                existing = got.get(asset_id)
                if existing is not None:
                    filled.append(existing)
                else:
                    filled.append(
                        RoleAssetPlan(
                            asset_id=asset_id,
                            concept=role_plan.concept,
                            image_direction=role_plan.image_direction,
                        )
                    )
            role_plan = role_plan.model_copy(update={"assets": filled})
        checked_roles.append(role_plan)

    return plan.model_copy(update={"theme": theme, "roles": checked_roles})


def resolve_api_key(cli_key: str | None) -> str:
    key = cli_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "Missing Gemini API key. Provide --api-key / --gemini-api-key "
            "or set GEMINI_API_KEY."
        )
    return key


def _is_retryable(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "429" in message
        or "503" in message
        or "unavailable" in message
        or "high demand" in message
        or "timeout" in message
        or "timed out" in message
        or "resource exhausted" in message
    )


def request_theme_plan(
    *,
    theme: str,
    catalog: dict[str, Any],
    prompt_text: str,
    model: str,
    api_key: str,
    client: Any | None = None,
) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    active_client = client or genai.Client(api_key=api_key)
    payload = {"theme": theme, "catalog": catalog}
    models = [model]
    if FALLBACK_MODEL not in models:
        models.append(FALLBACK_MODEL)
    last_error: Exception | None = None
    for active_model in models:
        for attempt in range(PLAN_ATTEMPTS):
            try:
                response = active_client.models.generate_content(
                    model=active_model,
                    contents=[prompt_text, json.dumps(payload, ensure_ascii=False)],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ThemePlan,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        ),
                    ),
                )
                if hasattr(response, "parsed") and response.parsed is not None:
                    parsed = response.parsed
                    if isinstance(parsed, ThemePlan):
                        return parsed.model_dump()
                    if isinstance(parsed, dict):
                        return parsed
                    raise RuntimeError("invalid_parsed_payload_type")
                text = getattr(response, "text", None)
                if isinstance(text, str) and text.strip():
                    return json.loads(text)
                raise RuntimeError("empty_model_response")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if _is_retryable(exc) and attempt < PLAN_ATTEMPTS - 1:
                    wait = 3 * (attempt + 1)
                    print(
                        f"[retry-plan] model={active_model} attempt={attempt + 1} "
                        f"wait={wait}s {exc}",
                        flush=True,
                    )
                    time.sleep(wait)
                    continue
                if _is_retryable(exc):
                    break
                raise
    raise RuntimeError(f"theme_plan_failed={last_error}") from last_error


def plan_theme(
    theme: str,
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
) -> Path:
    roles = load_roles()
    catalog = build_planner_catalog(roles)
    prompt_text = PROMPT_PATH.read_text(encoding="utf-8")
    raw = request_theme_plan(
        theme=theme,
        catalog=catalog,
        prompt_text=prompt_text,
        model=model,
        api_key=resolve_api_key(api_key),
        client=client,
    )
    try:
        plan = validate_against_catalog(ThemePlan.model_validate(raw), catalog, theme)
    except ValidationError as exc:
        raise RuntimeError(f"invalid_structured_response={exc}") from exc

    slug = theme_slug(theme)
    output_dir = THEMES_ROOT / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "theme_plan.json"
    metadata_path = output_dir / "theme_plan_metadata.json"
    plan_path.write_text(
        json.dumps(plan.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "theme": theme,
                "slug": slug,
                "model": model,
                "prompt_path": str(PROMPT_PATH.relative_to(TASK2_DIR.parent)),
                "roles_path": str(ROLES_PATH.relative_to(TASK2_DIR.parent)),
                "planned_roles": [role.role_id for role in plan.roles],
                "excluded_high_risk_roles": [
                    role["id"]
                    for role in roles["asset_roles"]
                    if role.get("risk") == "high"
                ],
                "created_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return plan_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task 2 theme planner")
    parser.add_argument("--theme", required=True, help="Arbitrary theme string")
    parser.add_argument("--api-key", dest="api_key", default=None)
    parser.add_argument("--gemini-api-key", dest="gemini_api_key", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    api_key = args.api_key or args.gemini_api_key
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    try:
        plan_path = plan_theme(args.theme, api_key=api_key, model=args.model)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"theme: {args.theme}")
    print(f"model: {args.model}")
    print(f"plan: {plan_path}")


if __name__ == "__main__":
    main()
