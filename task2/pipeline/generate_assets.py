"""Generate themed replacement sprites from a theme plan. No HTML writes."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import time
from collections import deque
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from task2.pipeline.theme_plan import (  # noqa: E402
    TASK2_DIR,
    THEMES_ROOT,
    load_roles,
    resolve_api_key,
    theme_slug,
)

PROMPT_PATH = TASK2_DIR / "prompts" / "asset_image.txt"
DEFAULT_MODEL = "gemini-3.1-flash-image"
FALLBACK_MODEL = "gemini-2.5-flash-image"
ASPECT_CHOICES = {
    "1:1": 1.0,
    "2:3": 2 / 3,
    "3:2": 1.5,
    "3:4": 0.75,
    "4:3": 4 / 3,
    "9:16": 9 / 16,
    "16:9": 16 / 9,
    "21:9": 21 / 9,
}
CORNER = 8
MIN_TRANSPARENT_RATIO = 0.08
VALIDATION_RETRIES = 1
SPRITE_BACKGROUND_RULE = (
    "Isolated game sprite on a TRUE transparent alpha background. "
    "Do not paint any backdrop. No checkerboard, no white background, "
    "no gray background, no black background, no floor, no scene."
)
TEXTURE_BACKGROUND_RULE = (
    "Full-bleed seamless game environment texture filling the entire frame. "
    "No empty margins and no transparency."
)
SPRITE_EXTRA_RULES = (
    "Never render a Photoshop-style checkerboard to imply transparency. "
    "Never put the subject on a square card, plate, or colored rectangle. "
    "No frame, no border, no drop shadow outside the sprite."
)
TEXTURE_EXTRA_RULES = (
    "Seamless and tileable on all four edges. Low visual noise; no high-frequency "
    "speckle or busy photo grain. Dark but readable. Game environment material only. "
    "No objects, no characters, no icons, no text, no frames."
)


def blocked_asset_ids(roles: dict[str, Any]) -> set[str]:
    blocked: set[str] = set()
    for item in roles.get("do_not_touch", []):
        blocked.update(str(asset_id) for asset_id in item.get("asset_ids", []))
    return blocked


def wants_transparency(role: dict[str, Any]) -> bool:
    return bool(role.get("transparent_background", True))


def is_tileable(role: dict[str, Any]) -> bool:
    return bool(role.get("tileable", False))


def closest_aspect_ratio(width: int, height: int) -> str:
    target = width / max(height, 1)
    return min(ASPECT_CHOICES, key=lambda key: abs(ASPECT_CHOICES[key] - target))


def extreme_aspect(width: int, height: int) -> bool:
    return max(width, height) / max(min(width, height), 1) >= 4


def fill_prompt(template: str, values: dict[str, str]) -> str:
    filled = template
    for key, value in values.items():
        filled = filled.replace(f"{{{{{key}}}}}", value)
    return filled


def build_generation_prompt(
    *,
    template: str,
    theme_plan: dict[str, Any],
    role: dict[str, Any],
    concept: str,
    image_direction: str,
    width: int,
    height: int,
) -> str:
    role_id = role["id"]
    if is_tileable(role):
        background_rule = TEXTURE_BACKGROUND_RULE
        extra_rules = TEXTURE_EXTRA_RULES
        silhouette_rule = "Fill the frame with a simple repeating material. No isolated object."
        aspect_rule = "Use the requested canvas. The texture must fill the frame edge to edge."
    elif wants_transparency(role):
        background_rule = SPRITE_BACKGROUND_RULE
        extra_rules = SPRITE_EXTRA_RULES
        if role.get("preserve_silhouette"):
            silhouette_rule = (
                "Keep a silhouette close to a typical game sprite for this role: "
                "one centered object, similar mass and outline, no extreme elongation."
            )
        else:
            silhouette_rule = "Keep the subject centered with transparent margins."
        if role.get("preserve_aspect"):
            aspect_rule = (
                f"Keep the subject fitted to a {width}x{height} sprite. "
                "Do not change the role's proportions. Leave transparent padding if needed."
            )
        else:
            aspect_rule = "Use the requested canvas. Subject may fill the frame."
    else:
        background_rule = "Full-bleed image filling the frame."
        extra_rules = "No frame, no border, no watermark."
        silhouette_rule = "Fill the frame."
        aspect_rule = f"Keep the {width}x{height} canvas."
    if role_id == "game_logo":
        text_rule = 'The only allowed text is exactly "ROYAL KINGDOM". No other letters or numbers.'
    else:
        text_rule = "No text, letters, numbers, captions, or UI labels of any kind."
    return fill_prompt(
        template,
        {
            "THEME": theme_plan["theme"],
            "ART_STYLE": theme_plan.get("art_style", ""),
            "VISUAL_THESIS": theme_plan.get("visual_thesis", ""),
            "ROLE_LABEL": role["label"],
            "ROLE_ID": role_id,
            "CONCEPT": concept,
            "IMAGE_DIRECTION": image_direction,
            "WIDTH": str(width),
            "HEIGHT": str(height),
            "ASPECT_RATIO": closest_aspect_ratio(width, height),
            "BACKGROUND_RULE": background_rule,
            "SILHOUETTE_RULE": silhouette_rule,
            "ASPECT_RULE": aspect_rule,
            "TEXT_RULE": text_rule,
            "EXTRA_RULES": extra_rules,
        },
    )


def planned_roles_by_id(theme_plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["role_id"]: item for item in theme_plan.get("roles", [])}


def build_jobs(roles: dict[str, Any], theme_plan: dict[str, Any]) -> list[dict[str, Any]]:
    blocked = blocked_asset_ids(roles)
    planned = planned_roles_by_id(theme_plan)
    jobs: list[dict[str, Any]] = []
    for role in roles["asset_roles"]:
        if not role.get("replaceable"):
            continue
        if not role.get("generate_image", True):
            continue
        if role.get("risk") == "high":
            continue
        plan = planned.get(role["id"])
        if plan is None:
            continue
        per_asset = {item["asset_id"]: item for item in plan.get("assets", [])}
        homogeneous = len(role["assets"]) > 8
        assets = []
        for asset in role["assets"]:
            asset_id = str(asset["id"])
            if asset_id in blocked:
                continue
            direction_source = per_asset.get(asset_id, plan)
            assets.append(
                {
                    "asset_id": asset_id,
                    "width": asset["width"],
                    "height": asset["height"],
                    "kind": asset["kind"],
                    "label": asset.get("label"),
                    "concept": direction_source.get("concept") or plan["concept"],
                    "image_direction": direction_source.get("image_direction") or plan["image_direction"],
                }
            )
        if not assets:
            continue
        jobs.append(
            {
                "role": role,
                "plan": plan,
                "homogeneous": homogeneous,
                "assets": assets,
            }
        )
    return jobs


def _response_text(response: Any) -> str:
    texts: list[str] = []
    direct = getattr(response, "text", None)
    if isinstance(direct, str) and direct.strip():
        texts.append(direct.strip())
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return " | ".join(texts)


def describe_image_failure(response: Any) -> str:
    reasons: list[str] = []
    prompt_feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(prompt_feedback, "block_reason", None) if prompt_feedback else None
    if block_reason:
        reasons.append(f"block_reason={block_reason}")
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        reasons.append("no_candidates")
    for candidate in candidates:
        finish = getattr(candidate, "finish_reason", None)
        if finish:
            reasons.append(f"finish_reason={finish}")
    text = _response_text(response)
    if text:
        reasons.append(text[:400])
    if not reasons:
        reasons.append("model returned no image bytes")
    return "; ".join(reasons)


def extract_image_bytes(response: Any) -> bytes:
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline is None:
                continue
            data = getattr(inline, "data", None)
            if isinstance(data, (bytes, bytearray)) and data:
                return bytes(data)
            if isinstance(data, str) and data:
                return base64.b64decode(data)
    raise RuntimeError(f"no_image_in_response ({describe_image_failure(response)})")


def generate_image_bytes(
    *,
    client: Any,
    prompt: str,
    model: str,
    width: int,
    height: int,
) -> bytes:
    from google.genai import types

    image_config = types.ImageConfig(
        aspect_ratio=closest_aspect_ratio(width, height),
        image_size="1K",
    )
    response = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=image_config,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    return extract_image_bytes(response)


def generate_with_retry(
    *,
    client: Any,
    prompt: str,
    models: list[str],
    width: int,
    height: int,
    attempts: int = 2,
) -> tuple[bytes, str]:
    last_error: Exception | None = None
    for model in models:
        for attempt in range(attempts):
            try:
                return (
                    generate_image_bytes(
                        client=client,
                        prompt=prompt,
                        model=model,
                        width=width,
                        height=height,
                    ),
                    model,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                message = str(exc).lower()
                retryable = (
                    "429" in message
                    or "503" in message
                    or "unavailable" in message
                    or "timeout" in message
                    or "timed out" in message
                )
                if retryable:
                    print(
                        f"[retry-api] model={model} attempt={attempt + 1} {exc}",
                        flush=True,
                    )
                    time.sleep(2 * (attempt + 1))
                    continue
                break
    raise RuntimeError(f"image_generation_failed={last_error}") from last_error


def relative_to_repo(path: Path) -> str:
    return str(path.relative_to(TASK2_DIR.parent))


def load_rgba(blob: bytes) -> Image.Image:
    with Image.open(BytesIO(blob)) as image:
        return image.convert("RGBA")


def color_distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1]) + abs(left[2] - right[2])


def is_grayish(color: tuple[int, int, int]) -> bool:
    red, green, blue = color
    return max(red, green, blue) - min(red, green, blue) <= 28


def is_near_white(color: tuple[int, int, int]) -> bool:
    return min(color) >= 220 and is_grayish(color)


def is_near_black(color: tuple[int, int, int]) -> bool:
    return max(color) <= 28 and is_grayish(color)


def is_near_gray(color: tuple[int, int, int]) -> bool:
    red, green, blue = color
    mean = (red + green + blue) / 3
    return is_grayish(color) and 55 <= mean <= 240


def is_matte_color(color: tuple[int, int, int]) -> bool:
    return is_near_white(color) or is_near_black(color) or is_near_gray(color)


def corner_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    size = max(4, min(CORNER, width // 4, height // 4))
    return [
        (0, 0, size, size),
        (width - size, 0, width, size),
        (0, height - size, size, height),
        (width - size, height - size, width, height),
    ]


def patch_mean_rgb(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    crop = image.crop(box)
    pixels = list(crop.getdata())
    count = max(len(pixels), 1)
    red = sum(pixel[0] for pixel in pixels) // count
    green = sum(pixel[1] for pixel in pixels) // count
    blue = sum(pixel[2] for pixel in pixels) // count
    return red, green, blue


def patch_mean_alpha(image: Image.Image, box: tuple[int, int, int, int]) -> float:
    crop = image.crop(box)
    pixels = list(crop.getdata())
    return sum(pixel[3] for pixel in pixels) / max(len(pixels), 1)


def patch_is_checkerboard(image: Image.Image, box: tuple[int, int, int, int]) -> bool:
    crop = image.crop(box).convert("RGB")
    width, height = crop.size
    cell = 8
    if width < cell * 2 or height < cell * 2:
        return False
    pixels = crop.load()
    quantized: dict[tuple[int, int, int], int] = {}
    for y in range(cell // 2, height, cell):
        for x in range(cell // 2, width, cell):
            red, green, blue = pixels[x, y]
            key = (red // 32, green // 32, blue // 32)
            quantized[key] = quantized.get(key, 0) + 1
    if len(quantized) != 2:
        return False
    (first, first_count), (second, second_count) = list(quantized.items())
    total = first_count + second_count
    if min(first_count, second_count) / max(total, 1) < 0.25:
        return False
    color_a = (first[0] * 32 + 16, first[1] * 32 + 16, first[2] * 32 + 16)
    color_b = (second[0] * 32 + 16, second[1] * 32 + 16, second[2] * 32 + 16)
    return is_matte_color(color_a) and is_matte_color(color_b) and color_distance(color_a, color_b) >= 24


def has_true_alpha(image: Image.Image) -> bool:
    extrema = image.getextrema()
    if len(extrema) < 4:
        return False
    return extrema[3][0] < 250


def transparent_pixel_ratio(image: Image.Image) -> float:
    histogram = image.getchannel("A").histogram()
    return sum(histogram[:32]) / max(sum(histogram), 1)


def validate_transparency(image: Image.Image, width: int, height: int) -> list[str]:
    errors: list[str] = []
    if not has_true_alpha(image):
        errors.append("missing_alpha_channel")
    boxes = corner_boxes(*image.size)
    opaque_corners = 0
    matte_corners = 0
    checker_corners = 0
    for box in boxes:
        if patch_is_checkerboard(image, box):
            checker_corners += 1
        mean_alpha = patch_mean_alpha(image, box)
        mean_rgb = patch_mean_rgb(image, box)
        if mean_alpha >= 32:
            opaque_corners += 1
            if is_matte_color(mean_rgb):
                matte_corners += 1
    if checker_corners:
        errors.append("checkerboard_background")
    if opaque_corners == 4 and not extreme_aspect(width, height):
        errors.append("opaque_corners")
    if matte_corners >= 3:
        errors.append("rectangular_matte_background")
    if not extreme_aspect(width, height) and transparent_pixel_ratio(image) < MIN_TRANSPARENT_RATIO:
        errors.append("full_rectangular_background")
    return errors


def iter_border(width: int, height: int):
    for x in range(width):
        yield x, 0
        yield x, height - 1
    for y in range(1, height - 1):
        yield 0, y
        yield width - 1, y


def punch_edge_background(image: Image.Image) -> Image.Image:
    pixels = image.load()
    width, height = image.size
    queue: deque[tuple[int, int]] = deque()
    visited = bytearray(width * height)
    for x, y in iter_border(width, height):
        red, green, blue, alpha = pixels[x, y]
        if alpha < 16:
            continue
        if is_matte_color((red, green, blue)):
            queue.append((x, y))
    if not queue:
        return image
    while queue:
        x, y = queue.popleft()
        index = y * width + x
        if visited[index]:
            continue
        visited[index] = 1
        red, green, blue, alpha = pixels[x, y]
        if alpha < 16 or not is_matte_color((red, green, blue)):
            continue
        pixels[x, y] = (0, 0, 0, 0)
        if x > 0:
            queue.append((x - 1, y))
        if x + 1 < width:
            queue.append((x + 1, y))
        if y > 0:
            queue.append((x, y - 1))
        if y + 1 < height:
            queue.append((x, y + 1))
    return image


def resize_to_source(image: Image.Image, width: int, height: int) -> Image.Image:
    if image.size == (width, height):
        return image
    return image.resize((width, height), Image.Resampling.LANCZOS)


def encode_png(image: Image.Image, *, transparent: bool) -> bytes:
    output = image.convert("RGBA" if transparent else "RGB")
    buffer = BytesIO()
    output.save(buffer, format="PNG", optimize=True, compress_level=9)
    return buffer.getvalue()


def fit_image_bytes(blob: bytes, width: int, height: int, *, transparent: bool) -> bytes:
    image = load_rgba(blob)
    if transparent:
        image = punch_edge_background(image)
    image = resize_to_source(image, width, height)
    if transparent:
        image = punch_edge_background(image)
    return encode_png(image, transparent=transparent)


def finalize_generated_image(
    blob: bytes,
    *,
    role: dict[str, Any],
    width: int,
    height: int,
) -> tuple[bytes, list[str], bool]:
    image = load_rgba(blob)
    recovered = False
    errors: list[str] = []
    if wants_transparency(role):
        before = validate_transparency(image, width, height)
        punched = punch_edge_background(image.copy())
        after = validate_transparency(punched, width, height)
        if after != before or transparent_pixel_ratio(punched) > transparent_pixel_ratio(image):
            recovered = True
        image = punched
        errors = after
    image = resize_to_source(image, width, height)
    if wants_transparency(role):
        image = punch_edge_background(image)
        errors = validate_transparency(image, width, height)
    return encode_png(image, transparent=wants_transparency(role)), errors, recovered


def cleanup_unlisted_assets(output_dir: Path, keep_ids: set[str]) -> None:
    if not output_dir.exists():
        return
    for path in output_dir.iterdir():
        if path.suffix.lower() != ".png":
            continue
        if path.stem not in keep_ids:
            path.unlink()


def generate_theme_assets(
    theme: str,
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    force: bool = False,
    client: Any | None = None,
) -> Path:
    from google import genai

    slug = theme_slug(theme)
    theme_dir = THEMES_ROOT / slug
    plan_path = theme_dir / "theme_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(
            f"theme plan not found: {plan_path}. Run theme_plan.py first."
        )
    theme_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    roles = load_roles()
    jobs = build_jobs(roles, theme_plan)
    if not jobs:
        raise RuntimeError("no_generatable_assets")

    output_dir = theme_dir / "generated_assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    template = PROMPT_PATH.read_text(encoding="utf-8")
    active_client = client or genai.Client(
        api_key=resolve_api_key(api_key),
        http_options={"timeout": 120_000},
    )
    models = [model]
    if FALLBACK_MODEL not in models:
        models.append(FALLBACK_MODEL)

    manifest_assets: list[dict[str, Any]] = []
    unique_count = 0
    copied_count = 0
    skipped_count = 0
    transparency_passed = 0
    transparency_failed = 0
    validation_retries = 0
    alpha_recovered = 0

    for job in jobs:
        role = job["role"]
        assets = job["assets"]
        groups: list[list[dict[str, Any]]]
        if job["homogeneous"]:
            groups = [assets]
        else:
            groups = [[asset] for asset in assets]

        for group in groups:
            primary = group[0]
            prompt = build_generation_prompt(
                template=template,
                theme_plan=theme_plan,
                role=role,
                concept=primary["concept"],
                image_direction=primary["image_direction"],
                width=primary["width"],
                height=primary["height"],
            )
            primary_path = output_dir / f"{primary['asset_id']}.png"
            generated_now = False
            used_model = model
            recovered = False
            retries_used = 0
            errors: list[str] = []
            if primary_path.exists() and not force:
                skipped_count += 1
            else:
                png_bytes: bytes | None = None
                for attempt in range(VALIDATION_RETRIES + 1):
                    image_bytes, used_model = generate_with_retry(
                        client=active_client,
                        prompt=prompt,
                        models=models,
                        width=primary["width"],
                        height=primary["height"],
                    )
                    png_bytes, errors, recovered = finalize_generated_image(
                        image_bytes,
                        role=role,
                        width=primary["width"],
                        height=primary["height"],
                    )
                    if not errors:
                        break
                    retries_used = attempt + 1
                    if attempt < VALIDATION_RETRIES:
                        validation_retries += 1
                        print(
                            f"[retry] {role['id']} {primary['asset_id']} {errors}",
                            flush=True,
                        )
                if errors or png_bytes is None:
                    transparency_failed += 1
                    raise RuntimeError(
                        f"asset {primary['asset_id']} ({role['id']}) failed validation: {errors}"
                    )
                primary_path.write_bytes(png_bytes)
                unique_count += 1
                generated_now = True
                if wants_transparency(role):
                    transparency_passed += 1
                    if recovered:
                        alpha_recovered += 1
                print(
                    f"[gen] {role['id']} {primary['asset_id']} "
                    f"{primary['width']}x{primary['height']} "
                    f"bytes={len(png_bytes)} model={used_model} "
                    f"retries={retries_used} recovered={recovered}",
                    flush=True,
                )

            for index, asset in enumerate(group):
                dest = output_dir / f"{asset['asset_id']}.png"
                if index > 0:
                    if dest != primary_path and (not dest.exists() or force or generated_now):
                        shutil.copyfile(primary_path, dest)
                        copied_count += 1
                manifest_assets.append(
                    {
                        "theme": theme_plan["theme"],
                        "role_id": role["id"],
                        "asset_id": asset["asset_id"],
                        "source_dimensions": {
                            "width": asset["width"],
                            "height": asset["height"],
                        },
                        "output_path": relative_to_repo(dest),
                        "transparent_background": wants_transparency(role),
                        "preserve_aspect": role["preserve_aspect"],
                        "preserve_silhouette": role["preserve_silhouette"],
                        "image_direction": asset["image_direction"],
                        "risk": role["risk"],
                        "shared_generation": job["homogeneous"],
                        "model": used_model,
                        "generation_prompt": prompt,
                        "validation_retries": retries_used,
                        "alpha_recovered": recovered,
                    }
                )

    keep_ids = {item["asset_id"] for item in manifest_assets}
    cleanup_unlisted_assets(output_dir, keep_ids)
    manifest = {
        "theme": theme_plan["theme"],
        "slug": slug,
        "model": model,
        "unique_generations": unique_count,
        "copied": copied_count,
        "skipped_existing": skipped_count,
        "asset_count": len(manifest_assets),
        "transparency_passed": transparency_passed,
        "transparency_failed": transparency_failed,
        "validation_retries": validation_retries,
        "alpha_recovered": alpha_recovered,
        "created_at": datetime.now(UTC).isoformat(),
        "assets": manifest_assets,
    }
    manifest_path = theme_dir / "asset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task 2 themed asset generator")
    parser.add_argument("--theme", required=True)
    parser.add_argument("--api-key", dest="api_key", default=None)
    parser.add_argument("--gemini-api-key", dest="gemini_api_key", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--force", action="store_true", help="Regenerate even if files exist")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    api_key = args.api_key or args.gemini_api_key
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    try:
        manifest_path = generate_theme_assets(
            args.theme,
            api_key=api_key,
            model=args.model,
            force=args.force,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roles = sorted({item["role_id"] for item in manifest["assets"]})
    print(f"theme: {manifest['theme']}")
    print(f"model: {manifest['model']}")
    print(f"unique_generations: {manifest['unique_generations']}")
    print(f"copied: {manifest['copied']}")
    print(f"skipped_existing: {manifest['skipped_existing']}")
    print(f"assets: {manifest['asset_count']}")
    print(f"roles: {', '.join(roles)}")
    print(f"transparency_passed: {manifest.get('transparency_passed', 0)}")
    print(f"transparency_failed: {manifest.get('transparency_failed', 0)}")
    print(f"validation_retries: {manifest.get('validation_retries', 0)}")
    print(f"alpha_recovered: {manifest.get('alpha_recovered', 0)}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
