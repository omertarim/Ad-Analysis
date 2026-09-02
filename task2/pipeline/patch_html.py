"""Patch generated sprites and safe visual overrides into a new playable HTML."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from html import unescape
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from task2.pipeline.generate_assets import fit_image_bytes  # noqa: E402
from task2.pipeline.inventory import (  # noqa: E402
    IMG_RE,
    SOURCE_HTML,
    decode_base122,
    sniff_kind,
)
from task2.pipeline.theme_plan import TASK2_DIR, THEMES_ROOT, load_roles, theme_slug  # noqa: E402

OUTPUT_DIR = TASK2_DIR / "outputs"
OVERRIDES_MARK = "window.playgroundOverrides = "
_SPECIALS = (0, 10, 13, 34, 38, 92)
_SPECIAL_INDEX = {value: index for index, value in enumerate(_SPECIALS)}
_MIME = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}
IMG_HEAD_RE = re.compile(
    r'(<img id="([^"]+)" data-src122=")(.*?)(" data-mime=")([^"]+)"',
    re.DOTALL,
)


def encode_base122(data: bytes) -> str:
    symbols: list[int] = []
    acc = 0
    nbits = 0
    for byte in data:
        acc = (acc << 8) | byte
        nbits += 8
        while nbits >= 7:
            nbits -= 7
            symbols.append((acc >> nbits) & 127)
            acc &= (1 << nbits) - 1
    if nbits:
        symbols.append((acc << (7 - nbits)) & 127)

    chars: list[str] = []
    index = 0
    while index < len(symbols):
        symbol = symbols[index]
        special_idx = _SPECIAL_INDEX.get(symbol)
        if special_idx is None:
            chars.append(chr(symbol))
            index += 1
            continue
        if index + 1 < len(symbols):
            nxt = symbols[index + 1]
            chars.append(chr((special_idx << 8) | 0x80 | nxt))
            index += 2
        else:
            chars.append(chr((7 << 8) | 0x80 | symbol))
            index += 1
    return "".join(chars)


def escape_attr(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def extract_js_object(source: str) -> str:
    if not source or source[0] != "{":
        raise RuntimeError("playgroundOverrides does not start with '{'")
    depth = 0
    in_str = False
    escape = False
    quote = ""
    for index, char in enumerate(source):
        if in_str:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                in_str = False
            continue
        if char in ('"', "'"):
            in_str = True
            quote = char
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[: index + 1]
    raise RuntimeError("playgroundOverrides object is unclosed")


def hex_to_rgba(color: str) -> list[float]:
    match = re.search(r"[0-9A-Fa-f]{6}", color.strip())
    if not match:
        raise ValueError(f"expected #RRGGBB color, got {color!r}")
    raw = match.group(0)
    red = int(raw[0:2], 16) / 255
    green = int(raw[2:4], 16) / 255
    blue = int(raw[4:6], 16) / 255
    return [red, green, blue, 1.0]


def set_override(overrides: dict[str, Any], component: str, field: str, typed_value: list[Any]) -> None:
    if component not in overrides or field not in overrides[component]:
        raise RuntimeError(f"override target missing: {component}.{field}")
    overrides[component][field] = typed_value


def apply_theme_overrides(overrides: dict[str, Any], theme_plan: dict[str, Any]) -> list[str]:
    applied: list[str] = []
    banner = theme_plan.get("banner") or {}
    plan_overrides = theme_plan.get("overrides") or {}

    if banner.get("text"):
        set_override(overrides, "BannerController", "messageText", ["string", banner["text"]])
        applied.append("BannerController.messageText")
    if banner.get("text_color"):
        set_override(
            overrides,
            "BannerController",
            "textColor",
            ["color", *hex_to_rgba(banner["text_color"])],
        )
        applied.append("BannerController.textColor")
    if banner.get("background_color"):
        set_override(
            overrides,
            "BannerController",
            "backgroundColor",
            ["color", *hex_to_rgba(banner["background_color"])],
        )
        applied.append("BannerController.backgroundColor")

    soil = plan_overrides.get("soil_color")
    if soil:
        color = hex_to_rgba(soil)
        for field in ("SoilColorPhase1", "SoilColorPhase2", "SoilColorPhase3"):
            set_override(
                overrides,
                "Dream.Playable.Configuration.ConfigurationManager",
                field,
                ["color", *color],
            )
            applied.append(f"ConfigurationManager.{field}")

    separator = plan_overrides.get("separator_wall_color")
    if separator:
        set_override(
            overrides,
            "LayoutController",
            "separatorWallColor",
            ["color", *hex_to_rgba(separator)],
        )
        applied.append("LayoutController.separatorWallColor")

    rock_colors = plan_overrides.get("rock_colors") or []
    if rock_colors:
        game = overrides.get("GameController") or {}
        layouts = sorted(
            {
                int(match.group(1))
                for key in game
                if (match := re.match(r"layout(\d+)Rock\d+Color$", key))
            }
        )
        if not layouts:
            layouts = [1]
        for layout in layouts:
            for index, color in enumerate(rock_colors, start=1):
                field = f"layout{layout}Rock{index}Color"
                if field in game:
                    set_override(
                        overrides,
                        "GameController",
                        field,
                        ["color", *hex_to_rgba(color)],
                    )
                    applied.append(f"GameController.{field}")
                grey_field = f"layout{layout}Rock{index}BaseGrey"
                if grey_field in game:
                    set_override(overrides, "GameController", grey_field, ["float", 0.0])
                    applied.append(f"GameController.{grey_field}")
    return applied


def high_risk_asset_ids(roles: dict[str, Any]) -> set[str]:
    blocked: set[str] = set()
    for role in roles.get("asset_roles", []):
        if role.get("risk") == "high":
            blocked.update(str(asset["id"]) for asset in role.get("assets", []))
    for item in roles.get("do_not_touch", []):
        blocked.update(str(asset_id) for asset_id in item.get("asset_ids", []))
    return blocked


def load_replacements(manifest: dict[str, Any], blocked: set[str], repo_root: Path) -> dict[str, tuple[str, str, bytes]]:
    replacements: dict[str, tuple[str, str, bytes]] = {}
    encoded_cache: dict[bytes, tuple[str, str]] = {}
    for item in manifest.get("assets", []):
        asset_id = str(item["asset_id"])
        if asset_id in blocked:
            raise RuntimeError(f"manifest includes a do_not_touch/high-risk asset: {asset_id}")
        source = repo_root / item["output_path"]
        if not source.exists():
            raise FileNotFoundError(f"generated asset missing: {source}")
        blob = source.read_bytes()
        dimensions = item.get("source_dimensions") or {}
        width = dimensions.get("width")
        height = dimensions.get("height")
        if width and height:
            blob = fit_image_bytes(
                blob,
                int(width),
                int(height),
                transparent=bool(item.get("transparent_background", True)),
            )
        cached = encoded_cache.get(blob)
        if cached is None:
            kind = sniff_kind(blob)
            mime = _MIME.get(kind)
            if mime is None:
                raise RuntimeError(f"unsupported generated image type for {asset_id}: {kind}")
            encoded = escape_attr(encode_base122(blob))
            if decode_base122(unescape(encoded)) != blob:
                raise RuntimeError(f"base122 roundtrip failed for {asset_id}")
            cached = (encoded, mime)
            encoded_cache[blob] = cached
        encoded, mime = cached
        replacements[asset_id] = (encoded, mime, blob)
    if not replacements:
        raise RuntimeError("no_patchable_assets_in_manifest")
    return replacements


def patch_images(html: str, replacements: dict[str, tuple[str, str, bytes]]) -> tuple[str, int]:
    found: set[str] = set()

    def replacer(match: re.Match[str]) -> str:
        path = match.group(2)
        asset_id = Path(path).stem
        if asset_id not in replacements:
            return match.group(0)
        found.add(asset_id)
        encoded, mime, _blob = replacements[asset_id]
        return f'{match.group(1)}{encoded}{match.group(4)}{mime}"'

    patched = IMG_HEAD_RE.sub(replacer, html)
    missing = sorted(set(replacements) - found)
    if missing:
        raise RuntimeError(f"asset IDs not found in HTML: {missing}")
    verified = 0
    for path, encoded, mime in IMG_RE.findall(patched):
        asset_id = Path(path).stem
        if asset_id not in replacements:
            continue
        _expected_encoded, expected_mime, blob = replacements[asset_id]
        if mime != expected_mime:
            raise RuntimeError(f"mime mismatch after patch for {asset_id}")
        if decode_base122(unescape(encoded)) != blob:
            raise RuntimeError(f"decoded payload mismatch after patch for {asset_id}")
        verified += 1
    if verified != len(replacements):
        raise RuntimeError(f"patched {verified} assets, expected {len(replacements)}")
    return patched, verified


def patch_overrides(html: str, theme_plan: dict[str, Any]) -> tuple[str, list[str]]:
    mark = html.find(OVERRIDES_MARK)
    if mark < 0:
        raise RuntimeError("window.playgroundOverrides not found")
    start = mark + len(OVERRIDES_MARK)
    original_obj = extract_js_object(html[start:])
    overrides = json.loads(original_obj)
    applied = apply_theme_overrides(overrides, theme_plan)
    dumped = json.dumps(overrides, ensure_ascii=True, separators=(",", ":"))
    return html[:start] + dumped + html[start + len(original_obj) :], applied


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_theme_html(theme: str) -> Path:
    slug = theme_slug(theme)
    theme_dir = THEMES_ROOT / slug
    manifest_path = theme_dir / "asset_manifest.json"
    plan_path = theme_dir / "theme_plan.json"
    if not SOURCE_HTML.exists():
        raise FileNotFoundError(f"source HTML not found: {SOURCE_HTML}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"asset manifest not found: {manifest_path}")
    if not plan_path.exists():
        raise FileNotFoundError(f"theme plan not found: {plan_path}")

    source_hash = file_sha256(SOURCE_HTML)
    html = SOURCE_HTML.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    theme_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    roles = load_roles()
    replacements = load_replacements(manifest, high_risk_asset_ids(roles), TASK2_DIR.parent)
    html, patched_count = patch_images(html, replacements)
    html, applied = patch_overrides(html, theme_plan)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{slug}.html"
    if output_path.resolve() == SOURCE_HTML.resolve():
        raise RuntimeError("refusing to overwrite source HTML")
    output_path.write_text(html, encoding="utf-8")

    if file_sha256(SOURCE_HTML) != source_hash:
        raise RuntimeError("source HTML changed during patch")
    report = {
        "theme": theme_plan.get("theme", theme),
        "slug": slug,
        "source_html": str(SOURCE_HTML.relative_to(TASK2_DIR.parent)),
        "output_html": str(output_path.relative_to(TASK2_DIR.parent)),
        "assets_patched": patched_count,
        "overrides_applied": applied,
        "source_sha256": source_hash,
    }
    report_path = theme_dir / "patch_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task 2 deterministic HTML patcher")
    parser.add_argument("--theme", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_path = patch_theme_html(args.theme)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    report_path = THEMES_ROOT / theme_slug(args.theme) / "patch_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(f"theme: {report['theme']}")
    print(f"assets_patched: {report['assets_patched']}")
    print(f"overrides: {len(report['overrides_applied'])}")
    for name in report["overrides_applied"]:
        print(f"  - {name}")
    print(f"output: {output_path}")
    print("source HTML unchanged")


if __name__ == "__main__":
    main()
