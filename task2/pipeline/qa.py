"""Playability and visual QA for a themed playable HTML."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from datetime import UTC, datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from task2.pipeline.theme_plan import (  # noqa: E402
    TASK2_DIR,
    THEMES_ROOT,
    resolve_api_key,
    theme_slug,
)

OUTPUT_DIR = TASK2_DIR / "outputs"
VIEWPORT = {"width": 720, "height": 1280}
LOAD_TIMEOUT_MS = 120_000
PRELOADER_TIMEOUT_MS = 90_000
CLICKS = ((0.50, 0.80), (0.36, 0.78), (0.64, 0.78), (0.50, 0.60))
CONSOLE_FATAL = ("webgl", "wasm", "not supported", "out of memory", "failed to compile")
VLM_MODEL = "gemini-3.6-flash"


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


def start_server(directory: Path) -> tuple[ThreadingHTTPServer, int]:
    class Handler(QuietHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(directory), **kwargs)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, int(server.server_address[1])


def relative_to_repo(path: Path) -> str:
    return str(path.relative_to(TASK2_DIR.parent))


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def largest_blob_ratio(mask: np.ndarray) -> float:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best = 0
    total = mask.size
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            size = 0
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                size += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if size > best:
                best = size
    return best / max(total, 1)


def pixel_diff_ratio(left: Path, right: Path) -> float:
    first = load_rgb(left)
    second = load_rgb(right)
    if first.shape != second.shape:
        return 1.0
    changed = np.any(first != second, axis=2)
    return float(changed.mean())


def analyze_screenshot(path: Path) -> dict[str, Any]:
    pixels = load_rgb(path)
    luminance = pixels.mean(axis=2)
    white = (pixels[:, :, 0] > 240) & (pixels[:, :, 1] > 240) & (pixels[:, :, 2] > 240)
    magenta = (pixels[:, :, 0] > 220) & (pixels[:, :, 2] > 220) & (pixels[:, :, 1] < 40)
    small = np.asarray(
        Image.fromarray(pixels).resize((80, 142), Image.Resampling.NEAREST)
    )
    small_white = (small[:, :, 0] > 240) & (small[:, :, 1] > 240) & (small[:, :, 2] > 240)
    unique_colors = len(
        {tuple(pixel) for pixel in small.reshape(-1, 3)[::8].tolist()}
    )
    black_ratio = float((luminance < 12).mean())
    white_ratio = float(white.mean())
    magenta_ratio = float(magenta.mean())
    white_blob_ratio = float(largest_blob_ratio(small_white))
    luminance_std = float(luminance.std())
    mostly_black = black_ratio >= 0.97
    large_white_block = white_blob_ratio >= 0.08
    asset_load_failure = magenta_ratio >= 0.01
    gameplay_visible = (not mostly_black) and luminance_std >= 18 and unique_colors >= 16
    checks = {
        "mostly_black": mostly_black,
        "asset_load_failure_marker": asset_load_failure,
        "gameplay_area_visible": gameplay_visible,
        "large_white_block": large_white_block,
    }
    return {
        "path": relative_to_repo(path),
        "black_ratio": round(black_ratio, 4),
        "white_ratio": round(white_ratio, 4),
        "magenta_ratio": round(magenta_ratio, 4),
        "white_blob_ratio": round(white_blob_ratio, 4),
        "luminance_std": round(luminance_std, 2),
        "unique_colors_sampled": unique_colors,
        "checks": checks,
        "pass": (not mostly_black)
        and (not asset_load_failure)
        and (not large_white_block)
        and gameplay_visible,
    }


def is_fatal_console(message: str) -> bool:
    lower = message.lower()
    return any(token in lower for token in CONSOLE_FATAL)


def run_vlm(screenshot: Path, theme: str, api_key: str) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    prompt = (
        "You are reviewing a mobile playable-ad screenshot. "
        f"Theme: {theme}. Return JSON only with keys "
        "theme_cohesion (0-1), obvious_artifact (boolean), "
        "visual_readability (0-1), notes (short string). "
        "Judge cohesion, obvious broken geometry/transparency artifacts, and readability."
    )
    response = client.models.generate_content(
        model=VLM_MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(data=screenshot.read_bytes(), mime_type="image/png"),
        ],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    text = getattr(response, "text", None) or "{}"
    parsed = json.loads(text)
    return {
        "theme_cohesion": parsed.get("theme_cohesion"),
        "obvious_artifact": parsed.get("obvious_artifact"),
        "visual_readability": parsed.get("visual_readability"),
        "notes": parsed.get("notes"),
        "model": VLM_MODEL,
    }


def run_browser_qa(html_name: str, screenshot_dir: Path) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    screenshot_dir.mkdir(parents=True, exist_ok=True)
    server, port = start_server(OUTPUT_DIR)
    url = f"http://127.0.0.1:{port}/{html_name}"
    console_errors: list[str] = []
    page_errors: list[str] = []
    screenshots: dict[str, str] = {}
    interactions: list[dict[str, Any]] = []
    notes: list[str] = []
    page_loaded = False
    canvas_present = False
    preloader_cleared = False
    canvas_size = {"width": 0, "height": 0}

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport=VIEWPORT)
            page.on(
                "console",
                lambda message: console_errors.append(message.text)
                if message.type == "error"
                else None,
            )
            page.on("pageerror", lambda exc: page_errors.append(str(exc)))
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=LOAD_TIMEOUT_MS)
                page_loaded = True
                page.wait_for_selector("#application-canvas", timeout=PRELOADER_TIMEOUT_MS)
                canvas_present = True
                page.wait_for_function(
                    "() => !document.getElementById('application-preloader')",
                    timeout=PRELOADER_TIMEOUT_MS,
                )
                preloader_cleared = True
                page.wait_for_timeout(2000)
                canvas_size = page.evaluate(
                    """() => {
                        const canvas = document.getElementById('application-canvas');
                        return {width: canvas ? canvas.width : 0, height: canvas ? canvas.height : 0};
                    }"""
                )
                initial_path = screenshot_dir / "initial.png"
                page.screenshot(path=str(initial_path), full_page=False)
                screenshots["initial"] = relative_to_repo(initial_path)

                box = page.locator("#application-canvas").bounding_box()
                if box:
                    for index, (nx, ny) in enumerate(CLICKS, start=1):
                        x = box["x"] + box["width"] * nx
                        y = box["y"] + box["height"] * ny
                        page.mouse.click(x, y)
                        page.wait_for_timeout(450)
                        interactions.append(
                            {
                                "name": f"canvas_tap_{index}",
                                "x": round(x, 1),
                                "y": round(y, 1),
                            }
                        )
                after_path = screenshot_dir / "after_interaction.png"
                page.screenshot(path=str(after_path), full_page=False)
                screenshots["after_interaction"] = relative_to_repo(after_path)
                page.wait_for_timeout(1500)
                final_path = screenshot_dir / "final.png"
                page.screenshot(path=str(final_path), full_page=False)
                screenshots["final"] = relative_to_repo(final_path)
            except Exception as exc:  # noqa: BLE001
                notes.append(f"browser_qa_error: {exc}")
                fail_path = screenshot_dir / "fail.png"
                try:
                    page.screenshot(path=str(fail_path), full_page=False)
                    screenshots["fail"] = relative_to_repo(fail_path)
                except Exception:
                    pass
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()

    fatal_console = [item for item in console_errors if is_fatal_console(item)]
    playability_pass = (
        page_loaded
        and canvas_present
        and preloader_cleared
        and int(canvas_size.get("width") or 0) > 50
        and int(canvas_size.get("height") or 0) > 50
        and not page_errors
        and not fatal_console
        and "initial" in screenshots
    )
    if not page_loaded:
        notes.append("page did not load")
    if page_loaded and not canvas_present:
        notes.append("application-canvas was not found")
    if canvas_present and not preloader_cleared:
        notes.append("preloader never cleared")
    if not interactions:
        notes.append("no canvas taps were sent")
    return {
        "url": url,
        "page_loaded": page_loaded,
        "canvas_present": canvas_present,
        "preloader_cleared": preloader_cleared,
        "canvas_size": canvas_size,
        "console_errors": console_errors,
        "page_errors": page_errors,
        "interactions_attempted": interactions,
        "screenshots": screenshots,
        "playability_pass": playability_pass,
        "notes": notes,
    }


def run_visual_qa(screenshots: dict[str, str]) -> dict[str, Any]:
    analyses = []
    keys = [key for key in ("initial", "after_interaction", "final", "fail") if key in screenshots]
    for key in keys:
        path = TASK2_DIR.parent / screenshots[key]
        if path.exists():
            analyses.append({"label": key, **analyze_screenshot(path)})
    state_changed = False
    if "initial" in screenshots and "after_interaction" in screenshots:
        state_changed = (
            pixel_diff_ratio(
                TASK2_DIR.parent / screenshots["initial"],
                TASK2_DIR.parent / screenshots["after_interaction"],
            )
            >= 0.002
        )
    visual_pass = bool(analyses) and all(
        item["pass"] for item in analyses if item["label"] != "fail"
    )
    notes = []
    if not analyses:
        notes.append("no screenshots to analyze")
    for item in analyses:
        checks = item["checks"]
        failed: list[str] = []
        if checks["mostly_black"]:
            failed.append("mostly_black")
        if checks["asset_load_failure_marker"]:
            failed.append("asset_load_failure_marker")
        if checks["large_white_block"]:
            failed.append("large_white_block")
        if not checks["gameplay_area_visible"]:
            failed.append("gameplay_area_not_visible")
        if failed and item["label"] != "fail":
            notes.append(f"{item['label']}: {', '.join(failed)}")
    if analyses and not state_changed:
        notes.append("taps did not produce a visible state change")
    return {
        "screenshots_analyzed": analyses,
        "interaction_state_changed": state_changed,
        "visual_pass": visual_pass,
        "notes": notes,
    }


def qa_theme(theme: str, *, vlm: bool = False, api_key: str | None = None) -> Path:
    slug = theme_slug(theme)
    html_path = OUTPUT_DIR / f"{slug}.html"
    if not html_path.exists():
        raise FileNotFoundError(f"themed HTML not found: {html_path}. Run patch_html.py first.")
    theme_dir = THEMES_ROOT / slug
    screenshot_dir = theme_dir / "screenshots"
    browser = run_browser_qa(html_path.name, screenshot_dir)
    visual = run_visual_qa(browser["screenshots"])
    notes = list(browser["notes"]) + list(visual["notes"])
    vlm_result = None
    if vlm:
        key = resolve_api_key(api_key)
        shot = browser["screenshots"].get("after_interaction") or browser["screenshots"].get("initial")
        if shot:
            try:
                vlm_result = run_vlm(TASK2_DIR.parent / shot, theme, key)
                if vlm_result.get("obvious_artifact"):
                    notes.append(f"vlm_obvious_artifact: {vlm_result.get('notes')}")
            except Exception as exc:  # noqa: BLE001
                notes.append(f"vlm_error: {exc}")
                vlm_result = {"error": str(exc)}
        else:
            notes.append("vlm skipped: no screenshot")
    overall_pass = bool(browser["playability_pass"] and visual["visual_pass"])
    report = {
        "theme": theme,
        "slug": slug,
        "html": relative_to_repo(html_path),
        "created_at": datetime.now(UTC).isoformat(),
        "page_loaded": browser["page_loaded"],
        "console_errors": browser["console_errors"],
        "page_errors": browser["page_errors"],
        "interactions_attempted": browser["interactions_attempted"],
        "screenshots": browser["screenshots"],
        "canvas_present": browser["canvas_present"],
        "preloader_cleared": browser["preloader_cleared"],
        "canvas_size": browser["canvas_size"],
        "playability_pass": browser["playability_pass"],
        "visual_pass": visual["visual_pass"],
        "interaction_state_changed": visual["interaction_state_changed"],
        "visual": visual["screenshots_analyzed"],
        "vlm": vlm_result,
        "overall_pass": overall_pass,
        "notes": notes,
    }
    report_path = theme_dir / "qa_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report_path


def _flag(value: bool) -> str:
    return "PASS" if value else "FAIL"


def print_qa_summary(report: dict, *, prefix: str = "") -> None:
    overall = _flag(bool(report.get("overall_pass")))
    print(f"{prefix}QA overall: {overall}")
    print(f"{prefix}  playability: {_flag(bool(report.get('playability_pass')))}")
    print(f"{prefix}    page_loaded: {report.get('page_loaded')}")
    print(f"{prefix}    canvas: {report.get('canvas_present')} {report.get('canvas_size')}")
    print(f"{prefix}    preloader_cleared: {report.get('preloader_cleared')}")
    print(f"{prefix}    console_errors: {len(report.get('console_errors') or [])}")
    print(f"{prefix}    page_errors: {len(report.get('page_errors') or [])}")
    print(f"{prefix}    taps: {len(report.get('interactions_attempted') or [])}")
    print(f"{prefix}    state_changed: {report.get('interaction_state_changed')}")
    print(f"{prefix}  visual: {_flag(bool(report.get('visual_pass')))}")
    for item in report.get("visual") or []:
        checks = item.get("checks") or {}
        failed = []
        if checks.get("mostly_black"):
            failed.append("black_screen")
        if checks.get("asset_load_failure_marker"):
            failed.append("missing_texture")
        if checks.get("large_white_block"):
            failed.append("white_block")
        if not checks.get("gameplay_area_visible", True):
            failed.append("no_gameplay")
        status = "PASS" if item.get("pass") else "FAIL " + ",".join(failed)
        print(
            f"{prefix}    {item.get('label')}: {status} "
            f"black={item.get('black_ratio')} white_blob={item.get('white_blob_ratio')} "
            f"std={item.get('luminance_std')}"
        )
    shots = report.get("screenshots") or {}
    if shots:
        print(f"{prefix}  screenshots: {', '.join(shots.values())}")
    for error in report.get("console_errors") or []:
        print(f"{prefix}  console: {error}")
    for error in report.get("page_errors") or []:
        print(f"{prefix}  page: {error}")
    for note in report.get("notes") or []:
        print(f"{prefix}  note: {note}")
    vlm = report.get("vlm")
    if vlm:
        print(f"{prefix}  vlm: {vlm}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task 2 playable QA")
    parser.add_argument("--theme", required=True)
    parser.add_argument("--vlm", action="store_true", help="Optional Gemini screenshot review")
    parser.add_argument("--api-key", dest="api_key", default=None)
    parser.add_argument("--gemini-api-key", dest="gemini_api_key", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    api_key = args.api_key or args.gemini_api_key
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    try:
        report_path = qa_theme(args.theme, vlm=args.vlm, api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print_qa_summary(report)
    print(f"report: {report_path}")
    if not report["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
