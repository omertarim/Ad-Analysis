# Task 2 — Playable Ad Theme Variation Workflow

Repo index: [`../README.md`](../README.md). Task 1: [`../task1/README.md`](../task1/README.md).

One command, one theme string, a new HTML. The original playable is never overwritten. There is no `if theme == "sushi"` in the code — an unseen theme should work.

```bash
python task2/run.py --theme sushi --api-key YOUR_GEMINI_API_KEY
```

```text
theme string
    → MAP     what is allowed to change
    → IDEATE  how this theme maps onto those fields
    → APPLY   generate sprites, patch a new HTML
    → QA      load in Chromium, tap, screenshot
    → outputs/<theme>.html
```

## How the architecture was built

The input is a ~4.2 MB Luna/Unity playable: WASM, embedded images (`data-src122`), and `window.playgroundOverrides`. Replacing every blob would break colliders, packed atlases, tutorial UVs, and the ad-network contract.

So the system is **parameterization**, not a full restyle.

Two kinds of work are split on purpose:

| Kind | Where | Examples |
| --- | --- | --- |
| Deterministic | Python | inventory scan, Luna encoding, HTML patch, screenshot metrics |
| Semantic | Gemini | “sushi → these tiles, this palette, this slogan” |

The catalog (`inventory/roles.json`) is theme-agnostic. The planner only ever sees: a free-form theme, role ids, sizes, and flags. Intermediate JSON is written after each stage, so a failed image gen does not force a re-plan.

Stages were built in order (Map, then Ideate, then Apply, then QA). Rules in [`CURSOR.md`](../CURSOR.md): no hardcoded themes, no hardcoded keys, keep artifacts, do not overwrite the source HTML.

## MAP — what we chose to vary

We ran the playable locally, listed 148 embedded images, and decided what is safe to restyle. That choice is the evaluation: high visual impact, low playability risk.

**Generated (image model)**

- match-3 tiles (7 icons — most of what the player looks at)
- falling rocks (52 frames; **one** sprite copied 51 times so physics radius stays the same)
- snake, wall texture, spike, tutorial hand, CTA, logo

Each role has flags later stages read: transparent vs tileable, generate or skip, source width × height.

**Patched without generating images**

- banner text and colors
- soil / wall colors
- rock tints on **every** layout, with `BaseGrey = 0`

**Left original**

| Leave | Why |
| --- | --- |
| Source HTML | variants go to `outputs/` |
| WASM | game logic |
| Chamber silhouette | colliders; generating it punched holes in the well |
| King atlas | packed UVs; regenerating it misaligned body parts |
| Board / tutorial coordinates | solvability |
| Endcard screenshots, store links, MRAID | ad-network contract |

Roles with `generate_image: false` are not sent to the planner, so the model cannot invent them back.

## IDEATE — theme in, plan out

`prompts/theme_plan.txt` + the catalog go to `gemini-3.6-flash`. Output is a pydantic `ThemePlan`: palette, banner, rock colors, one brief per role.

The planner does not get a sushi/football lookup table. It gets a theme string and has to map it. If a single-asset role (e.g. wall) is missing a per-id row, Python fills it from the role-level brief instead of crashing.

## APPLY — generate and patch

Sprites: `gemini-3.1-flash-image` (fallback `gemini-2.5-flash-image`). The image prompt is a Python-filled template (`prompts/asset_image.txt`), not a second LLM call.

The patcher encodes PNG bytes as Luna `base122`, replaces `data-src122` for catalog ids, writes `playgroundOverrides`, and saves `outputs/<slug>.html`. Missing ids fail loudly. After the write, the source file hash must still match.

## QA

Playwright Chromium, 720×1280, HTML served over HTTP with COOP/COEP (WASM will not load from `file://`).

- **Playable:** canvas exists, Luna preloader gone, four taps, pixels change
- **Visual:** not all-black, no magenta missing-texture, some color variance, no large white hole

Pass = both. An optional VLM comment exists; it does **not** decide the bit. A gate a human can audit beat an opaque “looks good” score.

## Bugs we found by running, not by guessing

The first sushi HTML looked themed and still failed as a game. That is how the MAP contract was tightened.

1. **Chamber layout.** Generating a new well broke the collider silhouette (white voids). Fix: stop generating it; theme that space with wall texture + colors.

2. **52 rock frames as 52 different objects.** Physics expected one radius. Fix: generate once, copy.

3. **Only layout 1 rocks were tinted.** Levels 2–3 stayed grey (`BaseGrey` still 0.43). Fix: copy theme rock colors onto every `layoutNRockKColor` and set `BaseGrey = 0`.

4. **Gemini returned JPEG grey cards, not PNG alpha.** Four-corner punch left grey around snake/CTA. Fix: prompt for true alpha, validate, retry once, then flood-fill **edge-connected** matte from the full border.

5. **1K generated sprites.** HTML jumped to ~50 MB. Fix: resize to the source sprite size before embed (~4–8 MB).

6. **Planner skipped wall asset `37162`.** Patch then failed. Fix: fill missing per-asset rows from the role brief.

7. **Copyrighted theme names** (e.g. Spider-Man). Image API refused; that is a provider block, not a missing key. Prefer generic themes.

8. **Theme-plan 503/429.** Retry with backoff and a Flash fallback.

## Submitted HTML files

| Theme | File |
| --- | --- |
| football (required) | `task2/outputs/football.html` |
| sushi (required) | `task2/outputs/sushi.html` |
| turkiye | `task2/outputs/t-rkiye.html` |
| jazz-concert | `task2/outputs/jazz-concert.html` |
| galatasaray | `task2/outputs/galatasaray.html` |

Use ASCII names (`turkiye`, not `türkiye`). Slugs are `[a-z0-9-]` only.

Trace per theme: `task2/artifacts/themes/<slug>/` (`theme_plan.json`, `asset_manifest.json`, `qa_report.json`, screenshots, generated assets).

Preview:

```bash
python -m http.server 8000 --directory task2/outputs
# http://localhost:8000/football.html
```

`--force` regenerates sprites even if files exist. `--api-key` / `--gemini-api-key` / `GEMINI_API_KEY` are equivalent.

## Models and tools

| Job | Choice | Why |
| --- | --- | --- |
| Theme plan | `gemini-3.6-flash` + JSON schema | Fast structured output; pydantic rejects bad plans |
| Sprites | `gemini-3.1-flash-image` | Current Gemini image model on this API (Imagen is deprecated here) |
| QA | Playwright Chromium | Real WebGL/WASM, not a static HTML parse |
| Embed | Luna base122 in-repo | Same encoding the playable already uses |
| Orchestration | `task2/run.py` | Fail-fast stages, one command |

One provider (Gemini) keeps keys and failure modes in one family.

## How we used AI

Cursor wrote most of the code. Decisions stayed in `roles.json`, the two prompt files, and the Python patch/QA. Constraints in [`CURSOR.md`](../CURSOR.md).

Prompts that actually shipped (not chat-only):

- [`prompts/theme_plan.txt`](prompts/theme_plan.txt) — map this theme onto the catalog; do not invent a different game
- [`prompts/asset_image.txt`](prompts/asset_image.txt) — filled in Python with transparent vs tileable flags

## What I would change in production

1. An image model that emits real PNG alpha, so matte recovery can go.
2. Lock each sprite to the original silhouette (mask / ControlNet-style).
3. Recolor the chamber without changing its collider.
4. Inpaint the king atlas without moving UVs.
5. Generate sprites in parallel, with a per-role cost cap.
6. Compare screenshots to a golden frame, not only black/white/magenta checks.
7. Unicode slugs (`türkiye` → `turkiye`, not `t-rkiye`).

## Layout

```text
CURSOR.md
task2/
  README.md
  run.py
  inventory/roles.json
  prompts/theme_plan.txt
  prompts/asset_image.txt
  pipeline/{inventory,theme_plan,generate_assets,patch_html,qa}.py
  source/RK_SaveTheKing_Playable.html
  outputs/<theme>.html
  artifacts/themes/<theme>/
```
