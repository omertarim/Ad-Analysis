# Dream Games AI Engineer Case

One private repository, two independent systems. Architecture and creative judgement are documented per task; this file is the landing page.

| | Task | Detail |
|---|---|---|
| 1 | Trend → Royal Match hook videos | [`task1/README.md`](task1/README.md) |
| 2 | Playable ad theme variants | [`task2/README.md`](task2/README.md) |

Working rules used while building: [`CURSOR.md`](CURSOR.md).

## Setup (once)

Python 3.11+ from the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

`requirements.txt` installs every Python library both tasks need. Same list, one by one:

```bash
pip install pydantic          # structured JSON schemas (both tasks)
pip install google-genai      # Gemini + Veo API client (both tasks)
pip install scikit-learn      # Random Forest ranker (Task 1)
pip install joblib            # save/load the ranker (Task 1)
pip install numpy             # ranker features + Task 2 screenshot metrics
pip install Pillow            # Task 2 sprite resize / alpha
pip install playwright        # Task 2 headless Chromium QA
pip install pytest            # unit tests under tests/task1/
```

After `pip install playwright`, browsers are a second step:

```bash
playwright install chromium
```

Task 1 video QA also needs **ffmpeg / ffprobe** on the machine (not a pip package):

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install -y ffmpeg
```

Keys are only passed on the command line (or via env). Do not commit them.

| Env / flag | Used by |
| --- | --- |
| `APIFY_API_TOKEN` / `--apify-api-token` | Task 1 TikTok discovery + download |
| `GEMINI_API_KEY` / `--gemini-api-key` / `--api-key` | Task 1 Gemini/Veo, Task 2 theme plan + sprites |

## Task 1 — one command

```bash
python task1/run.py \
  --full-pipeline \
  --apify-api-token YOUR_APIFY_TOKEN \
  --gemini-api-key YOUR_GEMINI_API_KEY
```

Submission run: `run_20260902_095910`  
Outputs: `task1/artifacts/runs/run_20260902_095910/` and three `final.mp4` files under `task1/artifacts/generation/`.

## Task 2 — one command

```bash
python task2/run.py --theme sushi --api-key YOUR_GEMINI_API_KEY
```

Submitted HTMLs: `task2/outputs/{football,sushi,t-rkiye,jazz-concert,galatasaray}.html`

## What lives where

```text
CURSOR.md
README.md                 this index
task1/README.md           architecture, models, weekly semantics, lineage
task1/run.py
task1/artifacts/          trends → videos (do not delete models/)
task2/README.md           Map → Ideate → Apply → QA
task2/run.py
task2/outputs/            themed playables
task2/source/             original HTML (never overwritten)
```

