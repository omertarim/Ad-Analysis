# Task 1 — Trend to Creative Agentic Pipeline

Repo index: [`../README.md`](../README.md). Task 2: [`../task2/README.md`](../task2/README.md).

### Engineering Decisions That Shaped the System

This project is built around a practical engineering objective: convert current TikTok trend signal into Royal Match-ready creative candidates while controlling cost, latency, and reliability risk. I designed the pipeline as a sequence of explicit decisions, each tied to a constraint and trade-off.

**A) Why I did not build a TikTok scraper from scratch**  
During this case, TikTok's developer onboarding flow indicated an approximately 10-day review period, which did not fit the delivery timeline. Building and maintaining an unofficial crawler was possible, but it would have redirected effort into anti-bot resilience, proxy management, and scraper break/fix cycles.  
I therefore separated source and infrastructure:
- **Public trend source:** TikTok Explore
- **Acquisition infrastructure:** Apify

This is intentional. TikTok Explore provides the trend signal. Apify provides repeatable extraction infrastructure so engineering effort stays focused on trend understanding and creative decision-making.

**B) Why I added ML before Gemini**  
A naive system would push all discovered trends through download + multimodal analysis + creative evaluation. That increases paid calls, increases upload volume, and amplifies exposure to provider-side capacity errors.  
I inserted a low-cost metadata ranking stage:

`metadata -> candidate prioritization -> expensive multimodal stages`

The ranker is not the final creative judge. It only decides whether a trend is worth expensive inference. Video-level semantics and creative suitability remain downstream.

**C) Why Random Forest**  
I evaluated Logistic Regression, Random Forest, and HistGradientBoosting. At this stage recall matters most:
- False positive: one extra Gemini call.
- False negative: potentially losing a strong trend before creative analysis.

Measured metrics from `task1/artifacts/models/training_summary.json`:
- Logistic Regression: precision 1.0000, recall 0.8229, f1 0.9028
- Random Forest: precision 0.8470, recall 0.8376, f1 0.8423
- HistGradientBoosting: precision 0.9605, recall 0.8081, f1 0.8778

Random Forest was selected due to highest measured recall.  
`engagement_rate` was used to create the training target and explicitly excluded from model features to avoid target leakage.

I also treat model probability as a relative ranking signal (top-percent selection), not absolute creative truth, because public training distribution and live TikTok Explore distribution are not identical.

**D) Why Gemini + Veo in one ecosystem**  
The system needed:
1. multimodal video understanding,
2. structured creative reasoning,
3. final video generation.

Using Gemini for analysis/evaluation and Veo for generation keeps those stages within one provider ecosystem and one credential family. This reduces integration surface area and cross-provider failure complexity.

**E) Why deterministic scoring after the LLM**  
The LLM outputs structured evaluation dimensions (hook strength, transferability, risks), but the application computes weighted `overall_score` and final `adaptable` gating.  
This prevents opaque model text from being the final business decision and keeps behavior reproducible, auditable, and testable.

**F) Why run semantics and artifacts are part of architecture**  
Each trend receives deterministic identity and stage artifacts. This supports cache reuse, resume behavior, and source-to-output lineage.  
Weekly correctness is explicit: every run starts from current discovery; historical adaptable trends are not auto-counted in new runs; historical artifacts are reused only if the same trend reappears in current discovery.

---

### System at a Glance

```text
TikTok Explore
      ↓
Apify Discovery
      ↓
Normalized Trend Metadata
      ↓
Random Forest Prioritization
      ↓
Video Download + Validation
      ↓
Gemini Multimodal Understanding
      ↓
Royal Match Creative Evaluation
      ↓
Deterministic Scoring
      ↓
10–20 Adaptable Hooks
      ↓
Top 3
      ↓
Creative Brief + Storyboard
      ↓
Veo
      ↓
QA
      ↓
3 Final Ads
```

---

### Real Run Results

Submission run: `run_20260902_095910` (clean cache, end-to-end, ~17 minutes)

- 61 unique current trends discovered (4 discovery rounds, region `TR`)
- 31 ML-selected for expensive stages
- 30 analyzed, 30 evaluated (`cache_hits: 0`)
- 8 adaptable hooks, 22 rejected
- top 3 generated with `veo-3.1-lite-generate-preview`
- 3/3 videos passed automated QA
- final output: 8 sec, 720×1280, 9:16

The design target is 10–20 adaptable hooks (`--target-adaptable 10`). This run stopped at 8 after four rounds because TikTok Explore started repeating already-seen ids and the remaining evaluated trends failed the adaptable gates. I did not loosen brand-safety / gameplay-bridge thresholds to fill the quota.

Reliability evidence: one real Gemini `503 UNAVAILABLE` on `trend_2f57d100d8c27e22`; the failure was recorded and the run continued.

---

### Discover: Current TikTok Trends

Discovery uses a source-independent collector abstraction with a TikTok implementation:
- `task1/collectors/base.py`
- `task1/collectors/apify_tiktok.py`

Actors used in code:
- `khadinakbar/tiktok-trending-videos-scraper`
- `khadinakbar/tiktok-no-watermark-downloader`

Raw data is normalized into `TrendSourceMetadata` with deterministic `trend_id`, stable schema, and preserved source metadata for traceability.

**Weekly operation:** the same command is the weekly job. Each invocation creates a new `run_*` under `task1/artifacts/runs/`. Discovery always starts from current TikTok Explore, not last week's adaptable list. If a previously seen `trend_id` reappears, stage artifacts (video, analysis, evaluation) can be reused; if it does not reappear, it is not counted in the new run. Ranking features are live engagement fields, so the same video can rank differently a week later.

---

### Cost & Latency Control: ML Candidate Ranking

The expensive path is video upload + multimodal inference + creative evaluation.  
The ranking layer intentionally sits before that path.

**Training data (where it came from)**  
The ranker was trained on a public GitHub dataset, not on Dream Games ads and not on the live Apify crawl.

- Public source: [https://github.com/Crawlora-org/tiktok-engagement-data](https://github.com/Crawlora-org/tiktok-engagement-data) (Crawlora TikTok Engagement Index, 4,509 videos across 13 niches)
- File I trained on, stored in this repo: `task1/data/training/tiktok_engagement_crawlora.jsonl`

That JSONL is the full training set (4,509 usable rows). `engagement_rate` is the target only; it is not an input feature. The fitted model and metrics are:

- `task1/artifacts/models/candidate_ranker.joblib`
- `task1/artifacts/models/training_summary.json`

Implemented model features:
- `views`, `likes`, `comments`, `shares`, `saves`, `followers`, `video_age_days`

Production selection mode is top-percent to prioritize relatively better candidates under domain shift.  
In the submission run, 61 current trends were discovered while 31 were selected by this layer, so expensive stages were not forced to process the entire discovered pool.
Candidate probabilities can change across weekly runs for the same `trend_id` because live engagement metadata (for example views/likes/comments/shares/saves/followers) can change between discoveries.  
For auditability, each ranking artifact persists an `inference_features` snapshot together with `model_version`, `model_path`, `inference_threshold`, and `candidate_probability`. The ranker probability is a funnel signal, not the creative score: several adaptable winners in this run had low metadata probability and were accepted only after Gemini evaluation.

---

### Multimodal Understanding & Creative Evaluation

After ranking and video validation:

1. `gemini-3.6-flash` performs structured multimodal understanding on actual downloaded video files.
2. `gemini-3.6-flash` performs structured Royal Match creative evaluation.
3. Application logic computes deterministic final score and adaptable/rejected decision.

Prompt files:
- `task1/prompts/video_analysis.txt`
- `task1/prompts/creative_evaluation.txt`

The system does not ask a single vague "is this good?" question. It decomposes suitability into explicit positive and risk dimensions, then applies deterministic gating.

Scoring logic (concise):

```text
overall_score = clamp(positive_weighted_sum - risk_weighted_sum, 0, 10)

adaptable = (
  overall_score >= 6.0
  and brand_safety_score >= 6.0
  and gameplay_bridge_score >= 5.0
  and royal_match_world_transferability_score >= 5.0
  and ip_or_identity_risk < 8.0
)
```

Positive dimensions include hook strength, first-seconds strength, transferability, player intervention, gameplay bridge, visual clarity, feasibility, tonal fit, trend transferability, and brand safety.  
Risk dimensions include context dependency, audio dependency, and IP/identity risk.

---

### Adaptable vs Rejected Trends

The submission run produced 8 adaptable hooks (target 10–20; see run results above). Index: `task1/artifacts/runs/run_20260902_095910/adaptable_hooks.json`.

Top 3 generated:

| Rank | Trend ID | Score | Adaptation |
|---|---|---:|---|
| 1 | `trend_4887f73cf1cb1b23` | 7.74 | King Robert balances a falling castle stone; match-3 shatters it |
| 2 | `trend_dc9b4e08aa3356b0` | 7.65 | Rolling barrel in a hallway; player solves to block / jump |
| 3 | `trend_068b1a1cbdf8e816` | 7.635 | Sleeping King woken by a royal parrot; player silences the gag |

Finished ads:
- `task1/artifacts/generation/trend_4887f73cf1cb1b23/final.mp4`
- `task1/artifacts/generation/trend_dc9b4e08aa3356b0/final.mp4`
- `task1/artifacts/generation/trend_068b1a1cbdf8e816/final.mp4`

Rejected example (required by the brief):
- `trend_c4eb24f5dd0bb588`
- Source URL: `https://www.tiktok.com/@amero63/video/7671937488486714645`
- Score: `1.86`
- Reasons: real-life building fire / trapped people (brand safety `0.0`); tone incompatible with a family-friendly puzzle ad
- Artifact: `task1/artifacts/evaluations/trend_c4eb24f5dd0bb588/evaluation.json`

More rejects (audio-only vlogs, no gameplay bridge) are in `task1/artifacts/runs/run_20260902_095910/rejected_examples.json`.

Popularity alone does not imply transferable creative value.

---

### Generate: Top 3 to Finished Ads

Generation flow:

`top trend -> creative_brief -> storyboard -> generation_prompt -> Veo -> ffprobe QA`

Model:
- `veo-3.1-lite-generate-preview` (cost-controlled text-conditioned generation)

Final outputs in the submission run:
- 3/3 generation success
- 3/3 QA pass (`qa.json` per video)
- 8 seconds each
- 720×1280 vertical (9:16)

Why 8 seconds: the Veo endpoint used supported up to 8 seconds, and 8 seconds still satisfies the case requirement of 8-15 seconds.

Mechanics are adapted, not copied. Creator likeness, celebrity likeness, TikTok branding, and unrelated copyrighted characters are excluded.
King Robert consistency is guided by a detailed canonical text description in the prompt rather than reference-image conditioning; this is a deliberate cost/consistency trade-off.
For higher-budget production, image-conditioned generation with approved brand assets can provide stronger character fidelity.
Final generation prompts enforce no spoken dialogue and no visible typography/text artifacts in rendered frames.
Technical QA (duration/resolution/playability) is automated; final character/brand fidelity and on-screen-text cleanliness remain human creative-review checkpoints.

---

### Traceability & Weekly Semantics

Concrete lineage example (rank 1, `trend_4887f73cf1cb1b23`):

TikTok URL  
→ `task1/artifacts/raw/trend_4887f73cf1cb1b23/trend_metadata.json`  
→ `task1/artifacts/rankings/trend_4887f73cf1cb1b23/ranking.json`  
→ `task1/artifacts/raw/trend_4887f73cf1cb1b23/source.mp4`  
→ `task1/artifacts/analysis/trend_4887f73cf1cb1b23/analysis.json`  
→ `task1/artifacts/evaluations/trend_4887f73cf1cb1b23/evaluation.json`  
→ `task1/artifacts/generation/trend_4887f73cf1cb1b23/creative_brief.json`  
→ `task1/artifacts/generation/trend_4887f73cf1cb1b23/storyboard.json`  
→ `task1/artifacts/generation/trend_4887f73cf1cb1b23/generation_prompt.txt`  
→ `task1/artifacts/generation/trend_4887f73cf1cb1b23/final.mp4`  
→ `task1/artifacts/generation/trend_4887f73cf1cb1b23/qa.json`

Run-level index files:
- `task1/artifacts/runs/run_20260902_095910/run_summary.json`
- `task1/artifacts/runs/run_20260902_095910/final_creatives.json`
- `task1/artifacts/runs/run_20260902_095910/adaptable_hooks.json`
- `task1/artifacts/runs/run_20260902_095910/rejected_examples.json`

Weekly rule remains strict: current run starts from current discovery; historical trends are cache only unless rediscovered.

---

### Run the Project

Setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Main one-command execution (API keys as the only required input):

```bash
python task1/run.py \
  --full-pipeline \
  --apify-api-token YOUR_APIFY_TOKEN \
  --gemini-api-key YOUR_GEMINI_API_KEY
```

Two flags because the pipeline uses two paid providers (Apify for TikTok acquisition, Gemini/Veo for understanding and generation). Environment fallback is supported (`APIFY_API_TOKEN`, `GEMINI_API_KEY`) for local convenience, but not required.

> **Paid API access is required.**  
> `APIFY_API_TOKEN` must map to an account with sufficient Apify billing/credits for the actors used.  
> `GEMINI_API_KEY` must have appropriate Gemini billing/model entitlement, including Veo generation access where required.

Observed run note only (not a pricing guarantee): one 20-trend discovery smoke run reported approximately `$1`.

---

### Repository Guide

Task 1 structure is intentionally role-based:
- `domain/` schemas and deterministic identity logic
- `collectors/` trend acquisition abstraction + TikTok implementation
- `services/` ranking, acquisition, validation, multimodal, evaluation, generation
- `pipeline/` orchestration flow
- `ml/` training utilities (`task1/ml/train_candidate_ranker.py`)
- `data/training/` public Crawlora JSONL used to train the ranker
- `prompts/` runtime prompt templates
- `artifacts/` run outputs and stage-level lineage/cache

This keeps policy logic, model logic, and provider integration concerns isolated.

---

### AI-Assisted Development

Cursor/AI-assisted coding was used as an implementation accelerator while engineering decisions remained explicit.

Working style:
- requirements/constraints defined iteratively,
- changes implemented in bounded steps,
- provider assumptions checked via real smoke runs,
- tests run before acceptance,
- weaker experiments measured and removed.

References:
- [`CURSOR.md`](../CURSOR.md) — step-by-step delivery, no hardcoded keys, keep intermediate artifacts, deterministic code vs creative LLM work
- `task1/prompts/video_analysis.txt`
- `task1/prompts/creative_evaluation.txt`

This project was not "AI autopilot"; AI accelerated implementation, while architecture and acceptance criteria were manually controlled.

---

### What I Would Change in Production

With real budget and production models:

1. Run the pipeline on a weekly schedule, with alerts when a stage fails.
2. Retrain the ranker on which ads actually performed, not only public TikTok stats.
3. Avoid near-duplicate hooks. In this run the top 3 are all “King Robert in danger, then match-3 rescue.”
4. Retry Gemini/Veo on 503s and process videos in parallel instead of one-by-one.
5. Let a human approve the brief before Veo spends money.
6. Tighten brand-safety and licensing checks before anything is generated.
7. Generate King Robert from approved reference images, not only a text description.
