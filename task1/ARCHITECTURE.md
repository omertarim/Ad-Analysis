# Task 1 Architecture

## Why TikTok + Apify

- TikTok is the primary short-form trend source aligned with the case objective.
- Public TikTok data access is unstable from direct scraping endpoints, so Apify is used
  as the practical integration layer for repeatable collection and downloader workflows.
- Apify returns enough metadata (views, likes, comments, shares, saves, creator signals)
  to support a low-cost pre-ranking stage before expensive multimodal inference.

## Why Random Forest + Gemini

- Random Forest powers metadata-only candidate prioritization.
- It is cheap, fast, and used only to rank/filter which trends are worth multimodal spend.
- Gemini is used for video-native understanding and creative adaptability reasoning, where
  semantic interpretation is needed and deterministic code is insufficient.

## Role Split: ML Ranker vs Gemini

- CandidateRanker does not make final adaptability decisions.
- CandidateRanker outputs probability/rank and selection flags for multimodal routing.
- MultimodalAnalyzer + CreativeEvaluator produce structured content analysis and creative
  scoring signals from actual video content.

## Deterministic Scoring in Application Layer

- Final `overall_score` and `adaptable` are computed in Python, not delegated to model output.
- This preserves traceability, prevents model drift from changing business rules, and avoids
  leaking policy decisions into untrusted free-form inference.

## Artifact and Traceability Design

For selected trends the chain is:

`source_url -> trend_metadata.json -> ranking.json -> source.mp4 -> analysis.json -> evaluation.json`

Rejected-by-ranker trends keep:

`source_url -> trend_metadata.json -> ranking.json`

All artifacts are trend-id keyed under `task1/artifacts/`.

## Weekly Production Flow

1. Discover fresh TikTok trend metadata (region-scoped batch).
2. Persist normalized metadata per trend id.
3. Rank candidates with ML (`top_percent` default) and persist ranking artifacts.
4. Process only selected candidates:
   - acquire video
   - technical validate
   - multimodal analysis
   - creative evaluation with deterministic final scoring
5. Build summary artifacts for adaptable and rejected sets.

## Cache and Resume Strategy

- Trend id is deterministic from source and source URL.
- Existing artifacts are reused:
  - existing metadata/ranking is not recomputed unnecessarily
  - existing `source.mp4` is not redownloaded
  - existing analysis/evaluation artifacts are not rerun
- Pipeline resumes from missing stage for each trend.

## Known Limitations

- TikTok trend distributions shift quickly, causing ranker-domain drift.
- Candidate selection is ranking-oriented, not a final quality guarantee.
- Multimodal latency and provider variability can still dominate runtime.
- Some videos can fail acquisition due to source/dependency availability changes.
