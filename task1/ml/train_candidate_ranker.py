from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from task1.ml.candidate_ranker_training import train_candidate_ranker


def main() -> None:
    parser = argparse.ArgumentParser(description="Train metadata candidate ranker")
    parser.add_argument(
        "--dataset",
        default="task1/data/training/tiktok_engagement_crawlora.jsonl",
        type=str,
    )
    parser.add_argument(
        "--model-output",
        default="task1/artifacts/models/candidate_ranker.joblib",
        type=str,
    )
    parser.add_argument(
        "--summary-output",
        default="task1/artifacts/models/training_summary.json",
        type=str,
    )
    args = parser.parse_args()

    summary = train_candidate_ranker(
        dataset_path=args.dataset,
        model_output_path=args.model_output,
        summary_output_path=args.summary_output,
    )
    print("training_summary_start")
    print(json.dumps(summary, ensure_ascii=False))
    print("training_summary_end")


if __name__ == "__main__":
    main()
