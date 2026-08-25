"""Check objective retention and the five-trait blind-review result."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neutral-report", type=Path, required=True)
    parser.add_argument("--style-report", type=Path, required=True)
    parser.add_argument("--review-form", type=Path, required=True)
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    neutral = json.loads(args.neutral_report.read_text(encoding="utf-8"))
    style = json.loads(args.style_report.read_text(encoding="utf-8"))
    review = json.loads(args.review_form.read_text(encoding="utf-8"))
    key = json.loads(args.answer_key.read_text(encoding="utf-8"))["mapping"]
    style_label = next(label for label, policy in key.items() if policy == "style")
    wins = sum(answer == style_label for answer in review["answers"].values())

    neutral_survival = sum(cell["survived"] for cell in neutral["cells"])
    style_survival = sum(cell["survived"] for cell in style["cells"])
    survival_retention = style_survival / max(neutral_survival, 1)
    neutral_error = sum(cell["mean_relative_tracking_error"] for cell in neutral["cells"]) / 9
    style_error = sum(cell["mean_relative_tracking_error"] for cell in style["cells"]) / 9
    neutral_tracking_score = max(0.0, 1.0 - neutral_error)
    style_tracking_score = max(0.0, 1.0 - style_error)
    tracking_retention = style_tracking_score / max(neutral_tracking_score, 1e-9)
    result = {
        "survival_retention": survival_retention,
        "tracking_retention": tracking_retention,
        "style_trait_wins": wins,
        "pass": survival_retention >= 0.90 and tracking_retention >= 0.90 and wins >= 3,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit("Style acceptance failed")


if __name__ == "__main__":
    main()
