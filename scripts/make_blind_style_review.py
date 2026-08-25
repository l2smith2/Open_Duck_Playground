"""Create randomized A/B review packs for the five BDX-inspired traits."""

import argparse
import json
import random
import shutil
from pathlib import Path

TRAITS = (
    "waddle",
    "slight crouch",
    "light bounce",
    "deliberate foot lift",
    "stable upper-body timing",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neutral-video", type=Path, required=True)
    parser.add_argument("--style-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    for video in (args.neutral_video, args.style_video):
        if not video.is_file():
            raise FileNotFoundError(video)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = [
        ("neutral", args.neutral_video),
        ("style", args.style_video),
    ]
    random.Random(args.seed).shuffle(sources)
    key = {}
    for label, (policy, source) in zip(("A", "B"), sources, strict=True):
        destination = args.output_dir / f"clip_{label}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        key[label] = policy

    review_form = {
        "instructions": "For each trait choose A, B, tie, or neither. Do not inspect answer_key.json.",
        "traits": list(TRAITS),
        "answers": {trait: "" for trait in TRAITS},
        "style_improvement_requires": "the style policy wins at least 3 of 5 traits",
    }
    (args.output_dir / "review_form.json").write_text(
        json.dumps(review_form, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "answer_key.json").write_text(
        json.dumps({"seed": args.seed, "mapping": key}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Blind review pack created at {args.output_dir}")


if __name__ == "__main__":
    main()
