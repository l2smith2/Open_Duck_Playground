"""Report how far the training pipeline has got, and what to run next.

Sessions get interrupted, restarted, and restored, and the notebook's Python
state does not survive any of that even when the files do. This inspects the
artifact directory on disk (the only thing that persists) and reports which
stages are genuinely complete, so the next action never has to be inferred
from memory of what was run before.
"""

import argparse
import json
from pathlib import Path

STAGES = (
    ("00_smoke_1m", "smoke test", 1_000_000),
    ("01_neutral_nominal_20m", "timed benchmark", 20_000_000),
    ("02_neutral_moderate_60m", "moderate randomization", 60_000_000),
    ("03_neutral_full_220m", "full randomization", 220_000_000),
    ("04_style_seed_201_30m", "style seed 201", 30_000_000),
    ("04_style_seed_202_30m", "style seed 202", 30_000_000),
    ("04_style_seed_203_30m", "style seed 203", 30_000_000),
    ("05_style_winner_additional_120m", "style winner extension", 120_000_000),
)


def stage_state(artifacts: Path, name: str, expected_steps: int) -> tuple[str, str]:
    stage_dir = artifacts / name
    if not stage_dir.is_dir():
        return "missing", ""
    result_path = stage_dir / "stage_result.json"
    if not result_path.is_file():
        if (stage_dir / "stage_failure.json").is_file():
            return "FAILED", "stage_failure.json present"
        return "partial", "no stage_result.json"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return "partial", f"unreadable stage_result.json ({exc})"
    if result.get("status") != "complete":
        return "partial", f"status={result.get('status')!r}"
    if result.get("steps_added") != expected_steps:
        return "partial", f"steps_added={result.get('steps_added')}"
    # stage_result.json records the absolute path the stage was trained at.
    # That resolves on Kaggle, but not for a bundle downloaded or restored
    # somewhere else, so fall back to looking the files up by name inside this
    # stage directory before calling the stage incomplete.
    checkpoint = Path(str(result.get("checkpoint", "")))
    onnx = Path(str(result.get("onnx", "")))
    if not checkpoint.is_dir():
        checkpoint = stage_dir / checkpoint.name
    if not onnx.is_file():
        onnx = stage_dir / onnx.name
    if not checkpoint.is_dir() or not onnx.is_file():
        return "partial", "checkpoint or ONNX missing on disk"
    return "complete", f"{result.get('elapsed_seconds', 0) / 60:.0f} min"


def installed_entries(repo_root: Path) -> int | None:
    """Number of command entries in the reference currently installed for training."""
    installed = repo_root / "playground" / "open_duck_mini_v2" / "data" / "polynomial_coefficients.pkl"
    if not installed.is_file():
        return None
    try:
        import pickle

        with open(installed, "rb") as handle:
            return len(pickle.load(handle))
    except Exception:  # noqa: BLE001 - a status check must not crash
        return None


def reference_state(artifacts: Path, repo_root: Path) -> list[str]:
    lines = []
    ref = artifacts / "bdx_reference"
    recordings = sorted((ref / "recordings").glob("*.json")) if (ref / "recordings").is_dir() else []
    lines.append(f"  recordings generated : {len(recordings)}/8")
    approved = (ref / "reference_review_approved.json").is_file()
    lines.append(f"  human review approved: {'yes' if approved else 'no'}")
    fitted = ref / "polynomial_coefficients.pkl"
    lines.append(f"  fitted coefficients  : {'yes' if fitted.is_file() else 'no'}")

    entries = installed_entries(repo_root)
    if entries is None:
        lines.append("  installed reference  : MISSING or unreadable")
    else:
        kind = "fitted style reference" if entries == 8 else "stock reference"
        lines.append(f"  installed reference  : {entries} entries ({kind})")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("/kaggle/working/artifacts"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    artifacts = args.artifacts
    print(f"Artifacts: {artifacts}")
    if not artifacts.is_dir():
        print("  (directory does not exist yet)")
        print("\nNext: run Setup, then section 1.")
        return

    print("\nTraining stages")
    states = {}
    for name, label, steps in STAGES:
        state, detail = stage_state(artifacts, name, steps)
        states[name] = state
        mark = {"complete": "[x]", "partial": "[~]", "FAILED": "[!]", "missing": "[ ]"}[state]
        suffix = f"  ({detail})" if detail else ""
        print(f"  {mark} {label:<26}{suffix}")

    print("\nBDX-inspired reference")
    for line in reference_state(artifacts, args.repo_root):
        print(line)

    print("\nNext")
    neutral = ["00_smoke_1m", "01_neutral_nominal_20m", "02_neutral_moderate_60m", "03_neutral_full_220m"]
    if any(states[n] != "complete" for n in neutral):
        first = next(n for n in neutral if states[n] != "complete")
        print(f"  Finish the neutral curriculum (next incomplete stage: {first}).")
        return
    ref = artifacts / "bdx_reference"
    if not (ref / "reference_review_approved.json").is_file():
        print("  Generate the reference, replay it locally, then approve and fit it (section 2).")
        return
    if not (ref / "polynomial_coefficients.pkl").is_file():
        print("  Reference approved but not fitted: run the fit + install lines (section 2).")
        return
    if installed_entries(args.repo_root) != 8:
        print("  Reference is fitted but NOT installed for training.")
        print("  Style training would imitate the stock reference instead. Run:")
        print("    uv run python scripts/install_reference_motion.py \\")
        print(f"      --source {ref / 'polynomial_coefficients.pkl'} --expect-entries 8")
        return
    seeds = ["04_style_seed_201_30m", "04_style_seed_202_30m", "04_style_seed_203_30m"]
    if any(states[s] != "complete" for s in seeds):
        print("  Run the three 30M style seeds (section 3, first cell).")
        return
    if states["05_style_winner_additional_120m"] != "complete":
        print("  Evaluate the seeds, pick a winner, then run the winner extension (section 3, second cell).")
        return
    print("  All stages complete: run acceptance (section 4).")


if __name__ == "__main__":
    main()
