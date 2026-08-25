"""Generate, review-gate, and fit an original BDX-inspired reference motion."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

STYLE_CONFIG = Path(__file__).parents[1] / "configs" / "bdx_inspired_reference.json"
MOTION_GRID = (
    ("stand", 0.0, 0.0, 0.0),
    ("forward_slow", 0.02, 0.0, 0.0),
    ("forward", 0.04, 0.0, 0.0),
    ("backward", -0.02, 0.0, 0.0),
    ("left", 0.0, 0.02, 0.0),
    ("right", 0.0, -0.02, 0.0),
    ("turn_left", 0.0, 0.0, 0.15),
    ("turn_right", 0.0, 0.0, -0.15),
)


def load_style() -> dict:
    return json.loads(STYLE_CONFIG.read_text(encoding="utf-8"))


def generator_script(generator_root: Path) -> Path:
    script = generator_root / "open_duck_reference_motion_generator" / "gait_generator.py"
    if not script.is_file():
        raise FileNotFoundError(f"Generator script not found: {script}")
    return script


def generate(generator_root: Path, artifact_dir: Path) -> None:
    script = generator_script(generator_root)
    params = load_style()["parameters"]
    recordings = artifact_dir / "recordings"
    recordings.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "style_config.json").write_text(
        json.dumps(load_style(), indent=2) + "\n", encoding="utf-8"
    )
    common = [
        "uv", "run", str(script), "--duck", "open_duck_mini_v2",
        "--output_dir", str(recordings), "--length", "10",
        "--walk_com_height", str(params["walk_com_height"]),
        "--walk_foot_height", str(params["walk_foot_height"]),
        "--walk_trunk_pitch", str(params["walk_trunk_pitch"]),
        "--single_support_duration", str(params["single_support_duration"]),
        "--feet_spacing", str(params["feet_spacing"]),
    ]
    for name, dx, dy, dtheta in MOTION_GRID:
        command = common + [
            "--name", f"bdx_inspired_{name}",
            "--dx", str(dx), "--dy", str(dy), "--dtheta", str(dtheta),
        ]
        subprocess.run(command, cwd=generator_root, check=True)
    print(f"Generated {len(MOTION_GRID)} original motions in {recordings}")
    print("Replay/inspect them, then run this script with the approve command.")


def approve(artifact_dir: Path, review_note: str) -> None:
    recordings = sorted((artifact_dir / "recordings").glob("*.json"))
    if len(recordings) != len(MOTION_GRID):
        raise RuntimeError(
            f"Expected {len(MOTION_GRID)} generated motions; found {len(recordings)}"
        )
    if len(review_note.strip()) < 8:
        raise ValueError("Give a short review note describing what you inspected")
    marker = {
        "approved": True,
        "review_note": review_note.strip(),
        "files": [path.name for path in recordings],
    }
    (artifact_dir / "reference_review_approved.json").write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8"
    )
    print("Review approval recorded. The reference is now eligible for fitting.")


def fit(generator_root: Path, artifact_dir: Path, playground_data: Path | None) -> None:
    approval = artifact_dir / "reference_review_approved.json"
    if not approval.is_file():
        raise RuntimeError("Inspect and approve the generated reference before fitting it")
    fit_script = generator_root / "scripts" / "fit_poly.py"
    if not fit_script.is_file():
        raise FileNotFoundError(f"Fit script not found: {fit_script}")
    subprocess.run(
        ["uv", "run", str(fit_script), "--ref_motion", str(artifact_dir / "recordings")],
        cwd=generator_root,
        check=True,
    )
    generated = generator_root / "polynomial_coefficients.pkl"
    if not generated.is_file():
        raise RuntimeError("fit_poly.py did not produce polynomial_coefficients.pkl")
    artifact_copy = artifact_dir / "polynomial_coefficients.pkl"
    shutil.copy2(generated, artifact_copy)
    if playground_data:
        playground_data.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact_copy, playground_data / artifact_copy.name)
    print(f"Fitted coefficients saved to {artifact_copy}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "approve", "fit"))
    parser.add_argument("--generator-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--review-note", default="")
    parser.add_argument("--playground-data", type=Path)
    args = parser.parse_args()
    generator_root = args.generator_root.resolve()
    artifact_dir = args.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "generate":
        generate(generator_root, artifact_dir)
    elif args.command == "approve":
        approve(artifact_dir, args.review_note)
    else:
        fit(generator_root, artifact_dir, args.playground_data)


if __name__ == "__main__":
    main()
