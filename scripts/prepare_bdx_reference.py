"""Generate, review-gate, and fit an original BDX-inspired reference motion."""

import argparse
import json
import pickle
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


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


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
    # gait_generator.py overwrites args.dx/dy/dtheta from its own preset JSON
    # right after argparse runs, and never reads args.walk_com_height/etc again
    # after parsing them, so the equivalent --dx/--walk_com_height/etc CLI flags
    # are silently ignored. Write a preset file per motion instead, since that
    # loaded file is the only input this generator version actually applies.
    base_preset_path = (
        generator_root / "open_duck_reference_motion_generator" / "robots"
        / "open_duck_mini_v2" / "placo_defaults.json"
    )
    base_preset = json.loads(base_preset_path.read_text())
    presets_dir = artifact_dir / "presets"
    presets_dir.mkdir(parents=True, exist_ok=True)
    motion_grid_record = {}
    for name, dx, dy, dtheta in MOTION_GRID:
        preset = dict(base_preset)
        preset.update({
            "dx": dx,
            "dy": dy,
            "dtheta": dtheta,
            "walk_com_height": params["walk_com_height"],
            "walk_foot_height": params["walk_foot_height"],
            "walk_trunk_pitch": params["walk_trunk_pitch"],
            "single_support_duration": params["single_support_duration"],
            "feet_spacing": params["feet_spacing"],
        })
        preset_path = presets_dir / f"bdx_inspired_{name}.json"
        preset_path.write_text(json.dumps(preset, indent=2) + "\n", encoding="utf-8")
        command = [
            "uv", "run", str(script), "--duck", "open_duck_mini_v2",
            "--output_dir", str(recordings), "--length", "10",
            "--preset", str(preset_path), "--name", f"bdx_inspired_{name}",
        ]
        subprocess.run(command, cwd=generator_root, check=True)
        # "forward" is a prefix of "forward_slow", so a plain glob wildcard
        # would match both recordings. Require the part after the name to be
        # exactly three numeric velocity tokens (nothing else, no more name).
        prefix = f"bdx_inspired_{name}_"
        matches = []
        for candidate in recordings.glob(f"{prefix}*.json"):
            suffix = candidate.name[len(prefix):].removesuffix(".json")
            parts = suffix.split("_")
            if len(parts) == 3 and all(_is_float(part) for part in parts):
                matches.append(candidate)
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one recording for {name!r}; found {[m.name for m in matches]}"
            )
        motion_grid_record[matches[0].name] = {"dx": dx, "dy": dy, "dtheta": dtheta}
    (artifact_dir / "motion_grid.json").write_text(
        json.dumps(motion_grid_record, indent=2) + "\n", encoding="utf-8"
    )
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


def _fit_poly_mangled_key(recording_filename: str) -> str:
    """Replicate fit_poly.py's key derivation exactly (including its bug).

    fit_poly.py assumes filenames look like PREFIX_dx_dy_dtheta.json (one
    prefix token) and blindly takes tmp[1]_tmp[2]_tmp[3]. Our filenames are
    bdx_inspired_{name}_{dx}_{dy}_{dtheta}.json, where {name} is often
    multi-word (forward_slow, turn_left), so that fixed-index slice does not
    land on the velocity values. This function reproduces exactly what key
    fit_poly.py will have written for a given recording filename, so it can
    be mapped back to the correct dx_dy_dtheta key afterwards.
    """
    stem = recording_filename.strip(".json")
    tmp = stem.split("_")
    return f"{tmp[1]}_{tmp[2]}_{tmp[3]}"


def fit(generator_root: Path, artifact_dir: Path, playground_data: Path | None) -> None:
    approval = artifact_dir / "reference_review_approved.json"
    if not approval.is_file():
        raise RuntimeError("Inspect and approve the generated reference before fitting it")
    motion_grid_path = artifact_dir / "motion_grid.json"
    if not motion_grid_path.is_file():
        raise RuntimeError("motion_grid.json missing; re-run generate before fitting")
    motion_grid = json.loads(motion_grid_path.read_text(encoding="utf-8"))
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

    with open(generated, "rb") as handle:
        coefficients = pickle.load(handle)

    # fit_poly.py derives its own dict keys from filenames using a fixed
    # token-index assumption that breaks on our multi-word motion names
    # (see _fit_poly_mangled_key). Rewrite every key to the dx_dy_dtheta
    # format playground/common/poly_reference_motion_numpy.py actually
    # requires, using motion_grid.json (recorded at generation time) as the
    # source of truth instead of re-parsing filenames.
    remapped = {}
    for filename, velocities in motion_grid.items():
        mangled_key = _fit_poly_mangled_key(filename)
        if mangled_key not in coefficients:
            raise RuntimeError(
                f"fit_poly.py output missing expected key {mangled_key!r} for {filename!r}; "
                f"available keys: {sorted(coefficients.keys())}"
            )
        correct_key = f"{velocities['dx']}_{velocities['dy']}_{velocities['dtheta']}"
        remapped[correct_key] = coefficients.pop(mangled_key)
    if coefficients:
        raise RuntimeError(f"Unmapped keys left in fit_poly.py output: {sorted(coefficients.keys())}")

    artifact_copy = artifact_dir / "polynomial_coefficients.pkl"
    with open(artifact_copy, "wb") as handle:
        pickle.dump(remapped, handle)
    if playground_data:
        playground_data.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact_copy, playground_data / artifact_copy.name)
    print(f"Fitted coefficients saved to {artifact_copy}")
    print(f"Remapped {len(remapped)} motions to dx_dy_dtheta keys: {sorted(remapped.keys())}")


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
