"""Run one resumable training stage and record its timing/artifacts."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

RUNNER = Path("playground/open_duck_mini_v2/runner.py")


def latest_artifacts(output_dir: Path) -> tuple[Path, Path]:
    checkpoints = sorted(
        (path for path in output_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    exports = sorted(output_dir.glob("*.onnx"), key=lambda path: path.stat().st_mtime)
    if not checkpoints or not exports:
        raise RuntimeError("Training completed without both a checkpoint and ONNX export")
    return checkpoints[-1], exports[-1]


def checkpoint_step(checkpoint: Path) -> int | None:
    try:
        return int(checkpoint.name.rsplit("_", 1)[-1])
    except ValueError:
        return None


def reusable_stage_result(
    result_path: Path,
    *,
    name: str,
    steps: int,
    randomization_stage: str,
    seed: int,
    restore: Path | None,
    imitation_reward_weight_scale: float,
    com_offset_scale: float,
) -> dict | None:
    if not result_path.exists():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    expected = {
        "name": name,
        "status": "complete",
        "steps_added": steps,
        "randomization_stage": randomization_stage,
        "seed": seed,
        "restore": str(restore.resolve()) if restore else None,
        "imitation_reward_weight_scale": imitation_reward_weight_scale,
        "com_offset_scale": com_offset_scale,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        return None

    checkpoint_value = result.get("checkpoint")
    onnx_value = result.get("onnx")
    if not isinstance(checkpoint_value, str) or not isinstance(onnx_value, str):
        return None
    checkpoint = Path(checkpoint_value)
    onnx = Path(onnx_value)
    output_dir = result_path.parent.resolve()
    try:
        checkpoint.resolve().relative_to(output_dir)
        onnx.resolve().relative_to(output_dir)
    except ValueError:
        return None
    if not checkpoint.is_dir() or not onnx.is_file():
        return None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True, help="Steps added by this invocation")
    parser.add_argument("--randomization-stage", choices=("nominal", "moderate", "full"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--imitation-reward-weight-scale", type=float, default=1.0)
    parser.add_argument("--com-offset-scale", type=float, choices=(0.5, 1.0), default=1.0)
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.restore and not args.restore.exists():
        raise FileNotFoundError(args.restore)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failure_path = args.output_dir / "stage_failure.json"
    result_path = args.output_dir / "stage_result.json"
    completed = reusable_stage_result(
        result_path,
        name=args.name,
        steps=args.steps,
        randomization_stage=args.randomization_stage,
        seed=args.seed,
        restore=args.restore,
        imitation_reward_weight_scale=args.imitation_reward_weight_scale,
        com_offset_scale=args.com_offset_scale,
    )
    if completed is not None:
        failure_path.unlink(missing_ok=True)
        print(f"Reusing completed stage: {args.name}")
        print(json.dumps(completed, indent=2))
        return

    command = [
        sys.executable,
        str(RUNNER),
        "--output_dir", str(args.output_dir.resolve()),
        "--num_timesteps", str(args.steps),
        "--env", "joystick",
        "--task", "flat_terrain_backlash",
        "--seed", str(args.seed),
        "--randomization_stage", args.randomization_stage,
        "--imitation_reward_weight_scale", str(args.imitation_reward_weight_scale),
        "--com_offset_scale", str(args.com_offset_scale),
    ]
    if args.restore:
        command.extend(["--restore_checkpoint_path", str(args.restore.resolve())])

    started = time.perf_counter()
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        failure = {
            "name": args.name,
            "status": "failed",
            "recovery": (
                "Resume the last good checkpoint with COM offsets halved; "
                "if failure persists, return to moderate mass ranges before changing rewards."
            ),
        }
        failure_path.write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8"
        )
        raise
    elapsed = time.perf_counter() - started
    checkpoint, onnx = latest_artifacts(args.output_dir)
    failure_path.unlink(missing_ok=True)
    result = {
        "name": args.name,
        "status": "complete",
        "steps_added": args.steps,
        "elapsed_seconds": elapsed,
        "randomization_stage": args.randomization_stage,
        "seed": args.seed,
        "restore": str(args.restore.resolve()) if args.restore else None,
        "checkpoint": str(checkpoint.resolve()),
        "onnx": str(onnx.resolve()),
        "imitation_reward_weight_scale": args.imitation_reward_weight_scale,
        "com_offset_scale": args.com_offset_scale,
        "final_training_step": checkpoint_step(checkpoint),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
