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
        (args.output_dir / "stage_failure.json").write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8"
        )
        raise
    elapsed = time.perf_counter() - started
    checkpoint, onnx = latest_artifacts(args.output_dir)
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
    }
    result_path = args.output_dir / "stage_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
