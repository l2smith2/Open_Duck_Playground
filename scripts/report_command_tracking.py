"""Report achieved forward speed per training stage, against a set of commands.

Neither speed nor the reward's per-step margin is visible from training logs,
and re-deriving "did this checkpoint actually get faster" from memory across a
session is exactly the kind of thing pipeline_status.py exists to avoid doing
for stage completion. This is the same idea for stage quality: it replays each
completed stage's latest checkpoint against a reference at a few commands and
reports measured root-frame speed, no reward weights or a second (marching)
policy required.

Uses the project's own mujoco/onnx stack, so run it the same way every other
script in this repo is run -- through uv, not the notebook kernel's Python:

    uv run python scripts/report_command_tracking.py --artifacts /kaggle/working/artifacts

Compare the printed table against a prior run's numbers (for example, the
figures recorded in AGENTS.md for the old reward) to see whether a reward or
reference change actually moved the achieved speed, not just the offline
incentive margin.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_reward_locomotion_incentive import rollout  # noqa: E402
from pipeline_status import STAGES  # noqa: E402

DEFAULT_REFERENCE = Path("playground/open_duck_mini_v2/data/polynomial_coefficients.pkl")
DEFAULT_MODEL = Path("playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml")
DEFAULT_COMMANDS = (0.05, 0.10, 0.15)


def latest_onnx(stage_dir: Path) -> Path | None:
    exports = sorted(
        stage_dir.glob("*.onnx"), key=lambda path: int(path.stem.rsplit("_", 1)[-1])
    )
    return exports[-1] if exports else None


def discover_stages(artifacts: Path) -> list[tuple[str, Path]]:
    """Every completed stage under artifacts, in pipeline_status.py's own order.

    Reusing STAGES rather than globbing artifacts directly keeps the stage list
    and its order a single source of truth shared with the status report.
    """
    found = []
    for name, _label, _steps in STAGES:
        stage_dir = artifacts / name
        onnx = latest_onnx(stage_dir) if stage_dir.is_dir() else None
        if onnx is not None:
            found.append((name, onnx))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--commands", type=float, nargs="+", default=list(DEFAULT_COMMANDS))
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--max-foot-height", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    stages = discover_stages(args.artifacts)
    if not stages:
        raise SystemExit(f"No completed stage with an ONNX export found under {args.artifacts}")

    report = {"reference": str(args.reference), "commands": args.commands, "stages": {}}
    header = f"{'stage':<32}" + "".join(f"{'cmd ' + format(c, '.2f'):>10}" for c in args.commands)
    print(header)
    for name, onnx in stages:
        speeds = [
            rollout(onnx, args.reference, args.model_path, cmd, args.seconds, args.max_foot_height)[
                "measured_speed_x"
            ]
            for cmd in args.commands
        ]
        report["stages"][name] = {"onnx": str(onnx), "measured_speed_x": speeds}
        print(f"{name:<32}" + "".join(f"{s:10.4f}" for s in speeds))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
