"""Headless ONNX push-recovery evaluation, standing and walking.

Training applies a disturbance every 5-10 s as an instantaneous change to the
floating base's linear x/y velocity, magnitude 0.1-1.0 m/s, random horizontal
direction (see push_config in playground/open_duck_mini_v2/joystick.py). This
reproduces exactly that disturbance model and sweeps magnitude past the trained
range, so the report says where recovery actually breaks down rather than
assuming the trained ceiling holds.

Only horizontal linear pushes are measured, because that is the only
disturbance the policy is trained against: rotational and toppling
disturbances are out of distribution and are deliberately not scored here.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from playground.open_duck_mini_v2.mujoco_infer import MjInfer, USE_MOTOR_SPEED_LIMITS

# Trained range is 0.1-1.0 m/s; the sweep runs past it to find the real ceiling.
DEFAULT_MAGNITUDES = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)
TRAINED_MAGNITUDE_MAX = 1.0


class PushRecoveryEvaluator:
    def __init__(self, args):
        self.args = args
        self.sim = MjInfer(
            str(args.model_path), str(args.reference_data), str(args.onnx), standing=False
        )
        self.model = self.sim.model
        self.data = self.sim.data
        # .adr is a length-1 array on this MuJoCo version, not a scalar.
        self.upvector_adr = int(np.asarray(self.model.sensor("upvector").adr).reshape(-1)[0])
        self.qvel_addr = self.sim._floating_base_qvel_addr
        self.qpos_addr = self.sim._floating_base_qpos_addr

    def _reset(self, commands) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.keyframe("home").qpos
        self.data.ctrl[:] = self.sim.default_actuator
        self.sim.last_action[:] = 0
        self.sim.last_last_action[:] = 0
        self.sim.last_last_last_action[:] = 0
        self.sim.motor_targets = self.sim.default_actuator.copy()
        self.sim.prev_motor_targets = self.sim.default_actuator.copy()
        self.sim.commands = list(commands)
        self.sim.imitation_i = 0
        self.sim.imitation_phase = np.array([0.0, 0.0])
        mujoco.mj_forward(self.model, self.data)

    def _control_step(self) -> None:
        self.sim.imitation_i = (
            self.sim.imitation_i + self.sim.phase_frequency_factor
        ) % self.sim.PRM.nb_steps_in_period
        phase = self.sim.imitation_i / self.sim.PRM.nb_steps_in_period * 2 * np.pi
        self.sim.imitation_phase = np.array([np.cos(phase), np.sin(phase)])
        obs = self.sim.get_obs(self.data, self.sim.commands)
        action = self.sim.policy.infer(obs)
        self.sim.last_last_last_action = self.sim.last_last_action.copy()
        self.sim.last_last_action = self.sim.last_action.copy()
        self.sim.last_action = action.copy()
        self.sim.motor_targets = self.sim.default_actuator + action * self.sim.action_scale
        if USE_MOTOR_SPEED_LIMITS:
            delta = self.sim.max_motor_velocity * (self.sim.sim_dt * self.sim.decimation)
            self.sim.motor_targets = np.clip(
                self.sim.motor_targets,
                self.sim.prev_motor_targets - delta,
                self.sim.prev_motor_targets + delta,
            )
            self.sim.prev_motor_targets = self.sim.motor_targets.copy()
        self.data.ctrl[:] = self.sim.motor_targets

    def _upright_and_finite(self) -> bool:
        finite = np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))
        upright = float(self.data.sensordata[self.upvector_adr + 2]) >= 0.0
        return bool(finite and upright)

    def _run_window(self, seconds: float) -> tuple[bool, float]:
        """Step for `seconds`, stopping early on a fall. Returns (survived, elapsed)."""
        steps = int(seconds / self.sim.sim_dt)
        for step in range(steps):
            mujoco.mj_step(self.model, self.data)
            if step % self.sim.decimation == 0:
                self._control_step()
            if not self._upright_and_finite():
                return False, (step + 1) * self.sim.sim_dt
        return True, steps * self.sim.sim_dt

    def run_episode(self, commands, magnitude: float, theta: float) -> dict:
        self._reset(commands)
        settled, settle_elapsed = self._run_window(self.args.settle_seconds)
        if not settled:
            # It fell before being pushed, so this episode says nothing about
            # push recovery. Reported separately rather than counted as a fall.
            return {
                "magnitude": magnitude,
                "push_heading_rad": theta,
                "fell_before_push": True,
                "survived": False,
                "recovery_seconds": 0.0,
                "settle_seconds": settle_elapsed,
                "displacement_m": None,
            }

        pre_push_xy = self.data.qpos[self.qpos_addr:self.qpos_addr + 2].copy()
        # Same disturbance model as training: an instantaneous delta on the
        # floating base's linear x/y velocity, not a force and not angular.
        push = np.array([np.cos(theta), np.sin(theta)]) * magnitude
        self.data.qvel[self.qvel_addr:self.qvel_addr + 2] += push

        survived, recovery_elapsed = self._run_window(self.args.recovery_seconds)
        post_push_xy = self.data.qpos[self.qpos_addr:self.qpos_addr + 2]
        displacement = float(np.linalg.norm(post_push_xy - pre_push_xy))
        return {
            "magnitude": magnitude,
            "push_heading_rad": float(theta),
            "fell_before_push": False,
            "survived": survived,
            "recovery_seconds": recovery_elapsed,
            "settle_seconds": settle_elapsed,
            "displacement_m": displacement,
        }

    def evaluate_condition(self, name: str, commands) -> dict:
        rng = np.random.default_rng(self.args.seed)
        levels = []
        for magnitude in self.args.magnitudes:
            episodes = []
            for _ in range(self.args.episodes):
                theta = float(rng.uniform(0.0, 2 * np.pi))
                episodes.append(self.run_episode(commands, magnitude, theta))
            pushed = [episode for episode in episodes if not episode["fell_before_push"]]
            survivors = [episode for episode in pushed if episode["survived"]]
            levels.append(
                {
                    "magnitude": magnitude,
                    "within_trained_range": magnitude <= TRAINED_MAGNITUDE_MAX,
                    "episodes": self.args.episodes,
                    "pushed": len(pushed),
                    "fell_before_push": len(episodes) - len(pushed),
                    "survived": len(survivors),
                    "survival_rate": (len(survivors) / len(pushed)) if pushed else None,
                    "mean_displacement_m": (
                        float(np.mean([episode["displacement_m"] for episode in survivors]))
                        if survivors else None
                    ),
                    "episode_results": episodes,
                }
            )
        return {"condition": name, "commands": list(commands), "levels": levels}


def summarize(condition: dict) -> dict:
    """Highest fully-survived magnitude, and where survival first drops below half."""
    levels = condition["levels"]
    full = [
        level["magnitude"] for level in levels
        if level["survival_rate"] is not None and level["survival_rate"] >= 1.0
    ]
    half = [
        level["magnitude"] for level in levels
        if level["survival_rate"] is not None and level["survival_rate"] < 0.5
    ]
    return {
        "condition": condition["condition"],
        "max_magnitude_all_survived": max(full) if full else None,
        "first_magnitude_below_half_survival": min(half) if half else None,
        "survival_by_magnitude": {
            str(level["magnitude"]): level["survival_rate"] for level in levels
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"),
    )
    parser.add_argument(
        "--reference-data",
        type=Path,
        default=Path("playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"),
    )
    parser.add_argument("--episodes", type=int, default=12, help="episodes per magnitude")
    parser.add_argument(
        "--magnitudes", type=float, nargs="+", default=list(DEFAULT_MAGNITUDES),
        help="push magnitudes in m/s; training used 0.1-1.0",
    )
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    parser.add_argument("--recovery-seconds", type=float, default=4.0)
    parser.add_argument("--command-x", type=float, default=0.10, help="walking condition speed")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--require-magnitude", type=float,
        help="fail if any condition does not fully survive this magnitude",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if args.settle_seconds <= 0 or args.recovery_seconds <= 0:
        raise ValueError("settle/recovery seconds must be positive")
    if not args.magnitudes or any(magnitude <= 0 for magnitude in args.magnitudes):
        raise ValueError("magnitudes must be positive")
    args.magnitudes = sorted(args.magnitudes)

    evaluator = PushRecoveryEvaluator(args)
    conditions = [
        evaluator.evaluate_condition("standing", [0.0] * 7),
        evaluator.evaluate_condition("walking", [args.command_x] + [0.0] * 6),
    ]
    summaries = [summarize(condition) for condition in conditions]
    report = {
        "onnx": str(args.onnx),
        "disturbance_model": "instantaneous floating-base linear x/y velocity delta",
        "trained_magnitude_range_m_s": [0.1, TRAINED_MAGNITUDE_MAX],
        "episodes_per_magnitude": args.episodes,
        "settle_seconds": args.settle_seconds,
        "recovery_seconds": args.recovery_seconds,
        "seed": args.seed,
        "summary": summaries,
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))

    if args.require_magnitude is not None:
        failed = [
            summary["condition"] for summary in summaries
            if summary["max_magnitude_all_survived"] is None
            or summary["max_magnitude_all_survived"] < args.require_magnitude
        ]
        if failed:
            raise SystemExit(
                f"Push recovery below {args.require_magnitude} m/s for: {', '.join(failed)}"
            )


if __name__ == "__main__":
    main()
