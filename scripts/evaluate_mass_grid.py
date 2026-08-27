"""Headless ONNX survival/tracking evaluation over the 3x3 mass grid."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from playground.common.randomize import PAIRED_LEG_BODY_NAMES
from playground.open_duck_mini_v2.mujoco_infer import MjInfer, USE_MOTOR_SPEED_LIMITS

TRUNK_GRID = {"light": 0.80, "nominal": 1.0, "heavy": 1.30}
HEAD_GRID = {"light": 0.65, "nominal": 1.0, "heavy": 1.40}
TRUNK_COM_LIMIT = np.array([0.015, 0.010, 0.015])
HEAD_COM_LIMIT = np.array([0.015, 0.010, 0.020])


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = int(model.body(name).id)
    if body_id < 0:
        raise ValueError(f"Required body not found: {name}")
    return body_id


class MassGridEvaluator:
    def __init__(self, args):
        self.args = args
        self.sim = MjInfer(
            str(args.model_path), str(args.reference_data), str(args.onnx), standing=False
        )
        self.model = self.sim.model
        self.data = self.sim.data
        self.nominal_mass = self.model.body_mass.copy()
        self.nominal_inertia = self.model.body_inertia.copy()
        self.nominal_ipos = self.model.body_ipos.copy()
        self.trunk_id = _body_id(self.model, "trunk_assembly")
        self.head_id = _body_id(self.model, "head_assembly")
        self.pairs = tuple(
            (_body_id(self.model, left), _body_id(self.model, right))
            for left, right in PAIRED_LEG_BODY_NAMES
        )
        # .adr is a length-1 array on this MuJoCo version, not a scalar.
        self.upvector_adr = int(np.asarray(self.model.sensor("upvector").adr).reshape(-1)[0])
        if not np.isclose(self.nominal_mass[self.trunk_id], 0.698526, atol=0.002):
            raise ValueError("Unexpected nominal trunk mass")
        if not np.isclose(self.nominal_mass[self.head_id], 0.406607, atol=0.002):
            raise ValueError("Unexpected nominal head mass")

    def _apply_variant(self, trunk_scale: float, head_scale: float, rng) -> None:
        scales = np.ones(self.model.nbody)
        positive = self.nominal_mass > 0
        scales[positive] = rng.uniform(0.95, 1.05, size=np.count_nonzero(positive))
        for left_id, right_id in self.pairs:
            main = rng.uniform(0.97, 1.03)
            mismatch = rng.uniform(0.98, 1.02, size=2)
            scales[[left_id, right_id]] = np.clip(main * mismatch, 0.95, 1.05)
        scales[self.trunk_id] = trunk_scale
        scales[self.head_id] = head_scale

        self.model.body_mass[:] = self.nominal_mass * scales
        self.model.body_inertia[:] = self.nominal_inertia * scales[:, None]
        self.model.body_ipos[:] = self.nominal_ipos
        self.model.body_ipos[self.trunk_id] += rng.uniform(-TRUNK_COM_LIMIT, TRUNK_COM_LIMIT)
        self.model.body_ipos[self.head_id] += rng.uniform(-HEAD_COM_LIMIT, HEAD_COM_LIMIT)
        if np.any(self.model.body_mass[positive] <= 0):
            raise ValueError("A randomized body mass was not positive")
        if np.any(self.model.body_inertia[positive] <= 0):
            raise ValueError("A randomized body inertia was not positive")
        mujoco.mj_setConst(self.model, self.data)

    def _reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.keyframe("home").qpos
        self.data.ctrl[:] = self.sim.default_actuator
        self.sim.last_action[:] = 0
        self.sim.last_last_action[:] = 0
        self.sim.last_last_last_action[:] = 0
        self.sim.motor_targets = self.sim.default_actuator.copy()
        self.sim.prev_motor_targets = self.sim.default_actuator.copy()
        self.sim.commands = [self.args.command_x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.sim.imitation_i = 0
        self.sim.imitation_phase = np.array([0.0, 0.0])
        mujoco.mj_forward(self.model, self.data)

    def run_episode(self, trunk_scale: float, head_scale: float, rng) -> dict:
        self._apply_variant(trunk_scale, head_scale, rng)
        self._reset()
        start_x = float(self.data.qpos[0])
        control_steps = 0
        total_steps = int(self.args.seconds / self.sim.sim_dt)
        survived = True

        for step in range(total_steps):
            mujoco.mj_step(self.model, self.data)
            if step % self.sim.decimation == 0:
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
                    delta = self.sim.max_motor_velocity * (
                        self.sim.sim_dt * self.sim.decimation
                    )
                    self.sim.motor_targets = np.clip(
                        self.sim.motor_targets,
                        self.sim.prev_motor_targets - delta,
                        self.sim.prev_motor_targets + delta,
                    )
                    self.sim.prev_motor_targets = self.sim.motor_targets.copy()
                self.data.ctrl[:] = self.sim.motor_targets
                control_steps += 1

            finite = np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))
            upright = float(self.data.sensordata[self.upvector_adr + 2]) >= 0.0
            if not finite or not upright:
                survived = False
                break

        elapsed = (step + 1) * self.sim.sim_dt
        actual_speed = (float(self.data.qpos[0]) - start_x) / elapsed
        tracking_error = abs(actual_speed - self.args.command_x) / abs(self.args.command_x)
        return {
            "survived": survived,
            "survival_seconds": elapsed,
            "command_x": self.args.command_x,
            "actual_speed_x": actual_speed,
            "relative_tracking_error": tracking_error,
            "control_steps": control_steps,
        }

    def evaluate(self) -> dict:
        rng = np.random.default_rng(self.args.seed)
        cells = []
        for trunk_name, trunk_scale in TRUNK_GRID.items():
            for head_name, head_scale in HEAD_GRID.items():
                episodes = [
                    self.run_episode(trunk_scale, head_scale, rng)
                    for _ in range(self.args.episodes)
                ]
                survivors = [episode for episode in episodes if episode["survived"]]
                survival_count = len(survivors)
                mean_error = (
                    float(np.mean([episode["relative_tracking_error"] for episode in survivors]))
                    if survivors else float("inf")
                )
                corner = trunk_name != "nominal" and head_name != "nominal"
                required_survival = min(16 if corner else 18, self.args.episodes)
                cells.append(
                    {
                        "trunk": trunk_name,
                        "head": head_name,
                        "trunk_scale": trunk_scale,
                        "head_scale": head_scale,
                        "survived": survival_count,
                        "episodes": self.args.episodes,
                        "required_survival": required_survival,
                        "mean_relative_tracking_error": mean_error,
                        "survival_pass": survival_count >= required_survival,
                        "tracking_pass": mean_error <= 0.15,
                        "episode_results": episodes,
                    }
                )
        return {
            "onnx": str(self.args.onnx),
            "seconds_per_episode": self.args.seconds,
            "seed": self.args.seed,
            "cells": cells,
            "acceptance_pass": all(
                cell["survival_pass"] and cell["tracking_pass"] for cell in cells
            ),
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
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--command-x", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.seconds <= 0 or args.command_x == 0:
        raise ValueError("episodes/seconds must be positive and command-x must be non-zero")
    report = MassGridEvaluator(args).evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "cells"}, indent=2))
    if not report["acceptance_pass"]:
        raise SystemExit("Mass-grid acceptance failed; inspect the JSON report")


if __name__ == "__main__":
    main()
