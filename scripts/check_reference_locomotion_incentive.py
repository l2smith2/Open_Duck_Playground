"""Prove a candidate reference rewards walking more than marching in place.

Style fine-tuning has now failed twice in a way no existing check caught: the
policy stayed upright, survived every mass-grid cell, tracked the reference
pose closely, and did not locomote (0.002 m/s against a 0.10 m/s command).

The cause is a property of the reference itself, not of training. The imitation
reward is dominated by `joint_pos`, an unbounded quadratic on joint angles, plus
a foot-contact match. Both can be satisfied by cycling the legs on the spot. If
a reference's fore-aft leg swing is small relative to its lateral sway, then
marching in place scores *higher* than really walking, the reward ordering is
inverted, and 30M steps of fine-tuning reliably find the standing optimum. There
is no gradient path back out, so this cannot be fixed by training longer.

That ordering is measurable from the fitted reference alone, in about a minute,
using two policies we already have: one that walks and one that collapsed. Run
this before spending GPU on a candidate reference.

    uv run python scripts/check_reference_locomotion_incentive.py \
        --reference artifacts/bdx_reference/polynomial_coefficients.pkl \
        --walking-onnx NEUTRAL.onnx --marching-onnx COLLAPSED_STYLE.onnx \
        --output artifacts/bdx_reference/locomotion_incentive.json

It exits non-zero unless walking wins, which is the precondition for the style
stage to be worth running at all.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from playground.open_duck_mini_v2.mujoco_infer import MjInfer, USE_MOTOR_SPEED_LIMITS

# Slices into the 40-wide fitted reference frame, matching custom_rewards.py.
JOINT_POS = slice(0, 16)
JOINT_VEL = slice(16, 32)
CONTACTS = slice(32, 34)
LIN_VEL = slice(34, 37)
ANG_VEL = slice(37, 40)
# reward_imitation drops neck/head/antennas from both sides before comparing.
REF_LEGS = list(range(0, 5)) + list(range(11, 16))
ACT_LEGS = list(range(0, 5)) + list(range(9, 14))
WEIGHTS = {
    "lin_vel_xy": 1.0,
    "lin_vel_z": 1.0,
    "ang_vel_xy": 0.5,
    "ang_vel_z": 0.5,
    "joint_pos": 15.0,
    "joint_vel": 1.0e-3,
    "contact": 1.0,
}


def imitation_terms(base_qvel, joints_qpos, joints_qvel, contacts, reference_frame):
    """The per-term imitation reward, mirroring custom_rewards.reward_imitation.

    Torso orientation is excluded because reward_imitation itself leaves it out
    of the sum. Everything else uses the same weights and exponent scales.
    """
    ref_lin = reference_frame[LIN_VEL]
    ref_ang = reference_frame[ANG_VEL]
    ref_joint_pos = reference_frame[JOINT_POS][REF_LEGS]
    ref_joint_vel = reference_frame[JOINT_VEL][REF_LEGS]
    ref_contacts = (reference_frame[CONTACTS] > 0.5).astype(float)
    joint_pos = joints_qpos[ACT_LEGS]
    joint_vel = joints_qvel[ACT_LEGS]
    return {
        "lin_vel_xy": np.exp(-8.0 * np.sum((base_qvel[:2] - ref_lin[:2]) ** 2)) * WEIGHTS["lin_vel_xy"],
        "lin_vel_z": np.exp(-8.0 * np.sum((base_qvel[2] - ref_lin[2]) ** 2)) * WEIGHTS["lin_vel_z"],
        "ang_vel_xy": np.exp(-2.0 * np.sum((base_qvel[3:5] - ref_ang[:2]) ** 2)) * WEIGHTS["ang_vel_xy"],
        "ang_vel_z": np.exp(-2.0 * np.sum((base_qvel[5] - ref_ang[2]) ** 2)) * WEIGHTS["ang_vel_z"],
        "joint_pos": -np.sum((joint_pos - ref_joint_pos) ** 2) * WEIGHTS["joint_pos"],
        "joint_vel": -np.sum((joint_vel - ref_joint_vel) ** 2) * WEIGHTS["joint_vel"],
        "contact": float(np.sum(contacts == ref_contacts)) * WEIGHTS["contact"],
    }


def score_policy(onnx: Path, reference: Path, model_path: Path, command_x: float, seconds: float) -> dict:
    """Mean per-step imitation reward this policy earns against this reference."""
    sim = MjInfer(str(model_path), str(reference), str(onnx), standing=False)
    model, data = sim.model, sim.data
    mujoco.mj_resetData(model, data)
    data.qpos[:] = model.keyframe("home").qpos
    data.ctrl[:] = sim.default_actuator
    sim.last_action[:] = 0
    sim.last_last_action[:] = 0
    sim.last_last_last_action[:] = 0
    sim.motor_targets = sim.default_actuator.copy()
    sim.prev_motor_targets = sim.default_actuator.copy()
    sim.commands = [command_x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    sim.imitation_i = 0
    sim.imitation_phase = np.array([0.0, 0.0])
    mujoco.mj_forward(model, data)

    start_x = float(data.qpos[0])
    totals = None
    steps = 0
    for step in range(int(seconds / sim.sim_dt)):
        mujoco.mj_step(model, data)
        if step % sim.decimation:
            continue
        sim.imitation_i = (sim.imitation_i + sim.phase_frequency_factor) % sim.PRM.nb_steps_in_period
        phase = sim.imitation_i / sim.PRM.nb_steps_in_period * 2 * np.pi
        sim.imitation_phase = np.array([np.cos(phase), np.sin(phase)])
        action = sim.policy.infer(sim.get_obs(data, sim.commands))
        sim.last_last_last_action = sim.last_last_action.copy()
        sim.last_last_action = sim.last_action.copy()
        sim.last_action = action.copy()
        sim.motor_targets = sim.default_actuator + action * sim.action_scale
        if USE_MOTOR_SPEED_LIMITS:
            delta = sim.max_motor_velocity * (sim.sim_dt * sim.decimation)
            sim.motor_targets = np.clip(
                sim.motor_targets, sim.prev_motor_targets - delta, sim.prev_motor_targets + delta
            )
            sim.prev_motor_targets = sim.motor_targets.copy()
        data.ctrl[:] = sim.motor_targets
        frame = np.asarray(sim.PRM.get_reference_motion(command_x, 0.0, 0.0, sim.imitation_i))
        terms = imitation_terms(
            sim.get_floating_base_qvel(data.qvel),
            sim.get_actuator_joints_qpos(data.qpos),
            sim.get_actuator_joints_qvel(data.qvel),
            sim.get_feet_contacts(data),
            frame,
        )
        totals = terms if totals is None else {k: totals[k] + v for k, v in terms.items()}
        steps += 1

    means = {k: v / steps for k, v in totals.items()}
    return {
        "onnx": str(onnx),
        "terms": means,
        "imitation_total": float(sum(means.values())),
        "measured_speed_x": (float(data.qpos[0]) - start_x) / seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True, help="candidate fitted polynomial_coefficients.pkl")
    parser.add_argument("--walking-onnx", type=Path, required=True, help="a policy known to locomote")
    parser.add_argument("--marching-onnx", type=Path, required=True, help="a policy known to have collapsed to standing")
    parser.add_argument(
        "--model-path", type=Path, default=Path("playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml")
    )
    parser.add_argument("--command-x", type=float, default=0.10)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.25,
        help="how far walking must beat marching before the reference is worth training on",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    walking = score_policy(args.walking_onnx, args.reference, args.model_path, args.command_x, args.seconds)
    marching = score_policy(args.marching_onnx, args.reference, args.model_path, args.command_x, args.seconds)
    margin = walking["imitation_total"] - marching["imitation_total"]
    report = {
        "reference": str(args.reference),
        "command_x": args.command_x,
        "walking": walking,
        "marching": marching,
        "margin": margin,
        "min_margin": args.min_margin,
        "pass": margin >= args.min_margin,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"reference: {args.reference}")
    print(f"{'term':>12}{'walking':>12}{'marching':>12}")
    for term in WEIGHTS:
        print(f"{term:>12}{walking['terms'][term]:12.3f}{marching['terms'][term]:12.3f}")
    print(f"{'TOTAL':>12}{walking['imitation_total']:12.3f}{marching['imitation_total']:12.3f}")
    print(f"{'speed m/s':>12}{walking['measured_speed_x']:12.4f}{marching['measured_speed_x']:12.4f}")
    print(f"\nmargin (walking - marching) = {margin:+.3f}, need >= {args.min_margin:.2f}")
    if not report["pass"]:
        raise SystemExit(
            "Refusing this reference: marching in place scores at least as well as walking, so "
            "style fine-tuning will collapse to standing. Increase the fore-aft leg swing "
            "(walk_foot_height is the strongest lever; lowering walk_com_height also helps) and "
            "regenerate before spending GPU on it."
        )
    print("PASS: walking is the better-rewarded behaviour under this reference.")


if __name__ == "__main__":
    main()
