"""Score a whole reward configuration on whether walking beats marching in place.

check_reference_locomotion_incentive.py asks "is this reference safe to train
on", holding the reward fixed. This asks the other half of the same question:
"is this reward configuration safe to train with", holding the reference fixed.

Both style attempts collapsed because the total per-step reward was higher for
marching in place than for walking. That total is a sum over every term in
Joystick._get_reward, not just the imitation term, so a fix may come from any of
them -- and a fix that only looks plausible can be refuted here in a minute
instead of in a 75-minute training run.

The expensive part is the rollout, and it does not depend on the weights. So
each policy is rolled out once, the per-step mean of every term is recorded
UNWEIGHTED, and any number of candidate weight sets are then scored from the
cache for free. The cache also carries the five gait-shaping terms that
playground/common/rewards.py defines but joystick.py does not currently wire in,
so a candidate that adds them can be measured before it is added.

    uv run python scripts/check_reward_locomotion_incentive.py \
        --reference artifacts/bdx_reference/polynomial_coefficients.pkl \
        --walking-onnx NEUTRAL.onnx --marching-onnx COLLAPSED_STYLE.onnx \
        --cache artifacts/reward_incentive_rollouts.json \
        --scales feet_air_time=2.0 --scales alive=2.0

It exits non-zero unless walking wins by --min-margin, which is the precondition
for the configuration to be worth training with.
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from playground.common import rewards_numpy as rn
from playground.open_duck_mini_v2 import constants
from playground.open_duck_mini_v2.joystick import default_config
from playground.open_duck_mini_v2.mujoco_infer import MjInfer, USE_MOTOR_SPEED_LIMITS

from check_reference_locomotion_incentive import imitation_terms

# Terms joystick.py does not wire in today. Defaulting them to zero means an
# empty --scales reproduces the reward that is actually training right now.
CANDIDATE_SCALES = {
    "feet_air_time": 0.0,
    "feet_phase": 0.0,
    "feet_clearance": 0.0,
    "feet_height": 0.0,
    "feet_slip": 0.0,
}

# Bump when the rollout records something new, so stale caches are refitted
# rather than silently scored against a shorter record.
CACHE_VERSION = 3

# The four imitation terms that are exponentials of a velocity error. Their beta
# was tuned by Disney for a 0.7 m/s robot; ours is commanded to 0.15, so beta is
# worth sweeping alongside the weights. Recomputing them needs only the squared
# error, so that is what the rollout records.
IMITATION_BETAS = {
    "lin_vel_xy": ("lin", 1.0),
    "lin_vel_z": ("lin", 1.0),
    "ang_vel_xy": ("ang", 0.5),
    "ang_vel_z": ("ang", 0.5),
}
# The rest do not depend on beta and are cached as they are scored.
IMITATION_FLAT = ("joint_pos", "joint_vel", "contact")


def default_scales() -> dict:
    """Today's live reward weights, plus the unwired candidates at zero."""
    scales = {k: float(v) for k, v in default_config().reward_config.scales.items()}
    scales.update(CANDIDATE_SCALES)
    return scales


def get_rz(phase: np.ndarray, swing_height: float) -> np.ndarray:
    """Desired foot height over one gait cycle (mujoco_playground's gait.get_rz)."""

    def bezier(y_start, y_end, x):
        return y_start + (y_end - y_start) * (x**3 + 3 * (x**2 * (1 - x)))

    x = (phase + np.pi) / (2 * np.pi)
    return np.where(
        x <= 0.5, bezier(0.0, swing_height, 2 * x), bezier(swing_height, 0.0, 2 * x - 1)
    )


def sensor_slice(model, name: str) -> slice:
    """Read a sensor by address and width, never by sensor id."""
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    adr = model.sensor_adr[sensor_id]
    return slice(adr, adr + model.sensor_dim[sensor_id])


def rollout(
    onnx: Path,
    reference: Path,
    model_path: Path,
    command_x: float,
    seconds: float,
    max_foot_height: float,
) -> dict:
    """Per-step mean of every reward term, unweighted, for one policy."""
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

    local_linvel = sensor_slice(model, constants.LOCAL_LINVEL_SENSOR)
    global_linvel = sensor_slice(model, constants.GLOBAL_LINVEL_SENSOR)
    feet_site_id = np.array([model.site(name).id for name in constants.FEET_SITES])
    feet_linvel = [
        sensor_slice(model, f"{site}_global_linvel") for site in constants.FEET_SITES
    ]
    ctrl_dt = sim.sim_dt * sim.decimation

    # Mirrors the per-step bookkeeping Joystick.step keeps in state.info.
    feet_air_time = np.zeros(2)
    last_contact = np.zeros(2, dtype=bool)
    swing_peak = np.zeros(2)

    command = np.array(sim.commands)
    start_x = float(data.qpos[0])
    totals, steps = None, 0
    air_times, swing_peaks = [], []
    # The two tracking terms depend on tracking_sigma, and four imitation terms
    # depend on their beta. Both are worth sweeping as freely as the weights
    # are, so keep their raw inputs rather than only their value at one setting.
    linvel_trace, gyro_trace, imitation_trace = [], [], []
    for step in range(int(seconds / sim.sim_dt)):
        mujoco.mj_step(model, data)
        if step % sim.decimation:
            continue
        sim.imitation_i = (
            sim.imitation_i + sim.phase_frequency_factor
        ) % sim.PRM.nb_steps_in_period
        phase = sim.imitation_i / sim.PRM.nb_steps_in_period * 2 * np.pi
        sim.imitation_phase = np.array([np.cos(phase), np.sin(phase)])
        action = sim.policy.infer(sim.get_obs(data, sim.commands))
        prev_action = sim.last_action.copy()
        sim.last_last_last_action = sim.last_last_action.copy()
        sim.last_last_action = sim.last_action.copy()
        sim.last_action = action.copy()
        sim.motor_targets = sim.default_actuator + action * sim.action_scale
        if USE_MOTOR_SPEED_LIMITS:
            delta = sim.max_motor_velocity * ctrl_dt
            sim.motor_targets = np.clip(
                sim.motor_targets,
                sim.prev_motor_targets - delta,
                sim.prev_motor_targets + delta,
            )
            sim.prev_motor_targets = sim.motor_targets.copy()
        data.ctrl[:] = sim.motor_targets

        contact = np.array(sim.get_feet_contacts(data), dtype=bool)
        contact_filt = contact | last_contact
        first_contact = (feet_air_time > 0.0) * contact_filt
        feet_air_time = feet_air_time + ctrl_dt
        foot_pos = data.site_xpos[feet_site_id]
        swing_peak = np.maximum(swing_peak, foot_pos[:, -1])
        feet_vel = np.array([data.sensordata[s] for s in feet_linvel])

        frame = np.asarray(
            sim.PRM.get_reference_motion(command_x, 0.0, 0.0, sim.imitation_i)
        )
        imitation = imitation_terms(
            sim.get_floating_base_qvel(data.qvel),
            sim.get_actuator_joints_qpos(data.qpos),
            sim.get_actuator_joints_qvel(data.qvel),
            contact,
            frame,
        )
        foot_phase = np.array([phase, phase + np.pi])
        linvel_trace.append(data.sensordata[local_linvel][:2].tolist())
        gyro_trace.append(float(sim.get_gyro(data)[2]))
        base_qvel = sim.get_floating_base_qvel(data.qvel)
        imitation_trace.append(
            [
                float(np.sum((base_qvel[:2] - frame[34:36]) ** 2)),
                float((base_qvel[2] - frame[36]) ** 2),
                float(np.sum((base_qvel[3:5] - frame[37:39]) ** 2)),
                float((base_qvel[5] - frame[39]) ** 2),
            ]
        )
        terms = {
            "torques": float(rn.cost_torques(data.actuator_force)),
            "action_rate": float(rn.cost_action_rate(action, prev_action)),
            "alive": 1.0,
            "imitation": float(sum(imitation.values())),
            "stand_still": float(
                rn.cost_stand_still(
                    command,
                    sim.get_actuator_joints_qpos(data.qpos),
                    sim.get_actuator_joints_qvel(data.qvel),
                    sim.default_actuator,
                    ignore_head=False,
                )
            ),
            "feet_air_time": float(
                rn.reward_feet_air_time(
                    feet_air_time, first_contact, command, 0.10, 0.25
                )
            ),
            "feet_phase": float(
                rn.reward_feet_phase(foot_pos, get_rz(foot_phase, max_foot_height))
            ),
            "feet_clearance": float(
                rn.cost_feet_clearance(feet_vel, foot_pos, max_foot_height)
            ),
            "feet_height": float(
                rn.cost_feet_height(swing_peak, first_contact, max_foot_height)
            ),
            "feet_slip": float(
                rn.cost_feet_slip(contact, data.sensordata[global_linvel])
            ),
        }
        terms.update({f"imitation/{k}": float(imitation[k]) for k in IMITATION_FLAT})
        totals = terms if totals is None else {k: totals[k] + v for k, v in terms.items()}
        steps += 1

        if first_contact.any():
            air_times.extend(feet_air_time[first_contact].tolist())
            swing_peaks.extend(swing_peak[first_contact].tolist())
        feet_air_time = feet_air_time * ~contact
        last_contact = contact
        swing_peak = swing_peak * ~contact

    return {
        "onnx": str(onnx),
        "terms": {k: v / steps for k, v in totals.items()},
        "command": command.tolist(),
        "local_linvel_xy": linvel_trace,
        "gyro_z": gyro_trace,
        "imitation_sq_error": imitation_trace,
        "steps": steps,
        "measured_speed_x": (float(data.qpos[0]) - start_x) / seconds,
        # Diagnostics for the gait terms: what the policy's stride actually does.
        "mean_air_time": float(np.mean(air_times)) if air_times else 0.0,
        "mean_swing_peak": float(np.mean(swing_peaks)) if swing_peaks else 0.0,
        "steps_per_second": len(air_times) / seconds,
    }


def tracking_terms(record: dict, tracking_sigma: float, ang_tracking_sigma: float) -> dict:
    """The two sigma-dependent terms, recomputed from the recorded traces."""
    command = np.array(record["command"])
    linvel = np.array(record["local_linvel_xy"])
    gyro_z = np.array(record["gyro_z"])
    lin = [
        rn.reward_tracking_lin_vel(command, v, tracking_sigma) for v in linvel
    ]
    ang = np.exp(-np.square(command[2] - gyro_z) / ang_tracking_sigma)
    return {
        "tracking_lin_vel": float(np.mean(lin)),
        "tracking_ang_vel": float(np.mean(np.nan_to_num(ang))),
    }


def imitation_breakdown(record: dict, lin_beta: float, ang_beta: float) -> dict:
    """Every term of reward_imitation, with the four exponentials at these betas.

    Summed, this is exactly what check_reference_locomotion_incentive.py reports,
    so one run answers both "is this reference safe" and "is this reward safe".
    """
    errors = np.array(record["imitation_sq_error"])
    betas = {"lin": lin_beta, "ang": ang_beta}
    out = {}
    for column, (name, (axis, weight)) in enumerate(IMITATION_BETAS.items()):
        out[name] = float(np.mean(np.exp(-betas[axis] * errors[:, column])) * weight)
    out.update({name: record["terms"][f"imitation/{name}"] for name in IMITATION_FLAT})
    return out


def score(
    record: dict,
    scales: dict,
    tracking_sigma: float,
    ang_tracking_sigma: float,
    lin_beta: float = 8.0,
    ang_beta: float = 2.0,
) -> dict:
    """Weighted per-step reward, split into terms, mirroring Joystick.step."""
    imitation = imitation_breakdown(record, lin_beta, ang_beta)
    terms = dict(record["terms"])
    terms.update(tracking_terms(record, tracking_sigma, ang_tracking_sigma))
    terms["imitation"] = float(sum(imitation.values()))
    weighted = {k: terms[k] * w for k, w in scales.items()}
    return {
        "terms": weighted,
        "imitation_terms": imitation,
        "total": float(sum(weighted.values())),
    }


def policies_from_bundle(bundle: Path) -> tuple[Path, Path]:
    """The known-walking and known-collapsed policy inside an artifact bundle.

    Saves pasting two long checkpoint paths on every run. Walking comes from the
    furthest-trained neutral stage, marching from the first style seed, each the
    highest-step ONNX in its directory.
    """

    def pick(pattern: str, what: str, index: int) -> Path:
        stages = sorted(d for d in bundle.glob(pattern) if d.is_dir())
        if not stages:
            raise SystemExit(f"no {what} stage matching {pattern!r} under {bundle}")
        stage = stages[index]
        exports = sorted(
            stage.glob("*.onnx"), key=lambda path: int(path.stem.rsplit("_", 1)[-1])
        )
        if not exports:
            raise SystemExit(f"{stage} has no ONNX export")
        return exports[-1]

    return pick("*neutral_full*", "neutral", -1), pick("*style_seed*", "style", 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference", type=Path, required=True, help="fitted polynomial_coefficients.pkl"
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        help="artifact bundle to take both policies from, instead of naming each ONNX",
    )
    parser.add_argument("--walking-onnx", type=Path, help="a policy known to locomote")
    parser.add_argument(
        "--marching-onnx", type=Path, help="a policy known to have collapsed to standing"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"),
    )
    parser.add_argument("--command-x", type=float, default=0.10)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument(
        "--max-foot-height",
        type=float,
        default=0.02,
        help="swing-height target for the foot clearance/height/phase terms",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        help="reuse rollouts from this file when it matches, otherwise write them to it",
    )
    parser.add_argument(
        "--scales",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override one reward weight; repeatable",
    )
    parser.add_argument(
        "--config", type=Path, help="JSON object of reward weights to override"
    )
    parser.add_argument(
        "--tracking-sigma",
        type=float,
        default=default_config().reward_config.tracking_sigma,
        help="sigma for tracking_lin_vel",
    )
    parser.add_argument(
        "--ang-tracking-sigma",
        type=float,
        default=default_config().reward_config.ang_tracking_sigma,
        help="sigma for tracking_ang_vel",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.25,
        help="how far walking must beat marching before the configuration is worth training with",
    )
    parser.add_argument(
        "--imitation-lin-beta",
        type=float,
        default=8.0,
        help="beta for the imitation reward's two linear-velocity exponentials",
    )
    parser.add_argument(
        "--imitation-ang-beta",
        type=float,
        default=2.0,
        help="beta for the imitation reward's two angular-velocity exponentials",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.bundle:
        if args.walking_onnx or args.marching_onnx:
            raise SystemExit("--bundle already names both policies")
        args.walking_onnx, args.marching_onnx = policies_from_bundle(args.bundle)

    scales = default_scales()
    if args.config:
        scales.update({k: float(v) for k, v in json.loads(args.config.read_text()).items()})
    for override in args.scales:
        name, _, value = override.partition("=")
        if name not in scales:
            raise SystemExit(f"unknown reward term {name!r}; known terms: {sorted(scales)}")
        scales[name] = float(value)

    ang_tracking_sigma = args.ang_tracking_sigma
    key = {
        "cache_version": CACHE_VERSION,
        "reference": str(args.reference),
        "walking_onnx": str(args.walking_onnx),
        "marching_onnx": str(args.marching_onnx),
        "command_x": args.command_x,
        "seconds": args.seconds,
        "max_foot_height": args.max_foot_height,
    }
    cached = None
    if args.cache and args.cache.exists():
        cached = json.loads(args.cache.read_text())
        if cached.get("key") != key:
            cached = None
    if cached is None:
        if not (args.walking_onnx and args.marching_onnx):
            raise SystemExit(
                "no usable cache, so --walking-onnx and --marching-onnx are required"
            )
        cached = {
            "key": key,
            "walking": rollout(
                args.walking_onnx,
                args.reference,
                args.model_path,
                args.command_x,
                args.seconds,
                args.max_foot_height,
            ),
            "marching": rollout(
                args.marching_onnx,
                args.reference,
                args.model_path,
                args.command_x,
                args.seconds,
                args.max_foot_height,
            ),
        }
        if args.cache:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            args.cache.write_text(json.dumps(cached, indent=2) + "\n", encoding="utf-8")
    walking, marching = cached["walking"], cached["marching"]

    betas = (args.imitation_lin_beta, args.imitation_ang_beta)
    walking_score = score(walking, scales, args.tracking_sigma, ang_tracking_sigma, *betas)
    marching_score = score(marching, scales, args.tracking_sigma, ang_tracking_sigma, *betas)
    margin = walking_score["total"] - marching_score["total"]
    # PPO sees relative advantage, so a margin buried in a large constant per-step
    # reward is weaker than the same margin against a small one.
    share = margin / marching_score["total"] if marching_score["total"] else float("nan")
    report = {
        "reference": str(args.reference),
        "command_x": args.command_x,
        "scales": scales,
        "tracking_sigma": args.tracking_sigma,
        "ang_tracking_sigma": ang_tracking_sigma,
        "walking": {**walking_score, "measured_speed_x": walking["measured_speed_x"]},
        "marching": {**marching_score, "measured_speed_x": marching["measured_speed_x"]},
        "imitation_lin_beta": args.imitation_lin_beta,
        "imitation_ang_beta": args.imitation_ang_beta,
        "margin": margin,
        "imitation_margin": (
            walking_score["terms"]["imitation"] - marching_score["terms"]["imitation"]
        ),
        "margin_share": share,
        "min_margin": args.min_margin,
        "pass": margin >= args.min_margin,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"reference: {args.reference}")
    print(
        f"tracking_sigma: {args.tracking_sigma:g} lin / {ang_tracking_sigma:g} ang"
    )
    print(f"{'term':>18}{'weight':>9}{'walking':>10}{'marching':>10}{'diff':>10}")
    for name in sorted(scales):
        w, a, b = scales[name], walking_score["terms"][name], marching_score["terms"][name]
        flag = "" if w else "   (off)"
        print(f"{name:>18}{w:9.3f}{a:10.3f}{b:10.3f}{a - b:+10.3f}{flag}")
    print(
        f"{'TOTAL':>18}{'':9}{walking_score['total']:10.3f}"
        f"{marching_score['total']:10.3f}{margin:+10.3f}"
    )
    # The same breakdown check_reference_locomotion_incentive.py prints, so this
    # one run also says whether the reference itself is safe to train on.
    imitation_margin = (
        walking_score["terms"]["imitation"] - marching_score["terms"]["imitation"]
    )
    print(
        f"\n  reference (imitation only), beta {args.imitation_lin_beta:g} lin / "
        f"{args.imitation_ang_beta:g} ang"
    )
    for name, value in walking_score["imitation_terms"].items():
        other = marching_score["imitation_terms"][name]
        print(f"{name:>18}{'':9}{value:10.3f}{other:10.3f}{value - other:+10.3f}")
    print(
        f"{'imitation total':>18}{'':9}{sum(walking_score['imitation_terms'].values()):10.3f}"
        f"{sum(marching_score['imitation_terms'].values()):10.3f}{imitation_margin:+10.3f}"
    )
    print()
    print(
        f"{'speed m/s':>18}{'':9}{walking['measured_speed_x']:10.4f}"
        f"{marching['measured_speed_x']:10.4f}"
    )
    print(
        f"{'air time s':>18}{'':9}{walking['mean_air_time']:10.3f}"
        f"{marching['mean_air_time']:10.3f}"
    )
    print(
        f"{'swing peak m':>18}{'':9}{walking['mean_swing_peak']:10.3f}"
        f"{marching['mean_swing_peak']:10.3f}"
    )
    print(
        f"{'steps/s':>18}{'':9}{walking['steps_per_second']:10.2f}"
        f"{marching['steps_per_second']:10.2f}"
    )
    print(f"\nmargin (walking - marching) = {margin:+.3f}, need >= {args.min_margin:.2f}")
    print(f"margin as a share of the marching total = {share:+.1%}")
    # A margin only means something when the two policies really do differ in
    # what they are doing. The 220M neutral policy stands still below about a
    # 0.05 m/s command, and comparing two standing policies always looks fine.
    if abs(walking["measured_speed_x"]) < 0.25 * abs(args.command_x):
        raise SystemExit(
            f"The reference policy only reached {walking['measured_speed_x']:.4f} m/s against a "
            f"{args.command_x:.2f} m/s command, so it is not walking here and this comparison "
            "says nothing. Pick a command where it does walk."
        )
    if not report["pass"]:
        raise SystemExit(
            "Refusing this reward configuration: marching in place earns at least as much "
            "per step as walking, so fine-tuning under it will collapse to standing."
        )
    print("PASS: walking is the better-rewarded behaviour under this configuration.")


if __name__ == "__main__":
    main()
