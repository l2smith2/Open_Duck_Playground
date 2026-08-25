"""Audit at least 10,000 batched randomized MJX models."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from playground.common import randomize
from playground.open_duck_mini_v2.joystick import Joystick


def _assert_between(values, bounds, label, tolerance=1e-6):
    low, high = bounds
    if np.min(values) < low - tolerance or np.max(values) > high + tolerance:
        raise AssertionError(
            f"{label} outside {bounds}: observed {np.min(values)} to {np.max(values)}"
        )


def audit(stage: str, samples: int, seed: int) -> dict:
    if samples < 10_000:
        raise ValueError("The acceptance audit requires at least 10,000 samples")
    env = Joystick(task="flat_terrain_backlash")
    config = randomize.get_mass_randomization_config(stage)
    trunk_id, head_id, pairs = randomize.resolve_and_validate_bodies(env.mj_model)
    callback = randomize.make_domain_randomizer(env.mj_model, stage)
    keys = jax.random.split(jax.random.PRNGKey(seed), samples)
    randomized, _ = callback(env.mjx_model, keys)

    nominal_mass = np.asarray(env.mjx_model.body_mass)
    nominal_inertia = np.asarray(env.mjx_model.body_inertia)
    mass = np.asarray(randomized.body_mass)
    inertia = np.asarray(randomized.body_inertia)
    body_ipos = np.asarray(randomized.body_ipos)
    positive_ids = np.flatnonzero(nominal_mass > 0)
    mass_scale = mass[:, positive_ids] / nominal_mass[positive_ids]

    if not np.all(np.isfinite(mass)) or not np.all(mass[:, positive_ids] > 0):
        raise AssertionError("Masses must be finite and positive")
    if not np.all(np.isfinite(inertia)) or not np.all(inertia[:, positive_ids] > 0):
        raise AssertionError("Diagonal inertias must be finite and positive")
    _assert_between(mass[:, trunk_id] / nominal_mass[trunk_id], config.trunk_mass_scale, "trunk")
    _assert_between(mass[:, head_id] / nominal_mass[head_id], config.head_mass_scale, "head")
    other_ids = [
        body_id for body_id in positive_ids if body_id not in {trunk_id, head_id}
    ]
    _assert_between(
        mass[:, other_ids] / nominal_mass[other_ids], config.other_mass_scale, "other links"
    )

    trunk_offset = body_ipos[:, trunk_id] - np.asarray(env.mjx_model.body_ipos[trunk_id])
    head_offset = body_ipos[:, head_id] - np.asarray(env.mjx_model.body_ipos[head_id])
    if np.any(np.abs(trunk_offset) > np.asarray(config.trunk_com_offset_m) + 1e-6):
        raise AssertionError("Trunk COM offset exceeded its configured axis bound")
    if np.any(np.abs(head_offset) > np.asarray(config.head_com_offset_m) + 1e-6):
        raise AssertionError("Head COM offset exceeded its configured axis bound")

    expected_inertia = nominal_inertia[None, positive_ids, :] * mass_scale[:, :, None]
    if not np.allclose(inertia[:, positive_ids], expected_inertia, rtol=1e-5, atol=1e-8):
        raise AssertionError("Inertia was not scaled with mass")

    maximum_pair_ratio = (
        (1 + config.paired_leg_mismatch) / max(1 - config.paired_leg_mismatch, 1e-9)
    )
    for left_id, right_id in pairs:
        left_scale = mass[:, left_id] / nominal_mass[left_id]
        right_scale = mass[:, right_id] / nominal_mass[right_id]
        ratio = np.maximum(left_scale / right_scale, right_scale / left_scale)
        if np.max(ratio) > maximum_pair_ratio + 1e-6:
            raise AssertionError("A paired leg mismatch exceeded the independent +/-2% rule")

    return {
        "stage": stage,
        "samples": samples,
        "seed": seed,
        "trunk_mass_kg": [float(np.min(mass[:, trunk_id])), float(np.max(mass[:, trunk_id]))],
        "head_mass_kg": [float(np.min(mass[:, head_id])), float(np.max(mass[:, head_id]))],
        "all_checks_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(randomize.MASS_RANDOMIZATION_STAGES), default="full")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.stage, args.samples, args.seed)
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
