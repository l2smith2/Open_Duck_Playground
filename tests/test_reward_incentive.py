"""Tests for the offline reward-configuration gate.

The gate itself needs the ONNX policies and a fitted reference, which live
outside Git. These cover the parts that can go wrong silently: the weights it
scores must be the weights that are actually training, and the two tracking
terms must respond to their own sigma rather than a shared one.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_reward_locomotion_incentive import (  # noqa: E402
    CANDIDATE_SCALES,
    IMITATION_BETAS,
    IMITATION_FLAT,
    default_scales,
    get_rz,
    imitation_breakdown,
    policies_from_bundle,
    score,
    tracking_terms,
)
from playground.open_duck_mini_v2.joystick import default_config  # noqa: E402


def _record(linvel, gyro_z, command=(0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), **terms):
    base = {name: 0.0 for name in default_scales()}
    base.update({f"imitation/{name}": 0.0 for name in IMITATION_FLAT})
    base.update(terms)
    return {
        "terms": base,
        "command": list(command),
        "local_linvel_xy": [list(v) for v in linvel],
        "gyro_z": list(gyro_z),
        "imitation_sq_error": [[0.0] * len(IMITATION_BETAS)] * len(gyro_z),
    }


def test_default_scales_are_the_live_training_weights():
    # A gate scoring weights the trainer does not use would pass configurations
    # that then collapse, which is the failure it exists to prevent.
    live = default_config().reward_config.scales
    scales = default_scales()
    for name, weight in live.items():
        assert scales[name] == pytest.approx(weight)
    # The unwired candidates default to zero, so an unmodified run reproduces
    # today's reward exactly rather than a hypothetical one.
    for name in CANDIDATE_SCALES:
        assert name not in live
        assert scales[name] == 0.0


def test_angular_tracking_has_its_own_sigma():
    # Walking swings the torso in yaw about a correct mean. Under the shared
    # sigma that reads as a large error and taxes every real gait, so the two
    # tracking terms have to be able to move independently.
    record = _record(linvel=[[0.10, 0.0]] * 4, gyro_z=[0.13, -0.13, 0.13, -0.13])
    tight = tracking_terms(record, 0.01, 0.01)["tracking_ang_vel"]
    loose = tracking_terms(record, 0.01, 0.25)["tracking_ang_vel"]
    assert tight < 0.25
    assert loose > 0.9
    # Widening the angular sigma must not touch the linear term.
    assert tracking_terms(record, 0.01, 0.01)["tracking_lin_vel"] == pytest.approx(
        tracking_terms(record, 0.01, 0.25)["tracking_lin_vel"]
    )


def test_a_yaw_wobble_is_priced_like_a_yaw_error_of_the_same_size():
    # Regression guard on the direction of the fix: a gait that oscillates in
    # yaw about zero must not score worse than one that holds still by more
    # than the configured sigma implies.
    still = _record(linvel=[[0.10, 0.0]] * 4, gyro_z=[0.0] * 4)
    wobbling = _record(linvel=[[0.10, 0.0]] * 4, gyro_z=[0.13, -0.13, 0.13, -0.13])
    sigma = default_config().reward_config.ang_tracking_sigma
    penalty = (
        tracking_terms(still, 0.01, sigma)["tracking_ang_vel"]
        - tracking_terms(wobbling, 0.01, sigma)["tracking_ang_vel"]
    )
    assert penalty < 0.1


def test_score_applies_weights_and_totals_them():
    record = _record(linvel=[[0.10, 0.0]], gyro_z=[0.0], alive=1.0)
    scales = default_scales()
    result = score(record, scales, 0.01, 0.25)
    assert result["terms"]["alive"] == pytest.approx(scales["alive"])
    # imitation is derived from the breakdown rather than read back as a scalar,
    # so it always matches the per-term table printed beside it.
    assert result["terms"]["imitation"] == pytest.approx(
        sum(result["imitation_terms"].values()) * scales["imitation"]
    )
    assert result["total"] == pytest.approx(sum(result["terms"].values()))


def test_get_rz_traces_one_swing_per_cycle():
    # Mirrors mujoco_playground's gait.get_rz: on the ground at the ends of the
    # cycle, at the swing height in the middle.
    phase = np.linspace(-np.pi, np.pi, 101)
    rz = get_rz(phase, 0.02)
    assert rz[0] == pytest.approx(0.0, abs=1e-6)
    assert rz[-1] == pytest.approx(0.0, abs=1e-6)
    assert rz.max() == pytest.approx(0.02, rel=1e-3)
    assert np.argmax(rz) == pytest.approx(50, abs=2)


def test_imitation_breakdown_reproduces_the_reference_gate_terms():
    # This is the number check_reference_locomotion_incentive.py reports, so one
    # run has to answer both "is this reference safe" and "is this reward safe".
    # At zero velocity error every exponential is 1.0 and only the weights show.
    record = _record(linvel=[[0.10, 0.0]] * 3, gyro_z=[0.0] * 3)
    terms = imitation_breakdown(record, 8.0, 2.0)
    assert terms["lin_vel_xy"] == pytest.approx(1.0)
    assert terms["lin_vel_z"] == pytest.approx(1.0)
    assert terms["ang_vel_xy"] == pytest.approx(0.5)
    assert terms["ang_vel_z"] == pytest.approx(0.5)
    assert set(terms) == set(IMITATION_BETAS) | set(IMITATION_FLAT)


def test_imitation_beta_sharpens_the_velocity_terms():
    # beta is sweepable because Disney tuned it for a 0.7 m/s robot and ours is
    # commanded to 0.15. Raising it must make the same error cost more.
    record = _record(linvel=[[0.10, 0.0]], gyro_z=[0.0])
    record["imitation_sq_error"] = [[0.04, 0.0, 0.0, 0.0]]
    assert imitation_breakdown(record, 8.0, 2.0)["lin_vel_xy"] == pytest.approx(
        np.exp(-8.0 * 0.04)
    )
    assert imitation_breakdown(record, 100.0, 2.0)["lin_vel_xy"] == pytest.approx(
        np.exp(-100.0 * 0.04)
    )


def test_bundle_picks_the_furthest_neutral_stage_and_the_first_style_seed(tmp_path):
    for stage, steps in (
        ("02_neutral_moderate_60m", (1, 2)),
        ("03_neutral_full_220m", (10, 20)),
        ("04_style_seed_201_30m", (5,)),
        ("04_style_seed_202_30m", (5,)),
    ):
        directory = tmp_path / stage
        directory.mkdir()
        for step in steps:
            (directory / f"2026_01_01_000000_{step}.onnx").write_bytes(b"x")
    walking, marching = policies_from_bundle(tmp_path)
    assert walking.parent.name == "03_neutral_full_220m"
    assert walking.name.endswith("_20.onnx")
    assert marching.parent.name == "04_style_seed_201_30m"


def test_bundle_without_the_expected_stages_fails_loudly(tmp_path):
    with pytest.raises(SystemExit):
        policies_from_bundle(tmp_path)
