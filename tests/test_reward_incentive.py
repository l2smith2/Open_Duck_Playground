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
    default_scales,
    get_rz,
    score,
    tracking_terms,
)
from playground.open_duck_mini_v2.joystick import default_config  # noqa: E402


def _record(linvel, gyro_z, command=(0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), **terms):
    base = {name: 0.0 for name in default_scales()}
    base.update(terms)
    return {
        "terms": base,
        "command": list(command),
        "local_linvel_xy": [list(v) for v in linvel],
        "gyro_z": list(gyro_z),
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
    record = _record(
        linvel=[[0.10, 0.0]],
        gyro_z=[0.0],
        imitation=2.0,
        alive=1.0,
    )
    scales = default_scales()
    result = score(record, scales, 0.01, 0.25)
    assert result["terms"]["imitation"] == pytest.approx(2.0 * scales["imitation"])
    assert result["terms"]["alive"] == pytest.approx(scales["alive"])
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
