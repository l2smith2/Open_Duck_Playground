import pickle

import numpy as np
import pytest

from playground.common.poly_reference_motion_numpy import PolyReferenceMotion


def _make_pkl(tmp_path, keys):
    """Build a minimal valid polynomial_coefficients.pkl with the given dx_dy_dtheta keys.

    Each entry's coefficients are derived from its own key so distinct
    commands are guaranteed to produce distinct sampled values, letting tests
    tell whether the loader actually picked the right entry.
    """
    data = {}
    for key in keys:
        dx, dy, dtheta = (float(v) for v in key.split("_"))
        offset = dx * 10 + dy * 3 + dtheta
        data[key] = {
            "coefficients": {"dim_0": [offset, 1.0], "dim_1": [1.0, offset]},
            "period": 0.4,
            "fps": 50,
            "frame_offsets": {"joints_pos": 7},
            "startend_double_support_ratio": 1.5,
        }
    path = tmp_path / "polynomial_coefficients.pkl"
    with open(path, "wb") as handle:
        pickle.dump(data, handle)
    return path


def test_sparse_cross_grid_loads_and_distinguishes_commands(tmp_path):
    # Mirrors this fork's eight bdx_inspired motions: one axis varies at a
    # time, not a dense dx*dy*dtheta grid. This used to crash with a
    # KeyError because the loader assumed every combination was present.
    keys = [
        "0.0_0.0_0.0",
        "0.02_0.0_0.0",
        "0.04_0.0_0.0",
        "-0.02_0.0_0.0",
        "0.0_0.02_0.0",
        "0.0_-0.02_0.0",
        "0.0_0.0_0.15",
        "0.0_0.0_-0.15",
    ]
    path = _make_pkl(tmp_path, keys)
    prm = PolyReferenceMotion(str(path))
    assert len(prm.data_array) == len(keys)

    stand = prm.get_reference_motion(0.0, 0.0, 0.0, 5)
    forward = prm.get_reference_motion(0.04, 0.0, 0.0, 5)
    turn = prm.get_reference_motion(0.0, 0.0, 0.15, 5)
    assert not np.allclose(stand, forward)
    assert not np.allclose(stand, turn)


def test_off_grid_command_interpolates_between_recordings(tmp_path):
    # Nearest-neighbour lookup made the reference a staircase: every command
    # from 0.037 to 0.111 m/s was served the same 0.074 m/s gait, so a 0.10 m/s
    # command was asked to imitate a gait translating at 74% of it while
    # tracking_lin_vel asked for the full speed.
    keys = ["0.0_0.0_0.0", "0.04_0.0_0.0"]
    path = _make_pkl(tmp_path, keys)
    prm = PolyReferenceMotion(str(path))
    stand = np.array(prm.get_reference_motion(0.0, 0.0, 0.0, 5))
    forward = np.array(prm.get_reference_motion(0.04, 0.0, 0.0, 5))
    # The recorded commands still reproduce their own recording exactly.
    assert np.allclose(prm.get_reference_motion(0.0, 0.0, 0.0, 5), stand)
    assert np.allclose(prm.get_reference_motion(0.04, 0.0, 0.0, 5), forward)
    # A command a quarter of the way between them lands a quarter of the way
    # between the two motions, not on top of the nearer one.
    quarter = np.array(prm.get_reference_motion(0.01, 0.0, 0.0, 5))
    assert np.allclose(quarter, 0.75 * stand + 0.25 * forward)
    assert not np.allclose(quarter, stand)


def test_command_outside_the_recorded_range_clamps_to_the_extreme(tmp_path):
    # Blending must not extrapolate past the fastest reviewed motion.
    keys = ["0.0_0.0_0.0", "0.04_0.0_0.0"]
    path = _make_pkl(tmp_path, keys)
    prm = PolyReferenceMotion(str(path))
    forward = prm.get_reference_motion(0.04, 0.0, 0.0, 5)
    assert np.allclose(prm.get_reference_motion(0.4, 0.0, 0.0, 5), forward)


def test_blend_partner_is_chosen_to_close_the_gap_not_by_proximity(tmp_path):
    # On the dense grid the two nearest commands normally differ along an axis
    # the query does not need, so blending with the second-nearest entry moves
    # sideways and leaves dx untouched. The partner is picked by how much error
    # it removes instead.
    keys = ["0.0_0.0_0.0", "0.0_0.02_0.0", "0.04_0.0_0.0"]
    path = _make_pkl(tmp_path, keys)
    prm = PolyReferenceMotion(str(path))
    stand = np.array(prm.get_reference_motion(0.0, 0.0, 0.0, 5))
    forward = np.array(prm.get_reference_motion(0.04, 0.0, 0.0, 5))
    lateral = np.array(prm.get_reference_motion(0.0, 0.02, 0.0, 5))
    blended = np.array(prm.get_reference_motion(0.02, 0.0, 0.0, 5))
    assert np.allclose(blended, 0.5 * stand + 0.5 * forward)
    assert not np.allclose(blended, lateral)


def test_axes_are_compared_on_their_own_scale(tmp_path):
    # dx runs over about +-0.15 m/s and dtheta over +-1.0 rad/s. Measured raw,
    # a physically trivial yaw difference outweighs a large speed difference and
    # the lookup picks a turning motion for a straight-ahead command.
    keys = ["0.0_0.0_0.0", "0.15_0.0_0.0", "0.0_0.0_1.0"]
    path = _make_pkl(tmp_path, keys)
    prm = PolyReferenceMotion(str(path))
    assert prm.vel_to_index(0.14, 0.0, 0.0) == 1


def test_dense_grid_still_works(tmp_path):
    # Backward-compat: the original auto-generated reference used a full
    # dx*dy*dtheta grid. Confirm that still loads correctly.
    keys = [
        f"{dx}_{dy}_{dtheta}"
        for dx in (-0.1, 0.0, 0.1)
        for dy in (-0.05, 0.05)
        for dtheta in (-0.2, 0.0, 0.2)
    ]
    path = _make_pkl(tmp_path, keys)
    prm = PolyReferenceMotion(str(path))
    assert len(prm.data_array) == len(keys)
    vals = prm.get_reference_motion(0.1, -0.05, 0.2, 3)
    assert len(vals) == 2
