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


def test_off_grid_command_falls_back_to_nearest(tmp_path):
    keys = ["0.0_0.0_0.0", "0.04_0.0_0.0"]
    path = _make_pkl(tmp_path, keys)
    prm = PolyReferenceMotion(str(path))
    # Should not raise, and should match whichever recorded command is closer.
    near_stand = prm.get_reference_motion(0.005, 0.0, 0.0, 5)
    near_forward = prm.get_reference_motion(0.035, 0.0, 0.0, 5)
    stand = prm.get_reference_motion(0.0, 0.0, 0.0, 5)
    forward = prm.get_reference_motion(0.04, 0.0, 0.0, 5)
    assert np.allclose(near_stand, stand)
    assert np.allclose(near_forward, forward)


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
