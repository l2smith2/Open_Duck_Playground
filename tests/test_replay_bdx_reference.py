import json

from scripts import prepare_bdx_reference as pbr
from scripts import replay_bdx_reference as rbr


class FakeJointModel:
    """Minimal stand-in for the MJCF model check_joint_ranges reads ranges from."""

    def __init__(self, ranges):
        self._order = list(ranges)
        self.jnt_range = [ranges[name] for name in ranges]

    def joint_id(self, name):
        return self._order.index(name)


def _base(ranges):
    model = FakeJointModel(ranges)

    class Base:
        pass

    base = Base()
    base.model = model
    base.get_joint_id_from_name = model.joint_id
    return base


def _motion(joint_names, frames):
    return {"Joints": joint_names, "Frame_offset": [{"joints_pos": 0}], "Frames": frames}


def test_overshoot_within_tolerance_is_accepted():
    """--check must agree with generate()'s JointRangeChecker (same RANGE_TOLERANCE),
    or a bundle that already passed on Kaggle would show as OVER LIMIT here and
    the review fingerprint would never print."""
    base = _base({"left_knee": (-1.5708, 1.5708)})
    motion = _motion(["left_knee"], [[1.0], [1.9], [1.4]])
    problems = rbr.check_joint_ranges(base, motion, ["left_knee"], [0], [0])
    assert problems == []


def test_overshoot_beyond_tolerance_is_reported():
    base = _base({"left_hip_pitch": (-0.524, 1.222)})
    over = 1.222 + pbr.RANGE_TOLERANCE + 0.1
    motion = _motion(["left_hip_pitch"], [[0.0], [over]])
    problems = rbr.check_joint_ranges(base, motion, ["left_hip_pitch"], [0], [0])
    assert len(problems) == 1 and "tolerance" in problems[0]


def test_tolerance_matches_generate():
    """The two checks must share one constant, not two copies that can drift apart."""
    assert rbr.RANGE_TOLERANCE is pbr.RANGE_TOLERANCE
