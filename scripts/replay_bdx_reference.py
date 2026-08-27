"""Replay a generated reference motion in this fork's own MuJoCo viewer.

The upstream generator ships scripts/replay_motion.py, but it depends on
FramesViewer/placo, which has no Windows wheels and hits a GLX context error
under WSLg. This replays the same recorded JSON using the MuJoCo model and
viewer this fork already depends on, so review works on any platform that can
run mujoco_infer.py.

    uv run python scripts/replay_bdx_reference.py -f MOTION.json

Motions play back as recorded joint targets (mj_forward, no physics stepping),
so joint limits are not enforced by the viewer. Use --check to report whether
the recording stays inside the model's joint ranges.
"""

import argparse
import json
import time

import mujoco
import mujoco.viewer

from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase

DEFAULT_MODEL = "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"


def xyzw_to_wxyz(quat):
    x, y, z, w = quat
    return [w, x, y, z]


def check_joint_ranges(base, motion, joint_names, joint_addrs, joint_indices):
    """Report joint values in the recording that fall outside the model's ranges."""
    offsets = motion["Frame_offset"][0]
    joints_pos = offsets["joints_pos"]
    problems = []
    for name, addr, index in zip(joint_names, joint_addrs, joint_indices):
        joint_id = base.get_joint_id_from_name(name)
        low, high = base.model.jnt_range[joint_id]
        values = [
            frame[joints_pos + index] for frame in motion["Frames"]
        ]
        if min(values) < low or max(values) > high:
            problems.append(
                f"  {name}: recorded [{min(values):+.3f}, {max(values):+.3f}] "
                f"exceeds model range [{low:+.3f}, {high:+.3f}]"
            )
    if problems:
        print("Joint range violations (the robot cannot reach these):")
        print("\n".join(problems))
    else:
        print("All joints stay within the model's declared ranges.")
    return not problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--motion_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--loops", type=int, default=0, help="0 replays until interrupted")
    parser.add_argument(
        "--check",
        action="store_true",
        help="report joint-range violations and exit without opening the viewer",
    )
    args = parser.parse_args()

    with open(args.motion_file, encoding="utf-8") as handle:
        motion = json.load(handle)

    joint_names = motion["Joints"]
    offsets = motion["Frame_offset"][0]
    frames = motion["Frames"]
    dt = 1.0 / motion["FPS"]

    base = MJInferBase(args.model_path)
    model, data = base.model, base.data

    # Antennas and other joints in the recording may not exist in this model.
    known = set(base.joint_names)
    usable = [(i, name) for i, name in enumerate(joint_names) if name in known]
    skipped = [name for name in joint_names if name not in known]
    if skipped:
        print(f"Skipping joints not present in this model: {skipped}")
    joint_indices = [index for index, _ in usable]
    joint_addrs = [base.get_joint_addr_from_name(name)[0] for _, name in usable]
    usable_names = [name for _, name in usable]

    if args.check:
        ok = check_joint_ranges(base, motion, usable_names, joint_addrs, joint_indices)
        raise SystemExit(0 if ok else 1)

    print(f"Replaying {args.motion_file}: {len(frames)} frames at {motion['FPS']} FPS")

    qpos = data.qpos.copy()
    base_addr = base._floating_base_qpos_addr
    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        loops = 0
        while args.loops == 0 or loops < args.loops:
            for frame in frames:
                step_start = time.time()

                root_pos = frame[offsets["root_pos"]:offsets["root_pos"] + 3]
                root_quat = frame[offsets["root_quat"]:offsets["root_quat"] + 4]
                joints = frame[offsets["joints_pos"]:offsets["joints_pos"] + len(joint_names)]

                qpos[base_addr:base_addr + 3] = root_pos
                qpos[base_addr + 3:base_addr + 7] = xyzw_to_wxyz(root_quat)
                for addr, index in zip(joint_addrs, joint_indices):
                    qpos[addr] = joints[index]

                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                viewer.sync()

                remaining = dt - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
            loops += 1


if __name__ == "__main__":
    main()
