"""Render an ONNX policy to an mp4 under a fixed, reproducible command schedule.

Blind A/B style review is only meaningful if the two clips differ by policy and
nothing else, so this fixes everything a reviewer could otherwise read as style:
the command schedule, the camera, the duration, and the frame rate. Running it
twice with different --onnx produces directly comparable clips.

The model defines no cameras, so a free camera tracks the floating base at a
fixed distance and angle.

Headless machines need a software/EGL GL backend, for example:

    MUJOCO_GL=egl uv run python scripts/record_policy_video.py ...
"""

import argparse
import json
from pathlib import Path

import mediapy
import mujoco
import numpy as np

from playground.open_duck_mini_v2.mujoco_infer import MjInfer, USE_MOTOR_SPEED_LIMITS

# (seconds, [dx, dy, dtheta]) -- exercises the motions the style reference covers.
DEFAULT_SCHEDULE = (
    (3.0, [0.0, 0.0, 0.0]),
    (6.0, [0.10, 0.0, 0.0]),
    (3.0, [0.0, 0.0, 0.0]),
    (4.0, [0.0, 0.08, 0.0]),
    (4.0, [0.0, 0.0, 0.5]),
    (6.0, [0.15, 0.0, 0.0]),
    (2.0, [0.0, 0.0, 0.0]),
)


class PolicyRecorder:
    def __init__(self, args):
        self.args = args
        self.sim = MjInfer(
            str(args.model_path), str(args.reference_data), str(args.onnx), standing=False
        )
        self.model = self.sim.model
        self.data = self.sim.data
        self.qpos_addr = self.sim._floating_base_qpos_addr
        # .adr is a length-1 array on this MuJoCo version, not a scalar.
        self.upvector_adr = int(np.asarray(self.model.sensor("upvector").adr).reshape(-1)[0])

        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = args.camera_distance
        self.camera.azimuth = args.camera_azimuth
        self.camera.elevation = args.camera_elevation

    def _reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.keyframe("home").qpos
        self.data.ctrl[:] = self.sim.default_actuator
        self.sim.last_action[:] = 0
        self.sim.last_last_action[:] = 0
        self.sim.last_last_last_action[:] = 0
        self.sim.motor_targets = self.sim.default_actuator.copy()
        self.sim.prev_motor_targets = self.sim.default_actuator.copy()
        self.sim.commands = [0.0] * 7
        self.sim.imitation_i = 0
        self.sim.imitation_phase = np.array([0.0, 0.0])
        mujoco.mj_forward(self.model, self.data)

    def _control_step(self) -> None:
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
            delta = self.sim.max_motor_velocity * (self.sim.sim_dt * self.sim.decimation)
            self.sim.motor_targets = np.clip(
                self.sim.motor_targets,
                self.sim.prev_motor_targets - delta,
                self.sim.prev_motor_targets + delta,
            )
            self.sim.prev_motor_targets = self.sim.motor_targets.copy()
        self.data.ctrl[:] = self.sim.motor_targets

    def record(self, schedule) -> dict:
        self._reset()
        steps_per_frame = max(1, int(round(1.0 / (self.args.fps * self.sim.sim_dt))))
        frames = []
        fell_at = None
        step = 0
        with mujoco.Renderer(self.model, self.args.height, self.args.width) as renderer:
            for seconds, command in schedule:
                self.sim.commands = list(command) + [0.0] * (7 - len(command))
                for _ in range(int(seconds / self.sim.sim_dt)):
                    mujoco.mj_step(self.model, self.data)
                    if step % self.sim.decimation == 0:
                        self._control_step()
                    if step % steps_per_frame == 0:
                        self.camera.lookat[:] = self.data.qpos[
                            self.qpos_addr:self.qpos_addr + 3
                        ]
                        renderer.update_scene(self.data, camera=self.camera)
                        frames.append(renderer.render())
                    if fell_at is None and not self._upright_and_finite():
                        fell_at = step * self.sim.sim_dt
                    step += 1

        if not frames:
            raise RuntimeError("No frames rendered; check the schedule and fps")
        self.args.output.parent.mkdir(parents=True, exist_ok=True)
        mediapy.write_video(str(self.args.output), frames, fps=self.args.fps)
        return {
            "onnx": str(self.args.onnx),
            "output": str(self.args.output),
            "frames": len(frames),
            "fps": self.args.fps,
            "duration_seconds": len(frames) / self.args.fps,
            "schedule": [[seconds, list(command)] for seconds, command in schedule],
            "fell_at_seconds": fell_at,
        }

    def _upright_and_finite(self) -> bool:
        finite = np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))
        upright = float(self.data.sensordata[self.upvector_adr + 2]) >= 0.0
        return bool(finite and upright)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="mp4 path")
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
    parser.add_argument(
        "--schedule", type=Path,
        help="JSON list of [seconds, [dx, dy, dtheta]]; defaults to the built-in schedule",
    )
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=1.2)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-15.0)
    args = parser.parse_args()
    if args.fps < 1 or args.width < 1 or args.height < 1:
        raise ValueError("fps/width/height must be positive")
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("--output must be an .mp4 path")

    if args.schedule:
        schedule = [
            (float(seconds), list(command))
            for seconds, command in json.loads(args.schedule.read_text(encoding="utf-8"))
        ]
        if not schedule or any(seconds <= 0 for seconds, _ in schedule):
            raise ValueError("schedule entries need positive durations")
    else:
        schedule = list(DEFAULT_SCHEDULE)

    report = PolicyRecorder(args).record(schedule)
    print(json.dumps(report, indent=2))
    if report["fell_at_seconds"] is not None:
        print(
            f"NOTE: the policy fell at {report['fell_at_seconds']:.2f}s. "
            "The clip still renders, but it is not a fair style comparison."
        )


if __name__ == "__main__":
    main()
