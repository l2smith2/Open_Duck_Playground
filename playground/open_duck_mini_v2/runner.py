"""Runs training and evaluation loop for Open Duck Mini V2."""

import argparse
import json
from pathlib import Path

from playground.common import randomize
from playground.common.runner import BaseRunner
from playground.open_duck_mini_v2 import joystick, standing


class OpenDuckMiniV2Runner(BaseRunner):

    def __init__(self, args):
        super().__init__(args)
        available_envs = {
            "joystick": (joystick, joystick.Joystick),
            "standing": (standing, standing.Standing),
        }
        if args.env not in available_envs:
            raise ValueError(f"Unknown env {args.env}")

        self.env_file = available_envs[args.env]
        self.env_config = self.env_file[0].default_config()
        eval_config = self.env_file[0].default_config()
        if args.imitation_reward_weight_scale != 1.0:
            if args.env != "joystick":
                raise ValueError(
                    "Imitation reward scaling is only valid for the joystick environment"
                )
            self.env_config.reward_config.scales.imitation *= (
                args.imitation_reward_weight_scale
            )
            eval_config.reward_config.scales.imitation *= (
                args.imitation_reward_weight_scale
            )

        self.env = self.env_file[1](task=args.task, config=self.env_config)
        self.eval_env = self.env_file[1](task=args.task, config=eval_config)
        self.randomization_config = randomize.get_mass_randomization_config(
            args.randomization_stage, args.com_offset_scale
        )
        self.randomizer = randomize.make_domain_randomizer(
            self.env.mj_model, args.randomization_stage, args.com_offset_scale
        )
        self.action_size = self.env.action_size
        self.obs_size = int(
            self.env.observation_size["state"][0]
        )  # 0: state 1: privileged_state
        self.restore_checkpoint_path = args.restore_checkpoint_path
        print(f"Observation size: {self.obs_size}")
        print(f"Mass randomization stage: {args.randomization_stage}")

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "seed": args.seed,
            "num_timesteps": args.num_timesteps,
            "env": args.env,
            "task": args.task,
            "restore_checkpoint_path": args.restore_checkpoint_path,
            "randomization": self.randomization_config.to_dict(),
            "policy_interface": "unchanged; masses and COM offsets are hidden",
            "head_imu": False,
            "imitation_reward_weight_scale": args.imitation_reward_weight_scale,
            "com_offset_scale": args.com_offset_scale,
        }
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Open Duck Mini Runner Script")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Where to save the checkpoints",
    )
    parser.add_argument("--num_timesteps", type=int, default=150000000)
    parser.add_argument("--env", type=str, default="joystick", help="env")
    parser.add_argument("--task", type=str, default="flat_terrain", help="Task to run")
    parser.add_argument("--seed", type=int, default=0, help="PPO random seed")
    parser.add_argument(
        "--imitation_reward_weight_scale",
        type=float,
        choices=(0.5, 1.0, 1.5, 2.0),
        default=1.0,
        help="Only approved reward adjustment for a stable policy that ignores the reference",
    )
    parser.add_argument(
        "--com_offset_scale",
        type=float,
        choices=(0.5, 1.0),
        default=1.0,
        help="Use 0.5 only when retrying a failed stage with halved COM offsets",
    )
    parser.add_argument(
        "--randomization_stage",
        choices=tuple(randomize.MASS_RANDOMIZATION_STAGES),
        default="nominal",
        help="Structured trunk/head mass and COM randomization stage",
    )
    parser.add_argument(
        "--restore_checkpoint_path",
        type=str,
        default=None,
        help="Resume training from this checkpoint",
    )
    args = parser.parse_args()

    runner = OpenDuckMiniV2Runner(args)
    runner.train()


if __name__ == "__main__":
    main()
