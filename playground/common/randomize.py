# Copyright 2025 DeepMind Technologies Limited
# Copyright 2025 Antoine Pirrone - Steve Nguyen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Named-body domain randomization for Open Duck Mini V2.

Masses remain hidden from the policy, preserving the exported ONNX interface.
"""

from dataclasses import asdict, dataclass, replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jp
from mujoco import mjx

FLOOR_GEOM_ID = 0
TRUNK_BODY_NAME = "trunk_assembly"
HEAD_BODY_NAME = "head_assembly"
EXPECTED_TRUNK_MASS_KG = 0.698526
EXPECTED_HEAD_MASS_KG = 0.406607
NOMINAL_MASS_TOLERANCE_KG = 0.002

PAIRED_LEG_BODY_NAMES = (
    ("hip_roll_assembly", "hip_roll_assembly_2"),
    ("left_roll_to_pitch_assembly", "right_roll_to_pitch_assembly"),
    ("knee_and_ankle_assembly", "knee_and_ankle_assembly_3"),
    ("knee_and_ankle_assembly_2", "knee_and_ankle_assembly_4"),
    ("foot_assembly", "foot_assembly_2"),
)


@dataclass(frozen=True)
class MassRandomizationConfig:
    """Mass/COM bounds for one cumulative training stage."""

    stage: str
    trunk_mass_scale: tuple[float, float]
    head_mass_scale: tuple[float, float]
    other_mass_scale: tuple[float, float]
    trunk_com_offset_m: tuple[float, float, float]
    head_com_offset_m: tuple[float, float, float]
    paired_leg_mismatch: float = 0.02

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MASS_RANDOMIZATION_STAGES = {
    "nominal": MassRandomizationConfig(
        "nominal", (1.0, 1.0), (1.0, 1.0), (1.0, 1.0),
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0,
    ),
    "moderate": MassRandomizationConfig(
        "moderate", (0.90, 1.10), (0.85, 1.15), (0.95, 1.05),
        (0.005, 0.005, 0.005), (0.005, 0.005, 0.005),
    ),
    "full": MassRandomizationConfig(
        "full", (0.80, 1.30), (0.65, 1.40), (0.95, 1.05),
        (0.015, 0.010, 0.015), (0.015, 0.010, 0.020),
    ),
}


def get_mass_randomization_config(
    stage: str, com_offset_scale: float = 1.0
) -> MassRandomizationConfig:
    try:
        config = MASS_RANDOMIZATION_STAGES[stage]
    except KeyError as exc:
        choices = ", ".join(MASS_RANDOMIZATION_STAGES)
        raise ValueError(f"Unknown randomization stage {stage!r}; choose {choices}") from exc
    if com_offset_scale not in (0.5, 1.0):
        raise ValueError("com_offset_scale must be 0.5 (recovery) or 1.0")
    if com_offset_scale != 1.0:
        config = replace(
            config,
            trunk_com_offset_m=tuple(
                value * com_offset_scale for value in config.trunk_com_offset_m
            ),
            head_com_offset_m=tuple(
                value * com_offset_scale for value in config.head_com_offset_m
            ),
        )
    validate_mass_randomization_config(config)
    return config


def validate_mass_randomization_config(config: MassRandomizationConfig) -> None:
    for label, bounds in (
        ("trunk_mass_scale", config.trunk_mass_scale),
        ("head_mass_scale", config.head_mass_scale),
        ("other_mass_scale", config.other_mass_scale),
    ):
        low, high = bounds
        if low <= 0 or high <= 0 or low > high:
            raise ValueError(f"Invalid {label}: {bounds}")
    for label, offsets in (
        ("trunk_com_offset_m", config.trunk_com_offset_m),
        ("head_com_offset_m", config.head_com_offset_m),
    ):
        if len(offsets) != 3 or any(value < 0 for value in offsets):
            raise ValueError(f"Invalid {label}: {offsets}")
    if not 0.0 <= config.paired_leg_mismatch <= 0.02:
        raise ValueError("paired_leg_mismatch must be between 0 and 0.02")


def _body_id(mj_model: Any, name: str) -> int:
    try:
        body_id = int(mj_model.body(name).id)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Required MuJoCo body {name!r} was not found") from exc
    if body_id < 0:
        raise ValueError(f"Required MuJoCo body {name!r} was not found")
    return body_id


def resolve_and_validate_bodies(
    mj_model: Any,
) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    """Resolve body names before JIT compilation and assert the expected MJCF."""

    trunk_id = _body_id(mj_model, TRUNK_BODY_NAME)
    head_id = _body_id(mj_model, HEAD_BODY_NAME)
    trunk_mass = float(mj_model.body_mass[trunk_id])
    head_mass = float(mj_model.body_mass[head_id])
    if abs(trunk_mass - EXPECTED_TRUNK_MASS_KG) > NOMINAL_MASS_TOLERANCE_KG:
        raise ValueError(
            f"Unexpected nominal trunk mass {trunk_mass:.6f} kg; "
            f"expected about {EXPECTED_TRUNK_MASS_KG:.6f} kg"
        )
    if abs(head_mass - EXPECTED_HEAD_MASS_KG) > NOMINAL_MASS_TOLERANCE_KG:
        raise ValueError(
            f"Unexpected nominal head mass {head_mass:.6f} kg; "
            f"expected about {EXPECTED_HEAD_MASS_KG:.6f} kg"
        )
    pairs = tuple(
        (_body_id(mj_model, left), _body_id(mj_model, right))
        for left, right in PAIRED_LEG_BODY_NAMES
    )
    if len({body_id for pair in pairs for body_id in pair}) != 2 * len(pairs):
        raise ValueError("Paired leg body names must resolve to distinct bodies")
    for body_id, mass in enumerate(mj_model.body_mass):
        mass = float(mass)
        inertia = tuple(float(value) for value in mj_model.body_inertia[body_id])
        if mass < 0:
            raise ValueError(f"Body {body_id} has a negative nominal mass")
        if mass > 0 and any(value <= 0 for value in inertia):
            raise ValueError(f"Body {body_id} has non-positive nominal inertia")
    for body_id in {trunk_id, head_id, *(item for pair in pairs for item in pair)}:
        if float(mj_model.body_mass[body_id]) <= 0:
            raise ValueError(f"Randomized body {body_id} must have a positive nominal mass")
    return trunk_id, head_id, pairs


def make_domain_randomizer(
    mj_model: Any, stage: str, com_offset_scale: float = 1.0
):
    """Build the two-argument callback expected by Brax training."""

    config = get_mass_randomization_config(stage, com_offset_scale)
    trunk_id, head_id, paired_ids = resolve_and_validate_bodies(mj_model)
    return partial(
        domain_randomize,
        config=config,
        trunk_body_id=trunk_id,
        head_body_id=head_id,
        paired_leg_body_ids=paired_ids,
    )


def domain_randomize(
    model: mjx.Model,
    rng: jax.Array,
    *,
    config: MassRandomizationConfig,
    trunk_body_id: int,
    head_body_id: int,
    paired_leg_body_ids: tuple[tuple[int, int], ...],
):
    """Randomize a batch of MJX models while preserving physical validity."""

    dof_id = jp.array(
        [idx for idx, has_friction in enumerate(model.dof_hasfrictionloss) if has_friction]
    )
    jnt_id = model.dof_jntid[dof_id]
    dof_addr = jp.array([addr for addr in model.jnt_dofadr if addr in dof_id])
    joint_addr = model.jnt_qposadr[jnt_id]

    @jax.vmap
    def rand_dynamics(one_rng):
        one_rng, key = jax.random.split(one_rng)
        geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
            jax.random.uniform(key, minval=0.5, maxval=1.0)
        )

        one_rng, key = jax.random.split(one_rng)
        frictionloss = model.dof_frictionloss[dof_addr] * jax.random.uniform(
            key, shape=(model.nu,), minval=0.9, maxval=1.1
        )
        dof_frictionloss = model.dof_frictionloss.at[dof_addr].set(frictionloss)

        one_rng, key = jax.random.split(one_rng)
        armature = model.dof_armature[dof_addr] * jax.random.uniform(
            key, shape=(model.nu,), minval=1.0, maxval=1.05
        )
        dof_armature = model.dof_armature.at[dof_addr].set(armature)

        one_rng, key = jax.random.split(one_rng)
        body_scale = jax.random.uniform(
            key, shape=(model.nbody,), minval=config.other_mass_scale[0],
            maxval=config.other_mass_scale[1]
        )
        body_scale = jp.where(model.body_mass > 0, body_scale, 1.0)

        one_rng, trunk_key, head_key = jax.random.split(one_rng, 3)
        trunk_scale = jax.random.uniform(
            trunk_key, minval=config.trunk_mass_scale[0], maxval=config.trunk_mass_scale[1]
        )
        head_scale = jax.random.uniform(
            head_key, minval=config.head_mass_scale[0], maxval=config.head_mass_scale[1]
        )
        body_scale = body_scale.at[trunk_body_id].set(trunk_scale)
        body_scale = body_scale.at[head_body_id].set(head_scale)

        for left_id, right_id in paired_leg_body_ids:
            one_rng, main_key, mismatch_key = jax.random.split(one_rng, 3)
            margin = config.paired_leg_mismatch
            main_low = config.other_mass_scale[0] / max(1.0 - margin, 1e-6)
            main_high = config.other_mass_scale[1] / (1.0 + margin)
            main_scale = jax.random.uniform(main_key, minval=main_low, maxval=main_high)
            mismatch = jax.random.uniform(
                mismatch_key, shape=(2,), minval=1.0 - margin, maxval=1.0 + margin
            )
            pair_scale = jp.clip(
                main_scale * mismatch, config.other_mass_scale[0], config.other_mass_scale[1]
            )
            body_scale = body_scale.at[left_id].set(pair_scale[0])
            body_scale = body_scale.at[right_id].set(pair_scale[1])

        body_mass = model.body_mass * body_scale
        body_inertia = model.body_inertia * body_scale[:, None]

        one_rng, trunk_key, head_key = jax.random.split(one_rng, 3)
        trunk_limit = jp.asarray(config.trunk_com_offset_m)
        head_limit = jp.asarray(config.head_com_offset_m)
        trunk_offset = jax.random.uniform(
            trunk_key, (3,), minval=-trunk_limit, maxval=trunk_limit
        )
        head_offset = jax.random.uniform(
            head_key, (3,), minval=-head_limit, maxval=head_limit
        )
        body_ipos = model.body_ipos.at[trunk_body_id].set(
            model.body_ipos[trunk_body_id] + trunk_offset
        )
        body_ipos = body_ipos.at[head_body_id].set(
            model.body_ipos[head_body_id] + head_offset
        )

        one_rng, key = jax.random.split(one_rng)
        qpos0 = model.qpos0.at[joint_addr].set(
            model.qpos0[joint_addr]
            + jax.random.uniform(key, shape=(model.nu,), minval=-0.03, maxval=0.03)
        )

        one_rng, key = jax.random.split(one_rng)
        kp_factor = jax.random.uniform(key, shape=(model.nu,), minval=0.9, maxval=1.1)
        current_kp = model.actuator_gainprm[:, 0]
        actuator_gainprm = model.actuator_gainprm.at[:, 0].set(current_kp * kp_factor)
        actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-current_kp * kp_factor)

        return (
            geom_friction, body_ipos, dof_frictionloss, dof_armature,
            body_mass, body_inertia, qpos0, actuator_gainprm, actuator_biasprm,
        )

    values = rand_dynamics(rng)
    fields = (
        "geom_friction", "body_ipos", "dof_frictionloss", "dof_armature",
        "body_mass", "body_inertia", "qpos0", "actuator_gainprm", "actuator_biasprm",
    )
    replacements = dict(zip(fields, values, strict=True))
    in_axes = jax.tree_util.tree_map(lambda _: None, model)
    in_axes = in_axes.tree_replace({field: 0 for field in fields})
    return model.tree_replace(replacements), in_axes
