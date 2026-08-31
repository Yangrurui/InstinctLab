"""Explicit Isaac-native fixture asset declaration."""

from __future__ import annotations

from dataclasses import dataclass

from instinctlab_engine.assets import NativeActuatorGroup

from .asset_common import JOINT, NativeJoint, resource

INSTINCTLAB_NATIVE_ASSET_API = "0.1"


@dataclass(frozen=True)
class NativeConfig:
    name: str
    schema_version: str
    asset_id: str
    root_body: str
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    frame_names: tuple[str, ...]
    collision_body_names: tuple[str, ...]
    joint_properties: tuple[NativeJoint, ...]
    contact_body_aliases: dict[str, str]
    default_root_pos: tuple[float, float, float]
    default_root_quat_wxyz: tuple[float, float, float, float]
    soft_joint_pos_limit_factor: float
    actuator_delay: tuple[int, int]
    actuator_model_ids: tuple[str, ...]
    actuator_group_count: int
    actuator_groups: tuple[NativeActuatorGroup, ...]
    length_unit: str
    angle_unit: str
    effort_unit: str
    urdf_path: str
    import_options: dict[str, object]


_CONFIG = NativeConfig(
    name="fixture_bot",
    schema_version="dfs_v1",
    asset_id="fixture_bot/v1",
    root_body="base",
    joint_names=("joint",),
    body_names=("base", "link"),
    frame_names=(),
    collision_body_names=("base", "link"),
    joint_properties=(JOINT,),
    contact_body_aliases={},
    default_root_pos=(0.0, 0.0, 0.5),
    default_root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    soft_joint_pos_limit_factor=0.9,
    actuator_delay=(1, 1),
    actuator_model_ids=("fixture.stateful.v1",),
    actuator_group_count=1,
    actuator_groups=(
        NativeActuatorGroup(
            name="joint",
            model_id="fixture.stateful.v1",
            selectors=("joint",),
        ),
    ),
    length_unit="m",
    angle_unit="rad",
    effort_unit="N*m",
    urdf_path=resource("fixture_bot.urdf"),
    import_options={},
)


def native_config(variant: str) -> NativeConfig:
    if variant != "v1":
        raise KeyError(variant)
    return _CONFIG


def articulation(variant: str, robot):
    """Materialize a genuine Isaac Lab articulation after Kit bootstrap."""
    import isaaclab.sim as sim_utils
    from instinctlab_engine.actuators import native_actuator_factory
    from instinctlab_engine.assets import validate_native_actuator_groups
    from isaaclab.assets import ArticulationCfg

    config = native_config(variant)
    actuator_cfg = native_actuator_factory("isaacsim", "fixture.stateful.v1")(
        joint_names_expr=["joint"],
        effort_limit=3.0,
        velocity_limit=4.0,
        stiffness=2.0,
        damping=0.1,
        armature=0.01,
        min_delay=1,
        max_delay=1,
    )
    groups = {"joint": actuator_cfg}
    validate_native_actuator_groups(
        config.asset_id,
        groups,
        tuple(robot.joint_names),
        selector_field="joint_names_expr",
        expected_groups=config.actuator_groups,
    )
    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            asset_path=config.urdf_path,
            fix_base=False,
            activate_contact_sensors=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0,
                    damping=0.0,
                )
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=robot.default_root_pos,
            rot=robot.default_root_quat_wxyz,
            joint_pos={"joint": 0.0},
            joint_vel={"joint": 0.0},
        ),
        soft_joint_pos_limit_factor=config.soft_joint_pos_limit_factor,
        actuators=groups,
    )
