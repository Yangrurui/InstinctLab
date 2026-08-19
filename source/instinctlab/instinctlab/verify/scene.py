"""The scene the sim2sim assertions run against, described in the engine-neutral sim contract.

Extracted from the retired unified locomotion config, which paired this description with a full set
of MDP terms. The behavioural checks in ``tests/simulators/`` never used the terms: they write a
root state, step, and read a sensor back, so what they need is a robot, a floor, contact sensors
and the capabilities a backend must have to serve them.

Kept in the declaration the two ``SimulatorBackend`` implementations speak, not in ``spec/``. The
new compiler stack lowers a :class:`~instinctlab.spec.task.TaskSpec` into each engine's own config
and never goes through this contract; a sim2sim check deliberately stays underneath both of them,
which is what lets it compare the engines rather than the compilation.
"""

from __future__ import annotations

from dataclasses import dataclass

from instinctlab.assets import ASSETS
from instinctlab.sim.backend import JOINT_ACC_SOURCES, RuntimeRequirements
from instinctlab.sim.capabilities import Capability
from instinctlab.sim.scene import ContactSensorSpec, SceneSpec, SimulationSpec, TerrainSpec

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")

ILLEGAL_CONTACT_BODIES = (
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
)


@dataclass(frozen=True)
class VerificationScene:
    """What a backend needs to be brought up, and nothing about what is learned in it."""

    scene: SceneSpec
    simulation: SimulationSpec
    requirements: RuntimeRequirements


def locomotion_flat_scene(*, num_envs: int = 2) -> VerificationScene:
    """Flat-ground G1, with the two contact sensors the behavioural checks read.

    The numbers are the retired unified config's, unchanged, so a sim2sim result stays comparable
    to the ones recorded against it. ``num_envs`` defaults small because these checks assert on
    physics rather than on throughput.
    """
    return VerificationScene(
        scene=SceneSpec(
            num_envs=num_envs,
            env_spacing=2.5,
            robot=ASSETS.make("unitree_g1_29dof"),
            terrain=TerrainSpec(terrain_type="plane", sliding_friction=1.0, restitution=0.0),
            contact_sensors=(
                ContactSensorSpec(
                    name="feet_contact_forces",
                    entity_name="robot",
                    body_names=FEET,
                    history_length=3,
                    force_threshold=1.0,
                    track_air_time=True,
                ),
                ContactSensorSpec(
                    name="base_contact_forces",
                    entity_name="robot",
                    body_names=ILLEGAL_CONTACT_BODIES,
                    history_length=3,
                    force_threshold=1.0,
                    track_air_time=False,
                ),
            ),
            backend_options={
                "isaacsim": {
                    "scene": {
                        "lazy_sensor_update": True,
                        "replicate_physics": True,
                        "filter_collisions": True,
                    },
                    "robot_spawn": {
                        "self_collision": True,
                        "rigid_props": {
                            "disable_gravity": False,
                            "retain_accelerations": False,
                            "linear_damping": 0.0,
                            "angular_damping": 0.0,
                            "max_linear_velocity": 1000.0,
                            "max_angular_velocity": 1000.0,
                            "max_depenetration_velocity": 1.0,
                        },
                        "articulation_props": {
                            "enabled_self_collisions": True,
                            "solver_position_iteration_count": 8,
                            "solver_velocity_iteration_count": 4,
                        },
                    },
                }
            },
        ),
        simulation=SimulationSpec(
            sim_dt=0.005,
            decimation=4,
            engine_options={
                "mjlab": {
                    "njmax": 300,
                    "solver": "newton",
                    "iterations": 10,
                    "ls_iterations": 20,
                    "ccd_iterations": 500,
                }
            },
        ),
        requirements=RuntimeRequirements(
            capabilities=frozenset(
                {
                    Capability.BATCHED_SIMULATION,
                    Capability.PLANE_TERRAIN,
                    Capability.ROOT_STATE,
                    Capability.JOINT_STATE,
                    Capability.BODY_STATE,
                    Capability.IMPLICIT_POSITION_CONTROL,
                    Capability.CONTACT_ACTIVE,
                    Capability.CONTACT_HISTORY,
                    Capability.CONTACT_AIR_TIME,
                    Capability.CONTACT_FORCE_VECTOR,
                    Capability.DR_SLIDING_FRICTION,
                    Capability.BODY_MASS_PROPERTIES,
                    Capability.ROOT_VELOCITY_WRITE,
                }
            ),
            optional_capabilities=frozenset({Capability.DR_RESTITUTION}),
            randomization_fields=frozenset({"sliding_friction", "mass", "root_pose", "root_velocity", "joint_state"}),
            accepted_joint_acc_sources=JOINT_ACC_SOURCES,
        ),
    )


__all__ = ["FEET", "ILLEGAL_CONTACT_BODIES", "VerificationScene", "locomotion_flat_scene"]
