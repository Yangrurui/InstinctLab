"""What holds for the proprioceptive parkour G1 declaration.

Modelled on ``test_rough_g1_declaration.py``. The reward set is a snapshot, not a claim about
any other implementation: changing the objective takes two edits and a sentence in the commit
message. ``engine_params`` is pinned to exactly two divergences -- a silently growing map is how
a task stops being portable.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

import instinctlab.mdp as mdp
from instinctlab.spec.capability import Requirement
from instinctlab.tasks.parkour.config.g1 import parkour_target_g1

REPO = Path(__file__).resolve().parent.parent
DECLARATION = REPO / "source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py"
PACKAGE_INIT = REPO / "source/instinctlab/instinctlab/tasks/parkour/config/g1/__init__.py"
AGENT = REPO / "source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py"
NOT_PORTABLE = {
    "feet_slide",
    "dof_torques_l2",
    "dof_acc_l2",
    "energy",
    "torque_limits",
    "undesired_contacts",
}
"""Rewards each backend implements itself. Present in the task, named by kind rather than by function."""

REQUIRED_KIND = frozenset(NOT_PORTABLE)

REWARDS = {
    "track_lin_vel_xy_exp": 2.0,
    "track_ang_vel_z_exp": 2.0,
    "heading_error": -1.0,
    "dont_wait": -0.5,
    "is_alive": 3.0,
    "stand_still": -0.3,
    "volume_points_penetration": -4.0,
    "feet_air_time": 0.5,
    "feet_slide": -0.4,
    "joint_deviation_hip": -0.5,
    "ang_vel_xy_l2": -0.05,
    "dof_torques_l2": -1.5e-7,
    "dof_acc_l2": -1.25e-7,
    "dof_vel_l2": -1e-4,
    "action_rate_l2": -0.005,
    "flat_orientation_l2": -3.0,
    "pelvis_orientation_l2": -3.0,
    "feet_flat_ori": -0.4,
    "feet_at_plane": -0.1,
    "feet_close_xy": 0.4,
    "energy": -5e-5,
    "freeze_upper_body": -0.004,
    "dof_pos_limits": -1.0,
    "dof_vel_limits": -1.0,
    "torque_limits": -0.01,
    "undesired_contacts": -1.0,
}
"""A snapshot of the proprioceptive objective, not a reference."""

POLICY_ORDER = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
    "depth_image",
)
CRITIC_ORDER = ("base_lin_vel",) + POLICY_ORDER
AMP_ORDER = (
    "projected_gravity",
    "joint_pos_rel",
    "joint_vel",
    "base_lin_vel",
    "base_ang_vel",
)
JOINT_TERMS = {
    "policy": ("joint_pos", "joint_vel"),
    "critic": ("joint_pos", "joint_vel"),
    "amp_policy": ("joint_pos_rel", "joint_vel"),
    "amp_reference": ("joint_pos_rel", "joint_vel"),
}

TENTH_TERRAIN = {"isaacsim": "mesh_boxes", "mjlab": "dense_boxes"}
ISAAC_RESTITUTION = {
    "restitution_range": (0.05, 0.5),
    "num_buckets": 64,
    "make_consistent": True,
}


@pytest.fixture(scope="module")
def task():
    return parkour_target_g1()


def test_the_objective_is_the_one_recorded_here(task) -> None:
    declared = {name: term.weight for name, term in task.mdp.rewards["rewards"].items()}
    assert declared == REWARDS, (
        "the reward set or a weight changed. If that was deliberate, update REWARDS in the same "
        "commit and say why; existing checkpoints were trained against the old objective."
    )


def test_the_terms_named_by_kind_are_the_ones_the_design_names(task) -> None:
    rewards = task.mdp.rewards["rewards"]
    by_kind = {name for name, term in rewards.items() if term.kind is not None}
    by_func = {name for name, term in rewards.items() if term.func is not None}
    assert by_kind == NOT_PORTABLE
    assert by_func == set(REWARDS) - NOT_PORTABLE
    for name in REQUIRED_KIND:
        assert rewards[name].level is Requirement.REQUIRED, name


def test_the_foot_scanners_are_declared_and_feet_at_plane_is_required(task) -> None:
    """Absence of this reward silently drops a stance-height penalty; it is not optional."""
    from instinctlab.spec.sensor import RayCasterRef

    names = [sensor.name for sensor in task.scene.ray_casters]
    assert names[:2] == ["left_height_scanner", "right_height_scanner"]
    for sensor in task.scene.ray_casters[:2]:
        assert isinstance(sensor, RayCasterRef)
        assert sensor.offset == (0.04, 0.0, 20.0)
        assert sensor.hit == "terrain"
        assert sensor.miss == "infinity"
        assert sensor.ray_alignment == "yaw"
        assert sensor.pattern.kind == "grid"
        assert sensor.pattern.resolution == 0.12
        assert sensor.pattern.size == (0.12, 0.0)
    term = task.mdp.rewards["rewards"]["feet_at_plane"]
    assert term.func is not None
    assert term.weight == -0.1
    assert term.params["height_offset"] == 0.058
    assert term.level is Requirement.REQUIRED
    assert term.params["left_scanner"].name == "left_height_scanner"
    assert term.params["right_scanner"].name == "right_height_scanner"


def test_the_curriculum_is_required_and_reads_the_command_metrics(task) -> None:
    term = task.mdp.curriculum["terrain_levels"]
    assert term.func is not None
    assert term.params["command_name"] == "base_velocity"
    assert term.params["lin_vel_threshold"] == (0.3, 0.6)
    assert term.params["ang_vel_threshold"] == (0.0, 0.0)
    assert term.level is Requirement.REQUIRED


def test_the_command_is_pose_velocity_and_required(task) -> None:
    term = task.mdp.commands["base_velocity"]
    assert term.kind == "pose_velocity"
    assert term.level is Requirement.REQUIRED
    assert term.params["resampling_time_range"] == (8.0, 12.0)
    assert "debug_vis" not in term.params


def test_engine_params_carry_exactly_the_two_intended_divergences(task) -> None:
    """A tenth terrain name the engines spell differently, and Isaac-only restitution.

    Any other ``engine_params`` entry is a silent per-engine fork. Growing this map is how a
    task stops being portable.
    """
    found = {key: dict(term.engine_params) for key, term in task.mdp.terms().items() if term.engine_params}
    assert set(found) == {"command/base_velocity", "event/physics_material"}

    command = found["command/base_velocity"]
    assert set(command) == {"isaacsim", "mjlab"}
    assert command["isaacsim"] == {
        "velocity_ranges": {
            TENTH_TERRAIN["isaacsim"]: {
                "lin_vel_x": (0.45, 0.8),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-1.0, 1.0),
            }
        }
    }
    assert command["mjlab"] == {
        "velocity_ranges": {
            TENTH_TERRAIN["mjlab"]: {
                "lin_vel_x": (0.45, 0.8),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-1.0, 1.0),
            }
        }
    }

    material = found["event/physics_material"]
    assert set(material) == {"isaacsim"}
    assert material["isaacsim"] == ISAAC_RESTITUTION
    assert "restitution_range" not in task.mdp.events["physics_material"].params

    command_term = task.mdp.commands["base_velocity"]
    isaac_ranges = command_term.resolved_params("isaacsim")["velocity_ranges"]
    mjlab_ranges = command_term.resolved_params("mjlab")["velocity_ranges"]
    assert TENTH_TERRAIN["isaacsim"] in isaac_ranges
    assert TENTH_TERRAIN["mjlab"] not in isaac_ranges
    assert TENTH_TERRAIN["mjlab"] in mjlab_ranges
    assert TENTH_TERRAIN["isaacsim"] not in mjlab_ranges
    assert len(isaac_ranges) == 10
    assert len(mjlab_ranges) == 10


def test_the_task_declares_no_engine_specific_escape_hatch(task) -> None:
    assert not task.engine_extras
    assert set(task.engines) == {"isaacsim", "mjlab"}
    assert task.scene.terrain.kind == "rough"
    assert task.scene.terrain.generator is None
    assert task.task_id == "Instinct-Parkour-Target-G1"


def test_observation_group_order(task) -> None:
    assert tuple(task.mdp.observations["policy"].terms) == POLICY_ORDER
    assert tuple(task.mdp.observations["critic"].terms) == CRITIC_ORDER
    assert tuple(task.mdp.observations["amp_policy"].terms) == AMP_ORDER
    assert tuple(task.mdp.observations["amp_reference"].terms) == AMP_ORDER
    assert task.mdp.observations["policy"].concatenate_terms is False
    assert task.mdp.observations["critic"].concatenate_terms is False
    assert task.mdp.observations["amp_policy"].concatenate_terms is False
    assert task.mdp.observations["amp_reference"].concatenate_terms is False
    assert task.mdp.observations["amp_policy"].enable_corruption is False
    assert task.mdp.observations["amp_reference"].enable_corruption is False
    assert task.mdp.observations["critic"].terms["base_lin_vel"].noise is None
    for name in AMP_ORDER:
        assert task.mdp.observations["amp_policy"].terms[name].history_length == 10
        assert task.mdp.observations["amp_reference"].terms[name].history_length == 10
    assert task.mdp.observations["amp_policy"].terms["joint_vel"].scale == 0.05
    assert task.mdp.observations["amp_reference"].terms["joint_vel"].scale == 0.05


def test_volume_points_and_edge_cylinders_are_declared_and_required(task) -> None:
    """Absence of this penalty is a robot that never learns to avoid edges."""
    from instinctlab.spec.capability import Requirement
    from instinctlab.spec.sensor import VirtualObstacleRef, VolumePointsRef

    sensor = task.scene.volume_point("leg_volume_points")
    assert isinstance(sensor, VolumePointsRef)
    assert sensor.attach == ("left_ankle_roll_link", "right_ankle_roll_link")
    assert sensor.grid.z_min == -0.063
    assert sensor.grid.z_max == -0.023
    assert sensor.grid.x_num * sensor.grid.y_num * sensor.grid.z_num == 100
    assert sensor.frame == "attach"
    assert sensor.quaternion == "wxyz"
    assert sensor.velocity == "attach_link"
    obstacles = task.scene.terrain.virtual_obstacles
    assert len(obstacles) == 1
    assert isinstance(obstacles[0], VirtualObstacleRef)
    assert obstacles[0].name == "edges"
    assert obstacles[0].kind == "greedy_edge_cylinder"
    assert obstacles[0].cylinder_radius == 0.05
    assert obstacles[0].min_points == 2
    term = task.mdp.rewards["rewards"]["volume_points_penetration"]
    assert term.func is not None
    assert term.weight == -4.0
    assert term.level is Requirement.REQUIRED
    assert term.params["sensor"] is sensor
    event = task.mdp.events["register_virtual_obstacles"]
    assert event.kind == "register_virtual_obstacles"
    assert event.mode == "startup"
    assert event.level is Requirement.REQUIRED
    assert event.params["sensor"] is sensor


def test_the_depth_camera_is_declared_and_required(task) -> None:
    """Absence of this observation silently drops the policy's exteroception."""
    from instinctlab.assets.unitree_g1.isaacsim import G1_29DOF_LINKS
    from instinctlab.spec.capability import Requirement

    camera = task.scene.ray_caster("camera")
    assert camera.pattern.kind == "pinhole"
    assert camera.attach == "torso_link"
    assert camera.offset_convention == "world"
    assert camera.ray_alignment == "base"
    assert camera.miss == "infinity"
    assert camera.max_distance == 2.5
    assert camera.min_distance == 0.1
    assert camera.crop == (18, 0, 16, 16)
    assert camera.cropped_hw() == (18, 32)
    assert camera.pattern.width == 64
    assert camera.pattern.height == 36
    assert camera.hits_terrain()
    assert camera.hit_bodies() == tuple(G1_29DOF_LINKS)
    for group_name in ("policy", "critic"):
        term = task.mdp.observations[group_name].terms["depth_image"]
        assert term.func is not None
        assert term.history_length == 0
        assert term.level is Requirement.REQUIRED
        assert term.params["sensor"].name == "camera"
        assert term.params["history_skip_frames"] == 5
        assert term.params["num_output_frames"] == 8
        assert term.params["delayed_frame_ranges"] == (0, 1)
        assert term.params["history_length"] == 37


def test_the_joint_axis_is_pinned_to_the_canonical_order(task) -> None:
    canonical = tuple(task.robot.joint_names)
    selectors = {"actions.joint_pos": task.mdp.actions["joint_pos"].target}
    assert set(task.mdp.observations) == set(JOINT_TERMS)
    for group, names in JOINT_TERMS.items():
        group_spec = task.mdp.observations[group]
        for name in names:
            selectors[f"observations.{group}.{name}"] = group_spec.terms[name].params["asset_cfg"]
    for path, ref in selectors.items():
        assert ref is not None, f"{path} selects no entity"
        assert tuple(ref.joints) == canonical, (
            f"{path} selects {list(ref.joints)!r}; a lone pattern leaves the engine's own order in "
            "place, so the joints have to be named in the canonical depth-first order"
        )
        assert ref.preserve_order is True, f"{path} names the joints but does not ask for their order"


def test_joint_vel_limits_reads_the_catalog_in_canonical_order(task) -> None:
    term = task.mdp.rewards["rewards"]["dof_vel_limits"]
    assert term.func is not None
    assert term.params["soft_ratio"] == 0.9
    assert term.params["limits"] == tuple(joint.velocity_limit for joint in task.robot.joint_properties)
    assert tuple(term.params["asset_cfg"].joints) == tuple(task.robot.joint_names)


def test_base_contact_is_the_per_engine_force_threshold_term(task) -> None:
    """A portable illegal_contact has no newton gate; parkour needs one per engine."""
    term = task.mdp.terminations["base_contact"]
    assert term.kind == "illegal_contact"
    assert term.func is None
    assert term.level is Requirement.REQUIRED
    assert term.params["sensor"].elements == ("torso_link",)
    assert "threshold" not in term.params


def test_undesired_contacts_is_the_per_engine_force_threshold_term(task) -> None:
    """Shared params must not write 1.0 twice and call the two quantities aligned."""
    term = task.mdp.rewards["rewards"]["undesired_contacts"]
    assert term.kind == "undesired_contacts"
    assert term.func is None
    assert term.level is Requirement.REQUIRED
    assert "threshold" not in term.params


def test_feet_air_time_stays_on_the_portable_path(task) -> None:
    """Isaac already gates air-time at 1 N; mjlab uses found. Near-zero was early death."""
    term = task.mdp.rewards["rewards"]["feet_air_time"]
    assert term.func is not None
    assert term.func.__name__ == "feet_air_time"
    assert term.kind is None


def test_dataset_exhausted_matches_original_silent_reset(task) -> None:
    term = task.mdp.terminations["dataset_exhausted"]
    assert term.func is mdp.dataset_exhausted
    assert term.time_out is True
    assert term.params["sensor"].name == "motion_reference"
    assert term.params["reset_without_notice"] is True
    assert term.params["print_reason"] is False


def test_the_motion_reference_and_amp_groups_are_declared(task) -> None:
    """Clip sensor, both AMP branches, and the original silent exhaustion reset are declared."""
    from instinctlab.mdp.amp import AMP_TERM_ORDER
    from instinctlab.tasks.parkour.config.g1.g1_parkour_target_amp_cfg import PARKOUR_MOTION_CLIP, PARKOUR_MOTION_LINKS

    sensor = task.scene.motion_reference("motion_reference")
    assert sensor.clip == PARKOUR_MOTION_CLIP
    assert tuple(sensor.joints) == tuple(task.robot.joint_names)
    assert tuple(sensor.links) == PARKOUR_MOTION_LINKS
    assert sensor.num_frames == 10
    assert sensor.frame_interval_s == 0.02
    assert sensor.update_period == 0.02
    assert sensor.clip_target_fps == 50.0
    assert sensor.velocity_method == "frontward"
    assert sensor.start_range == (0.0, 0.9)
    assert sensor.exhaustion == "freeze_last_and_flag"
    assert sensor.quaternion == "wxyz"
    assert sensor.data_start_from == "one_frame_interval"
    assert AMP_ORDER == AMP_TERM_ORDER
    assert set(task.mdp.observations) >= {"amp_policy", "amp_reference"}
    for name, term in task.mdp.observations["amp_reference"].terms.items():
        assert term.params["sensor"] is sensor, name
        assert tuple(term.params["asset_cfg"].joints) == tuple(task.robot.joint_names)
    assert task.sim.step_dt == pytest.approx(1.0 / sensor.clip_target_fps)
    assert task.sim.step_dt == pytest.approx(sensor.frame_interval_s)


def test_the_agent_is_wasabi_moe_and_names_the_depth_component(task) -> None:
    """A depth encoder whose component_names miss the declared term is a blind policy."""
    from instinctlab.tasks.parkour.config.g1.agents.instinct_rl_amp_cfg import G1ParkourTargetPPORunnerCfg

    agent = G1ParkourTargetPPORunnerCfg()
    assert agent.algorithm.class_name == "WasabiPPO"
    assert agent.algorithm.actor_state_key == "amp_policy"
    assert agent.algorithm.reference_state_key == "amp_reference"
    assert agent.algorithm.discriminator_reward_coef == 0.25
    assert agent.algorithm.discriminator_reward_type == "quad"
    assert agent.algorithm.discriminator_kwargs["hidden_sizes"] == [1024, 512]
    assert agent.policy.class_name == "EncoderMoEActorCritic"
    assert agent.policy.num_moe_experts == 4
    depth = agent.policy.encoder_configs.depth_encoder
    assert depth.component_names == ["depth_image"]
    assert depth.takeout_input_components is True
    assert "depth_image" in task.mdp.observations["policy"].terms
    critic_depth = agent.policy.critic_encoder_configs.depth_encoder
    assert critic_depth.component_names == ["depth_image"]
    dumped = agent.to_dict()
    assert dumped["policy"]["encoder_configs"]["depth_encoder"]["component_names"] == ["depth_image"]
    assert task.agent.runner.endswith("G1ParkourTargetPPORunnerCfg")


def _imported_roots(path: Path) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


def test_the_declaration_imports_no_engine() -> None:
    forbidden = {"isaaclab", "mjlab", "omni", "mujoco", "isaacsim", "gymnasium"}
    for source in (DECLARATION, PACKAGE_INIT, AGENT):
        leaked = _imported_roots(source) & forbidden
        assert not leaked, f"{source.name} imports {sorted(leaked)}."


def test_the_declaration_imports_with_both_engines_blocked(monkeypatch) -> None:
    for name in list(sys.modules):
        if name.startswith(("instinctlab.engines.isaacsim", "instinctlab.engines.mjlab", "isaaclab", "mjlab")):
            monkeypatch.delitem(sys.modules, name, raising=False)

    class Blocker:
        def find_module(self, name, path=None):
            if name.split(".")[0] in {"isaaclab", "omni", "pxr", "carb", "isaacsim", "mjlab", "mujoco", "gymnasium"}:
                raise AssertionError(f"Importing the declaration pulled in {name!r}.")

    monkeypatch.setattr(sys, "meta_path", [Blocker(), *sys.meta_path])
    import importlib

    module = importlib.import_module("instinctlab.tasks.parkour.config.g1.g1_parkour_target_amp_cfg")
    assert module.parkour_target_g1().task_id == "Instinct-Parkour-Target-G1"


def test_both_backends_report_that_they_can_run_this_task(task) -> None:
    from instinctlab.engines.isaacsim import IsaacSimAdapter
    from instinctlab.engines.mjlab import MjlabAdapter

    task.validate()
    for adapter in (IsaacSimAdapter(), MjlabAdapter()):
        report = adapter.contract_report(task)
        assert report["missing"] == {}, report["missing"]
        assert report["engine_extras_used"] == []


def test_mjlab_compilation_preserves_declared_observation_history(task) -> None:
    """Term history must survive compilation. Group ``0`` used to overwrite every term.

    Asserts the declared length, not a pasted 8: if the task later asks for 4, this
    still asks whether that 4 reached the native cfg. The manager applies
    ``group.history_length`` whenever it is not ``None``, so the group sentinel is
    part of the assertion — checking only the term cfg would pass today, because
    the term still says 8 while the group silently zeros it at env init.
    """
    pytest.importorskip("mjlab")
    from instinctlab.engines.mjlab import MjlabAdapter

    compiled = MjlabAdapter().compile(task, num_envs=1, device="cpu")
    for group_name, group_spec in task.mdp.observations.items():
        group_cfg = compiled.env_cfg.observations[group_name]
        assert group_cfg.history_length is group_spec.history_length, group_name
        for term_name, term_spec in group_spec.terms.items():
            term_cfg = group_cfg.terms[term_name]
            assert term_cfg.history_length == term_spec.history_length, f"{group_name}/{term_name}"
            effective = group_cfg.history_length if group_cfg.history_length is not None else term_cfg.history_length
            assert effective == term_spec.history_length, (
                f"{group_name}/{term_name}: manager would use {effective}, "
                f"declaration asked for {term_spec.history_length}"
            )


def test_mjlab_compiles_every_kind_this_task_declares(task) -> None:
    """Enter every ``kind=`` builder body. ``contract_report`` never does.

    Today's crash was ``ValueError: mutable default list`` inside
    ``build_command`` while defining the cfg class. Registry membership and the
    declaration tests were green. This constructs the native cfg objects — no
    simulator, no GPU. Isaac's builders import ``isaaclab.managers``, which
    imports ``omni`` and needs AppLauncher; they are not cheap to exercise here.
    """
    pytest.importorskip("mjlab")
    from mjlab.envs.mdp import dr, joint_acc_l2, reset_root_state_uniform

    from instinctlab.engines.mjlab import MjlabAdapter
    from instinctlab.engines.mjlab.events import reset_joints_by_offset
    from instinctlab.engines.mjlab.rewards import (
        applied_torque_limits_by_ratio,
        contact_slide,
        illegal_contact,
        joint_torques_l2,
        motors_power_square,
        undesired_contacts,
    )
    from instinctlab.mdp.events import register_virtual_obstacles
    from tests.parkour_live_expect import PARKOUR_KIND_NAMES

    compiled = MjlabAdapter().compile(task, num_envs=16, device="cpu")
    assert compiled.resolution.skipped == {}, compiled.resolution.skipped
    kinds = {key: term.kind for key, term in task.mdp.terms().items() if term.kind is not None}
    assert PARKOUR_KIND_NAMES <= set(kinds.values()), set(kinds.values())
    for key, kind in kinds.items():
        assert key in compiled.resolution.resolved, f"{key} (kind={kind}) was not built"

    command = compiled.env_cfg.commands["base_velocity"]
    assert list(command.random_velocity_terrain) == ["perlin_rough_stand"]
    assert set(command.velocity_ranges) == {
        "perlin_rough",
        "perlin_rough_stand",
        "square_gaps",
        "pyramid_stairs",
        "pyramid_stairs_high",
        "pyramid_stairs_inv",
        "pyramid_stairs_inv_high",
        "boxes",
        "dense_boxes",
        "hf_pyramid_slope_inv",
    }
    assert "mesh_boxes" not in command.velocity_ranges
    assert command.velocity_ranges["dense_boxes"]["lin_vel_x"] == (0.45, 0.8)

    action = compiled.env_cfg.actions["joint_pos"]
    assert tuple(action.actuator_names) == tuple(task.robot.joint_names)
    assert action.preserve_order is True

    rewards = compiled.env_cfg.rewards
    assert rewards["energy"].func is motors_power_square
    assert rewards["energy"].params["normalize_by_stiffness"] is True
    assert rewards["torque_limits"].func is applied_torque_limits_by_ratio
    assert rewards["torque_limits"].params["limit_ratio"] == 0.8
    assert rewards["dof_torques_l2"].func is joint_torques_l2
    assert rewards["dof_acc_l2"].func is joint_acc_l2
    assert rewards["feet_slide"].func is contact_slide
    assert rewards["undesired_contacts"].func is undesired_contacts
    assert rewards["undesired_contacts"].params["threshold"] == 1.0
    assert compiled.env_cfg.terminations["base_contact"].func is illegal_contact
    assert compiled.env_cfg.terminations["base_contact"].params["threshold"] == 1.0

    events = compiled.env_cfg.events
    assert events["reset_robot_joints"].func is reset_joints_by_offset
    assert events["physics_material"].func is dr.geom_friction
    assert events["physics_material"].params["ranges"] == (0.3, 1.6)
    assert "restitution_range" not in events["physics_material"].params
    assert events["reset_base"].func is reset_root_state_uniform

    sensors = {sensor.name: sensor for sensor in compiled.env_cfg.scene.sensors}
    assert "motion_reference" in sensors
    assert sensors["motion_reference"].name == "motion_reference"
    assert "leg_volume_points" in sensors
    assert sensors["leg_volume_points"].name == "leg_volume_points"
    assert tuple(sensors["leg_volume_points"].body_names) == (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    assert sensors["leg_volume_points"].velocity == "attach_link"
    terrain_cfg = compiled.env_cfg.scene.terrain
    assert "edges" in terrain_cfg.virtual_obstacles
    assert terrain_cfg.virtual_obstacle_source == "mesh"
    assert terrain_cfg.virtual_obstacle_hfield_height_threshold == 0.04
    assert events["register_virtual_obstacles"].func is register_virtual_obstacles
    assert events["register_virtual_obstacles"].params["sensor"].name == "leg_volume_points"
    assert compiled.env_cfg.rewards["volume_points_penetration"].weight == -4.0
    assert "found" in sensors["contact_forces"].fields
    assert sensors["left_height_scanner"].ray_alignment == "yaw"
    assert sensors["left_height_scanner"].max_distance == 10.0
    exhausted = compiled.env_cfg.terminations["dataset_exhausted"]
    assert exhausted.func is mdp.dataset_exhausted
    assert exhausted.time_out is True
    assert exhausted.params["sensor"].name == "motion_reference"
    assert exhausted.params["reset_without_notice"] is True
    assert not hasattr(sensors["left_height_scanner"], "origin_offset")
    assert not hasattr(sensors["right_height_scanner"], "origin_offset")
    camera = sensors["camera"]
    assert camera.pattern.width == 64
    assert camera.pattern.height == 36
    assert camera.max_distance == 2.5
    assert camera.image_plane_max == 2.5
    assert camera.min_distance == 0.1
    assert camera.origin_offset_rot[0] == pytest.approx(0.9135367613482678)
    from instinctlab.engines.mjlab.camera import pinhole_camera_geom_groups

    assert camera.include_geom_groups == pinhole_camera_geom_groups()
    depth = compiled.env_cfg.observations["policy"].terms["depth_image"]
    assert depth.history_length == 0
    assert getattr(depth, "delay_max_lag", 0) == 0
    for group_name in ("amp_policy", "amp_reference"):
        group = compiled.env_cfg.observations[group_name]
        assert list(group.terms) == list(AMP_ORDER)
        assert group.terms["joint_vel"].scale == 0.05
        assert group.terms["joint_pos_rel"].history_length == 10
        asset_cfg = group.terms["joint_pos_rel"].params["asset_cfg"]
        assert tuple(asset_cfg.joint_names) == tuple(task.robot.joint_names)
        assert asset_cfg.preserve_order is True
    generator = compiled.env_cfg.scene.terrain.terrain_generator
    assert generator.num_cols == 10
    assert generator.curriculum is True
    assert list(generator.sub_terrains) == [
        "perlin_rough",
        "perlin_rough_stand",
        "square_gaps",
        "pyramid_stairs",
        "pyramid_stairs_high",
        "pyramid_stairs_inv",
        "pyramid_stairs_inv_high",
        "boxes",
        "dense_boxes",
        "hf_pyramid_slope_inv",
    ]
