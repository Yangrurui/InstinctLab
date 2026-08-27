"""Guard: a task declaration rejects the mistakes that would otherwise be found at runtime.

The declaration layer has no engine to check it against, so almost everything it can do for a task
author has to happen in ``__post_init__`` and :meth:`TaskSpec.validate`. The mistakes worth
catching there are the quiet ones -- an engine key spelled ``isaac`` instead of ``isaacsim`` is not
an error anywhere, it simply means the override never applies and the run silently uses defaults.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from instinctlab.sim.robot_spec import BackendAsset, JointProperties, RobotSpec
from instinctlab.spec import (
    ActionTermSpec,
    AgentSpec,
    CommandTermSpec,
    ContactSensorRef,
    DoneTermSpec,
    EntityRef,
    EventTermSpec,
    Grid3dPointsRef,
    MdpSpec,
    MotionReferenceRef,
    NoiseSpec,
    ObsGroupSpec,
    ObsTermSpec,
    RayCasterRef,
    Requirement,
    RewardTermSpec,
    SceneSpec,
    SimSpec,
    SubTerrainSpec,
    TaskSpec,
    TerrainGeneratorSpec,
    TerrainSpec,
    VirtualObstacleRef,
    VolumePointsRef,
)


def _robot() -> RobotSpec:
    return RobotSpec(
        name="tiny",
        schema_version="dfs_v1",
        asset_id="tiny_v1",
        root_body="root",
        joint_names=("hip", "knee"),
        body_names=("root", "foot"),
        joint_properties=(
            JointProperties("hip", 0.0, 2.0, 0.1, 0.0, 100.0, 10.0, 0.5),
            JointProperties("knee", 0.0, 2.0, 0.1, 0.0, 100.0, 10.0, 0.5),
        ),
        assets=(
            BackendAsset(backend="isaacsim", path="robot.urdf"),
            BackendAsset(backend="mjlab", path="robot.xml"),
        ),
        default_root_pos=(0.0, 0.0, 1.0),
        default_root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        soft_joint_pos_limit_factor=0.9,
    )


FEET = ContactSensorRef(name="feet", elements=("foot",), track_air_time=True)


def _observed(env, asset_cfg=None):  # noqa: ANN001 - a stand-in for a portable term
    return env


def _task(**overrides) -> TaskSpec:
    fields = dict(
        task_id="Test-Task-v0",
        robot=_robot(),
        scene=SceneSpec(contact_sensors=(FEET,)),
        sim=SimSpec(physics_dt=0.005, decimation=4, episode_length_s=20.0),
        mdp=MdpSpec(
            observations={"policy": ObsGroupSpec(terms={"joint_pos": ObsTermSpec(_observed)})},
            actions={"joint_pos": ActionTermSpec(kind="joint_position")},
            rewards={"rewards": {"alive": RewardTermSpec(_observed, weight=1.0)}},
            terminations={"time_out": DoneTermSpec(_observed, time_out=True)},
            commands={"base_velocity": CommandTermSpec(_observed)},
        ),
        agent=AgentSpec(runner="instinctlab.spec.task:AgentSpec"),
        engines=("isaacsim", "mjlab"),
    )
    fields.update(overrides)
    return TaskSpec(**fields)


"""
Term invariants.
"""


def test_a_term_carries_either_a_function_or_a_kind_but_not_both():
    """The distinction is what tells the compiler whether it needs a per-engine mapping at all."""
    with pytest.raises(ValueError, match="both was given"):
        ObsTermSpec(func=_observed, kind="joint_pos")
    with pytest.raises(ValueError, match="neither was given"):
        ObsTermSpec()


def test_portability_follows_from_which_one_was_set():
    assert ObsTermSpec(_observed).is_portable
    assert not EventTermSpec(kind="randomize_friction", mode="startup").is_portable


def test_engine_params_override_params_for_that_engine_only():
    term = RewardTermSpec(
        _observed,
        params={"std": 0.5, "command_name": "base_velocity"},
        engine_params={"mjlab": {"std": 0.6}},
    )
    assert term.resolved_params("isaacsim") == {"std": 0.5, "command_name": "base_velocity"}
    assert term.resolved_params("mjlab") == {"std": 0.6, "command_name": "base_velocity"}
    assert term.params == {"std": 0.5, "command_name": "base_velocity"}  # not mutated


def test_task_rejects_a_term_that_reads_an_unknown_command():
    task = _task(
        mdp=MdpSpec(
            commands={"base_velocity": CommandTermSpec(_observed)},
            rewards={
                "rewards": {
                    "tracking": RewardTermSpec(
                        _observed,
                        weight=1.0,
                        params={"command_name": "misspelled_velocity"},
                    )
                }
            },
        )
    )
    with pytest.raises(ValueError, match="reads command 'misspelled_velocity'"):
        task.validate()


def test_engine_params_merge_nested_mappings() -> None:
    """A shared dict keeps its keys; the engine only adds the ones it names."""
    term = CommandTermSpec(
        kind="pose_velocity",
        params={"velocity_ranges": {"shared": (0.0, 1.0)}},
        engine_params={"mjlab": {"velocity_ranges": {"extra": (0.45, 0.8)}}},
    )
    assert term.resolved_params("isaacsim")["velocity_ranges"] == {"shared": (0.0, 1.0)}
    assert term.resolved_params("mjlab")["velocity_ranges"] == {"shared": (0.0, 1.0), "extra": (0.45, 0.8)}
    assert term.params["velocity_ranges"] == {"shared": (0.0, 1.0)}


def test_default_levels_follow_the_consequences_of_dropping_the_term():
    assert ObsTermSpec(_observed).level is Requirement.REQUIRED
    assert DoneTermSpec(_observed).level is Requirement.REQUIRED
    assert ActionTermSpec(kind="joint_position").level is Requirement.REQUIRED
    assert CommandTermSpec(_observed).level is Requirement.REQUIRED
    assert RewardTermSpec(_observed).level is Requirement.OPTIONAL
    assert EventTermSpec(kind="x").level is Requirement.OPTIONAL


def test_an_interval_event_must_say_how_often():
    with pytest.raises(ValueError, match="interval range is required"):
        EventTermSpec(kind="push_robot", mode="interval")
    with pytest.raises(ValueError, match="meaningless otherwise"):
        EventTermSpec(kind="push_robot", mode="reset", interval_range_s=(1.0, 2.0))
    assert EventTermSpec(kind="push_robot", mode="interval", interval_range_s=(10.0, 15.0))
    with pytest.raises(ValueError, match="0 < min <= max"):
        EventTermSpec(kind="push_robot", mode="interval", interval_range_s=(2.0, 1.0))
    with pytest.raises(ValueError, match="Unknown event mode"):
        EventTermSpec(kind="push_robot", mode="sometimes")  # type: ignore[arg-type]


def test_noise_bounds_are_checked_in_the_direction_each_distribution_needs():
    with pytest.raises(ValueError, match="lo=0.2 above hi=0.1"):
        NoiseSpec("uniform", 0.2, 0.1)
    with pytest.raises(ValueError, match="negative standard deviation"):
        NoiseSpec("gaussian", 0.0, -1.0)


def test_an_observation_group_with_no_terms_is_rejected():
    with pytest.raises(ValueError, match="empty input tensor"):
        ObsGroupSpec(terms={})


def test_a_group_history_of_none_is_unset_and_zero_is_an_override():
    """``0`` is a real instruction to drop history; unspecified is ``None``."""
    unset = ObsGroupSpec(terms={"joint_pos": ObsTermSpec(_observed)})
    assert unset.history_length is None
    assert ObsGroupSpec(terms=unset.terms, history_length=0).history_length == 0
    with pytest.raises(ValueError, match="negative"):
        ObsGroupSpec(terms=unset.terms, history_length=-1)


"""
The MDP as a whole.
"""


def test_terms_are_keyed_by_family_and_group():
    keys = set(_task().mdp.terms())
    assert "observation/policy/joint_pos" in keys
    assert "reward/rewards/alive" in keys
    assert "termination/time_out" in keys
    assert "action/joint_pos" in keys


def test_reward_groups_may_reuse_a_term_name_without_colliding():
    mdp = MdpSpec(
        rewards={
            "task": {"alive": RewardTermSpec(_observed, weight=1.0)},
            "style": {"alive": RewardTermSpec(_observed, weight=2.0)},
        }
    )
    assert set(mdp.terms()) == {"reward/task/alive", "reward/style/alive"}


def test_entity_refs_are_found_in_params_as_well_as_in_target():
    """Portable terms take their entity as a parameter, the way Isaac Lab tasks always have."""
    target, in_params, in_override = EntityRef(bodies=("foot",)), EntityRef(joints=("hip",)), EntityRef()
    mdp = MdpSpec(
        rewards={
            "rewards": {
                "a": RewardTermSpec(_observed, target=target),
                "b": RewardTermSpec(_observed, params={"asset_cfg": in_params}),
                "c": RewardTermSpec(_observed, engine_params={"mjlab": {"asset_cfg": in_override}}),
            }
        }
    )
    found = mdp.entity_refs()
    assert target in found and in_params in found and in_override in found


def test_entity_refs_are_found_inside_nested_parameters():
    nested = EntityRef(bodies=("foot",))
    mdp = MdpSpec(
        rewards={
            "rewards": {
                "nested": RewardTermSpec(
                    _observed,
                    engine_params={"mjlab": {"selectors": {"feet": [nested]}}},
                )
            }
        }
    )
    assert nested in mdp.entity_refs()


"""
Whole-task validation.
"""


def test_a_valid_task_validates():
    _task().validate()


def test_a_task_refuses_an_engine_it_did_not_declare():
    task = _task(engines=("mjlab",))
    with pytest.raises(ValueError, match="does not declare engine 'isaacsim'"):
        task.validate_for_engine("isaacsim")


def test_isaac_contract_rejects_observation_names_that_are_group_settings():
    from instinctlab.engines.isaacsim import IsaacSimAdapter

    task = _task(
        mdp=MdpSpec(observations={"policy": ObsGroupSpec(terms={"enable_corruption": ObsTermSpec(_observed)})})
    )
    with pytest.raises(ValueError, match="reserved term names"):
        IsaacSimAdapter().contract_report(task)


def test_a_misspelled_engine_key_is_rejected_rather_than_ignored():
    """The failure this exists for: the override simply never applies, and nothing says so."""
    with pytest.raises(ValueError, match=r"keys sim.profiles by \['isaac'\]"):
        _task(sim=SimSpec(0.005, 4, 20.0, profiles={"isaac": {"solver": 4}})).validate()

    task = _task(
        mdp=MdpSpec(rewards={"rewards": {"alive": RewardTermSpec(_observed, engine_params={"newton": {"w": 1}})}})
    )
    with pytest.raises(ValueError, match=r"engine_params for \['newton'\]"):
        task.validate()


def test_a_term_may_not_read_a_sensor_the_scene_does_not_declare():
    task = _task(
        scene=SceneSpec(),
        mdp=MdpSpec(rewards={"rewards": {"air": RewardTermSpec(_observed, params={"sensor": FEET})}}),
    )
    with pytest.raises(ValueError, match="scene does not declare"):
        task.validate()
    scanner = RayCasterRef(name="left_height_scanner", attach="foot")
    task = _task(
        scene=SceneSpec(),
        mdp=MdpSpec(rewards={"rewards": {"plane": RewardTermSpec(_observed, params={"left_scanner": scanner})}}),
    )
    with pytest.raises(ValueError, match="ray caster"):
        task.validate()

    task = _task(
        scene=SceneSpec(),
        mdp=MdpSpec(
            rewards={
                "rewards": {
                    "nested": RewardTermSpec(
                        _observed,
                        engine_params={"mjlab": {"sensors": {"feet": [FEET]}}},
                    )
                }
            }
        ),
    )
    with pytest.raises(ValueError, match="scene does not declare"):
        task.validate()
    motion = MotionReferenceRef(
        name="motion_reference",
        clip="clip.npz",
        joints=("hip", "knee"),
        links=("root",),
    )
    task = _task(
        scene=SceneSpec(),
        mdp=MdpSpec(rewards={"rewards": {"amp": RewardTermSpec(_observed, params={"sensor": motion})}}),
    )
    with pytest.raises(ValueError, match="motion reference"):
        task.validate()
    volume = VolumePointsRef(name="leg_volume_points", attach=("foot",))
    task = _task(
        scene=SceneSpec(),
        mdp=MdpSpec(rewards={"rewards": {"pen": RewardTermSpec(_observed, params={"sensor": volume})}}),
    )
    with pytest.raises(ValueError, match="volume-points sensor"):
        task.validate()


def test_scene_sensors_must_name_real_robot_parts_and_entities():
    bad_ray = RayCasterRef(name="scanner", attach="missing")
    with pytest.raises(ValueError, match="not a declared robot body"):
        _task(scene=SceneSpec(ray_casters=(bad_ray,))).validate()

    bad_contact = ContactSensorRef(name="contact", elements=("missing_.*",))
    with pytest.raises(ValueError, match="match no robot body"):
        _task(scene=SceneSpec(contact_sensors=(bad_contact,))).validate()

    bad_volume = VolumePointsRef(name="volume", attach=("missing",))
    with pytest.raises(ValueError, match="unknown bodies"):
        _task(scene=SceneSpec(volume_points=(bad_volume,))).validate()


def test_motion_reference_joint_axis_must_follow_the_robot_canonical_order():
    motion = MotionReferenceRef(
        name="motion_reference",
        clip="clip.npz",
        joints=("knee", "hip"),
        links=("root",),
    )
    task = _task(scene=SceneSpec(contact_sensors=(FEET,), motion_references=(motion,)))

    with pytest.raises(ValueError, match="joint axis is not the robot's canonical order"):
        task.validate()


def test_policy_joint_selector_must_be_ordered_and_canonical():
    task = _task()
    observations = {
        "policy": ObsGroupSpec(
            terms={
                "joint_pos": ObsTermSpec(
                    _observed,
                    params={"asset_cfg": EntityRef("robot", joints=("knee", "hip"), preserve_order=True)},
                )
            }
        )
    }
    reversed_axis = replace(task, mdp=replace(task.mdp, observations=observations))
    with pytest.raises(ValueError, match="joint axis is not the RobotSpec canonical order"):
        reversed_axis.validate()

    native_axis = replace(
        task,
        mdp=replace(
            task.mdp,
            observations={
                "policy": ObsGroupSpec(
                    terms={
                        "joint_pos": ObsTermSpec(
                            _observed,
                            params={"asset_cfg": EntityRef("robot", joints=".*")},
                        )
                    }
                )
            },
        ),
    )
    with pytest.raises(ValueError, match="without preserve_order=True"):
        native_axis.validate()


def test_joint_action_selector_must_follow_the_robot_canonical_order():
    task = _task()
    action = replace(
        task.mdp.actions["joint_pos"],
        target=EntityRef("robot", joints=("knee", "hip"), preserve_order=True),
    )
    changed = replace(task, mdp=replace(task.mdp, actions={"joint_pos": action}))

    with pytest.raises(ValueError, match="joint axis is not the RobotSpec canonical order"):
        changed.validate()


def test_order_sensitive_reward_joint_selector_must_also_be_canonical():
    task = _task()
    reward = RewardTermSpec(
        _observed,
        params={"asset_cfg": EntityRef("robot", joints=("knee", "hip"), preserve_order=True)},
    )
    changed = replace(task, mdp=replace(task.mdp, rewards={"rewards": {"joint_vector": reward}}))

    with pytest.raises(ValueError, match="joint axis is not the RobotSpec canonical order"):
        changed.validate()


def test_scene_sensor_names_may_not_shadow_scene_entities_or_isaac_fields():
    with pytest.raises(ValueError, match="collide with scene entities"):
        _task(scene=SceneSpec(contact_sensors=(ContactSensorRef(name="robot", elements="foot"),))).validate()

    from instinctlab.engines.isaacsim import IsaacSimAdapter

    task = _task(scene=SceneSpec(contact_sensors=(ContactSensorRef(name="lazy_sensor_update", elements="foot"),)))
    with pytest.raises(ValueError, match="InteractiveSceneCfg fields"):
        IsaacSimAdapter().contract_report(task)


def test_term_sensor_references_must_match_what_the_scene_built():
    scene_contact = ContactSensorRef(name="contact", elements="foot")
    term_contact = ContactSensorRef(name="contact", elements="root")
    task = _task(
        scene=SceneSpec(contact_sensors=(scene_contact,)),
        mdp=MdpSpec(rewards={"rewards": {"contact": RewardTermSpec(_observed, params={"sensor": term_contact})}}),
    )
    with pytest.raises(ValueError, match="outside sensor"):
        task.validate()

    scene_ray = RayCasterRef(name="scanner", attach="foot")
    term_ray = replace(scene_ray, max_distance=2.0)
    task = _task(
        scene=SceneSpec(ray_casters=(scene_ray,)),
        mdp=MdpSpec(rewards={"rewards": {"height": RewardTermSpec(_observed, params={"sensor": term_ray})}}),
    )
    with pytest.raises(ValueError, match="different from scene sensor"):
        task.validate()


def test_a_task_must_name_at_least_one_engine_and_may_not_repeat_one():
    with pytest.raises(ValueError, match="names no engines"):
        _task(engines=())
    with pytest.raises(ValueError, match="repeats engines"):
        _task(engines=("mjlab", "mjlab"))


def test_a_task_requires_a_robot_asset_for_every_declared_engine():
    robot = replace(_robot(), assets=(BackendAsset(backend="mjlab", path="robot.xml"),))
    with pytest.raises(ValueError, match="without a robot asset"):
        _task(robot=robot).validate()


def test_a_plane_cannot_carry_a_generator():
    generator = TerrainGeneratorSpec(sub_terrains={"flat": SubTerrainSpec(kind="random_rough")})
    with pytest.raises(ValueError, match="cannot carry a generator"):
        TerrainSpec(kind="plane", generator=generator)


def test_terrain_material_coefficients_are_validated():
    with pytest.raises(ValueError, match="non-negative"):
        TerrainSpec(dynamic_friction=-0.1)
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        TerrainSpec(restitution=1.1)


def test_a_generator_must_carry_a_recipe():
    with pytest.raises(ValueError, match="needs a TerrainGeneratorSpec"):
        TerrainSpec(kind="generator")


def test_a_rough_terrain_must_carry_a_recipe():
    generator = TerrainGeneratorSpec(sub_terrains={"flat": SubTerrainSpec(kind="random_rough")})
    assert TerrainSpec(kind="rough", generator=generator).generator is generator
    with pytest.raises(ValueError, match="needs a TerrainGeneratorSpec"):
        TerrainSpec(kind="rough")


def test_a_generator_with_no_tiles_is_rejected():
    with pytest.raises(ValueError, match="no sub-terrains"):
        TerrainGeneratorSpec(sub_terrains={})


def test_a_tile_proportion_must_be_positive():
    with pytest.raises(ValueError, match="proportion must be positive"):
        SubTerrainSpec(kind="random_rough", proportion=0.0)


def test_duplicate_sensor_names_are_rejected():
    with pytest.raises(ValueError, match="must be unique"):
        SceneSpec(contact_sensors=(FEET, ContactSensorRef(name="feet", elements=("root",))))
    scanner = RayCasterRef(name="feet", attach="foot")
    with pytest.raises(ValueError, match="must be unique"):
        SceneSpec(contact_sensors=(FEET,), ray_casters=(scanner,))
    motion = MotionReferenceRef(name="feet", clip="clip.npz", joints=("hip",), links=("root",))
    with pytest.raises(ValueError, match="must be unique"):
        SceneSpec(contact_sensors=(FEET,), motion_references=(motion,))
    volume = VolumePointsRef(name="feet", attach=("foot",))
    with pytest.raises(ValueError, match="must be unique"):
        SceneSpec(contact_sensors=(FEET,), volume_points=(volume,))
    with pytest.raises(ValueError, match="must be unique"):
        TerrainSpec(virtual_obstacles=(VirtualObstacleRef(name="edges"), VirtualObstacleRef(name="edges")))


def test_the_scene_finds_a_declared_sensor_and_says_what_it_has_when_it_cannot():
    assert SceneSpec(contact_sensors=(FEET,)).sensor("feet") is FEET
    with pytest.raises(KeyError, match="Declared: feet"):
        SceneSpec(contact_sensors=(FEET,)).sensor("hands")
    scanner = RayCasterRef(name="left_height_scanner", attach="foot")
    assert SceneSpec(ray_casters=(scanner,)).ray_caster("left_height_scanner") is scanner
    with pytest.raises(KeyError, match="Declared: left_height_scanner"):
        SceneSpec(ray_casters=(scanner,)).ray_caster("right_height_scanner")
    motion = MotionReferenceRef(name="motion_reference", clip="clip.npz", joints=("hip",), links=("root",))
    assert SceneSpec(motion_references=(motion,)).motion_reference("motion_reference") is motion
    with pytest.raises(KeyError, match="Declared: motion_reference"):
        SceneSpec(motion_references=(motion,)).motion_reference("other")
    volume = VolumePointsRef(
        name="leg_volume_points",
        attach=("foot",),
        grid=Grid3dPointsRef(
            x_min=-0.025,
            x_max=0.12,
            x_num=10,
            y_min=-0.03,
            y_max=0.03,
            y_num=5,
            z_min=-0.063,
            z_max=-0.023,
            z_num=2,
        ),
    )
    assert volume.grid.count == 100
    assert SceneSpec(volume_points=(volume,)).volume_point("leg_volume_points") is volume
    with pytest.raises(KeyError, match="Declared: leg_volume_points"):
        SceneSpec(volume_points=(volume,)).volume_point("other")


"""
Sim and agent.
"""


def test_step_dt_is_the_product_both_engines_compute():
    assert SimSpec(physics_dt=0.005, decimation=4, episode_length_s=20.0).step_dt == pytest.approx(0.02)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"physics_dt": 0.0}, "physics_dt must be positive"),
        ({"decimation": 0}, "decimation must be positive"),
        ({"episode_length_s": -1.0}, "episode_length_s must be positive"),
    ],
)
def test_sim_timing_must_be_positive(kwargs: dict, message: str):
    fields = {"physics_dt": 0.005, "decimation": 4, "episode_length_s": 20.0} | kwargs
    with pytest.raises(ValueError, match=message):
        SimSpec(**fields)


def test_profiles_carry_only_overrides_and_are_empty_by_default():
    """Defaults come from the adapter, so that each engine gets its own reference values."""
    sim = SimSpec(0.005, 4, 20.0, profiles={"mjlab": {"iterations": 10}})
    assert sim.profile_for("mjlab") == {"iterations": 10}
    assert sim.profile_for("isaacsim") == {}


def test_the_agent_is_imported_only_when_asked_for():
    """Runner configs are built on ``isaaclab.utils.configclass``; naming one must not import it."""
    agent = AgentSpec(runner="instinctlab.spec.task:SimSpec", engine_overrides={"mjlab": {"num_steps": 24}})
    assert agent.resolve() is SimSpec
    assert agent.resolved_overrides("mjlab") == {"num_steps": 24}
    assert agent.resolved_overrides("isaacsim") == {}


def test_a_runner_path_without_a_module_is_rejected():
    with pytest.raises(ValueError, match="dotted path"):
        AgentSpec(runner="SimSpec").resolve()


def test_portability_means_more_than_one_engine_and_no_escape_hatch():
    assert _task().is_portable
    assert not _task(engines=("isaacsim",)).is_portable
    assert not _task(engine_extras={"isaacsim": {"tiled_camera": True}}).is_portable
