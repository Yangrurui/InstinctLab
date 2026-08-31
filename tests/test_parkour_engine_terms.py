"""Guard: parkour's per-engine reward/event kinds do what the task asked, not a neighbour.

``applied_torque`` is on the denylist and mjlab has no attribute of that name. The stock mjlab
``joint_torques_l2`` reads ``actuator_force`` (nu) and ignores a joint-only selection. Friction
ranges used to come only from the solver profile. Each of those is a silent failure: the run
converges, the objective is not the one written down.
"""

from __future__ import annotations

import ast
import pathlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from instinctlab.tasks.parkour.mdp.rewards import (
    applied_torque_limits_by_ratio,
    joint_torques_l2,
    motors_power_square,
    undesired_contacts_by_force,
)
from instinctlab.tasks.parkour.mdp.terminations import illegal_contact_by_force
from instinctlab_engine.actuators import STIFFNESS, ActuatorRegistry
from instinctlab_engine.bridge import robot as robot_bridge
from instinctlab_engine.bridge.robot import joint_effort_limits
from instinctlab_engine.spec.sensor import ContactSensorRef
from instinctlab_engine_isaacsim import terms as isaac_terms
from instinctlab_engine_isaacsim.event_terms import (
    merge_friction_params as isaac_merge_friction,
)
from instinctlab_engine_isaacsim.terms import TERMS as ISAAC_TERMS
from instinctlab_engine_mjlab.event_terms import (
    merge_friction_params as mjlab_merge_friction,
)
from instinctlab_engine_mjlab.native_event_functions import (
    reset_joints_by_offset,
    reset_joints_by_scale,
)
from instinctlab_engine_mjlab.terms import TERMS as MJLAB_TERMS

from tests.engine_packages import MJLAB_ENGINE

EVENTS = MJLAB_ENGINE / "native_event_functions.py"


def _evaluate_class_term(term_type, env, **params):
    term = term_type(SimpleNamespace(params=params), env)
    return term(env, **params)


def _function(path: pathlib.Path, name: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{path} has no function {name}")


def _aug_ops(function: ast.FunctionDef) -> list[type]:
    return [
        type(node.op) for node in ast.walk(function) if isinstance(node, ast.AugAssign)
    ]


"""
Friction: honor what the engine can apply, refuse the rest.
"""


def test_isaac_friction_overlays_task_ranges_and_rejects_mjlab_keys() -> None:
    profile = {
        "static_friction_range": (0.25, 0.8),
        "dynamic_friction_range": (0.2, 0.6),
        "restitution_range": (0.0, 0.8),
        "num_buckets": 64,
    }
    merged = isaac_merge_friction(
        profile,
        {
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.6),
            "restitution_range": (0.05, 0.5),
        },
    )
    assert merged["static_friction_range"] == (0.3, 1.6)
    assert merged["dynamic_friction_range"] == (0.3, 1.6)
    assert merged["restitution_range"] == (0.05, 0.5)
    assert merged["num_buckets"] == 64
    with pytest.raises(ValueError, match="does not honor \\['ranges'\\]"):
        isaac_merge_friction(profile, {"ranges": (0.3, 1.6)})


def test_mjlab_friction_maps_static_dynamic_to_their_union_and_rejects_restitution() -> (
    None
):
    profile = {"ranges": (0.2, 0.8), "operation": "abs", "shared_random": True}
    merged = mjlab_merge_friction(
        profile,
        {"static_friction_range": (0.3, 1.6), "dynamic_friction_range": (0.4, 1.2)},
    )
    assert merged["ranges"] == (0.3, 1.6)
    assert merged["operation"] == "abs"
    with pytest.raises(ValueError, match="cannot honor restitution_range"):
        mjlab_merge_friction(profile, {"restitution_range": (0.05, 0.5)})
    empty = mjlab_merge_friction(profile, {})
    assert empty["ranges"] == (0.2, 0.8)


"""
Rewards that read joint-space actuator force.
"""


def test_mjlab_joint_torques_l2_slices_joint_ids_on_qfrc_actuator() -> None:
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(qfrc_actuator=torch.tensor([[1.0, 2.0, 3.0]]))
            )
        }
    )
    out = joint_torques_l2(
        env, asset_cfg=SimpleNamespace(name="robot", joint_ids=[0, 2])
    )
    assert torch.equal(out, torch.tensor([10.0]))


def test_mjlab_motors_power_square_uses_qfrc_times_joint_vel() -> None:
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    qfrc_actuator=torch.tensor([[1.0, 2.0]]),
                    joint_vel=torch.tensor([[3.0, 4.0]]),
                )
            )
        }
    )
    out = _evaluate_class_term(
        motors_power_square,
        env,
        asset_cfg=SimpleNamespace(name="robot", joint_ids=slice(None)),
        normalize_by_stiffness=False,
    )
    assert torch.equal(out, torch.tensor([9.0 + 64.0]))


def test_isaac_reward_lowering_wraps_portable_class_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NativeRewardCfg:
        def __init__(self, **kwargs):
            vars(self).update(kwargs)

    wrapped = object()
    monkeypatch.setattr(
        isaac_terms, "_import_cfgs", lambda: {"reward": NativeRewardCfg}
    )
    monkeypatch.setattr(
        isaac_terms,
        "_as_isaac_manager_term",
        lambda term_type: wrapped if term_type is motors_power_square else None,
    )
    spec = SimpleNamespace(
        func=motors_power_square,
        weight=-5e-5,
        params={"normalize_by_stiffness": True},
    )
    ctx = SimpleNamespace(params=lambda term: dict(term.params))

    native = ISAAC_TERMS.lookup_portable("reward")(spec, ctx)

    assert native.func is wrapped
    assert native.params == {"normalize_by_stiffness": True}


def test_isaac_class_term_without_state_uses_native_reset_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NativeManagerTermBase:
        def __init__(self, cfg, env):
            del cfg
            self.env = env

        def reset(self, env_ids=None):
            self.env.native_reset_ids = env_ids

    class PortableTerm:
        def __init__(self, cfg, env):
            del cfg, env

        def __call__(self, env):
            return env

    isaaclab = ModuleType("isaaclab")
    isaaclab.__path__ = []
    managers = ModuleType("isaaclab.managers")
    managers.ManagerTermBase = NativeManagerTermBase
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    env = SimpleNamespace()

    wrapped = isaac_terms._as_isaac_manager_term(PortableTerm)
    term = wrapped(SimpleNamespace(), env)
    term.reset([1])

    assert env.native_reset_ids == [1]


def test_mjlab_reward_manager_instantiates_computes_and_resets_class_term() -> None:
    from mjlab.managers import RewardManager, RewardTermCfg

    robot = SimpleNamespace(
        data=SimpleNamespace(
            qfrc_actuator=torch.tensor([[2.0, 4.0]]),
            joint_vel=torch.tensor([[3.0, 2.0]]),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene={"robot": robot},
        max_episode_length_s=1.0,
    )
    manager = RewardManager(
        {
            "energy": RewardTermCfg(
                func=motors_power_square,
                weight=-0.5,
                params={"normalize_by_stiffness": False},
            )
        },
        env,
    )

    reward = manager.compute(dt=0.1)
    extras = manager.reset()

    torch.testing.assert_close(reward, torch.tensor([-5.0]))
    torch.testing.assert_close(extras["Episode_Reward/energy"], torch.tensor(-5.0))
    assert manager._class_term_cfgs == []


def test_mjlab_motors_power_square_rejects_an_unregistered_auxiliary_actuator() -> None:
    pd = SimpleNamespace(
        transmission_type="joint",
        target_ids=torch.tensor([0, 1]),
        cfg=SimpleNamespace(stiffness=2.0),
    )
    limiter = SimpleNamespace(
        transmission_type="joint",
        target_ids=torch.tensor([0, 1]),
        cfg=SimpleNamespace(velocity_limit=20.0),
    )
    robot = SimpleNamespace(
        actuators=(pd, limiter),
        data=SimpleNamespace(
            qfrc_actuator=torch.tensor([[2.0, 4.0]]),
            joint_vel=torch.tensor([[3.0, 2.0]]),
        ),
    )
    env = SimpleNamespace(scene={"robot": robot})

    with pytest.raises(RuntimeError, match="no registered runtime adapter") as error:
        _evaluate_class_term(
            motors_power_square,
            env,
            asset_cfg=SimpleNamespace(name="robot", joint_ids=slice(None)),
        )
    assert "motors_power_square" in str(error.value)
    assert "stiffness" in str(error.value)


def test_mjlab_motors_power_square_maps_stiffness_by_native_joint_ids() -> None:
    from mjlab.actuator import BuiltinPdActuator

    first = object.__new__(BuiltinPdActuator)
    first.cfg = SimpleNamespace(transmission_type="joint", stiffness=2.0)
    first._target_ids = torch.tensor([2, 0])
    first._global_ctrl_ids = torch.tensor([0, 1, 3, 4])
    second = object.__new__(BuiltinPdActuator)
    second.cfg = SimpleNamespace(transmission_type="joint", stiffness=4.0)
    second._target_ids = torch.tensor([1])
    second._global_ctrl_ids = torch.tensor([2, 5])
    gain_parameters = torch.zeros(6, 10)
    gain_parameters[[0, 1], 0] = 2.0
    gain_parameters[2, 0] = 4.0
    robot = SimpleNamespace(
        actuators=(first, second),
        data=SimpleNamespace(
            qfrc_actuator=torch.tensor([[2.0, 8.0, 6.0]]),
            joint_vel=torch.tensor([[3.0, 1.0, 2.0]]),
        ),
    )
    env = SimpleNamespace(
        scene={"robot": robot},
        sim=SimpleNamespace(
            model=SimpleNamespace(actuator_gainprm=gain_parameters)
        ),
    )

    out = _evaluate_class_term(
        motors_power_square,
        env,
        asset_cfg=SimpleNamespace(name="robot", joint_ids=[1, 2]),
    )

    assert torch.equal(
        out, torch.tensor([(8.0 * 1.0 / 4.0) ** 2 + (6.0 * 2.0 / 2.0) ** 2])
    )


def test_motors_power_square_does_not_read_device_ids_after_term_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tensor.tolist() synchronizes CUDA, so it is forbidden in the reward hot path."""
    from mjlab.actuator import BuiltinPdActuator

    actuator = object.__new__(BuiltinPdActuator)
    actuator.cfg = SimpleNamespace(transmission_type="joint", stiffness=2.0)
    actuator._target_ids = torch.tensor([1, 0])
    actuator._global_ctrl_ids = torch.tensor([0, 1, 2, 3])
    gain_parameters = torch.zeros(4, 10)
    gain_parameters[[0, 1], 0] = 2.0
    robot = SimpleNamespace(
        actuators=(actuator,),
        data=SimpleNamespace(
            qfrc_actuator=torch.tensor([[2.0, 8.0]]),
            joint_vel=torch.tensor([[3.0, 1.0]]),
        ),
    )
    env = SimpleNamespace(
        scene={"robot": robot},
        sim=SimpleNamespace(
            model=SimpleNamespace(actuator_gainprm=gain_parameters)
        ),
    )
    asset_cfg = SimpleNamespace(name="robot", joint_ids=torch.tensor([0, 1]))
    term = motors_power_square(
        SimpleNamespace(params={"asset_cfg": asset_cfg}),
        env,
    )

    def reject_tolist(_tensor):
        raise AssertionError("reward evaluation copied static joint ids to the host")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "tolist", reject_tolist)
        out = term(env, asset_cfg=asset_cfg)

    assert torch.equal(
        out, torch.tensor([(2.0 * 3.0 / 2.0) ** 2 + (8.0 * 1.0 / 2.0) ** 2])
    )


def test_motors_power_square_reads_partial_group_stiffness_after_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NativeActuator:
        instinctlab_model_id = "test.dynamic_stiffness.v1"
        transmission_type = "joint"
        target_ids = (0, 1)

        def __init__(self) -> None:
            self.stiffness = torch.tensor([[2.0, 8.0]])

    class RuntimeAdapter:
        def matches(self, actuator: object) -> bool:
            return isinstance(actuator, NativeActuator)

        def stiffness_groups(self, env, asset, actuator: NativeActuator):
            del env, asset
            return ((actuator.target_ids, actuator.stiffness),)

    registry = ActuatorRegistry(load_entry_points=False)
    registry.register(
        engine="mjlab",
        model_id="test.dynamic_stiffness.v1",
        config_factory=lambda: None,
        runtime_adapter=RuntimeAdapter,
        capabilities={STIFFNESS},
    )
    monkeypatch.setattr(robot_bridge, "ACTUATORS", registry)
    actuator = NativeActuator()
    robot = SimpleNamespace(
        actuators=(actuator,),
        data=SimpleNamespace(
            qfrc_actuator=torch.tensor([[4.0, 1.0]]),
            joint_vel=torch.tensor([[1.0, 1.0]]),
        ),
    )
    env = SimpleNamespace(scene={"robot": robot})
    asset_cfg = SimpleNamespace(name="robot", joint_ids=[0])
    term = motors_power_square(
        SimpleNamespace(params={"asset_cfg": asset_cfg}),
        env,
    )

    torch.testing.assert_close(term(env, asset_cfg=asset_cfg), torch.tensor([4.0]))
    actuator.stiffness[:, 0] = 4.0

    torch.testing.assert_close(term(env, asset_cfg=asset_cfg), torch.tensor([1.0]))


@pytest.mark.mjlab
def test_motors_power_square_has_no_cuda_sync_after_term_initialization() -> None:
    """A partial-group gain mutation stays live without synchronizing CUDA."""
    from mjlab.actuator import BuiltinPdActuator

    from tests.parkour_live_expect import require_live_device

    device = require_live_device()
    actuator = object.__new__(BuiltinPdActuator)
    actuator.cfg = SimpleNamespace(transmission_type="joint", stiffness=2.0)
    actuator._target_ids = torch.tensor([1, 0], device=device)
    actuator._global_ctrl_ids = torch.tensor([0, 1, 2, 3], device=device)
    gain_parameters = torch.zeros(1, 4, 10, device=device)
    gain_parameters[:, [0, 1], 0] = 2.0
    robot = SimpleNamespace(
        actuators=(actuator,),
        data=SimpleNamespace(
            qfrc_actuator=torch.tensor([[2.0, 8.0]], device=device),
            joint_vel=torch.tensor([[3.0, 1.0]], device=device),
        ),
    )
    env = SimpleNamespace(
        scene={"robot": robot},
        sim=SimpleNamespace(
            model=SimpleNamespace(actuator_gainprm=gain_parameters)
        ),
    )
    asset_cfg = SimpleNamespace(
        name="robot",
        joint_ids=torch.tensor([0], device=device),
    )
    term = motors_power_square(
        SimpleNamespace(params={"asset_cfg": asset_cfg}),
        env,
    )

    original_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        before = term(env, asset_cfg=asset_cfg)
        gain_parameters[:, 1, 0] = 4.0
        after = term(env, asset_cfg=asset_cfg)
    finally:
        torch.cuda.set_sync_debug_mode(original_mode)

    torch.testing.assert_close(
        before,
        torch.tensor([(2.0 * 3.0 / 2.0) ** 2], device=device),
    )
    torch.testing.assert_close(
        after,
        torch.tensor([(2.0 * 3.0 / 4.0) ** 2], device=device),
    )


def test_mixed_actuator_reward_skips_unselected_group_without_stiffness() -> None:
    from mjlab.actuator import BuiltinPdActuator

    unrelated = SimpleNamespace(
        transmission_type="joint",
        target_ids=torch.tensor([0]),
        cfg=SimpleNamespace(velocity_limit=20.0),
    )
    selected = object.__new__(BuiltinPdActuator)
    selected.cfg = SimpleNamespace(transmission_type="joint", stiffness=4.0)
    selected._target_ids = torch.tensor([1])
    selected._global_ctrl_ids = torch.tensor([0, 1])
    gain_parameters = torch.zeros(2, 10)
    gain_parameters[0, 0] = 4.0
    robot = SimpleNamespace(
        actuators=(unrelated, selected),
        data=SimpleNamespace(
            qfrc_actuator=torch.tensor([[2.0, 8.0]]),
            joint_vel=torch.tensor([[3.0, 2.0]]),
        ),
    )
    env = SimpleNamespace(
        scene={"robot": robot},
        sim=SimpleNamespace(
            model=SimpleNamespace(actuator_gainprm=gain_parameters)
        ),
    )

    out = _evaluate_class_term(
        motors_power_square,
        env,
        asset_cfg=SimpleNamespace(name="robot", joint_ids=[1]),
    )

    assert torch.equal(out, torch.tensor([(8.0 * 2.0 / 4.0) ** 2]))


@pytest.mark.parametrize(
    ("returned_ids", "stiffness", "message"),
    (
        ((0,), 2.0, "outside its owning group"),
        ((1,), torch.ones(3), "broadcast-compatible"),
    ),
)
def test_stiffness_adapter_output_is_bounded_by_its_native_group(
    monkeypatch: pytest.MonkeyPatch,
    returned_ids,
    stiffness,
    message: str,
) -> None:
    class NativeActuator:
        instinctlab_model_id = "test.bad_stiffness.v1"
        transmission_type = "joint"
        target_ids = torch.tensor([1])

    class RuntimeAdapter:
        def matches(self, actuator: object) -> bool:
            return isinstance(actuator, NativeActuator)

        def stiffness_groups(self, env, asset, actuator: object):
            del env, asset, actuator
            return ((returned_ids, stiffness),)

    registry = ActuatorRegistry(load_entry_points=False)
    registry.register(
        engine="mjlab",
        model_id="test.bad_stiffness.v1",
        config_factory=lambda: None,
        runtime_adapter=RuntimeAdapter,
        capabilities={STIFFNESS},
    )
    monkeypatch.setattr(robot_bridge, "ACTUATORS", registry)
    robot = SimpleNamespace(
        actuators=(NativeActuator(),),
        data=SimpleNamespace(
            qfrc_actuator=torch.tensor([[2.0, 8.0], [3.0, 9.0]]),
            joint_vel=torch.tensor([[3.0, 2.0], [4.0, 1.0]]),
        ),
    )
    env = SimpleNamespace(scene={"robot": robot})

    with pytest.raises(RuntimeError, match=message):
        _evaluate_class_term(
            motors_power_square,
            env,
            asset_cfg=SimpleNamespace(name="robot", joint_ids=[1]),
        )


@pytest.mark.parametrize(
    ("actuator_groups", "selected_ids", "message"),
    (
        (
            (((0,), (((0,), 2.0), ((0,), 2.0))),),
            [0],
            "duplicate selected joint ids",
        ),
        (
            (((0, 1), (((0,), 2.0),)),),
            [0, 1],
            "did not return stiffness",
        ),
        (
            (
                ((0,), (((0,), 2.0),)),
                ((0,), (((0,), 2.0),)),
            ),
            [0],
            "more than once",
        ),
    ),
)
def test_stiffness_groups_reject_duplicate_missing_and_overlapping_ids(
    monkeypatch: pytest.MonkeyPatch,
    actuator_groups,
    selected_ids,
    message: str,
) -> None:
    class NativeActuator:
        instinctlab_model_id = "test.invalid_stiffness_groups.v1"
        transmission_type = "joint"

        def __init__(self, target_ids, groups):
            self.target_ids = target_ids
            self.groups = groups

    class RuntimeAdapter:
        def matches(self, actuator: object) -> bool:
            return isinstance(actuator, NativeActuator)

        def stiffness_groups(self, env, asset, actuator: NativeActuator):
            del env, asset
            return actuator.groups

    registry = ActuatorRegistry(load_entry_points=False)
    registry.register(
        engine="mjlab",
        model_id="test.invalid_stiffness_groups.v1",
        config_factory=lambda: None,
        runtime_adapter=RuntimeAdapter,
        capabilities={STIFFNESS},
    )
    monkeypatch.setattr(robot_bridge, "ACTUATORS", registry)
    actuators = tuple(NativeActuator(*group) for group in actuator_groups)
    robot = SimpleNamespace(
        actuators=actuators,
        data=SimpleNamespace(
            qfrc_actuator=torch.tensor([[2.0, 3.0]]),
            joint_vel=torch.tensor([[4.0, 5.0]]),
        ),
    )
    env = SimpleNamespace(scene={"robot": robot})

    with pytest.raises(RuntimeError, match=message):
        _evaluate_class_term(
            motors_power_square,
            env,
            asset_cfg=SimpleNamespace(name="robot", joint_ids=selected_ids),
        )


def test_stiffness_adapter_preserves_per_environment_broadcasting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NativeActuator:
        instinctlab_model_id = "test.per_env_stiffness.v1"
        transmission_type = "joint"
        target_ids = (0,)

    class RuntimeAdapter:
        def matches(self, actuator: object) -> bool:
            return isinstance(actuator, NativeActuator)

        def stiffness_groups(self, env, asset, actuator: object):
            del env, asset, actuator
            return (((0,), torch.tensor([[2.0], [4.0]])),)

    registry = ActuatorRegistry(load_entry_points=False)
    registry.register(
        engine="mjlab",
        model_id="test.per_env_stiffness.v1",
        config_factory=lambda: None,
        runtime_adapter=RuntimeAdapter,
        capabilities={STIFFNESS},
    )
    monkeypatch.setattr(robot_bridge, "ACTUATORS", registry)
    robot = SimpleNamespace(
        actuators=(NativeActuator(),),
        data=SimpleNamespace(
            qfrc_actuator=torch.tensor([[2.0], [2.0]]),
            joint_vel=torch.tensor([[3.0], [3.0]]),
        ),
    )
    env = SimpleNamespace(scene={"robot": robot})

    out = _evaluate_class_term(
        motors_power_square,
        env,
        asset_cfg=SimpleNamespace(name="robot", joint_ids=[0]),
    )

    torch.testing.assert_close(out, torch.tensor([9.0, 2.25]))


@pytest.mark.parametrize(
    "returned_ids",
    (
        (0.5,),
        (True,),
        ("0",),
        torch.tensor([0.5]),
        torch.tensor([True]),
    ),
)
def test_stiffness_adapter_rejects_non_integral_joint_ids(
    monkeypatch: pytest.MonkeyPatch,
    returned_ids,
) -> None:
    class NativeActuator:
        instinctlab_model_id = "test.invalid_stiffness_ids.v1"
        transmission_type = "joint"
        target_ids = torch.tensor([0])

    class RuntimeAdapter:
        def matches(self, actuator: object) -> bool:
            return isinstance(actuator, NativeActuator)

        def stiffness_groups(self, env, asset, actuator: object):
            del env, asset, actuator
            return ((returned_ids, 2.0),)

    registry = ActuatorRegistry(load_entry_points=False)
    registry.register(
        engine="mjlab",
        model_id="test.invalid_stiffness_ids.v1",
        config_factory=lambda: None,
        runtime_adapter=RuntimeAdapter,
        capabilities={STIFFNESS},
    )
    monkeypatch.setattr(robot_bridge, "ACTUATORS", registry)
    robot = SimpleNamespace(
        actuators=(NativeActuator(),),
        data=SimpleNamespace(
            qfrc_actuator=torch.tensor([[2.0]]),
            joint_vel=torch.tensor([[3.0]]),
        ),
    )
    env = SimpleNamespace(scene={"robot": robot})

    with pytest.raises(RuntimeError, match="integer tensor dtype|only integral"):
        _evaluate_class_term(
            motors_power_square,
            env,
            asset_cfg=SimpleNamespace(name="robot", joint_ids=[0]),
        )


@pytest.mark.parametrize(
    "native_ids",
    (
        (0.5,),
        (True,),
        ("0",),
        torch.tensor([0.5]),
        torch.tensor([True]),
    ),
)
def test_stiffness_reader_rejects_non_integral_native_ownership_ids(
    native_ids,
) -> None:
    actuator = SimpleNamespace(
        transmission_type="joint",
        target_ids=native_ids,
    )
    robot = SimpleNamespace(
        actuators=(actuator,),
        data=SimpleNamespace(
            qfrc_actuator=torch.tensor([[2.0]]),
            joint_vel=torch.tensor([[3.0]]),
        ),
    )
    env = SimpleNamespace(scene={"robot": robot})

    with pytest.raises(RuntimeError, match="integer tensor dtype|only integral"):
        _evaluate_class_term(
            motors_power_square,
            env,
            asset_cfg=SimpleNamespace(name="robot", joint_ids=[0]),
        )


def test_mjlab_applied_torque_limits_by_ratio_reads_joint_effort_limits_when_present() -> (
    None
):
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    qfrc_actuator=torch.tensor([[10.0, 1.0]]),
                    joint_effort_limits=torch.tensor([[10.0, 10.0]]),
                )
            )
        }
    )
    out = applied_torque_limits_by_ratio(
        env,
        asset_cfg=SimpleNamespace(name="robot", joint_ids=slice(None)),
        limit_ratio=0.8,
    )
    assert torch.equal(out, torch.tensor([4.0]))


def test_mjlab_model_effort_limits_map_global_ranges_to_selected_local_joints(
    monkeypatch,
) -> None:
    class BuiltinPdActuator:
        transmission_type = "joint"
        target_names = ("hip", "ankle")
        target_ids = torch.tensor([0, 1])

    monkeypatch.setattr("mjlab.actuator.BuiltinPdActuator", BuiltinPdActuator)
    ranges = torch.zeros(2, 6, 2)
    ranges[:, 2] = torch.tensor([[-11.0, 11.0], [-13.0, 13.0]])
    ranges[:, 5] = torch.tensor([[-7.0, 7.0], [-9.0, 9.0]])
    robot = SimpleNamespace(
        num_joints=2,
        joint_names=("hip", "ankle"),
        indexing=SimpleNamespace(joint_ids=torch.tensor([2, 5])),
        actuators=(BuiltinPdActuator(),),
        data=SimpleNamespace(qfrc_actuator=torch.zeros(2, 2)),
    )
    env = SimpleNamespace(
        scene={"robot": robot},
        sim=SimpleNamespace(
            model=SimpleNamespace(
                jnt_actfrcrange=ranges,
                actuator_forcerange=torch.zeros(2, 0, 2),
            )
        ),
    )

    limits = joint_effort_limits(env, robot, [1, 0])

    torch.testing.assert_close(limits, torch.tensor([[7.0, 11.0], [9.0, 13.0]]))


"""
Offset reset is addition, not a scale.
"""


def test_force_threshold_formulas_are_not_registered_on_engines() -> None:
    assert ISAAC_TERMS.kinds("termination") == set()
    assert ISAAC_TERMS.kinds("reward") == set()
    assert MJLAB_TERMS.kinds("termination") == set()
    assert MJLAB_TERMS.kinds("reward") == set()


class _MjlabContact:
    """Weak-ref'able stand-in; SimpleNamespace cannot key the compat cache."""

    def __init__(self, names, force):
        self.name = "contact_forces"
        self.primary_names = names
        self.data = SimpleNamespace(force_history=force)


def test_mjlab_illegal_contact_thresholds_full_force_history() -> None:
    """1 N on ‖force‖, max over history. A 0.4 N brush must not terminate."""
    from instinctlab_engine.bridge.sensors import forget

    ref = ContactSensorRef(
        name="contact_forces", elements="torso_link", history_length=3
    )
    force = torch.zeros(2, 2, 3, 3)
    force[0, 0, 0] = torch.tensor([0.4, 0.0, 0.0])
    force[1, 0, 1] = torch.tensor([0.0, 0.0, 1.2])
    sensor = _MjlabContact(["torso_link", "pelvis"], force)
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"contact_forces": sensor}))
    try:
        out = illegal_contact_by_force(env, ref, threshold=1.0)
    finally:
        forget(sensor)
    assert out.tolist() == [False, True]


def test_mjlab_undesired_contacts_counts_bodies_above_one_newton() -> None:
    from instinctlab_engine.bridge.sensors import forget

    ref = ContactSensorRef(
        name="contact_forces", elements="(?!.*_ankle_roll_link).*", history_length=3
    )
    force = torch.zeros(1, 3, 2, 3)
    force[0, 0, 0] = torch.tensor([2.0, 0.0, 0.0])
    force[0, 1, 0] = torch.tensor([0.2, 0.0, 0.0])
    sensor = _MjlabContact(["torso_link", "pelvis", "left_ankle_roll_link"], force)
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"contact_forces": sensor}))
    try:
        out = undesired_contacts_by_force(env, ref, threshold=1.0)
    finally:
        forget(sensor)
    assert torch.equal(out, torch.tensor([1.0]))


def test_mjlab_offset_reset_adds_and_scale_reset_multiplies() -> None:
    offset = _aug_ops(_function(EVENTS, "reset_joints_by_offset"))
    scale = _aug_ops(_function(EVENTS, "reset_joints_by_scale"))
    assert ast.Add in offset
    assert ast.Mult not in offset
    assert ast.Mult in scale
    assert ast.Add not in scale


@pytest.mark.parametrize(
    ("reset", "position_range", "expected_pos", "velocity_range", "expected_vel"),
    [
        (reset_joints_by_offset, (0.2, 0.2), (1.2, 2.2), (0.3, 0.3), (0.4, 0.5)),
        (reset_joints_by_scale, (2.0, 2.0), (2.0, 4.0), (3.0, 3.0), (0.3, 0.6)),
    ],
)
def test_mjlab_joint_resets_broadcast_model_defaults_to_arbitrary_env_ids(
    reset, position_range, expected_pos, velocity_range, expected_vel, monkeypatch
) -> None:
    """MJLab stores defaults and limits in one row; resetting env 2 must not index row 2."""

    class Asset:
        data = SimpleNamespace(
            default_joint_pos=torch.tensor([[1.0, 2.0]]),
            default_joint_vel=torch.tensor([[0.1, 0.2]]),
            soft_joint_pos_limits=torch.tensor([[[-10.0, 10.0], [-10.0, 10.0]]]),
        )

        def write_joint_state_to_sim(self, pos, vel, **kwargs):
            self.written = pos, vel, kwargs

    asset = Asset()
    env = SimpleNamespace(num_envs=3, device="cpu", scene={"robot": asset})
    cfg = SimpleNamespace(name="robot", joint_ids=slice(None))
    monkeypatch.setattr(
        "instinctlab_engine.bridge.math.sample_uniform",
        lambda lo, hi, shape, device: torch.full(shape, (lo + hi) / 2, device=device),
    )

    reset(
        env,
        torch.tensor([2, 0]),
        position_range=position_range,
        velocity_range=velocity_range,
        asset_cfg=cfg,
    )

    pos, vel, kwargs = asset.written
    torch.testing.assert_close(pos, torch.tensor([expected_pos, expected_pos]))
    torch.testing.assert_close(vel, torch.tensor([expected_vel, expected_vel]))
    assert kwargs["env_ids"].tolist() == [2, 0]
