"""Guard: parkour's per-engine reward/event kinds do what the task asked, not a neighbour.

``applied_torque`` is on the denylist and mjlab has no attribute of that name. The stock mjlab
``joint_torques_l2`` reads ``actuator_force`` (nu) and ignores a joint-only selection. Friction
ranges used to come only from the solver profile. Each of those is a silent failure: the run
converges, the objective is not the one written down.
"""

from __future__ import annotations

import ast
import pathlib
import torch
from types import SimpleNamespace

import pytest

from instinctlab.engines.isaacsim.terms import ISAAC_CONTACT_FORCE_THRESHOLD_N
from instinctlab.engines.isaacsim.terms import TERMS as ISAAC_TERMS
from instinctlab.engines.isaacsim.terms import merge_friction_params as isaac_merge_friction
from instinctlab.engines.mjlab.events import reset_joints_by_offset, reset_joints_by_scale
from instinctlab.engines.mjlab.rewards import (
    CONTACT_FORCE_THRESHOLD_N,
    applied_torque_limits_by_ratio,
    illegal_contact,
    joint_torques_l2,
    motors_power_square,
    undesired_contacts,
)
from instinctlab.engines.mjlab.terms import TERMS as MJLAB_TERMS
from instinctlab.engines.mjlab.terms import merge_friction_params as mjlab_merge_friction
from instinctlab.spec.sensor import ContactSensorRef

EVENTS = pathlib.Path(__file__).resolve().parents[1] / "source/instinctlab/instinctlab/engines/mjlab/events.py"


def _function(path: pathlib.Path, name: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{path} has no function {name}")


def _aug_ops(function: ast.FunctionDef) -> list[type]:
    return [type(node.op) for node in ast.walk(function) if isinstance(node, ast.AugAssign)]


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
        {"static_friction_range": (0.3, 1.6), "dynamic_friction_range": (0.3, 1.6), "restitution_range": (0.05, 0.5)},
    )
    assert merged["static_friction_range"] == (0.3, 1.6)
    assert merged["dynamic_friction_range"] == (0.3, 1.6)
    assert merged["restitution_range"] == (0.05, 0.5)
    assert merged["num_buckets"] == 64
    with pytest.raises(ValueError, match="does not honor \\['ranges'\\]"):
        isaac_merge_friction(profile, {"ranges": (0.3, 1.6)})


def test_mjlab_friction_maps_static_dynamic_to_their_union_and_rejects_restitution() -> None:
    profile = {"ranges": (0.2, 0.8), "operation": "abs", "shared_random": True}
    merged = mjlab_merge_friction(profile, {"static_friction_range": (0.3, 1.6), "dynamic_friction_range": (0.4, 1.2)})
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
        scene={"robot": SimpleNamespace(data=SimpleNamespace(qfrc_actuator=torch.tensor([[1.0, 2.0, 3.0]])))}
    )
    out = joint_torques_l2(env, asset_cfg=SimpleNamespace(name="robot", joint_ids=[0, 2]))
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
    out = motors_power_square(
        env, asset_cfg=SimpleNamespace(name="robot", joint_ids=slice(None)), normalize_by_stiffness=False
    )
    assert torch.equal(out, torch.tensor([9.0 + 64.0]))


def test_mjlab_applied_torque_limits_by_ratio_reads_joint_effort_limits_when_present() -> None:
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
        env, asset_cfg=SimpleNamespace(name="robot", joint_ids=slice(None)), limit_ratio=0.8
    )
    assert torch.equal(out, torch.tensor([4.0]))


"""
Offset reset is addition, not a scale.
"""


def test_force_threshold_kinds_are_registered_on_both_engines() -> None:
    assert ISAAC_TERMS.lookup("termination", "illegal_contact") is not None
    assert ISAAC_TERMS.lookup("reward", "undesired_contacts") is not None
    assert MJLAB_TERMS.lookup("termination", "illegal_contact") is not None
    assert MJLAB_TERMS.lookup("reward", "undesired_contacts") is not None
    assert CONTACT_FORCE_THRESHOLD_N == 1.0
    assert ISAAC_CONTACT_FORCE_THRESHOLD_N == 1.0


class _MjlabContact:
    """Weak-ref'able stand-in; SimpleNamespace cannot key the compat cache."""

    def __init__(self, names, force):
        self.name = "contact_forces"
        self.primary_names = names
        self.data = SimpleNamespace(force_history=force)


def test_mjlab_illegal_contact_thresholds_full_force_history() -> None:
    """1 N on ‖force‖, max over history. A 0.4 N brush must not terminate."""
    from instinctlab.compat.sensors import forget

    ref = ContactSensorRef(name="contact_forces", elements="torso_link", history_length=3)
    force = torch.zeros(2, 2, 3, 3)
    force[0, 0, 0] = torch.tensor([0.4, 0.0, 0.0])
    force[1, 0, 1] = torch.tensor([0.0, 0.0, 1.2])
    sensor = _MjlabContact(["torso_link", "pelvis"], force)
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"contact_forces": sensor}))
    try:
        out = illegal_contact(env, ref, threshold=1.0)
    finally:
        forget(sensor)
    assert out.tolist() == [False, True]


def test_mjlab_undesired_contacts_counts_bodies_above_one_newton() -> None:
    from instinctlab.compat.sensors import forget

    ref = ContactSensorRef(name="contact_forces", elements="(?!.*_ankle_roll_link).*", history_length=3)
    force = torch.zeros(1, 3, 2, 3)
    force[0, 0, 0] = torch.tensor([2.0, 0.0, 0.0])
    force[0, 1, 0] = torch.tensor([0.2, 0.0, 0.0])
    sensor = _MjlabContact(["torso_link", "pelvis", "left_ankle_roll_link"], force)
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"contact_forces": sensor}))
    try:
        out = undesired_contacts(env, ref, threshold=1.0)
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
        "instinctlab.compat.math.sample_uniform",
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
