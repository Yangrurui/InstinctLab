"""MJLab must apply policy actions in the order declared by the portable task."""

from __future__ import annotations

import pytest

pytest.importorskip("mjlab")

from instinctlab.engines.mjlab.actions import PreservingJointPositionAction, PreservingJointPositionActionCfg


class _Entity:
    joint_names = ["hip", "knee", "ankle"]

    def find_joints_by_actuator_names(self, names):
        selected = [name for name in self.joint_names if name in names]
        return [self.joint_names.index(name) for name in selected], selected

    def find_joints(self, names, joint_subset=None, preserve_order=False):
        pool = list(joint_subset or self.joint_names)
        selected = (
            [name for name in names if name in pool] if preserve_order else [name for name in pool if name in names]
        )
        return [pool.index(name) for name in selected], selected


def test_joint_action_preserves_declared_order_instead_of_entity_order() -> None:
    action = object.__new__(PreservingJointPositionAction)
    action._entity = _Entity()
    cfg = PreservingJointPositionActionCfg(entity_name="robot", actuator_names=("ankle", "hip"), preserve_order=True)

    ids, names = action._find_targets(cfg)

    assert names == ["ankle", "hip"]
    assert ids == [2, 0]


def test_joint_action_keeps_upstream_natural_order_when_not_requested() -> None:
    action = object.__new__(PreservingJointPositionAction)
    action._entity = _Entity()
    cfg = PreservingJointPositionActionCfg(entity_name="robot", actuator_names=("ankle", "hip"), preserve_order=False)

    ids, names = action._find_targets(cfg)

    assert names == ["hip", "ankle"]
    assert ids == [0, 2]
