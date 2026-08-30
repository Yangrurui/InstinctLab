"""Guards for the runtime compatibility boundary.

The production package intentionally has no static hub/denylist catalog. Tasks
state portable fields directly and use a small compat reader only where native
names or physical quantities genuinely differ.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import instinctlab_engine.bridge as compat
from instinctlab_engine.bridge import robot as compat_robot
from instinctlab_engine.bridge.errors import PortabilityError


def _asset(**data):
    return SimpleNamespace(data=SimpleNamespace(**data))


def test_package_front_exports_runtime_interfaces_not_audit_catalogs() -> None:
    assert compat.PortabilityError is PortabilityError
    for obsolete in ("DENYLIST", "HUB", "assert_portable", "hub_entry", "spoke_attr"):
        assert not hasattr(compat, obsolete)


def test_root_angular_velocity_prefers_the_direct_com_property() -> None:
    com = torch.tensor([[1.0, 2.0, 3.0]])
    link = torch.tensor([[4.0, 5.0, 6.0]])
    asset = _asset(root_com_ang_vel_b=com, root_link_ang_vel_b=link)
    assert compat_robot.root_angular_velocity_b(asset) is com


def test_root_angular_velocity_accepts_the_equivalent_link_property() -> None:
    link = torch.tensor([[1.0, 2.0, 3.0]])
    assert compat_robot.root_angular_velocity_b(
        _asset(root_link_ang_vel_b=link)
    ) is link


def test_root_angular_velocity_fails_when_neither_native_property_exists() -> None:
    with pytest.raises(PortabilityError, match="exposes neither"):
        compat_robot.root_angular_velocity_b(_asset())


def test_root_linear_velocity_requires_an_explicit_physical_anchor() -> None:
    com = torch.tensor([[1.0, 2.0, 3.0]])
    link = torch.tensor([[4.0, 5.0, 6.0]])
    asset = _asset(root_com_lin_vel_b=com, root_link_lin_vel_b=link)
    assert compat_robot.root_linear_velocity_b(asset, anchor="com") is com
    assert compat_robot.root_linear_velocity_b(asset, anchor="link") is link
    with pytest.raises(ValueError, match="must be 'com' or 'link'"):
        compat_robot.root_linear_velocity_b(asset, anchor="native")


def test_root_linear_velocity_does_not_fall_back_to_another_anchor() -> None:
    asset = _asset(root_link_lin_vel_b=torch.zeros(1, 3))
    with pytest.raises(PortabilityError, match="root_com_lin_vel_b"):
        compat_robot.root_linear_velocity_b(asset, anchor="com")
