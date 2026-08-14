"""Engine-neutral robot asset registry.

Tasks reference a robot by a stable ``name`` (or ``asset_id``) instead of
importing a concrete factory function. This keeps task configs decoupled from
asset file layout and gives checkpoint tooling a single place to resolve an
``asset_id`` back to its :class:`RobotSpec`.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

from instinctlab.sim.robot_spec import RobotSpec


def _load(path: str) -> Callable[..., RobotSpec]:
    module_name, attribute = path.split(":", maxsplit=1)
    return getattr(importlib.import_module(module_name), attribute)


@dataclass(frozen=True)
class AssetRegistration:
    """A lazily-resolved binding from a robot name to its ``RobotSpec`` factory."""

    name: str
    asset_id: str
    spec_entry_point: str

    def make_spec(self) -> RobotSpec:
        spec = _load(self.spec_entry_point)()
        if spec.name != self.name:
            raise ValueError(f"asset {self.name!r} factory produced RobotSpec name {spec.name!r}")
        if spec.asset_id != self.asset_id:
            raise ValueError(
                f"asset {self.name!r} factory produced asset_id {spec.asset_id!r}, registry declares {self.asset_id!r}"
            )
        return spec


class AssetRegistry:
    """Lazy registry mapping robot names and asset ids to ``RobotSpec`` factories."""

    def __init__(self) -> None:
        self._by_name: dict[str, AssetRegistration] = {}
        self._name_by_asset_id: dict[str, str] = {}

    def register(self, registration: AssetRegistration) -> None:
        previous = self._by_name.get(registration.name)
        if previous is not None and previous != registration:
            raise ValueError(f"asset {registration.name!r} is already registered")
        owner = self._name_by_asset_id.get(registration.asset_id)
        if owner is not None and owner != registration.name:
            raise ValueError(f"asset_id {registration.asset_id!r} is already owned by asset {owner!r}")
        self._by_name[registration.name] = registration
        self._name_by_asset_id[registration.asset_id] = registration.name

    def get(self, name: str) -> AssetRegistration:
        try:
            return self._by_name[name]
        except KeyError as error:
            raise KeyError(f"unknown asset {name!r}; available: {', '.join(self.names())}") from error

    def make(self, name: str) -> RobotSpec:
        """Build the ``RobotSpec`` registered under ``name``."""
        return self.get(name).make_spec()

    def make_by_asset_id(self, asset_id: str) -> RobotSpec:
        """Build the ``RobotSpec`` whose ``asset_id`` matches (checkpoint resolution)."""
        try:
            name = self._name_by_asset_id[asset_id]
        except KeyError as error:
            raise KeyError(
                f"unknown asset_id {asset_id!r}; available: {', '.join(sorted(self._name_by_asset_id))}"
            ) from error
        return self.make(name)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_name))

    def asset_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._name_by_asset_id))


ASSETS = AssetRegistry()
ASSETS.register(
    AssetRegistration(
        name="unitree_g1_29dof",
        asset_id="popsicle_torsobase_v1",
        spec_entry_point="instinctlab.assets.unitree_g1:make_g1_29dof_robot_spec",
    )
)


__all__ = ["ASSETS", "AssetRegistration", "AssetRegistry"]
