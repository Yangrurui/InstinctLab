"""MJLab-native external actuator loaded only after backend selection."""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.actuator import BuiltinPdActuator, BuiltinPdActuatorCfg


class StatefulActuator(BuiltinPdActuator):
    """External model using MJLab's native PD and delay-buffer paths."""


@dataclass(kw_only=True)
class StatefulActuatorCfg(BuiltinPdActuatorCfg):
    """Build :class:`StatefulActuator` without engine-side model knowledge."""

    def build(self, entity, target_ids, target_names) -> StatefulActuator:
        return StatefulActuator(self, entity, target_ids, target_names)
