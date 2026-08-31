"""Isaac-native external actuator loaded only after Kit bootstrap."""

from __future__ import annotations

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass


class StatefulActuator(DelayedPDActuator):
    """External model using Isaac Lab's native delayed-PD state path."""


@configclass
class StatefulActuatorCfg(DelayedPDActuatorCfg):
    """Build :class:`StatefulActuator` without engine-side model knowledge."""

    class_type: type = StatefulActuator
