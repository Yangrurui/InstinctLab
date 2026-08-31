"""SDK-free registration of InstinctLab's application actuator models."""

from __future__ import annotations

from instinctlab_engine.actuators import (
    APPLIED_EFFORT,
    EFFORT_LIMITS,
    GAIN_RANDOMIZATION,
    JOINT_POSITION_COMMAND,
    STATEFUL_RESET,
    STIFFNESS,
)

DELAYED_PD_MODEL_ID = "instinctlab.delayed_pd.v1"

_CAPABILITIES = {
    JOINT_POSITION_COMMAND,
    APPLIED_EFFORT,
    EFFORT_LIMITS,
    STIFFNESS,
    GAIN_RANDOMIZATION,
    STATEFUL_RESET,
}


def register_isaacsim(registry) -> None:
    """Bind the application delayed-PD identity to Isaac Lab's native model."""
    registry.register(
        model_id=DELAYED_PD_MODEL_ID,
        config_factory="isaaclab.actuators:DelayedPDActuatorCfg",
        runtime_adapter="instinctlab.actuators.runtime:ISAACSIM_DELAYED_PD_RUNTIME",
        capabilities=_CAPABILITIES,
    )


def register_mjlab(registry) -> None:
    """Bind the application delayed-PD identity to MJLab's native model."""
    registry.register(
        model_id=DELAYED_PD_MODEL_ID,
        config_factory="mjlab.actuator:BuiltinPdActuatorCfg",
        runtime_adapter="instinctlab.actuators.runtime:MJLAB_DELAYED_PD_RUNTIME",
        capabilities=_CAPABILITIES,
    )


register_isaacsim.instinctlab_engine_api = ">=0.1,<0.2"
register_mjlab.instinctlab_engine_api = ">=0.1,<0.2"

__all__ = ["DELAYED_PD_MODEL_ID", "register_isaacsim", "register_mjlab"]
