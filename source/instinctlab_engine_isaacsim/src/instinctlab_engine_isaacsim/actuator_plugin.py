"""SDK-free registration of Isaac Lab's built-in actuator models."""

from instinctlab_engine.actuators import (
    APPLIED_EFFORT,
    EFFORT_LIMITS,
    GAIN_RANDOMIZATION,
    JOINT_POSITION_COMMAND,
    STATEFUL_RESET,
    STIFFNESS,
)


def register(registry) -> None:
    registry.register(
        model_id="isaaclab.implicit_pd.v1",
        config_factory="isaaclab.actuators:ImplicitActuatorCfg",
        runtime_adapter=(
            "instinctlab_engine_isaacsim.actuator_runtime:IMPLICIT_PD_RUNTIME"
        ),
        capabilities={
            JOINT_POSITION_COMMAND,
            APPLIED_EFFORT,
            EFFORT_LIMITS,
            STIFFNESS,
            GAIN_RANDOMIZATION,
        },
    )
    registry.register(
        model_id="isaaclab.delayed_pd.v1",
        config_factory="isaaclab.actuators:DelayedPDActuatorCfg",
        runtime_adapter=(
            "instinctlab_engine_isaacsim.actuator_runtime:DELAYED_PD_RUNTIME"
        ),
        capabilities={
            JOINT_POSITION_COMMAND,
            APPLIED_EFFORT,
            EFFORT_LIMITS,
            STIFFNESS,
            GAIN_RANDOMIZATION,
            STATEFUL_RESET,
        },
    )


register.instinctlab_engine_api = ">=0.1,<0.2"

__all__ = ["register"]
