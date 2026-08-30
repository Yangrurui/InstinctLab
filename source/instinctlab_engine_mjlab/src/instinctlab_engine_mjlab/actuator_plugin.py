"""SDK-free registration of MJLab's built-in PD actuator model."""

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
        model_id="mjlab.builtin_pd.v1",
        config_factory="mjlab.actuator:BuiltinPdActuatorCfg",
        runtime_adapter="instinctlab_engine_mjlab.actuator_runtime:BUILTIN_PD_RUNTIME",
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
