"""Identity-scoped runtime adapters for application actuator aliases."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from .registration import DELAYED_PD_MODEL_ID


def _model_id(actuator: object) -> object:
    declared = getattr(actuator, "instinctlab_model_id", None)
    if declared is not None:
        return declared
    return getattr(getattr(actuator, "cfg", None), "instinctlab_model_id", None)


@dataclass(frozen=True, slots=True)
class DeclaredModelRuntimeAdapter:
    """Delegate only native groups carrying this application's model identity.

    The backends also publish adapters for their native base classes. Reusing
    those adapters directly would make an unlabelled native PD group match both
    the backend model and this application alias. Requiring the declared model
    id preserves the registry's one-model-per-group invariant.
    """

    delegate_path: str

    def _delegate(self) -> Any:
        module_name, _, attribute = self.delegate_path.partition(":")
        return getattr(import_module(module_name), attribute)

    def matches(self, actuator: object) -> bool:
        if _model_id(actuator) != DELAYED_PD_MODEL_ID:
            return False
        return bool(self._delegate().matches(actuator))

    def stiffness_groups(self, env: object, asset: object, actuator: object):
        return self._delegate().stiffness_groups(env, asset, actuator)

    def effort_limit_for_joint(
        self,
        env: object,
        asset: object,
        actuator: object,
        local_index: int,
    ):
        return self._delegate().effort_limit_for_joint(
            env, asset, actuator, local_index
        )


ISAACSIM_DELAYED_PD_RUNTIME = DeclaredModelRuntimeAdapter(
    "instinctlab_engine_isaacsim.actuator_runtime:DELAYED_PD_RUNTIME"
)
MJLAB_DELAYED_PD_RUNTIME = DeclaredModelRuntimeAdapter(
    "instinctlab_engine_mjlab.actuator_runtime:BUILTIN_PD_RUNTIME"
)

__all__ = [
    "ISAACSIM_DELAYED_PD_RUNTIME",
    "MJLAB_DELAYED_PD_RUNTIME",
    "DeclaredModelRuntimeAdapter",
]
