"""Adapt the unified MJLab environment to MJLab's ``ViserPlayViewer``."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable


class _ViserSceneShim:
    sensors: dict[str, Any] = {}


class _ViserRewardBridge:
    def __init__(self, reward_manager: Any) -> None:
        self._manager = reward_manager

    def get_active_iterable_terms(self, env_idx: int) -> list[tuple[str, list[float]]]:
        terms: list[tuple[str, list[float]]] = []
        for (group_name, term_name), values in self._manager._episode_sums.items():
            terms.append((f"{group_name}/{term_name}", [float(values[env_idx].item())]))
        return terms

    def get_visualizable_terms(self) -> list[tuple[str, Any]]:
        return []


class _ViserCommandBridge:
    def __init__(self, command_manager: Any) -> None:
        self._manager = command_manager

    @property
    def active_terms(self) -> list[str]:
        return list(self._manager.term_names)

    def create_gui(
        self,
        server: Any,
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        del request_action
        if "base_velocity" not in self._manager.term_names:
            return
        term = dict(self._manager._terms)["base_velocity"]
        ranges = term.cfg.params.get("ranges", {})
        lin_x = ranges.get("lin_vel_x", (-0.5, 1.0))
        lin_y = ranges.get("lin_vel_y", (-0.5, 0.5))
        ang_z = ranges.get("ang_vel_z", (-1.5, 1.5))
        with server.gui.add_folder("Base velocity"):
            enabled = server.gui.add_checkbox("Enable", initial_value=True)
            sliders = [
                server.gui.add_slider(
                    "lin_vel_x", min=float(lin_x[0]), max=float(lin_x[1]), step=0.05, initial_value=0.8
                ),
                server.gui.add_slider(
                    "lin_vel_y", min=float(lin_y[0]), max=float(lin_y[1]), step=0.05, initial_value=0.0
                ),
                server.gui.add_slider(
                    "ang_vel_z", min=float(ang_z[0]), max=float(ang_z[1]), step=0.05, initial_value=0.0
                ),
            ]

        def _apply(_event: Any = None) -> None:
            if not enabled.value:
                return
            idx = int(get_env_idx())
            term._standing[idx] = False
            term._heading[idx] = False
            term._time_left[idx] = 1.0e9
            term._command[idx, 0] = float(sliders[0].value)
            term._command[idx, 1] = float(sliders[1].value)
            term._command[idx, 2] = float(sliders[2].value)
            if on_change is not None:
                on_change()

        enabled.on_update(_apply)
        for slider in sliders:
            slider.on_update(_apply)
        _apply()

    def on_viewer_pause(self, paused: bool) -> None:
        del paused

    def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
        del env_ids
        return False

    def create_debug_vis_gui(self, server: Any, on_change: Callable[[], None] | None = None) -> None:
        del server, on_change


class _ViserUnwrapped:
    def __init__(self, env: Any, sim: Any) -> None:
        self._env = env
        self.sim = sim
        self.step_dt = env.step_dt
        self.num_envs = env.num_envs
        self.device = env.device
        self.scene = _ViserSceneShim()
        self.command_manager = _ViserCommandBridge(env.command_manager)
        self.reward_manager = _ViserRewardBridge(env.reward_manager)

    def reset(self, env_ids: torch.Tensor | None = None, **kwargs: Any) -> Any:
        if env_ids is None:
            return self._env.reset(**kwargs)
        self._env._reset_idx(env_ids)
        return self._env.observation_manager.compute(), self._env.extras


@dataclass
class MjlabViserPlayEnv:
    """``EnvProtocol`` wrapper that exposes MJLab's native ``Simulation`` to Viser."""

    env: Any
    command: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        from mjlab.viewer.viewer_config import ViewerConfig

        backend = self.env.unwrapped.backend
        if not hasattr(backend, "native_sim"):
            raise RuntimeError("Viser play requires an initialized MJLab backend")
        self.cfg = SimpleNamespace(
            viewer=ViewerConfig(
                origin_type=ViewerConfig.OriginType.ASSET_ROOT,
                entity_name="robot",
                distance=2.8,
                elevation=-18.0,
                azimuth=140.0,
            )
        )
        self.unwrapped = _ViserUnwrapped(self.env.unwrapped, backend.native_sim)
        self.num_envs = self.env.num_envs
        self.device = self.env.device
        if self.command is not None:
            self._pin_command()

    def get_observations(self) -> torch.Tensor:
        obs, _ = self.env.get_observations()
        return obs

    def step(self, actions: torch.Tensor) -> tuple[Any, ...]:
        result = self.env.step(actions)
        if self.command is not None and bool(result[2].any()):
            self._pin_command()
        return result

    def reset(self) -> torch.Tensor:
        obs, _ = self.env.reset()
        if self.command is not None:
            self._pin_command()
        return obs

    def close(self) -> None:
        self.env.close()

    def _pin_command(self) -> None:
        if self.command is None:
            return
        term = dict(self.env.unwrapped.command_manager._terms)["base_velocity"]
        term._standing[:] = False
        term._heading[:] = False
        term._time_left[:] = 1.0e9
        term._command[:, 0] = self.command[0]
        term._command[:, 1] = self.command[1]
        term._command[:, 2] = self.command[2]


def play_with_viser(
    env: Any,
    policy: Any,
    *,
    command: tuple[float, float, float] | None = None,
    port: int = 8080,
) -> None:
    """Run MJLab's Viser play viewer against a unified Instinct-RL environment."""
    import viser
    from mjlab.viewer.viser.viewer import ViserPlayViewer

    adapter = MjlabViserPlayEnv(env, command=command)
    server = viser.ViserServer(label="instinctlab", port=port)
    print(f"[INFO] Viser viewer: http://localhost:{port}")
    try:
        ViserPlayViewer(adapter, policy, viser_server=server).run()
    finally:
        server.stop()


__all__ = ["MjlabViserPlayEnv", "play_with_viser"]
