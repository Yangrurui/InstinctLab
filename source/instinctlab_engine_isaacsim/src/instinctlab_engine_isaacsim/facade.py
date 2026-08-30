"""Pre-bootstrap facade for the Isaac Sim adapter."""

from __future__ import annotations

import argparse
from typing import Any

from instinctlab_engine.base import require_supported_version


class _ExplicitAction(argparse.Action):
    """Match AppLauncher's explicit-value marker without importing Isaac Lab."""

    def __call__(self, parser, namespace, values, option_string=None):
        del parser, option_string
        setattr(namespace, self.dest, values)
        setattr(namespace, f"{self.dest}_explicit", True)


def _implementation():
    from .adapter import IsaacSimAdapter as NativeIsaacSimAdapter

    return NativeIsaacSimAdapter()


class IsaacSimAdapter:
    """Keep discovery and CLI parsing on the safe side of AppLauncher."""

    name = "isaacsim"
    SUPPORTED_VERSIONS = ">=0.54,<0.55"

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group(
            "app_launcher arguments",
            description="Arguments forwarded to Isaac Lab's AppLauncher.",
        )
        group.add_argument("--headless", action="store_true", default=False)
        group.add_argument("--livestream", type=int, default=-1, choices=(0, 1, 2))
        group.add_argument("--enable_cameras", action="store_true", default=False)
        group.add_argument("--xr", action="store_true", default=False)
        group.add_argument(
            "--device", type=str, action=_ExplicitAction, default="cuda:0"
        )
        group.add_argument("--cpu", action="store_true", help=argparse.SUPPRESS)
        group.add_argument("--verbose", action="store_true")
        group.add_argument("--info", action="store_true")
        group.add_argument("--experience", type=str, default="")
        group.add_argument(
            "--rendering_mode",
            type=str,
            action=_ExplicitAction,
            choices=("performance", "balanced", "quality"),
        )
        group.add_argument("--kit_args", type=str, default="")
        group.add_argument("--anim_recording_enabled", action="store_true")
        group.add_argument("--anim_recording_start_time", type=float, default=0)
        group.add_argument("--anim_recording_stop_time", type=float, default=10)

    @staticmethod
    def bootstrap(args: argparse.Namespace) -> object:
        require_supported_version(
            "isaaclab",
            IsaacSimAdapter.SUPPORTED_VERSIONS,
            engine=IsaacSimAdapter.name,
        )
        from isaaclab.app import AppLauncher

        app = AppLauncher(args).app

        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        return app

    @staticmethod
    def wrap_for_rl(env: Any) -> Any:
        return _implementation().wrap_for_rl(env)

    def capabilities(self):
        return _implementation().capabilities()

    def robot_spec(self, asset_id: str):
        return _implementation().robot_spec(asset_id)

    def profile(self, spec):
        return _implementation().profile(spec)

    def compile(
        self, spec, *, num_envs: int, device: str, strict: bool = False
    ):
        return _implementation().compile(
            spec, num_envs=num_envs, device=device, strict=strict
        )

    def contract_report(self, spec):
        return _implementation().contract_report(spec)

    @staticmethod
    def finalize_process(exit_code: int) -> int:
        """Avoid Isaac Sim's known post-close teardown hang."""
        import os
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
