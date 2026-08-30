"""Pre-bootstrap facade for the MJLab adapter."""

from __future__ import annotations

import argparse
from typing import Any, ClassVar

from instinctlab_engine.base import require_supported_version


def _implementation():
    from .adapter import MjlabAdapter as NativeMjlabAdapter

    return NativeMjlabAdapter()


class MjlabAdapter:
    """Expose CLI/bootstrap without importing MJLab or torch during discovery."""

    name = "mjlab"
    SUPPORTED_VERSIONS = "==1.5.0"
    RUNTIME_VERSIONS: ClassVar[dict[str, str]] = {
        "mujoco": "==3.10.0",
        "mujoco-warp": "==3.10.0.1",
        "warp-lang": "==1.14.0",
    }

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--device",
            type=str,
            default="cuda:0",
            help="Device to simulate and learn on.",
        )

    @staticmethod
    def bootstrap(args: argparse.Namespace) -> object | None:
        del args
        require_supported_version(
            "mjlab", MjlabAdapter.SUPPORTED_VERSIONS, engine=MjlabAdapter.name
        )
        for distribution, supported in MjlabAdapter.RUNTIME_VERSIONS.items():
            require_supported_version(
                distribution, supported, engine=MjlabAdapter.name
            )
        from mjlab.utils.torch import configure_torch_backends

        configure_torch_backends()
        return None

    @staticmethod
    def wrap_for_rl(env: Any) -> Any:
        return _implementation().wrap_for_rl(env)

    def capabilities(self):
        return _implementation().capabilities()

    def robot_spec(self, asset_id: str):
        return _implementation().robot_spec(asset_id)

    def asset_conformance(self, asset_id: str):
        from .assets import asset_conformance

        return asset_conformance(asset_id)

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
        return exit_code
