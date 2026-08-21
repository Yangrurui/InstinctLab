"""Live Isaac Sim behavior cells. Default pytest deselects this module.

Kit must start before ``torch`` is imported. This file therefore imports only
pytest at module level. Do not collect it in the same process as mjlab cells.
"""

from __future__ import annotations

import pytest

from tests.isaacsim_app import ensure_isaac_app

pytestmark = pytest.mark.isaacsim


def _launch_isaac():
    pytest.importorskip("isaaclab")
    return ensure_isaac_app()


@pytest.fixture(scope="module")
def isaac_locomotion():
    launcher = _launch_isaac()
    from tests.simulators.harness import close_live_backend, initialize_locomotion_backend

    backend, cfg = initialize_locomotion_backend("isaacsim", num_envs=2, bootstrap_context=launcher)
    try:
        yield backend, cfg
    finally:
        close_live_backend(backend)


def test_reset_root_vel(isaac_locomotion) -> None:
    from tests.simulators.harness import assert_reset_root_vel

    backend, _ = isaac_locomotion
    assert_reset_root_vel(backend)


def test_air_time_advance(isaac_locomotion) -> None:
    from tests.simulators.harness import assert_air_time_advance

    backend, _ = isaac_locomotion
    assert_air_time_advance(backend)


def test_material_write_scope(isaac_locomotion) -> None:
    from tests.simulators.harness import assert_material_write_scope

    backend, cfg = isaac_locomotion
    assert_material_write_scope(backend, cfg.scene.robot)
