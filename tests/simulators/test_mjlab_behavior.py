"""Live MJLab behavior cells. Default pytest deselects this module's tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.mjlab


@pytest.fixture(scope="module")
def mjlab_locomotion():
    from tests.simulators.harness import close_live_backend, initialize_locomotion_backend

    backend, cfg = initialize_locomotion_backend("mjlab", num_envs=2)
    try:
        yield backend, cfg
    finally:
        close_live_backend(backend)


def test_reset_root_vel(mjlab_locomotion) -> None:
    from tests.simulators.harness import assert_reset_root_vel

    backend, _ = mjlab_locomotion
    assert_reset_root_vel(backend)


def test_air_time_advance(mjlab_locomotion) -> None:
    from tests.simulators.harness import assert_air_time_advance

    backend, _ = mjlab_locomotion
    assert_air_time_advance(backend)


def test_material_write_scope(mjlab_locomotion) -> None:
    from tests.simulators.harness import assert_material_write_scope

    backend, cfg = mjlab_locomotion
    assert_material_write_scope(backend, cfg.scene.robot)
