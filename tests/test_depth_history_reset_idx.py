"""Depth ring clearing must follow training auto-reset via ``_reset_idx``, not only ``reset()``."""

from __future__ import annotations

import torch

import pytest

from instinctlab.tasks.parkour.mdp.observations import DelayedDepthImage


def _depth_term(env) -> DelayedDepthImage:
    for group_cfgs in env.observation_manager._group_obs_term_cfgs.values():
        for cfg in group_cfgs:
            implementation = getattr(
                getattr(cfg, "func", None),
                "_impl",
                getattr(cfg, "func", None),
            )
            if isinstance(implementation, DelayedDepthImage):
                return implementation
    raise AssertionError("DelayedDepthImage not wired in observation manager")


def _fill_depth_ring(env, *, steps: int = 8) -> DelayedDepthImage:
    term = _depth_term(env)
    sensor = env.scene.sensors["camera"]
    raw = sensor.data.output["distance_to_image_plane"]
    for step in range(steps):
        raw.fill_(0.05 * (step + 1))
        env.observation_manager.compute(update_history=True)
    assert float(term._history.abs().max()) > 0.0
    return term


def _mjlab_post_reset_obs(env) -> None:
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.sim.sense()
    env.observation_manager.compute(update_history=True)


def _isaac_post_reset_obs(env) -> None:
    env.scene.write_data_to_sim()
    if hasattr(env.sim, "forward"):
        env.sim.forward()
    dt = float(env.step_dt)
    cam = env.scene.sensors["camera"]
    cam.update(dt, force_recompute=True)
    env.scene.update(dt)
    env.observation_manager.compute(update_history=True)


def _assert_depth_stack_primed(
    image: torch.Tensor, env_idx: int, *, tol: float = 1e-5
) -> None:
    stack = image[env_idx]
    newest = stack[-1]
    assert float(newest.abs().max()) > tol, "primed stack should not be all zeros"
    for slot in range(stack.shape[0]):
        delta = (stack[slot] - newest).abs().max()
        assert float(delta) <= tol, f"slot {slot} not primed (delta={float(delta)})"


@pytest.mark.mjlab
def test_mjlab_reset_idx_clears_subset_before_first_post_reset_obs() -> None:
    pytest.importorskip("mjlab")
    from instinctlab_engine_mjlab import MjlabAdapter
    from tests.task_specs import task_spec

    compiled = MjlabAdapter().compile(
        task_spec("Instinct-Parkour-Target-G1", "mjlab"),
        num_envs=8,
        device="cpu",
    )
    env = compiled.make_env()
    try:
        env.reset()
        term = _fill_depth_ring(env)
        kept = term._history[0].clone()
        write_before = term._write
        reset_ids = torch.tensor([1, 2, 3], device=env.device, dtype=torch.long)
        env._reset_idx(reset_ids)
        assert torch.equal(term._history[0], kept)
        assert float(term._history[1].abs().max()) == 0.0
        assert float(term._history[2].abs().max()) == 0.0
        assert float(term._history[3].abs().max()) == 0.0
        assert term._write == write_before
        assert not bool(term._primed[1])
        _mjlab_post_reset_obs(env)
        depth = env.obs_buf["policy"]["depth_image"]
        for idx in (1, 2, 3):
            _assert_depth_stack_primed(depth, idx)
        assert torch.equal(term._history[0], kept)
    finally:
        env.close()


@pytest.mark.mjlab
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="mjlab parkour stepping needs a GPU"
)
def test_mjlab_auto_reset_256_subset_clears_depth_via_reset_idx() -> None:
    from tests.parkour_live_expect import require_live_device

    pytest.importorskip("mjlab")
    from instinctlab_engine_mjlab import MjlabAdapter
    from tests.task_specs import task_spec

    device = require_live_device()
    compiled = MjlabAdapter().compile(
        task_spec("Instinct-Parkour-Target-G1", "mjlab"),
        num_envs=256,
        device=device,
    )
    compiled.env_cfg.seed = 42
    env = compiled.make_env()
    try:
        env.reset()
        term = _depth_term(env)
        _fill_depth_ring(env)
        action_dim = env.action_manager.total_action_dim
        zero = torch.zeros(env.num_envs, action_dim, device=device)
        seen = False
        for _ in range(3000):
            env.step(zero)
            reset_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if reset_ids.numel() == 0:
                continue
            seen = True
            depth = env.obs_buf["policy"]["depth_image"]
            for idx in reset_ids[:8].tolist():
                _assert_depth_stack_primed(depth, idx)
            reset_set = set(reset_ids.tolist())
            unreset = next(i for i in range(env.num_envs) if i not in reset_set)
            assert bool(term._primed[unreset])
            assert float(term._history[unreset].abs().max()) > 0.0
            break
        assert seen, "expected at least one auto-reset within 3000 zero-action steps"
    finally:
        env.close()


@pytest.mark.isaacsim
def test_isaac_reset_idx_clears_subset_before_first_post_reset_obs() -> None:
    from tests.isaacsim_app import ensure_isaac_app
    from tests.live_device import resolve_live_device

    pytest.importorskip("isaaclab")
    device = resolve_live_device()
    ensure_isaac_app(device=device)

    from instinctlab_engine_isaacsim import IsaacSimAdapter
    from tests.task_specs import task_spec

    compiled = IsaacSimAdapter().compile(
        task_spec("Instinct-Parkour-Target-G1", "isaacsim"),
        num_envs=4,
        device=device,
    )
    compiled.env_cfg.seed = 42
    env = compiled.make_env()
    try:
        env.reset()
        term = _fill_depth_ring(env)
        kept = term._history[0].clone()
        write_before = term._write
        reset_ids = torch.tensor([1, 2], device=env.device, dtype=torch.long)
        env._reset_idx(reset_ids)
        assert torch.equal(term._history[0], kept)
        assert float(term._history[1].abs().max()) == 0.0
        assert float(term._history[2].abs().max()) == 0.0
        assert term._write == write_before
        _isaac_post_reset_obs(env)
        depth = env.obs_buf["policy"]["depth_image"]
        _assert_depth_stack_primed(depth, 1)
        _assert_depth_stack_primed(depth, 2)
    finally:
        env.close()


@pytest.mark.isaacsim
def test_isaac_step_auto_reset_clears_depth_via_reset_idx() -> None:
    from tests.isaacsim_app import ensure_isaac_app
    from tests.live_device import resolve_live_device

    pytest.importorskip("isaaclab")
    device = resolve_live_device()
    ensure_isaac_app(device=device)

    from instinctlab_engine_isaacsim import IsaacSimAdapter
    from tests.task_specs import task_spec

    compiled = IsaacSimAdapter().compile(
        task_spec("Instinct-Parkour-Target-G1", "isaacsim"),
        num_envs=8,
        device=device,
    )
    compiled.env_cfg.seed = 43
    env = compiled.make_env()
    try:
        env.reset()
        action_dim = env.action_manager.total_action_dim
        zero = torch.zeros(env.num_envs, action_dim, device=device)
        seen = False
        for _ in range(2000):
            env.step(zero)
            reset_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if reset_ids.numel() == 0:
                continue
            seen = True
            depth = env.obs_buf["policy"]["depth_image"]
            _assert_depth_stack_primed(depth, int(reset_ids[0].item()))
            break
        assert seen, "expected at least one auto-reset within 2000 zero-action steps"
    finally:
        env.close()
