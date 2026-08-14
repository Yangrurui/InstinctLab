from __future__ import annotations

import torch
from types import SimpleNamespace

from instinctlab.backends.mjlab.simulator import MjlabBackend, _write_cvel_link_velocity
from instinctlab.sim.backend import CanonicalIndexMap, SensorReadPhase, contiguous_index_range
from instinctlab.sim.state import ArticulationState


def _reference_cvel_velocity(pos: torch.Tensor, subtree_com: torch.Tensor, cvel: torch.Tensor) -> torch.Tensor:
    if subtree_com.shape != pos.shape:
        subtree_com = subtree_com.unsqueeze(1).expand_as(pos)
    lin_vel_c = cvel[..., 3:6]
    ang_vel_c = cvel[..., 0:3]
    offset = subtree_com - pos
    lin_vel_w = lin_vel_c - torch.cross(ang_vel_c, offset, dim=-1)
    return torch.cat([lin_vel_w, ang_vel_c], dim=-1)


def test_contiguous_index_range_records_runtime_start() -> None:
    body_ids = torch.arange(2, 42)
    assert contiguous_index_range(body_ids, expected_count=40, require_positive_start=True) == (2, 40)
    assert contiguous_index_range(torch.arange(0, 40), expected_count=40, require_positive_start=True) is None
    assert contiguous_index_range(torch.tensor([2, 3, 5]), require_positive_start=True) is None
    assert contiguous_index_range(torch.arange(7, 36), expected_count=29) == (7, 29)
    assert contiguous_index_range(torch.arange(6, 35), expected_count=29) == (6, 29)
    assert contiguous_index_range(torch.arange(0, 2)) == (0, 2)


def test_qpos_and_qvel_slices_are_detected_separately() -> None:
    q_slice = contiguous_index_range(torch.arange(7, 36), expected_count=29)
    v_slice = contiguous_index_range(torch.arange(6, 35), expected_count=29)
    assert q_slice == (7, 29)
    assert v_slice == (6, 29)
    assert q_slice != v_slice


def test_write_cvel_link_velocity_matches_reference() -> None:
    pos = torch.randn(3, 4, 3)
    subtree_com = torch.randn(3, 3)
    cvel = torch.randn(3, 4, 6)
    lin_out = torch.zeros(3, 4, 3)
    ang_out = torch.zeros(3, 4, 3)
    offset = torch.zeros(3, 4, 3)
    _write_cvel_link_velocity(pos, subtree_com, cvel, lin_out, ang_out, offset)
    expected = _reference_cvel_velocity(pos, subtree_com, cvel)
    torch.testing.assert_close(lin_out, expected[..., :3], atol=0, rtol=0)
    torch.testing.assert_close(ang_out, expected[..., 3:6], atol=0, rtol=0)


def _make_fast_backend(*, alias: bool = False) -> tuple[MjlabBackend, ArticulationState]:
    num_envs = 2
    num_bodies = 3
    num_joints = 2
    nbody = 5
    nqpos = 10
    nqvel = 9
    body_start, q_start, v_start = 2, 7, 6
    data = SimpleNamespace(
        xpos=torch.randn(num_envs, nbody, 3),
        xquat=torch.randn(num_envs, nbody, 4),
        cvel=torch.randn(num_envs, nbody, 6),
        subtree_com=torch.randn(num_envs, nbody, 3),
        qpos=torch.randn(num_envs, nqpos),
        qvel=torch.randn(num_envs, nqvel),
    )
    native = SimpleNamespace(
        data=data,
        qfrc_actuator=torch.randn(num_envs, num_joints),
        joint_pos=data.qpos[:, q_start : q_start + num_joints],
        joint_vel=data.qvel[:, v_start : v_start + num_joints],
        body_link_pose_w=torch.cat(
            [data.xpos[:, body_start : body_start + num_bodies], data.xquat[:, body_start : body_start + num_bodies]],
            dim=-1,
        ),
        body_link_vel_w=_reference_cvel_velocity(
            data.xpos[:, body_start : body_start + num_bodies],
            data.subtree_com[:, body_start],
            data.cvel[:, body_start : body_start + num_bodies],
        ),
        root_link_pose_w=torch.cat([data.xpos[:, body_start], data.xquat[:, body_start]], dim=-1),
        root_link_vel_w=_reference_cvel_velocity(
            data.xpos[:, body_start],
            data.subtree_com[:, body_start],
            data.cvel[:, body_start],
        ),
    )
    indexing = SimpleNamespace(
        body_ids=torch.arange(body_start, body_start + num_bodies),
        joint_q_adr=torch.arange(q_start, q_start + num_joints),
        joint_v_adr=torch.arange(v_start, v_start + num_joints),
        root_body_id=body_start,
    )
    names = tuple(f"b{i}" for i in range(num_bodies))
    joints = tuple(f"j{i}" for i in range(num_joints))
    mapping = CanonicalIndexMap.build(joints, joints, device="cpu")
    body_map = CanonicalIndexMap.build(names, names, device="cpu")
    state = ArticulationState.allocate(
        num_envs=num_envs,
        num_joints=num_joints,
        num_bodies=num_bodies,
        device="cpu",
    )
    backend = object.__new__(MjlabBackend)
    backend._entity = SimpleNamespace(data=native, indexing=indexing)
    backend._entity_name = "robot"
    backend._joint_map = mapping
    backend._body_map = body_map
    backend._body_slice = (body_start, num_bodies)
    backend._joint_q_slice = (q_start, num_joints)
    backend._joint_v_slice = (v_start, num_joints)
    backend._sync_fast_path = True
    backend._alias_native_views = False
    backend._native_ptrs = None
    backend._tmp_body_offset = torch.zeros(num_envs, num_bodies, 3)
    backend._tmp_root_offset = torch.zeros(num_envs, 3)
    backend._last_joint_acc_native = torch.randn(num_envs, num_joints)
    backend._contact_bindings = {}
    backend.scene = SimpleNamespace(articulations={"robot": SimpleNamespace(data=state)}, sensors={})
    if alias:
        backend._bind_native_views(state)
    return backend, state


def test_mjlab_fast_sync_matches_property_path() -> None:
    backend, state = _make_fast_backend()
    backend._synchronize_articulation_fast()
    fast = {
        "root_pos_w": state.root_pos_w.clone(),
        "root_quat_w": state.root_quat_w.clone(),
        "root_lin_vel_w": state.root_lin_vel_w.clone(),
        "root_ang_vel_w": state.root_ang_vel_w.clone(),
        "body_pos_w": state.body_pos_w.clone(),
        "body_quat_w": state.body_quat_w.clone(),
        "body_lin_vel_w": state.body_lin_vel_w.clone(),
        "body_ang_vel_w": state.body_ang_vel_w.clone(),
        "joint_pos": state.joint_pos.clone(),
        "joint_vel": state.joint_vel.clone(),
        "joint_acc": state.joint_acc.clone(),
        "applied_joint_effort": state.applied_joint_effort.clone(),
    }
    backend._synchronize_articulation_properties()
    for name in (
        "root_pos_w",
        "root_quat_w",
        "body_pos_w",
        "body_quat_w",
        "joint_pos",
        "joint_vel",
        "joint_acc",
        "applied_joint_effort",
    ):
        assert torch.equal(fast[name], getattr(state, name)), name
    for name in ("root_lin_vel_w", "root_ang_vel_w", "body_lin_vel_w", "body_ang_vel_w"):
        torch.testing.assert_close(fast[name], getattr(state, name), atol=0, rtol=0)


def test_mjlab_fast_sync_does_not_reuse_qpos_slice_for_qvel() -> None:
    backend, state = _make_fast_backend()
    native = backend._entity.data
    backend._synchronize_articulation_fast()
    assert torch.equal(state.joint_pos, native.data.qpos[:, 7:9])
    assert torch.equal(state.joint_vel, native.data.qvel[:, 6:8])
    assert not torch.equal(state.joint_vel, native.data.qvel[:, 7:9])


def test_mjlab_native_view_alias_and_detach() -> None:
    backend, state = _make_fast_backend(alias=True)
    assert backend._alias_native_views
    assert state.joint_pos.data_ptr() == backend._entity.data.data.qpos[:, 7:9].data_ptr()
    backend._entity.data.data.qpos[:, 7:9] = 1.5
    assert torch.all(state.joint_pos == 1.5)
    backend._entity.data.data.qpos = backend._entity.data.data.qpos.clone()
    assert not backend._native_views_still_valid()
    backend._detach_native_views()
    assert not backend._alias_native_views
    backend._synchronize_articulation_fast()
    assert torch.equal(state.joint_pos, backend._entity.data.data.qpos[:, 7:9])


def test_mjlab_synchronize_uses_fast_path_without_forward() -> None:
    backend, _ = _make_fast_backend()
    backend._sim = SimpleNamespace(forward=lambda: (_ for _ in ()).throw(AssertionError("forward")))
    backend._mj_scene = SimpleNamespace(write_data_to_sim=lambda: (_ for _ in ()).throw(AssertionError("write")))
    backend.synchronize(SensorReadPhase.POST_PHYSICS)
