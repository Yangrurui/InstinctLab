"""VolumePoints IR, the loud unregistered path, and the known-geometry penalty.

A cross-engine equality check cannot see an error both sides share. These tests
ask for numbers a person can compute by hand, then document the detector
divergence that makes the generated cylinder sets incomparable.
"""

from __future__ import annotations

import math
import numpy as np
import torch
from types import SimpleNamespace

import pytest
from test_mdp_terms import _Env

from instinctlab.compat.denylist import PortabilityError
from instinctlab.compat.sensors import (
    registered_cylinder_count,
    require_volume_points_registered,
    volume_points_penetration_offset,
    volume_points_vel_w,
)
from instinctlab.engines.volume_points import (
    cylinder_penetration_offset,
    grid3d_points,
    link_linear_velocity_from_com,
    penetration_reward,
    point_velocity_from_link,
)
from instinctlab.mdp.events import register_virtual_obstacles
from instinctlab.mdp.rewards import volume_points_penetration
from instinctlab.spec.sensor import Grid3dPointsRef, VirtualObstacleRef, VolumePointsRef

SHOE = Grid3dPointsRef(
    x_min=-0.025,
    x_max=0.12,
    x_num=10,
    y_min=-0.03,
    y_max=0.03,
    y_num=5,
    z_min=-0.063,
    z_max=-0.023,
    z_num=2,
)


def test_the_shoe_grid_is_one_hundred_points_below_the_ankle() -> None:
    """z is body-local and negative. A sign flip puts the cloud in the shin."""
    assert SHOE.count == 100
    assert SHOE.z_min < 0.0
    assert SHOE.z_max < 0.0
    assert SHOE.z_min < SHOE.z_max
    points = grid3d_points(SHOE)
    assert len(points) == 100
    assert points[0] == pytest.approx((-0.025, -0.03, -0.063))
    assert points[-1] == pytest.approx((0.12, 0.03, -0.023))
    zs = {point[2] for point in points}
    assert zs == {-0.063, -0.023}


def test_volume_points_ref_refuses_a_frame_or_velocity_flip() -> None:
    with pytest.raises(ValueError, match="attach-body local frame"):
        VolumePointsRef(name="legs", attach=("foot",), frame="world")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="link-origin"):
        VolumePointsRef(name="legs", attach=("foot",), velocity="com")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no attach"):
        VolumePointsRef(name="legs", attach=())
    with pytest.raises(ValueError, match="repeats an attach body"):
        VolumePointsRef(name="legs", attach=("foot", "foot"))
    with pytest.raises(ValueError, match="non-positive update_period"):
        VolumePointsRef(name="legs", attach=("foot",), update_period=0.0)


def test_virtual_obstacle_ref_is_greedy_edge_cylinders_only() -> None:
    with pytest.raises(ValueError, match="greedy-concat"):
        VirtualObstacleRef(name="edges", kind="ransac")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-positive cylinder_radius"):
        VirtualObstacleRef(name="edges", cylinder_radius=0.0)
    ref = VirtualObstacleRef(name="edges", cylinder_radius=0.05, min_points=2)
    assert ref.kind == "greedy_edge_cylinder"
    assert ref.angle_threshold == 70.0


def test_an_unregistered_sensor_is_refused_by_the_compat_reader() -> None:
    """The silent-zero path: reward table has the term, logs 0.0, robot never learns."""
    sensor = SimpleNamespace(
        cfg=SimpleNamespace(name="leg_volume_points"),
        virtual_obstacles_registered=False,
        _virtual_obstacles={},
        registered_cylinder_count=0,
        data=SimpleNamespace(penetration_offset=torch.zeros(1, 2, 100, 3)),
    )
    with pytest.raises(RuntimeError, match="identically zero"):
        require_volume_points_registered(sensor)
    with pytest.raises(RuntimeError, match="identically zero"):
        volume_points_penetration_offset(sensor)
    empty = SimpleNamespace(
        cfg=SimpleNamespace(name="leg_volume_points"),
        virtual_obstacles_registered=True,
        _virtual_obstacles={"edges": SimpleNamespace(edges_pyt=torch.empty(0, 6))},
        registered_cylinder_count=0,
        data=SimpleNamespace(penetration_offset=torch.zeros(1, 2, 100, 3)),
    )
    with pytest.raises(RuntimeError, match="0 cylinders"):
        require_volume_points_registered(empty)


def test_register_virtual_obstacles_refuses_an_empty_terrain() -> None:
    sensor = SimpleNamespace(
        cfg=SimpleNamespace(name="leg_volume_points"),
        virtual_obstacles_registered=False,
        _virtual_obstacles={},
        register_virtual_obstacles=lambda obstacles: None,
    )
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"leg_volume_points": sensor}, terrain=SimpleNamespace()))
    env.scene.terrain.virtual_obstacles = {}
    with pytest.raises(RuntimeError, match="identically zero"):
        register_virtual_obstacles(env, None, "leg_volume_points")


def test_a_com_sensor_is_refused_by_the_velocity_reader() -> None:
    """The denylisted trap: same buffer, COM linear, ω ≠ 0 scales the penalty wrong."""
    sensor = SimpleNamespace(
        cfg=SimpleNamespace(name="leg_volume_points", velocity="com"),
        virtual_obstacles_registered=True,
        registered_cylinder_count=1,
        _virtual_obstacles={"known": SimpleNamespace(edges_pyt=torch.zeros(1, 6))},
        data=SimpleNamespace(points_vel_w=torch.zeros(1, 2, 100, 3)),
    )
    with pytest.raises(PortabilityError, match="attach-body link origin"):
        volume_points_vel_w(sensor)


def test_registered_cylinder_count_is_observable() -> None:
    sensor = SimpleNamespace(
        registered_cylinder_count=17,
        _virtual_obstacles={"edges": SimpleNamespace(edges_pyt=torch.zeros(17, 6))},
    )
    assert registered_cylinder_count(sensor) == 17


def test_hand_computed_cylinder_offset_and_penalty() -> None:
    """Point at (0.04, 0, 0.5), vertical cylinder r=0.10 through the origin.

    Radial distance 0.04, depth 0.06. Offset is surface → point, so −x.
    Speed 1.0, one point: (1 + 1e-6) * 0.06.
    """
    offset = cylinder_penetration_offset((0.04, 0.0, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.10)
    assert offset == pytest.approx((-0.06, 0.0, 0.0))
    depth = (offset[0] ** 2 + offset[1] ** 2 + offset[2] ** 2) ** 0.5
    assert depth == pytest.approx(0.06)
    assert penetration_reward(depth, 1.0) == pytest.approx((1.0 + 1e-6) * 0.06)
    assert cylinder_penetration_offset((0.20, 0.0, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.10) == (0.0, 0.0, 0.0)


def test_on_axis_offset_has_length_equal_to_the_radius() -> None:
    """A point on the axis used to divide by zero and write NaN into the reward."""
    offset = cylinder_penetration_offset((0.0, 0.0, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.10)
    depth = (offset[0] ** 2 + offset[1] ** 2 + offset[2] ** 2) ** 0.5
    assert depth == pytest.approx(0.10)
    assert offset[2] == pytest.approx(0.0)
    assert not any(math.isnan(c) for c in offset)


def test_cvel_at_the_body_subtree_adds_the_pelvis_lever() -> None:
    """mjwarp cvel linear is at the free-joint COM, 0.65 m above the ankle.

    Transporting from the foot's own subtree (3 cm) leaves ω × (pelvis − ankle)
    in v_link. That is the silent miss: both engines report a speed, they just
    disagree by ~1.8 m/s whenever the ankle spins.
    """
    omega = (2.0, 2.0, 0.0)
    origin = (0.0, 0.0, 0.0)
    root_com = (0.0, 0.0, 0.65)
    body_com = (0.0312, 0.0, 0.0)
    # Still origin: cvel_lin at the root COM is ω × (root − origin).
    cvel_lin = (
        omega[1] * root_com[2] - omega[2] * root_com[1],
        omega[2] * root_com[0] - omega[0] * root_com[2],
        omega[0] * root_com[1] - omega[1] * root_com[0],
    )
    v_from_root = link_linear_velocity_from_com(
        cvel_lin, omega, (root_com[0] - origin[0], root_com[1] - origin[1], root_com[2] - origin[2])
    )
    v_from_body = link_linear_velocity_from_com(
        cvel_lin, omega, (body_com[0] - origin[0], body_com[1] - origin[1], body_com[2] - origin[2])
    )
    assert v_from_root == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert (v_from_body[0] ** 2 + v_from_body[1] ** 2 + v_from_body[2] ** 2) ** 0.5 == pytest.approx(1.840, abs=1e-3)


def test_mjlab_sensor_transports_cvel_from_the_tree_root() -> None:
    """Reverting to per-body subtree would pass every cheap test and fail live ω ≠ 0."""
    from pathlib import Path

    text = (
        Path(__file__).resolve().parents[1] / "source/instinctlab/instinctlab/engines/mjlab/volume_points.py"
    ).read_text()
    assert "_cvel_frame_ids" in text
    assert "body_rootid" in text
    assert "free-joint subtree COM" in text


def test_point_velocity_from_link_is_v_plus_omega_cross_r() -> None:
    """ω = (0, 0, 2), r = (0.05, 0, 0) → ω × r = (0, 0.10, 0). COM mixing adds extra."""
    vel = point_velocity_from_link((1.0, 0.0, 0.0), (0.0, 0.0, 2.0), (0.0, 0.0, 0.0), (0.05, 0.0, 0.0))
    assert vel == pytest.approx((1.0, 0.10, 0.0))
    # COM 3 cm forward of the origin: using v_com with a link lever arm is wrong.
    v_link = link_linear_velocity_from_com((0.0, 0.0, 0.0), (0.0, 0.0, 2.0), (0.03, 0.0, 0.0))
    assert v_link == pytest.approx((0.0, -0.06, 0.0))
    mixed = point_velocity_from_link((0.0, 0.0, 0.0), (0.0, 0.0, 2.0), (0.0, 0.0, 0.0), (0.05, 0.0, 0.0))
    assert mixed != pytest.approx(point_velocity_from_link(v_link, (0.0, 0.0, 2.0), (0.0, 0.0, 0.0), (0.05, 0.0, 0.0)))


def test_volume_points_penetration_reward_matches_the_hand_sum() -> None:
    sensor_ref = VolumePointsRef(name="leg_volume_points", attach=("left_ankle_roll_link",))
    # One env, one body, two points: one inside at depth 0.06 / speed 1, one miss.
    offset = torch.tensor([[[[-0.06, 0.0, 0.0], [0.0, 0.0, 0.0]]]])
    vel = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]])
    sensor = SimpleNamespace(
        cfg=SimpleNamespace(name="leg_volume_points", velocity="attach_link"),
        virtual_obstacles_registered=True,
        registered_cylinder_count=1,
        _virtual_obstacles={"known": SimpleNamespace(edges_pyt=torch.zeros(1, 6))},
        data=SimpleNamespace(penetration_offset=offset, points_vel_w=vel),
    )
    env = _Env(sensors={"leg_volume_points": sensor})
    reward = volume_points_penetration(env, sensor_ref)
    assert reward.shape == (1,)
    assert reward.item() == pytest.approx((1.0 + 1e-6) * 0.06)


def _isaac_style_process_edges(edge_coords: np.ndarray, *, min_points: int = 2) -> np.ndarray:
    """Isaac's Greedyconcat: split threshold hardcoded at 0.05, no collinear merge."""
    import random

    line_pts = edge_coords.reshape(-1, 3)
    vertices, inv_idx = np.unique(line_pts, axis=0, return_inverse=True)
    edge_pairs = inv_idx.reshape(-1, 2)
    adj_list = {i: set() for i in range(vertices.shape[0])}
    for start, end in edge_pairs:
        start, end = int(start), int(end)
        if start != end:
            adj_list[start].add(end)
            adj_list[end].add(start)
    num_edges_v = np.array([len(adj_list[i]) for i in range(vertices.shape[0])], dtype=int)
    available = set(np.where(num_edges_v > 0)[0])
    cos_threshold = np.cos(np.deg2rad(30.0))
    processed: list[np.ndarray] = []

    def max_dist(vertex_set: list[int]) -> tuple[int, float]:
        start_point = vertices[vertex_set[0]]
        end_point = vertices[vertex_set[-1]]
        line = end_point - start_point
        norm = np.linalg.norm(line)
        if norm == 0:
            return vertex_set[0], 0.0
        pts = vertices[vertex_set]
        dists = np.linalg.norm(np.cross(pts - start_point, pts - end_point), axis=1) / norm
        return vertex_set[int(np.argmax(dists))], float(dists.max())

    while available:
        selected = random.choice(list(available))
        vertex_set = [selected]
        if adj_list[selected]:
            neighbor = next(iter(adj_list[selected]))
            vertex_set.append(neighbor)
            adj_list[selected].remove(neighbor)
            adj_list[neighbor].remove(selected)
            for vid in (selected, neighbor):
                num_edges_v[vid] -= 1
                if num_edges_v[vid] == 0:
                    available.discard(vid)
        while True:
            found = False
            start, end = vertex_set[0], vertex_set[-1]
            neighbors = list(adj_list[start])
            if neighbors:
                dirs = vertices[start] - vertices[neighbors]
                dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                start_dir = vertices[vertex_set[1]] - vertices[start]
                start_dir /= np.linalg.norm(start_dir)
                idx = np.where((dirs @ start_dir) > cos_threshold)[0]
                if idx.size > 0:
                    neighbor = neighbors[int(idx[0])]
                    vertex_set.insert(0, neighbor)
                    adj_list[start].remove(neighbor)
                    adj_list[neighbor].remove(start)
                    for vid in (start, neighbor):
                        num_edges_v[vid] -= 1
                        if num_edges_v[vid] == 0:
                            available.discard(vid)
                    found = True
            neighbors = list(adj_list[end])
            if neighbors:
                dirs = vertices[neighbors] - vertices[end]
                dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                end_dir = vertices[end] - vertices[vertex_set[-2]]
                end_dir /= np.linalg.norm(end_dir)
                idx = np.where((dirs @ end_dir) > cos_threshold)[0]
                if idx.size > 0:
                    neighbor = neighbors[int(idx[0])]
                    vertex_set.append(neighbor)
                    adj_list[end].remove(neighbor)
                    adj_list[neighbor].remove(end)
                    for vid in (end, neighbor):
                        num_edges_v[vid] -= 1
                        if num_edges_v[vid] == 0:
                            available.discard(vid)
                    found = True
            if not found:
                break
        while len(vertex_set) >= min_points:
            for split_idx in range(len(vertex_set) - 1):
                _vid, dist = max_dist(vertex_set[split_idx:])
                if dist < 0.05:
                    break
            if len(vertex_set) - split_idx >= min_points:
                processed.append(np.concatenate([vertices[vertex_set[split_idx]], vertices[vertex_set[-1]]]))
            vertex_set = vertex_set[: split_idx + 1]
    if not processed:
        return np.empty((0, 6), dtype=np.float32)
    return np.asarray(processed, dtype=np.float32).reshape(-1, 6)


def test_collinear_gap_merge_is_why_the_cylinder_sets_cannot_match() -> None:
    """Same two collinear segments, 0.07 m gap.

    Isaac has no post-merge → 2 cylinders. InstinctMJ / mjlab parkour merges
    gaps up to 0.09 m → 1 cylinder. The penalty then fires on a different edge
    set even when both sides are non-zero.
    """
    import random

    from instinctlab.engines.mjlab.terrains.virtual_obstacle.edge_cylinder import GreedyconcatEdgeCylinder
    from instinctlab.engines.mjlab.terrains.virtual_obstacle.edge_cylinder_cfg import GreedyconcatEdgeCylinderCfg

    edge_coords = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.07, 0.0, 0.0, 2.07, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    random.seed(0)
    isaac = _isaac_style_process_edges(edge_coords, min_points=2)
    cfg = GreedyconcatEdgeCylinderCfg(
        cylinder_radius=0.05,
        min_points=2,
        component_workers=1,
        merge_collinear_gap=0.09,
        merge_collinear_angle_threshold=30.0,
        merge_collinear_line_distance=0.04,
    )
    random.seed(0)
    mjlab = GreedyconcatEdgeCylinder(cfg).process_edges(edge_coords)
    assert isaac.shape[0] == 2, isaac
    assert mjlab.shape[0] == 1, mjlab


def test_a_box_mesh_yields_the_same_raw_edge_count_before_merge() -> None:
    """On a closed box the 12 edges are already isolated; merge does nothing."""
    import random
    import trimesh

    from instinctlab.engines.mjlab.terrains.virtual_obstacle.edge_cylinder import GreedyconcatEdgeCylinder
    from instinctlab.engines.mjlab.terrains.virtual_obstacle.edge_cylinder_cfg import GreedyconcatEdgeCylinderCfg

    mesh = trimesh.creation.box(extents=(1.0, 1.0, 0.2))
    angles = mesh.face_adjacency_angles
    sharp = mesh.face_adjacency_edges[angles > np.deg2rad(70.0)]
    edge_coords = np.hstack([mesh.vertices[sharp[:, 0]], mesh.vertices[sharp[:, 1]]]).astype(np.float32)
    random.seed(0)
    isaac = _isaac_style_process_edges(edge_coords, min_points=2)
    cfg = GreedyconcatEdgeCylinderCfg(
        cylinder_radius=0.05,
        min_points=2,
        component_workers=1,
        merge_collinear_gap=0.09,
        merge_collinear_angle_threshold=30.0,
        merge_collinear_line_distance=0.04,
    )
    random.seed(0)
    mjlab = GreedyconcatEdgeCylinder(cfg).process_edges(edge_coords)
    assert isaac.shape[0] == mjlab.shape[0]
    assert isaac.shape[0] == 12


def test_rough_py_records_that_the_penetration_penalty_is_not_a_parity_signal() -> None:
    """Same place as the stairs 6-vs-5. A reader comparing training curves must see this."""
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "source/instinctlab/instinctlab/engines/mjlab/rough.py").read_text()
    assert "penetration penalty is therefore not comparable" in text
    assert "35k vs 43k" in text
    assert "208" in text and "518" in text


def test_warp_kernel_on_axis_is_finite_and_has_depth_radius() -> None:
    """The python helper is not enough — training reads the Warp launch."""
    if not torch.cuda.is_available():
        pytest.skip("Warp cylinder grid is built for GPU")
    import numpy as np
    import os

    from instinctlab.utils.warp.cylinder import CylinderSpatialGrid

    device = os.environ.get("INSTINCTLAB_LIVE_DEVICE", "cuda:2")
    grid = CylinderSpatialGrid(
        cylinders=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.10]], dtype=np.float32),
        num_grid_cells=8**3,
        device=device,
    )
    points = torch.tensor([[0.0, 0.0, 0.5], [0.04, 0.0, 0.5]], device=device)
    offset = grid.get_points_penetration_offset(points)
    assert torch.isfinite(offset).all(), offset
    on_axis = float(offset[0].norm())
    off_axis = float(offset[1].norm())
    assert on_axis == pytest.approx(0.10, abs=1e-4)
    assert off_axis == pytest.approx(0.06, abs=1e-4)
