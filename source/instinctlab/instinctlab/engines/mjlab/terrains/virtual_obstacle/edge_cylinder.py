"""InstinctMJ's greedy-concat edge cylinders, minus visualisation and the other detectors.

``process_edges`` is the algorithm parkour actually runs. Warp is imported only when
the spatial grid is built so a CPU test can call ``process_edges`` without a GPU.
"""

from __future__ import annotations

import numpy as np
import os
import random
import torch
import trimesh
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

from .virtual_obstacle_base import VirtualObstacleBase

if TYPE_CHECKING:
    from mjlab.viewer.debug_visualizer import DebugVisualizer

    from .edge_cylinder_cfg import EdgeCylinderCfg, GreedyconcatEdgeCylinderCfg

_DEFAULT_DEBUG_CYLINDER_RGBA = (0.0, 0.0, 0.9, 0.2)


def _remaining_debug_geom_capacity(visualizer) -> int | None:
    scn = getattr(visualizer, "scn", None)
    if scn is None:
        return None
    geoms = getattr(scn, "geoms", None)
    ngeom = getattr(scn, "ngeom", None)
    if geoms is None or ngeom is None:
        return None
    return max(len(geoms) - int(ngeom), 0)


def _sample_debug_rows(rows: torch.Tensor, capacity: int | None) -> torch.Tensor:
    if rows.numel() == 0 or capacity is None:
        return rows
    if capacity <= 0:
        return rows[:0]
    count = int(rows.shape[0])
    if count <= capacity:
        return rows
    sample_ids = torch.linspace(0, count - 1, steps=capacity, device=rows.device)
    sample_ids = torch.round(sample_ids).to(torch.long)
    return rows.index_select(0, sample_ids)


def _greedyconcat_component_labels(num_vertices: int, edge_pairs: np.ndarray) -> np.ndarray:
    labels = np.full(num_vertices, -1, dtype=np.int32)
    if num_vertices == 0 or edge_pairs.size == 0:
        return labels

    parent = np.arange(num_vertices, dtype=np.int32)
    rank = np.zeros(num_vertices, dtype=np.int8)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for start, end in edge_pairs:
        start = int(start)
        end = int(end)
        if start == end:
            continue
        root_start = find(start)
        root_end = find(end)
        if root_start == root_end:
            continue
        if rank[root_start] < rank[root_end]:
            parent[root_start] = root_end
        elif rank[root_start] > rank[root_end]:
            parent[root_end] = root_start
        else:
            parent[root_end] = root_start
            rank[root_start] += 1

    root_to_label: dict[int, int] = {}
    next_label = 0
    active_vertices = np.unique(edge_pairs.reshape(-1))
    for vertex in active_vertices:
        vertex = int(vertex)
        root = find(vertex)
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        labels[vertex] = root_to_label[root]
    return labels


def _process_greedyconcat_component(
    vertices: np.ndarray,
    edge_pairs: np.ndarray,
    cos_threshold: float,
    point_distance_threshold: float,
    min_points: int,
    rng_seed: int | None = None,
) -> np.ndarray:
    if vertices.size == 0 or edge_pairs.size == 0:
        return np.empty((0, 6), dtype=np.float32)

    adj_list = {i: set() for i in range(vertices.shape[0])}
    for start, end in edge_pairs:
        start = int(start)
        end = int(end)
        if start == end:
            continue
        adj_list[start].add(end)
        adj_list[end].add(start)

    num_edges_v = np.array([len(adj_list[i]) for i in range(vertices.shape[0])], dtype=int)
    available_edges = set(np.where(num_edges_v > 0)[0])
    processed_edge_coords: list[np.ndarray] = []
    rng = random.Random(rng_seed) if rng_seed is not None else None

    def compute_max_distance_to_line_vec(points: np.ndarray, vertex_set: list[int]) -> tuple[int, float]:
        start_point = points[vertex_set[0]]
        end_point = points[vertex_set[-1]]
        line = end_point - start_point
        line_norm = np.linalg.norm(line)
        if line_norm == 0:
            return vertex_set[0], 0.0
        pts = points[vertex_set]
        dists = np.linalg.norm(np.cross(pts - start_point, pts - end_point), axis=1) / line_norm
        max_idx = int(np.argmax(dists))
        return vertex_set[max_idx], float(dists[max_idx])

    while available_edges:
        selectable_vertices = list(available_edges)
        selected_vertex = rng.choice(selectable_vertices) if rng is not None else random.choice(selectable_vertices)
        vertex_set = [selected_vertex]

        if adj_list[selected_vertex]:
            neighbor = next(iter(adj_list[selected_vertex]))
            vertex_set.append(neighbor)
            adj_list[selected_vertex].remove(neighbor)
            adj_list[neighbor].remove(selected_vertex)
            for vertex_id in [selected_vertex, neighbor]:
                num_edges_v[vertex_id] -= 1
                if num_edges_v[vertex_id] == 0:
                    available_edges.discard(vertex_id)

        while True:
            find_neighbor = False
            start_vertex, end_vertex = vertex_set[0], vertex_set[-1]

            neighbors = list(adj_list[start_vertex])
            if neighbors:
                dirs = vertices[start_vertex] - vertices[neighbors]
                dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                start_dir = vertices[vertex_set[1]] - vertices[start_vertex]
                start_dir /= np.linalg.norm(start_dir)
                dots = dirs @ start_dir
                idx = np.where(dots > cos_threshold)[0]
                if idx.size > 0:
                    neighbor = neighbors[int(idx[0])]
                    vertex_set.insert(0, neighbor)
                    adj_list[start_vertex].remove(neighbor)
                    adj_list[neighbor].remove(start_vertex)
                    for vertex_id in [start_vertex, neighbor]:
                        num_edges_v[vertex_id] -= 1
                        if num_edges_v[vertex_id] == 0:
                            available_edges.discard(vertex_id)
                    find_neighbor = True

            neighbors = list(adj_list[end_vertex])
            if neighbors:
                dirs = vertices[neighbors] - vertices[end_vertex]
                dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                end_dir = vertices[end_vertex] - vertices[vertex_set[-2]]
                end_dir /= np.linalg.norm(end_dir)
                dots = dirs @ end_dir
                idx = np.where(dots > cos_threshold)[0]
                if idx.size > 0:
                    neighbor = neighbors[int(idx[0])]
                    vertex_set.append(neighbor)
                    adj_list[end_vertex].remove(neighbor)
                    adj_list[neighbor].remove(end_vertex)
                    for vertex_id in [end_vertex, neighbor]:
                        num_edges_v[vertex_id] -= 1
                        if num_edges_v[vertex_id] == 0:
                            available_edges.discard(vertex_id)
                    find_neighbor = True

            if not find_neighbor:
                break

        while len(vertex_set) >= min_points:
            for split_idx in range(len(vertex_set) - 1):
                _max_vertex, max_dist = compute_max_distance_to_line_vec(vertices, vertex_set[split_idx:])
                if max_dist < point_distance_threshold:
                    break
            if len(vertex_set) - split_idx >= min_points:
                processed_edge_coords.append(
                    np.concatenate([vertices[vertex_set[split_idx]], vertices[vertex_set[-1]]])
                )
            vertex_set = vertex_set[: split_idx + 1]

    if len(processed_edge_coords) == 0:
        return np.empty((0, 6), dtype=np.float32)
    return np.asarray(processed_edge_coords, dtype=np.float32).reshape(-1, 6)


class EdgeCylinder(VirtualObstacleBase):
    def __init__(self, cfg: EdgeCylinderCfg):
        super().__init__(cfg)
        self.cfg: EdgeCylinderCfg = cfg
        self.angle_threshold = cfg.angle_threshold
        self.supports_edge_segment_generation = True
        self.edges_pyt = torch.empty(0, 6)
        self.cylinders = None
        self.device = torch.device("cpu")

    def _set_edge_cylinders(self, edge_end_points: np.ndarray, device="cpu") -> None:
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.edges_pyt = torch.tensor(edge_end_points, dtype=torch.float32, device=self.device)
        if edge_end_points.size > 0:
            from instinctlab.utils.warp.cylinder import CylinderSpatialGrid

            self.cylinders = CylinderSpatialGrid(
                cylinders=np.concatenate(
                    [edge_end_points, np.ones_like(edge_end_points[:, :1]) * self.cfg.cylinder_radius],
                    axis=1,
                ),
                num_grid_cells=self.cfg.num_grid_cells,
                device=self.device,
            )
        else:
            self.cylinders = None

    def generate_from_edge_segments(self, edge_segments: np.ndarray, device="cpu") -> None:
        if edge_segments.size == 0:
            edge_end_points = np.empty((0, 6), dtype=np.float32)
        else:
            edge_end_points = self.process_edges(edge_segments.astype(np.float32, copy=False))
            if edge_end_points.size == 0:
                edge_end_points = np.empty((0, 6), dtype=np.float32)
        self._set_edge_cylinders(edge_end_points, device=device)

    def generate(self, mesh: trimesh.Trimesh, device="cpu") -> None:
        angles = mesh.face_adjacency_angles
        threshold = np.deg2rad(self.angle_threshold)
        sharp_mask = angles > threshold
        if not np.any(sharp_mask):
            edge_end_points = np.empty((0, 6), dtype=np.float32)
            print("[WARNING] No sharp edges detected.")
        else:
            sharp_edges = mesh.face_adjacency_edges[sharp_mask]
            vertices = mesh.vertices
            edge_coords = np.hstack([vertices[sharp_edges[:, 0]], vertices[sharp_edges[:, 1]]])
            edge_end_points = self.process_edges(edge_coords)
            print(f"Detected {edge_end_points.shape[0]} edges after processing.")
        self._set_edge_cylinders(edge_end_points, device=device)

    def get_points_penetration_offset(self, points):
        return (
            self.cylinders.get_points_penetration_offset(points)
            if self.cylinders is not None
            else torch.zeros_like(points, device=self.device)
        )

    def debug_vis(self, visualizer: DebugVisualizer) -> None:
        if self.edges_pyt.numel() == 0:
            return
        radius = float(self.cfg.cylinder_radius)
        edge_rows = _sample_debug_rows(self.edges_pyt, _remaining_debug_geom_capacity(visualizer))
        edge_rows_np = edge_rows.cpu().numpy()
        for edge in edge_rows_np:
            visualizer.add_cylinder(
                start=edge[:3],
                end=edge[3:6],
                radius=radius,
                color=_DEFAULT_DEBUG_CYLINDER_RGBA,
            )

    def process_edges(self, edge_coords: np.ndarray) -> np.ndarray:
        return edge_coords


class GreedyconcatEdgeCylinder(EdgeCylinder):
    def __init__(self, cfg: GreedyconcatEdgeCylinderCfg):
        super().__init__(cfg)
        self.cfg: GreedyconcatEdgeCylinderCfg = cfg

    @staticmethod
    def _try_merge_collinear_pair(
        seg_a: np.ndarray,
        seg_b: np.ndarray,
        *,
        cos_threshold: float,
        gap_threshold: float,
        line_distance_threshold: float,
    ) -> np.ndarray | None:
        a0, a1 = seg_a[:3], seg_a[3:]
        b0, b1 = seg_b[:3], seg_b[3:]
        dir_a = a1 - a0
        dir_b = b1 - b0
        len_a = np.linalg.norm(dir_a)
        len_b = np.linalg.norm(dir_b)
        if len_a <= 1.0e-8 or len_b <= 1.0e-8:
            return None

        unit_a = dir_a / len_a
        unit_b = dir_b / len_b
        if abs(float(np.dot(unit_a, unit_b))) < cos_threshold:
            return None

        dist_b0 = np.linalg.norm(np.cross(b0 - a0, unit_a))
        dist_b1 = np.linalg.norm(np.cross(b1 - a0, unit_a))
        dist_a0 = np.linalg.norm(np.cross(a0 - b0, unit_b))
        dist_a1 = np.linalg.norm(np.cross(a1 - b0, unit_b))
        if max(dist_b0, dist_b1, dist_a0, dist_a1) > line_distance_threshold:
            return None

        a_min, a_max = 0.0, len_a
        b_t0 = float(np.dot(b0 - a0, unit_a))
        b_t1 = float(np.dot(b1 - a0, unit_a))
        b_min, b_max = min(b_t0, b_t1), max(b_t0, b_t1)
        projected_gap = max(b_min - a_max, a_min - b_max, 0.0)
        if projected_gap > gap_threshold:
            return None

        merged_min = min(a_min, b_min)
        merged_max = max(a_max, b_max)
        merged_start = a0 + merged_min * unit_a
        merged_end = a0 + merged_max * unit_a
        if np.linalg.norm(merged_end - merged_start) <= 1.0e-8:
            return None
        return np.concatenate([merged_start, merged_end]).astype(np.float32)

    def _post_merge_collinear_segments(self, segments: np.ndarray) -> np.ndarray:
        gap_threshold = float(self.cfg.merge_collinear_gap)
        if gap_threshold <= 0.0 or segments.shape[0] <= 1:
            return segments
        max_segments = int(self.cfg.merge_collinear_max_segments)
        if segments.shape[0] > max_segments:
            return segments

        angle_threshold = float(self.cfg.merge_collinear_angle_threshold)
        cos_threshold = np.cos(np.deg2rad(angle_threshold))
        line_distance_threshold = self.cfg.merge_collinear_line_distance
        if line_distance_threshold is None:
            line_distance_threshold = float(self.cfg.point_distance_threshold)
        else:
            line_distance_threshold = float(line_distance_threshold)

        max_passes = max(int(self.cfg.merge_collinear_max_passes), 1)
        active_segments = [seg.astype(np.float64, copy=False) for seg in segments]
        for _ in range(max_passes):
            used = [False] * len(active_segments)
            merged_segments: list[np.ndarray] = []
            changed = False
            for i, base_seg in enumerate(active_segments):
                if used[i]:
                    continue
                current_seg = base_seg.astype(np.float32, copy=False)
                used[i] = True
                keep_merging = True
                while keep_merging:
                    keep_merging = False
                    for j, candidate in enumerate(active_segments):
                        if used[j]:
                            continue
                        merged = self._try_merge_collinear_pair(
                            current_seg,
                            candidate.astype(np.float32, copy=False),
                            cos_threshold=cos_threshold,
                            gap_threshold=gap_threshold,
                            line_distance_threshold=line_distance_threshold,
                        )
                        if merged is None:
                            continue
                        current_seg = merged
                        used[j] = True
                        changed = True
                        keep_merging = True
                merged_segments.append(current_seg.astype(np.float64, copy=False))
            active_segments = merged_segments
            if not changed:
                break

        if len(active_segments) == 0:
            return np.empty((0, 6), dtype=np.float32)
        return np.asarray(active_segments, dtype=np.float32).reshape(-1, 6)

    def process_edges(self, edge_coords: np.ndarray) -> np.ndarray:
        line_pts = edge_coords.reshape(-1, 3)
        vertices, inv_idx = np.unique(line_pts, axis=0, return_inverse=True)
        edge_pairs = inv_idx.reshape(-1, 2)
        cos_threshold = np.cos(np.deg2rad(self.cfg.adjacent_angle_threshold))
        workers = int(self.cfg.component_workers)
        if workers == 0:
            workers = max(1, os.cpu_count() or 1)

        if workers <= 1 or edge_pairs.shape[0] < 2048:
            processed = _process_greedyconcat_component(
                vertices,
                edge_pairs,
                cos_threshold=cos_threshold,
                point_distance_threshold=float(self.cfg.point_distance_threshold),
                min_points=int(self.cfg.min_points),
                rng_seed=None,
            )
            processed_parts = [processed]
        else:
            component_labels = _greedyconcat_component_labels(vertices.shape[0], edge_pairs)
            active_labels = np.unique(component_labels[component_labels >= 0])
            if active_labels.size <= 1:
                processed = _process_greedyconcat_component(
                    vertices,
                    edge_pairs,
                    cos_threshold=cos_threshold,
                    point_distance_threshold=float(self.cfg.point_distance_threshold),
                    min_points=int(self.cfg.min_points),
                    rng_seed=None,
                )
                processed_parts = [processed]
            else:
                edge_component_labels = component_labels[edge_pairs[:, 0]]
                worker_count = min(workers, int(active_labels.size))
                futures = []
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    for label in active_labels:
                        component_edge_ids = np.where(edge_component_labels == label)[0]
                        if component_edge_ids.size == 0:
                            continue
                        component_edges = edge_pairs[component_edge_ids]
                        component_vertex_ids = np.flatnonzero(component_labels == label)
                        component_vertices = vertices[component_vertex_ids]
                        local_edges = np.searchsorted(component_vertex_ids, component_edges).astype(
                            np.int32, copy=False
                        )
                        futures.append(
                            executor.submit(
                                _process_greedyconcat_component,
                                component_vertices,
                                local_edges,
                                cos_threshold,
                                float(self.cfg.point_distance_threshold),
                                int(self.cfg.min_points),
                                random.getrandbits(32),
                            )
                        )
                processed_parts = [future.result() for future in futures]

        processed_parts = [part for part in processed_parts if part.size > 0]
        if len(processed_parts) == 0:
            return np.empty((0, 6), dtype=np.float32)
        processed = np.concatenate(processed_parts, axis=0).astype(np.float32, copy=False)
        return self._post_merge_collinear_segments(processed)
