"""Motion-matched MuJoCo terrain with the InstinctMJ CoACD collision contract."""

from __future__ import annotations

import hashlib
import numpy as np
import os
import trimesh
import uuid
import yaml

import mujoco
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput

_PARTS_CACHE: dict[tuple, list[tuple[np.ndarray, np.ndarray]]] = {}
_PREWARMED: set[tuple] = set()


def _crop(mesh: trimesh.Trimesh, size: tuple[float, float]) -> trimesh.Trimesh:
    for normal, origin in (
        ((-1, 0, 0), (size[0] / 2, 0, 0)),
        ((1, 0, 0), (-size[0] / 2, 0, 0)),
        ((0, -1, 0), (0, size[1] / 2, 0)),
        ((0, 1, 0), (0, -size[1] / 2, 0)),
    ):
        mesh = trimesh.intersections.slice_mesh_plane(mesh, plane_normal=normal, plane_origin=origin)
    return mesh


def _load_mesh(path: str, size: tuple[float, float], crop: bool, preserve_z: bool) -> tuple[trimesh.Trimesh, float]:
    mesh = trimesh.load(path, force="mesh")
    if crop:
        mesh = _crop(mesh, size)
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh, multibody=True)
    if preserve_z:
        border_height = 0.0
    else:
        border = mesh.vertices[
            np.logical_or(
                np.abs(mesh.vertices[:, 0]) > size[0] / 2 - 0.05,
                np.abs(mesh.vertices[:, 1]) > size[1] / 2 - 0.05,
            ),
            2,
        ]
        border_height = float(np.mean(border)) if border.size else float(np.percentile(mesh.vertices[:, 2], 10))
    transform = np.eye(4)
    transform[:2, 3] = np.asarray(size) / 2
    transform[2, 3] = -border_height
    mesh.apply_transform(transform)
    return mesh, border_height


def _key(cfg, source: str) -> tuple:
    stat = os.stat(source)
    return (
        3,
        os.path.abspath(source),
        stat.st_size,
        stat.st_mtime_ns,
        tuple(cfg.size),
        cfg.crop_to_size,
        cfg.use_input_origin_frame,
        cfg.collision_coacd_threshold,
        cfg.collision_coacd_resolution,
        cfg.collision_coacd_decimate,
        cfg.collision_coacd_max_ch_vertex,
    )


def _cache_path(cfg, source: str, key: tuple) -> str:
    digest = hashlib.sha1(repr(key).encode()).hexdigest()[:16]
    stem = os.path.splitext(os.path.basename(source))[0]
    return os.path.join(
        os.path.dirname(source),
        cfg.collision_coacd_cache_dirname,
        f"{stem}.{digest}.npz",
    )


def _save(path: str, parts: list[tuple[np.ndarray, np.ndarray]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: dict[str, np.ndarray] = {"num_parts": np.asarray([len(parts)], dtype=np.int32)}
    for index, (vertices, faces) in enumerate(parts):
        payload[f"verts_{index}"] = vertices.astype(np.float32)
        payload[f"faces_{index}"] = faces.astype(np.int32)
    temporary = f"{path}.{uuid.uuid4().hex}.tmp.npz"
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def _load(path: str) -> list[tuple[np.ndarray, np.ndarray]]:
    with np.load(path, allow_pickle=False) as cache:
        return [(cache[f"verts_{i}"], cache[f"faces_{i}"]) for i in range(int(cache["num_parts"][0]))]


def _decompose(cfg, source: str, mesh: trimesh.Trimesh) -> list[tuple[np.ndarray, np.ndarray]]:
    key = _key(cfg, source)
    if key in _PARTS_CACHE:
        return _PARTS_CACHE[key]
    cache_path = _cache_path(cfg, source, key)
    if cfg.collision_coacd_use_disk_cache and os.path.exists(cache_path):
        parts = _load(cache_path)
    else:
        try:
            import coacd
        except ImportError as exc:
            raise RuntimeError("MJLab perceptive shadowing requires coacd==1.0.7; install the 'mjlab' extra.") from exc
        coacd.set_log_level(cfg.collision_coacd_log_level)
        native = coacd.Mesh(
            vertices=np.asarray(mesh.vertices, dtype=np.float64),
            indices=np.asarray(mesh.faces, dtype=np.int32),
        )
        raw = coacd.run_coacd(
            native,
            threshold=cfg.collision_coacd_threshold,
            max_convex_hull=-1,
            preprocess_mode="auto",
            preprocess_resolution=50,
            resolution=cfg.collision_coacd_resolution,
            mcts_nodes=20,
            mcts_iterations=150,
            mcts_max_depth=3,
            pca=False,
            merge=False,
            decimate=cfg.collision_coacd_decimate,
            max_ch_vertex=cfg.collision_coacd_max_ch_vertex,
            extrude=False,
            extrude_margin=0.01,
            apx_mode="ch",
            seed=0,
        )
        parts = [(np.asarray(v, dtype=np.float32), np.asarray(f, dtype=np.int32)) for v, f in raw]
        if cfg.collision_coacd_use_disk_cache:
            _save(cache_path, parts)
    _PARTS_CACHE[key] = parts
    return parts


def _height_map(mesh: trimesh.Trimesh, resolution: float) -> dict[tuple[int, int], float]:
    face_mask = np.abs(mesh.face_normals[:, 2]) > 0.15
    samples = mesh.vertices[np.unique(mesh.faces[face_mask])] if face_mask.any() else mesh.vertices
    keys = np.round(samples[:, :2] / max(resolution * 0.5, 1e-6)).astype(np.int64)
    result: dict[tuple[int, int], float] = {}
    for key, z in zip(keys, samples[:, 2]):
        pair = (int(key[0]), int(key[1]))
        result[pair] = max(result.get(pair, -np.inf), float(z))
    return result


def _auto_align(mesh: trimesh.Trimesh, parts: list[tuple[np.ndarray, np.ndarray]], resolution: float) -> float:
    vertices, faces, offset = [], [], 0
    for part_vertices, part_faces in parts:
        vertices.append(part_vertices)
        faces.append(part_faces + offset)
        offset += len(part_vertices)
    hull = trimesh.Trimesh(np.concatenate(vertices), np.concatenate(faces), process=False)
    visual, collision = _height_map(mesh, resolution), _height_map(hull, resolution)
    common = visual.keys() & collision.keys()
    return float(np.median([visual[key] - collision[key] for key in common])) if common else 0.0


def motion_matched_terrain(cfg, difficulty: float, spec: mujoco.MjSpec, rng) -> TerrainOutput:
    del rng
    with open(cfg.metadata_yaml) as stream:
        terrains = yaml.safe_load(stream)["terrains"]
    prewarm_key = (
        os.path.abspath(cfg.path),
        tuple(cfg.size),
        cfg.collision_coacd_threshold,
    )
    if cfg.collision_coacd_prewarm_all and prewarm_key not in _PREWARMED:
        for entry in terrains:
            prewarm_source = os.path.join(cfg.path, entry["terrain_file"])
            prewarm_mesh, _ = _load_mesh(
                prewarm_source,
                (float(cfg.size[0]), float(cfg.size[1])),
                cfg.crop_to_size,
                cfg.use_input_origin_frame,
            )
            _decompose(cfg, prewarm_source, prewarm_mesh)
        _PREWARMED.add(prewarm_key)
    terrain_idx = int(np.clip(difficulty * len(terrains), 0, len(terrains) - 1))
    source = os.path.join(cfg.path, terrains[terrain_idx]["terrain_file"])
    size = (float(cfg.size[0]), float(cfg.size[1]))
    mesh, border_height = _load_mesh(source, size, cfg.crop_to_size, cfg.use_input_origin_frame)
    parts = _decompose(cfg, source, mesh)
    z_offset = cfg.collision_coacd_z_offset
    if cfg.collision_coacd_auto_align_top_surface:
        z_offset += _auto_align(mesh, parts, cfg.collision_coacd_auto_align_resolution)
    geometries = []
    for part_idx, (vertices, faces) in enumerate(parts):
        name = f"motion_matched_coacd_t{terrain_idx}_h{part_idx}_{uuid.uuid4().hex}"
        native = spec.add_mesh(
            name=name,
            uservert=vertices.reshape(-1).tolist(),
            userface=faces.reshape(-1).tolist(),
        )
        geom = spec.body("terrain").add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=native.name,
            pos=(0.0, 0.0, z_offset),
        )
        geom.group = 2 if cfg.collision_coacd_visualize_collision_hulls else 3
        geom.margin = cfg.collision_coacd_geom_margin
        geom.gap = 0.0
        geometries.append(TerrainGeometry(geom=geom))
    return TerrainOutput(
        origin=np.array([size[0] / 2, size[1] / 2, -border_height]),
        geometries=geometries,
    )


__all__ = ["motion_matched_terrain"]
