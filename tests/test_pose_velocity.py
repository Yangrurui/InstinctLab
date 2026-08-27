"""Guard: the pose-velocity command is one algorithm, keyed by sub-terrain name.

The two native parkour commands were copies of the same math on two ``CommandTerm`` bases, and
they already disagreed about which column a velocity box lands on. The mixin in
``engines/pose_velocity.py`` is the one copy; each engine's subclass only names columns. These
tests do not start a simulator.
"""

from __future__ import annotations

import ast
import importlib
import math
import pathlib
import sys
import torch
from types import SimpleNamespace

import pytest

from instinctlab.engines.pose_velocity import (
    PoseVelocityMixin,
    UnresolvableTerrainColumn,
    actual_column_count,
    command_params,
    curriculum_column_indices,
    even_column_assignment,
    resolve_named_columns,
    type_share_histogram,
)
from instinctlab.sim.robot_spec import BackendAsset, JointProperties, RobotSpec
from instinctlab.spec import (
    AgentSpec,
    CommandTermSpec,
    EventTermSpec,
    MdpSpec,
    RewardTermSpec,
    SceneSpec,
    SimSpec,
    TaskSpec,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
ENGINES = REPO / "source/instinctlab/instinctlab/engines"
_ENGINE_ROOTS = frozenset({"isaaclab", "isaacsim", "mjlab", "omni", "pxr", "carb", "mujoco", "warp", "usd"})
_SUBCLASS_METHODS = frozenset({"__init__", "_column_sub_terrain_names"})
_MIXIN_ARITHMETIC = frozenset(
    {
        "_bind_velocity_boxes",
        "_pose_velocity_setup",
        "_resample_command",
        "_update_command",
        "_update_metrics",
    }
)


def _robot() -> RobotSpec:
    return RobotSpec(
        name="tiny",
        schema_version="dfs_v1",
        asset_id="tiny_v1",
        root_body="root",
        joint_names=("hip", "knee"),
        body_names=("root", "foot"),
        joint_properties=(
            JointProperties("hip", 0.0, 2.0, 0.1, 0.0, 100.0, 10.0, 0.5),
            JointProperties("knee", 0.0, 2.0, 0.1, 0.0, 100.0, 10.0, 0.5),
        ),
        assets=(
            BackendAsset(backend="isaacsim", path="robot.urdf"),
            BackendAsset(backend="mjlab", path="robot.xml"),
        ),
        default_root_pos=(0.0, 0.0, 1.0),
        default_root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        soft_joint_pos_limit_factor=0.9,
    )


def _parkour_kinds_task() -> TaskSpec:
    """A spec that names every kind this slice registered, so ``contract_report`` has something to miss."""
    return TaskSpec(
        task_id="Instinct-Parkour-Target-G1",
        robot=_robot(),
        scene=SceneSpec(),
        sim=SimSpec(physics_dt=0.005, decimation=4, episode_length_s=20.0),
        mdp=MdpSpec(
            commands={
                "base_velocity": CommandTermSpec(
                    kind="pose_velocity",
                    params={
                        "resampling_time_range": (8.0, 12.0),
                        "lin_vel_x": (0.0, 0.0),
                        "lin_vel_y": (0.0, 0.0),
                        "ang_vel_z": (-1.0, 1.0),
                    },
                )
            },
            rewards={
                "rewards": {
                    "energy": RewardTermSpec(kind="motors_power_square", weight=-1e-5),
                    "torque_limits": RewardTermSpec(kind="applied_torque_limits_by_ratio", weight=-0.01),
                    "dof_torques": RewardTermSpec(kind="joint_torques_l2", weight=-1e-5),
                    "dof_acc": RewardTermSpec(kind="joint_acc_l2", weight=-1e-7),
                }
            },
            events={
                "reset_robot_joints": EventTermSpec(
                    kind="reset_joints_by_offset",
                    mode="reset",
                    params={"position_range": (-0.15, 0.15), "velocity_range": (0.0, 0.0)},
                )
            },
        ),
        agent=AgentSpec(runner="instinctlab.spec.task:AgentSpec"),
        engines=("isaacsim", "mjlab"),
    )


def _class_def(path: pathlib.Path, name: str) -> ast.ClassDef:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{path} has no class {name}")


def _methods(node: ast.ClassDef) -> set[str]:
    return {child.name for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))}


"""
Registration and isolation.
"""


def test_both_engines_advertise_the_new_kinds() -> None:
    from instinctlab.engines.isaacsim import TERMS as isaac_terms
    from instinctlab.engines.mjlab import TERMS as mjlab_terms

    for terms in (isaac_terms, mjlab_terms):
        assert "pose_velocity" in terms.kinds("command")
        assert "motors_power_square" in terms.kinds("reward")
        assert "applied_torque_limits_by_ratio" in terms.kinds("reward")
        assert "reset_joints_by_offset" in terms.kinds("event")
        assert "joint_torques_l2" in terms.kinds("reward")
        assert "joint_acc_l2" in terms.kinds("reward")
        assert "contact_slide" in terms.kinds("reward")


def test_contract_report_is_clean_for_the_new_kinds() -> None:
    from instinctlab.engines.isaacsim import IsaacSimAdapter
    from instinctlab.engines.mjlab import MjlabAdapter

    task = _parkour_kinds_task()
    for adapter in (IsaacSimAdapter(), MjlabAdapter()):
        report = adapter.contract_report(task)
        assert report["missing"] == {}, report["missing"]


def test_reset_joints_by_offset_provides_joint_state() -> None:
    from instinctlab.engines.isaacsim import TERMS as isaac_terms
    from instinctlab.engines.mjlab import TERMS as mjlab_terms
    from instinctlab.sim.capabilities import JOINT_STATE

    assert isaac_terms.provides()["event/reset_joints_by_offset"] == (JOINT_STATE,)
    assert mjlab_terms.provides()["event/reset_joints_by_offset"] == (JOINT_STATE,)
    assert "command/pose_velocity" not in isaac_terms.provides()
    assert "command/pose_velocity" not in mjlab_terms.provides()


def test_the_mixin_imports_with_engines_blocked(monkeypatch) -> None:
    for name in list(sys.modules):
        if name.startswith(("instinctlab.engines.pose_velocity", *_ENGINE_ROOTS)):
            monkeypatch.delitem(sys.modules, name, raising=False)

    class Blocker:
        def find_module(self, name, path=None):
            if name.split(".")[0] in _ENGINE_ROOTS:
                raise AssertionError(f"Importing the mixin pulled in {name!r}.")

    monkeypatch.setattr(sys, "meta_path", [Blocker(), *sys.meta_path])
    module = importlib.import_module("instinctlab.engines.pose_velocity")
    assert hasattr(module.PoseVelocityMixin, "_update_command")
    assert hasattr(module.PoseVelocityMixin, "_resample_command")
    assert hasattr(module.PoseVelocityMixin, "_update_metrics")


def test_the_mixin_holds_the_arithmetic_and_the_subclasses_do_not() -> None:
    mixin = _class_def(ENGINES / "pose_velocity.py", "PoseVelocityMixin")
    assert _MIXIN_ARITHMETIC <= _methods(mixin)
    for engine in ("isaacsim", "mjlab"):
        subclass = _class_def(ENGINES / engine / "pose_velocity.py", "PoseVelocityCommand")
        assert _methods(subclass) == _SUBCLASS_METHODS, f"{engine} subclass grew extra methods"


def test_per_engine_pose_velocity_modules_do_not_import_an_sdk_at_the_top() -> None:
    """``build_command`` may import the SDK; the module may not, or ``contract_report`` cannot answer."""
    for engine in ("isaacsim", "mjlab"):
        tree = ast.parse((ENGINES / engine / "pose_velocity.py").read_text())
        imported: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        leaked = imported & _ENGINE_ROOTS
        assert not leaked, f"{engine}/pose_velocity.py imports {sorted(leaked)} at module level"


"""
Name → column resolution.
"""


def test_resolve_named_columns_groups_by_name() -> None:
    mapping = resolve_named_columns(["flat", "flat", "stairs"], ["flat", "stairs"])
    assert mapping == {"flat": (0, 1), "stairs": (2,)}


def test_resolve_named_columns_raises_on_an_unknown_name() -> None:
    with pytest.raises(UnresolvableTerrainColumn, match="Unknown names: \\['dense_boxes'\\]") as caught:
        resolve_named_columns(["perlin_rough", "mesh_boxes"], ["dense_boxes"])
    assert "Columns: ['perlin_rough', 'mesh_boxes']" in str(caught.value)
    assert "Requested: ['dense_boxes']" in str(caught.value)


def test_resolve_named_columns_raises_on_an_unnamable_column() -> None:
    with pytest.raises(UnresolvableTerrainColumn, match="Unnamable columns: \\[1\\]") as caught:
        resolve_named_columns(["flat", None], ["flat"])
    assert "Columns: ['flat', None]" in str(caught.value)


def test_actual_column_count_reads_the_grid_not_num_cols() -> None:
    terrain = SimpleNamespace(
        flat_patches={"target": torch.zeros(10, 10, 50, 3)},
        cfg=SimpleNamespace(terrain_generator=SimpleNamespace(num_cols=20)),
    )
    assert actual_column_count(terrain) == 10


def test_both_engines_reexport_the_shared_column_namer() -> None:
    from instinctlab.engines.isaacsim.pose_velocity import column_sub_terrain_names as isaac_names
    from instinctlab.engines.mjlab.pose_velocity import column_sub_terrain_names as mjlab_names
    from instinctlab.engines.pose_velocity import column_sub_terrain_names as shared_names

    assert isaac_names is shared_names
    assert mjlab_names is shared_names


def test_isaac_columns_follow_proportion() -> None:
    from instinctlab.engines.isaacsim.pose_velocity import column_sub_terrain_names

    terrain = SimpleNamespace(
        flat_patches={"target": torch.zeros(2, 4, 1, 3)},
        cfg=SimpleNamespace(
            terrain_generator=SimpleNamespace(
                curriculum=True,
                num_cols=4,
                sub_terrains={
                    "a": SimpleNamespace(proportion=0.5),
                    "b": SimpleNamespace(proportion=0.5),
                },
            )
        ),
    )
    assert column_sub_terrain_names(terrain) == ["a", "a", "b", "b"]


def test_mjlab_curriculum_columns_follow_proportion() -> None:
    from instinctlab.engines.mjlab.pose_velocity import column_sub_terrain_names

    terrain = SimpleNamespace(
        flat_patches={"target": torch.zeros(2, 4, 1, 3)},
        cfg=SimpleNamespace(
            terrain_generator=SimpleNamespace(
                curriculum=True,
                num_cols=4,
                sub_terrains={
                    "a": SimpleNamespace(proportion=0.5),
                    "b": SimpleNamespace(proportion=0.5),
                },
            )
        ),
    )
    assert column_sub_terrain_names(terrain) == ["a", "a", "b", "b"]


def test_mjlab_parkour_curriculum_table_uses_shared_proportion_columns() -> None:
    from instinctlab.engines.mjlab.pose_velocity import column_sub_terrain_names
    from tests.parkour_live_expect import MJLAB_CURRICULUM_COLUMNS, PARKOUR_DECLARED_PROPORTIONS

    terrain = SimpleNamespace(
        flat_patches={"target": torch.zeros(10, 20, 50, 3)},
        cfg=SimpleNamespace(
            terrain_generator=SimpleNamespace(
                curriculum=True,
                num_cols=20,
                sub_terrains={name: SimpleNamespace(proportion=value) for name, value in PARKOUR_DECLARED_PROPORTIONS},
            )
        ),
    )
    assert column_sub_terrain_names(terrain) == list(MJLAB_CURRICULUM_COLUMNS)


def test_isaac_parkour_proportion_table_is_twenty_named_columns() -> None:
    from instinctlab.engines.isaacsim.pose_velocity import column_sub_terrain_names
    from tests.parkour_live_expect import ISAAC_NINTH_NAME, ISAAC_PROPORTION_COLUMNS, parkour_declared_shares

    proportions = parkour_declared_shares(ninth_name=ISAAC_NINTH_NAME)
    terrain = SimpleNamespace(
        flat_patches={"target": torch.zeros(10, 20, 50, 3)},
        cfg=SimpleNamespace(
            terrain_generator=SimpleNamespace(
                curriculum=True,
                num_cols=20,
                sub_terrains={name: SimpleNamespace(proportion=value) for name, value in proportions.items()},
            )
        ),
    )
    assert column_sub_terrain_names(terrain) == list(ISAAC_PROPORTION_COLUMNS)


def test_curriculum_namer_raises_when_built_width_disagrees_with_num_cols() -> None:
    from instinctlab.engines.pose_velocity import column_sub_terrain_names

    terrain = SimpleNamespace(
        flat_patches={"target": torch.zeros(10, 10, 50, 3)},
        cfg=SimpleNamespace(
            terrain_generator=SimpleNamespace(
                curriculum=True,
                num_cols=20,
                sub_terrains={"a": SimpleNamespace(proportion=1.0)},
            )
        ),
    )
    with pytest.raises(RuntimeError, match="10 columns but terrain_generator.num_cols=20"):
        column_sub_terrain_names(terrain)


def test_isaac_even_split_on_proportion_columns_reproduces_declared_shares() -> None:
    """512 envs, 20 columns: type histogram must match the declaration, not 10% each."""
    from tests.parkour_live_expect import (
        ISAAC_NINTH_NAME,
        ISAAC_PROPORTION_COLUMNS,
        assert_terrain_type_shares,
        parkour_declared_shares,
    )

    num_envs = 512
    types = even_column_assignment(num_envs, 20)
    declared = parkour_declared_shares(ninth_name=ISAAC_NINTH_NAME)
    assert_terrain_type_shares(types, ISAAC_PROPORTION_COLUMNS, declared, num_envs=num_envs)
    hist = type_share_histogram(types, ISAAC_PROPORTION_COLUMNS)
    assert hist["perlin_rough"] != pytest.approx(0.10, abs=0.01)
    assert hist["pyramid_stairs"] == pytest.approx(0.15, abs=3 / num_envs)


def test_uniform_type_histogram_is_rejected() -> None:
    from tests.parkour_live_expect import PARKOUR_DECLARED_PROPORTIONS, assert_terrain_type_shares

    names = [name for name, _ in PARKOUR_DECLARED_PROPORTIONS]
    # 10 types, 100 envs, 10 each — the silent fallback the change must not produce.
    types = torch.arange(100) % 10
    with pytest.raises(AssertionError, match="uniform at ~10%"):
        assert_terrain_type_shares(types, names, dict(PARKOUR_DECLARED_PROPORTIONS), num_envs=100)


def test_curriculum_plus_0_001_moves_a_column_off_a_near_boundary() -> None:
    """Isaac's ``+ 0.001`` is load-bearing, not decoration.

    First type 0.051, 20 columns: ``1/20 = 0.05`` stays in type 0 without the
    offset (``0.05 < 0.051``) and moves to type 1 with it (``0.051 < 0.051`` is
    false). Dropping the offset is a re-derivation and this test goes red.
    """
    proportions = (0.051, 0.949)
    ours = curriculum_column_indices(proportions, 20)
    naive = []
    cumulative = [0.051, 1.0]
    for index in range(20):
        matches = [i for i, value in enumerate(cumulative) if index / 20 < value]
        naive.append(matches[0])
    assert naive[1] == 0
    assert ours[1] == 1
    assert ours != naive


def test_mjlab_random_mode_cannot_name_a_column() -> None:
    from instinctlab.engines.mjlab.pose_velocity import column_sub_terrain_names

    terrain = SimpleNamespace(
        flat_patches={"target": torch.zeros(10, 4, 50, 3)},
        cfg=SimpleNamespace(
            terrain_generator=SimpleNamespace(
                curriculum=False,
                num_cols=4,
                sub_terrains={"a": SimpleNamespace(), "b": SimpleNamespace()},
            )
        ),
    )
    columns = column_sub_terrain_names(terrain)
    assert columns == [None, None, None, None]
    with pytest.raises(UnresolvableTerrainColumn, match="Unnamable columns"):
        resolve_named_columns(columns, ["a"])


"""
Command math.
"""


def _ranges(lin_x=(0.0, 1.0), lin_y=(0.0, 0.5), ang=(-1.0, 1.0)):
    return SimpleNamespace(lin_vel_x=lin_x, lin_vel_y=lin_y, ang_vel_z=ang)


class _StubCommand(PoseVelocityMixin):
    def __init__(self, *, n_env=1, columns=("flat",), types=(0,), cfg=None, patches=None):
        self.num_envs = n_env
        self.device = torch.device("cpu")
        self.metrics = {}
        self.cfg = cfg or SimpleNamespace(
            resampling_time_range=(8.0, 12.0),
            velocity_control_stiffness=2.0,
            heading_control_stiffness=2.0,
            only_positive_lin_vel_x=True,
            ranges=_ranges(),
            rel_standing_envs=0.0,
            random_velocity_terrain=None,
            velocity_ranges=None,
            lin_vel_threshold=0.15,
            ang_vel_threshold=0.15,
            lin_vel_metrics_std=0.5,
            ang_vel_metrics_std=0.5,
            target_dis_threshold=0.4,
        )
        self._columns = list(columns)
        n_cols = len(self._columns)
        self.terrain = SimpleNamespace(
            flat_patches={"target": patches if patches is not None else torch.zeros(2, n_cols, 1, 3)},
            terrain_types=torch.tensor(list(types), dtype=torch.long),
            terrain_levels=torch.zeros(n_env, dtype=torch.long),
        )
        self.robot = SimpleNamespace(
            data=SimpleNamespace(
                root_link_pos_w=torch.zeros(n_env, 3),
                root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n_env, 4).clone(),
                root_link_lin_vel_b=torch.zeros(n_env, 3),
                root_link_ang_vel_b=torch.zeros(n_env, 3),
            )
        )
        self._env = SimpleNamespace(step_dt=0.02, max_episode_length=500)
        self._pose_velocity_setup()

    def _column_sub_terrain_names(self):
        return self._columns


def test_debug_vis_draws_the_instinctmj_goal_cylinder() -> None:
    """Red cylinder at pos_command_w, same geom InstinctMJ queues for viser/native."""
    cmd = _StubCommand()
    cmd.robot.data.root_link_pos_w[:] = torch.tensor([[0.0, 0.0, 0.8]])
    cmd.pos_command_w[:] = torch.tensor([[1.5, -0.4, 0.8]])
    cmd.vel_command_b[:] = torch.tensor([[0.6, 0.0, 0.0]])
    drawn: list[tuple] = []

    class _Vis:
        def add_cylinder(self, *, start, end, radius, color, label):
            drawn.append(("cylinder", start.copy(), end.copy(), radius, color, label))

        def add_arrow(self, *, start, end, color, width, label):
            drawn.append(("arrow", start.copy(), end.copy(), color, width, label))

    cmd._debug_vis_impl(_Vis())
    cylinders = [item for item in drawn if item[0] == "cylinder"]
    assert len(cylinders) == 1
    _, start, end, radius, color, label = cylinders[0]
    assert label == "goal_0"
    assert color == (1.0, 0.0, 0.0, 0.6)
    assert radius == pytest.approx(0.4)
    assert start.tolist() == pytest.approx([1.5, -0.4, 0.8])
    assert end.tolist() == pytest.approx([1.5, -0.4, 0.9])
    arrows = [item for item in drawn if item[0] == "arrow"]
    assert [item[-1] for item in arrows] == ["cmd_vel_0", "actual_vel_0"]
    with pytest.raises(ValueError, match="debug visualization is not wired"):
        command_params(
            {
                "resampling_time_range": (8.0, 12.0),
                "lin_vel_x": (0.0, 1.0),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-1.0, 1.0),
                "debug_vis": True,
            }
        )
    with pytest.raises(ValueError, match="does not honor"):
        command_params(
            {
                "resampling_time_range": (8.0, 12.0),
                "lin_vel_x": (0.0, 1.0),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-1.0, 1.0),
                "flat_patch_visualizer_cfg": {},
            }
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"resampling_time_range": (0.0, 1.0)}, "must be positive"),
        ({"lin_vel_x": (1.0, 0.0)}, "finite and ordered"),
        ({"rel_standing_envs": 1.1}, "must be in \\[0, 1\\]"),
        ({"lin_vel_metrics_std": 0.0}, "at least"),
        ({"random_velocity_terrain": "flat"}, "must be a sequence"),
        ({"random_velocity_terrain": ["flat", "flat"]}, "duplicate"),
        ({"velocity_ranges": {"flat": {"lin_vel_x": (0.0, 1.0)}}}, "must contain exactly"),
    ],
)
def test_pose_velocity_rejects_invalid_numeric_and_terrain_ranges(override, message) -> None:
    params = {
        "resampling_time_range": (8.0, 12.0),
        "lin_vel_x": (0.0, 1.0),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-1.0, 1.0),
    }
    params.update(override)
    with pytest.raises(ValueError, match=message):
        command_params(params)


def test_body_frame_command_and_tracking_metrics() -> None:
    cmd = _StubCommand()
    cmd.pos_command_w[:] = torch.tensor([[1.0, 0.0, 0.8]])
    cmd.robot.data.root_link_pos_w[:] = torch.tensor([[0.0, 0.0, 0.8]])
    cmd.max_command_b[:] = torch.tensor([[0.8, 0.0, 1.0]])
    cmd._update_command()
    assert torch.equal(cmd.command, torch.tensor([[0.8, 0.0, 0.0]]))

    cmd.robot.data.root_link_lin_vel_b[:] = cmd.command
    cmd.robot.data.root_link_ang_vel_b[:] = torch.zeros(1, 3)
    cmd._update_metrics()
    assert cmd.metrics["tracking_exp_vel_xy"].item() == pytest.approx(1.0 / 500.0)
    assert cmd.metrics["tracking_exp_vel_yaw"].item() == pytest.approx(1.0 / 500.0)


def test_command_mix_ratios_separate_zero_commands_from_tracking_quality() -> None:
    """InstinctMJ's four debug ratios. Without them a command-distribution shift reads as a
    tracking regression: env 1 below scores a perfect tracking_exp on a command of zero."""
    cmd = _StubCommand(n_env=2, columns=("flat",), types=(0, 0))
    cmd.pos_command_w[:] = torch.tensor([[1.0, 0.0, 0.8], [0.1, 0.0, 0.8]])
    cmd.max_command_b[:] = torch.tensor([[0.8, 0.0, 1.0], [0.8, 0.0, 1.0]])
    cmd.is_standing_env[:] = torch.tensor([False, True])
    cmd.random_velocity_indices[:] = torch.tensor([False, True])
    cmd._update_command()
    assert torch.allclose(cmd.command, torch.tensor([[0.8, 0.0, 0.0], [0.0, 0.0, 0.0]]))

    cmd._update_metrics()
    step = 1.0 / 500.0
    assert cmd.metrics["command_nonzero_ratio"].tolist() == pytest.approx([step, 0.0])
    assert cmd.metrics["target_near_ratio"].tolist() == pytest.approx([0.0, step])
    assert cmd.metrics["standing_env_ratio"].tolist() == pytest.approx([0.0, step])
    assert cmd.metrics["random_velocity_env_ratio"].tolist() == pytest.approx([0.0, step])
    # env 1 stood still against a zero command and still scores a perfect tracking_exp.
    assert cmd.metrics["tracking_exp_vel_xy"][1].item() == pytest.approx(step)


def test_yaw_command_and_mismatched_tracking_score() -> None:
    cmd = _StubCommand()
    cmd.pos_command_w[:] = torch.tensor([[1.0, 0.0, 0.8]])
    cmd.robot.data.root_link_pos_w[:] = torch.tensor([[0.0, 0.0, 0.8]])
    cmd.max_command_b[:] = torch.tensor([[0.8, 0.0, 1.0]])
    cmd._update_command()
    cmd.robot.data.root_link_lin_vel_b[:] = torch.tensor([[0.3, 0.0, 0.0]])
    cmd._update_metrics()
    expected = math.exp(-0.25 / 0.25) / 500.0
    assert cmd.metrics["tracking_exp_vel_xy"].item() == pytest.approx(expected)
    assert cmd.metrics["tracking_exp_vel_yaw"].item() == pytest.approx(1.0 / 500.0)


def test_near_target_and_standing_envs_zero_the_command() -> None:
    cmd = _StubCommand()
    cmd.pos_command_w[:] = torch.tensor([[1.0, 0.0, 0.8]])
    cmd.robot.data.root_link_pos_w[:] = torch.tensor([[0.95, 0.0, 0.8]])
    cmd.max_command_b[:] = torch.tensor([[0.8, 0.0, 1.0]])
    cmd._update_command()
    assert torch.equal(cmd.command, torch.zeros(1, 3))

    cmd.robot.data.root_link_pos_w[:] = torch.tensor([[0.0, 0.0, 0.8]])
    cmd.is_standing_env[:] = True
    cmd._update_command()
    assert torch.equal(cmd.command, torch.zeros(1, 3))


def test_setup_records_the_column_name_table() -> None:
    cmd = _StubCommand(n_env=2, columns=("flat", "stairs"), types=(0, 1))
    assert cmd._column_names == ["flat", "stairs"]


def test_velocity_boxes_bind_by_sub_terrain_name() -> None:
    cfg = SimpleNamespace(
        resampling_time_range=(8.0, 12.0),
        velocity_control_stiffness=2.0,
        heading_control_stiffness=2.0,
        only_positive_lin_vel_x=True,
        ranges=_ranges(),
        rel_standing_envs=0.0,
        random_velocity_terrain=["stand"],
        velocity_ranges={
            "flat": {"lin_vel_x": (0.4, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "stand": {"lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
        },
        lin_vel_threshold=0.15,
        ang_vel_threshold=0.15,
        lin_vel_metrics_std=0.5,
        ang_vel_metrics_std=0.5,
        target_dis_threshold=0.4,
    )
    cmd = _StubCommand(n_env=2, columns=("flat", "stand"), types=(0, 1), cfg=cfg)
    assert torch.equal(cmd.lin_vel_x_range, torch.tensor([[0.4, 1.0], [0.0, 0.0]]))
    assert torch.equal(cmd.random_velocity_indices, torch.tensor([False, True]))


def test_velocity_boxes_raise_when_a_name_matches_no_column() -> None:
    cfg = SimpleNamespace(
        resampling_time_range=(8.0, 12.0),
        velocity_control_stiffness=1.0,
        heading_control_stiffness=1.0,
        only_positive_lin_vel_x=True,
        ranges=_ranges(),
        rel_standing_envs=0.0,
        random_velocity_terrain=None,
        velocity_ranges={"dense_boxes": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}},
        lin_vel_threshold=0.15,
        ang_vel_threshold=0.15,
        lin_vel_metrics_std=0.5,
        ang_vel_metrics_std=0.5,
        target_dis_threshold=0.4,
    )
    with pytest.raises(UnresolvableTerrainColumn, match="Unknown names: \\['dense_boxes'\\]"):
        _StubCommand(columns=("mesh_boxes",), cfg=cfg)
