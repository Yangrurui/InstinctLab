"""Guard: the portable terms are portable, and compute what the golden computes.

Two questions, checked separately because they fail differently.

**Does each term only read things that mean the same on both engines?** Answered statically, by
scanning every ``asset.data.<attr>`` access in task-owned ``mdp/`` modules and confronting it with
an explicit task-layer attribute set. This is the check that
catches the dangerous class of mistake -- a term reading ``root_lin_vel_b``, which exists on Isaac
Lab, means the centre-of-mass velocity, and does not exist on mjlab at all.

**Does each term compute the right thing?** Answered against stubs with known values. Isaac Sim is
not installed here (it needs the ``omni`` stack), so a live two-engine comparison is not available
for the terms themselves; what is available, and is used, is the engines' own source, from which
the one claim these ports rest on -- that Isaac Lab's link and centre-of-mass *angular* velocities
are the same tensor -- is verified rather than assumed.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import torch

import pytest

from instinctlab.compat import robot as compat_robot
from instinctlab.compat.errors import PortabilityError
from instinctlab.spec.sensor import ContactSensorRef
from instinctlab.tasks.locomotion.mdp import (
    commands,
    curriculums,
    observations,
    rewards,
    terminations,
)

_ENGINE_ROOTS = frozenset(
    {"isaaclab", "isaacsim", "mjlab", "omni", "pxr", "carb", "mujoco", "warp"}
)

# Direct native data fields allowed in task-owned MDP modules. Quantity reads
# that differ by engine belong behind a small runtime interface in compat/.
_PORTABLE_DATA_ATTRIBUTES = frozenset(
    {
        "body_link_pos_w",
        "body_link_quat_w",
        "default_joint_pos",
        "default_joint_vel",
        "heading_w",
        "joint_pos",
        "joint_vel",
        "projected_gravity_b",
        "root_link_ang_vel_b",
        "root_link_ang_vel_w",
        "root_link_lin_vel_b",
        "root_link_lin_vel_w",
        "root_link_pos_w",
        "root_link_quat_w",
        "soft_joint_pos_limits",
        "timestamp",
        "validity",
    }
)
_NONPORTABLE_NATIVE_ATTRIBUTES = frozenset(
    {
        "applied_torque",
        "default_root_state",
        "gravity_vec_w",
        "joint_acc",
        "net_forces_w",
        "points_vel_w",
    }
)
_ISAAC_LEGACY_ALIASES = frozenset(
    {
        "body_ang_acc_w",
        "body_ang_vel_w",
        "body_lin_acc_w",
        "body_lin_vel_w",
        "body_pos_w",
        "body_quat_w",
        "body_vel_w",
        "com_pos_b",
        "com_quat_b",
        "root_ang_vel_b",
        "root_ang_vel_w",
        "root_lin_vel_b",
        "root_lin_vel_w",
        "root_pos_w",
        "root_pose_w",
        "root_quat_w",
        "root_vel_w",
    }
)


def _mdp_modules() -> list[pathlib.Path]:
    task_root = pathlib.Path("source/instinctlab/instinctlab/tasks")
    return sorted(task_root.glob("*/mdp/*.py"))


def _data_attributes() -> dict[str, set[str]]:
    """``attr`` -> the term functions that read ``<something>.data.<attr>``."""
    found: dict[str, set[str]] = {}
    for source in _mdp_modules():
        tree = ast.parse(source.read_text())
        for function in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            for node in ast.walk(function):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "data"
                ):
                    found.setdefault(node.attr, set()).add(function.name)
    return found


"""
What the terms are allowed to read.
"""


def test_the_terms_read_something_at_all():
    """Otherwise the checks below would pass by having nothing to check."""
    assert len(_data_attributes()) >= 5
    assert len(_mdp_modules()) >= 15


def test_no_term_reads_a_denylisted_attribute():
    """The traps: same name on both engines, different meaning."""
    offenders = {
        attr: sorted(users)
        for attr, users in _data_attributes().items()
        if attr in _NONPORTABLE_NATIVE_ATTRIBUTES
    }
    assert not offenders, f"portable terms read denylisted attributes: {offenders}"


def test_no_term_reads_an_isaac_legacy_alias():
    """``root_lin_vel_b`` reads like a link quantity, is a centre-of-mass one, and mjlab has none."""
    offenders = {
        attr: sorted(users)
        for attr, users in _data_attributes().items()
        if attr in _ISAAC_LEGACY_ALIASES
    }
    assert not offenders, f"portable terms read Isaac Lab legacy aliases: {offenders}"


def test_every_attribute_read_is_either_in_the_hub_or_frame_free():
    unknown = {
        a: sorted(u)
        for a, u in _data_attributes().items()
        if a not in _PORTABLE_DATA_ATTRIBUTES
    }
    assert not unknown, (
        f"attributes outside the task-layer interface: {unknown}. Add an explicit field with "
        "documented frame/origin semantics or route the native read through compat."
    )


@pytest.mark.parametrize(
    "attr", ("default_joint_pos", "default_joint_vel", "soft_joint_pos_limits")
)
def test_the_frame_free_attributes_exist_on_both_engines(attr: str):
    """They sit outside the hub, so nothing else would notice if an engine renamed one."""
    isaaclab = pytest.importorskip("isaaclab")
    source = (
        pathlib.Path(isaaclab.__file__).parent
        / "assets/articulation/articulation_data.py"
    )
    isaac_names = {
        node.name if isinstance(node, ast.FunctionDef) else node.target.id
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.FunctionDef)
        or (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name))
    }
    assert attr in isaac_names, f"Isaac Lab no longer exposes {attr}"

    entity_data = pytest.importorskip("mjlab.entity.data").EntityData
    mjlab_names: set[str] = set()
    for klass in getattr(entity_data, "__mro__", [entity_data]):
        mjlab_names |= set(getattr(klass, "__annotations__", {}))
        mjlab_names |= {k for k in vars(klass) if not k.startswith("_")}
    assert attr in mjlab_names, f"mjlab no longer exposes {attr}"


@pytest.mark.parametrize("source", _mdp_modules(), ids=lambda p: p.name)
def test_no_engine_is_imported_by_a_portable_term(source: pathlib.Path):
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    leaked = imported & _ENGINE_ROOTS
    assert not leaked, f"{source.name} imports {sorted(leaked)}"


"""
The claim the velocity ports rest on.
"""


def test_isaac_copies_the_angular_rows_between_its_link_and_com_velocity_buffers():
    """Why ``base_ang_vel`` may read the link spelling without differing from the golden.

    ``root_link_vel_w`` is built by cloning ``root_com_vel_w`` and correcting rows ``:3`` alone, so
    rows ``3:6`` are the same numbers. If Isaac Lab ever corrects the angular rows too, this test
    fails and three terms in this package need whitelist entries they do not currently need.
    """
    isaaclab = pytest.importorskip("isaaclab")
    source = (
        pathlib.Path(isaaclab.__file__).parent
        / "assets/articulation/articulation_data.py"
    )
    class_def = next(
        node
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.ClassDef) and node.name == "ArticulationData"
    )
    body = next(
        n
        for n in class_def.body
        if isinstance(n, ast.FunctionDef) and n.name == "root_link_vel_w"
    )
    text = ast.unparse(body)
    assert "self.root_com_vel_w.clone()" in text
    assert "vel[:, :3] +=" in text
    assert "vel[:, 3:] +=" not in text and "vel[:, 3:6] +=" not in text


def test_linear_velocity_is_the_one_that_differs():
    """The counterpart: the correction that makes ``base_lin_vel`` a whitelist entry."""
    isaaclab = pytest.importorskip("isaaclab")
    source = (
        pathlib.Path(isaaclab.__file__).parent
        / "assets/articulation/articulation_data.py"
    )
    class_def = next(
        node
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.ClassDef) and node.name == "ArticulationData"
    )
    body = next(
        n
        for n in class_def.body
        if isinstance(n, ast.FunctionDef) and n.name == "root_link_vel_w"
    )
    assert "cross" in ast.unparse(body) and "body_com_pos_b" in ast.unparse(body)


"""
What the terms compute, against stubs.
"""


class _Data:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Entity:
    def __init__(self, **kwargs):
        self.data = _Data(**kwargs)


class _Scene(dict):
    def __init__(self, entities, sensors=None):
        super().__init__(entities)
        self.sensors = sensors or {}


class _ActionManager:
    def __init__(self, action, prev_action):
        self.action, self.prev_action = action, prev_action


class _CommandManager:
    def __init__(self, commands):
        self.active_terms = list(commands)
        self._commands = commands

    def get_command(self, name):
        return self._commands.get(name)


class _Env:
    def __init__(
        self,
        *,
        entities=None,
        sensors=None,
        commands=None,
        action=None,
        prev_action=None,
        terminated=None,
    ):
        self.scene = _Scene(entities or {}, sensors)
        self.command_manager = _CommandManager(commands or {})
        self.action_manager = _ActionManager(action, prev_action)
        self.termination_manager = _Data(terminated=terminated)
        self.episode_length_buf = torch.tensor([5, 20])
        self.max_episode_length = 20


class _Cfg:
    def __init__(self, name="robot", joint_ids=slice(None), body_ids=slice(None)):
        self.name, self.joint_ids, self.body_ids = name, joint_ids, body_ids


class _Sensor:
    """Shaped after Isaac Lab's contact sensor, which is what ``compat.sensors`` duck-types on."""

    def __init__(self, body_names, current_air_time=None, current_contact_time=None):
        self.body_names = list(body_names)
        self.data = _Data(
            current_air_time=current_air_time, current_contact_time=current_contact_time
        )


def test_observations_prefer_direct_com_angular_velocity_and_read_link_linear_velocity():
    com_ang_vel = torch.tensor([[1.0, 2.0, 3.0]])
    robot = _Entity(
        root_com_ang_vel_b=com_ang_vel,
        root_link_ang_vel_b=torch.tensor([[10.0, 20.0, 30.0]]),
        root_link_lin_vel_b=torch.tensor([[4.0, 5.0, 6.0]]),
        projected_gravity_b=torch.tensor([[0.0, 0.1, -1.0]]),
    )
    env = _Env(entities={"robot": robot})
    assert torch.equal(observations.base_ang_vel(env), com_ang_vel)
    assert torch.equal(observations.base_lin_vel(env), robot.data.root_link_lin_vel_b)
    assert torch.equal(
        observations.projected_gravity(env), robot.data.projected_gravity_b
    )


def test_base_ang_vel_falls_back_to_the_link_quantity():
    link_ang_vel = torch.tensor([[1.0, 2.0, 3.0]])
    env = _Env(entities={"robot": _Entity(root_link_ang_vel_b=link_ang_vel)})
    assert torch.equal(observations.base_ang_vel(env), link_ang_vel)


def test_root_linear_velocity_requires_the_task_to_name_its_anchor():
    com = torch.tensor([[1.0, 2.0, 3.0]])
    link = torch.tensor([[4.0, 5.0, 6.0]])
    robot = _Entity(root_com_lin_vel_b=com, root_link_lin_vel_b=link)

    assert compat_robot.root_linear_velocity_b(robot, anchor="com") is com
    assert compat_robot.root_linear_velocity_b(robot, anchor="link") is link
    with pytest.raises(ValueError, match="must be 'com' or 'link'"):
        compat_robot.root_linear_velocity_b(robot, anchor="native")


def test_velocity_command_metric_uses_its_declared_anchor():
    robot = _Entity(
        root_com_lin_vel_b=torch.zeros(1, 3),
        root_link_lin_vel_b=torch.tensor([[3.0, 4.0, 0.0]]),
        root_link_ang_vel_b=torch.zeros(1, 3),
    )
    env = _Env(entities={"robot": robot})
    env.device = "cpu"
    env.num_envs = 1
    env.step_dt = 0.02
    params = {
        "entity": "robot",
        "heading_command": False,
        "init_velocity_prob": 0.0,
        "resampling_time_range": (10.0, 10.0),
        "metric_velocity_anchor": "com",
    }
    command = commands.UniformVelocityCommand(env, params)
    command.vel_command_b[:] = torch.tensor([[3.0, 4.0, 0.0]])
    command._update_metrics()
    torch.testing.assert_close(command.metrics["error_vel_xy"], torch.tensor([0.01]))

    params["metric_velocity_anchor"] = "link"
    command = commands.UniformVelocityCommand(env, params)
    command.vel_command_b[:] = torch.tensor([[3.0, 4.0, 0.0]])
    command._update_metrics()
    torch.testing.assert_close(command.metrics["error_vel_xy"], torch.zeros(1))


def test_joint_observations_subtract_the_defaults_and_honour_the_selection():
    robot = _Entity(
        joint_pos=torch.tensor([[1.0, 2.0, 3.0]]),
        default_joint_pos=torch.tensor([[0.5, 0.5, 0.5]]),
        joint_vel=torch.tensor([[7.0, 8.0, 9.0]]),
        default_joint_vel=torch.zeros(1, 3),
    )
    env = _Env(entities={"robot": robot})
    assert torch.allclose(
        observations.joint_pos_rel(env), torch.tensor([[0.5, 1.5, 2.5]])
    )
    assert torch.allclose(
        observations.joint_pos_rel(env, _Cfg(joint_ids=[0, 2])),
        torch.tensor([[0.5, 2.5]]),
    )
    assert torch.allclose(
        observations.joint_vel(env, _Cfg(joint_ids=[1])), torch.tensor([[8.0]])
    )


def test_generated_commands_fails_loudly_when_the_command_is_absent():
    env = _Env(commands={"base_velocity": torch.ones(1, 3)})
    assert torch.equal(
        observations.generated_commands(env, "base_velocity"), torch.ones(1, 3)
    )
    with pytest.raises(PortabilityError):
        observations.generated_commands(_Env(), "base_velocity")


def test_velocity_tracking_measures_in_the_yaw_frame():
    """A robot yawed 90 degrees, moving along world +x, is moving along its own -y."""
    half = torch.tensor(torch.pi / 4)
    robot = _Entity(
        root_link_quat_w=torch.tensor([[torch.cos(half), 0.0, 0.0, torch.sin(half)]]),
        root_link_lin_vel_w=torch.tensor([[1.0, 0.0, 0.0]]),
    )
    env = _Env(
        entities={"robot": robot},
        commands={"base_velocity": torch.tensor([[0.0, -1.0, 0.0]])},
    )
    assert rewards.track_lin_vel_xy_yaw_frame_exp(
        env, std=0.5, command_name="base_velocity"
    ).item() == pytest.approx(1.0)

    env = _Env(
        entities={"robot": robot},
        commands={"base_velocity": torch.tensor([[1.0, 0.0, 0.0]])},
    )
    reward = rewards.track_lin_vel_xy_yaw_frame_exp(
        env, std=0.5, command_name="base_velocity"
    ).item()
    assert reward == pytest.approx(torch.exp(torch.tensor(-2.0 / 0.25)).item())


def test_velocity_tracking_ignores_pitch_and_roll():
    """The ``yaw_quat`` is load-bearing: a robot leaning forward is still tracking its command.

    Without it the whole orientation would be projected out, and the tracking reward would fall off
    whenever the base pitched -- which for a walking humanoid is continuously.
    """
    pitch = torch.tensor(0.4)
    robot = _Entity(
        root_link_quat_w=torch.tensor(
            [[torch.cos(pitch / 2), 0.0, torch.sin(pitch / 2), 0.0]]
        ),
        root_link_lin_vel_w=torch.tensor([[1.0, 0.0, 0.0]]),
    )
    env = _Env(
        entities={"robot": robot},
        commands={"base_velocity": torch.tensor([[1.0, 0.0, 0.0]])},
    )
    assert rewards.track_lin_vel_xy_yaw_frame_exp(
        env, std=0.5, command_name="base_velocity"
    ).item() == pytest.approx(1.0)


def test_yaw_rate_tracking_is_a_plain_world_frame_error():
    robot = _Entity(root_link_ang_vel_w=torch.tensor([[0.0, 0.0, 0.3]]))
    env = _Env(
        entities={"robot": robot},
        commands={"base_velocity": torch.tensor([[0.0, 0.0, 0.5]])},
    )
    reward = rewards.track_ang_vel_z_world_exp(
        env, command_name="base_velocity", std=0.5
    ).item()
    assert reward == pytest.approx(torch.exp(torch.tensor(-(0.2**2) / 0.25)).item())


def test_feet_air_time_rewards_single_stance_only():
    sensor_ref = ContactSensorRef(
        name="feet", elements=(".*foot",), track_air_time=True
    )
    sensor = _Sensor(
        body_names=["left_foot", "right_foot"],
        current_air_time=torch.tensor([[0.3, 0.0], [0.3, 0.4], [0.3, 0.0]]),
        current_contact_time=torch.tensor([[0.0, 0.2], [0.0, 0.0], [0.9, 0.8]]),
    )
    env = _Env(
        sensors={"feet": sensor},
        commands={
            "base_velocity": torch.tensor(
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
            )
        },
    )
    reward = rewards.feet_air_time_positive_biped(
        env, "base_velocity", threshold=0.5, sensor=sensor_ref
    )
    # Row 0 is single stance: min(air 0.3, contact 0.2). Rows 1 and 2 are flight and double stance.
    assert reward.tolist() == pytest.approx([0.2, 0.0, 0.0])


def test_feet_air_time_pays_nothing_for_a_near_zero_command():
    sensor_ref = ContactSensorRef(
        name="feet", elements=(".*foot",), track_air_time=True
    )
    sensor = _Sensor(
        body_names=["left_foot", "right_foot"],
        current_air_time=torch.tensor([[0.3, 0.0]]),
        current_contact_time=torch.tensor([[0.0, 0.2]]),
    )
    env = _Env(
        sensors={"feet": sensor},
        commands={"base_velocity": torch.tensor([[0.05, 0.0, 0.0]])},
    )
    assert (
        rewards.feet_air_time_positive_biped(
            env, "base_velocity", 0.5, sensor_ref
        ).item()
        == 0.0
    )


def test_feet_air_time_is_capped_at_the_threshold():
    sensor_ref = ContactSensorRef(
        name="feet", elements=(".*foot",), track_air_time=True
    )
    sensor = _Sensor(
        body_names=["left_foot", "right_foot"],
        current_air_time=torch.tensor([[9.0, 0.0]]),
        current_contact_time=torch.tensor([[0.0, 3.0]]),
    )
    env = _Env(
        sensors={"feet": sensor},
        commands={"base_velocity": torch.tensor([[1.0, 0.0, 0.0]])},
    )
    assert rewards.feet_air_time_positive_biped(
        env, "base_velocity", 0.5, sensor_ref
    ).item() == pytest.approx(0.5)


def test_stand_still_only_bites_when_the_command_is_near_zero():
    robot = _Entity(
        joint_pos=torch.tensor([[1.0, 1.0]] * 2), default_joint_pos=torch.zeros(2, 2)
    )
    env = _Env(
        entities={"robot": robot},
        commands={"base_velocity": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])},
    )
    assert rewards.stand_still(env, "base_velocity").tolist() == pytest.approx(
        [2.0, 0.0]
    )


def test_joint_limit_and_deviation_penalties():
    robot = _Entity(
        joint_pos=torch.tensor([[-1.5, 0.0, 2.5]]),
        default_joint_pos=torch.tensor([[0.0, 0.0, 0.0]]),
        soft_joint_pos_limits=torch.tensor([[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]]),
    )
    env = _Env(entities={"robot": robot})
    assert rewards.joint_pos_limits(env).item() == pytest.approx(0.5 + 1.5)
    assert rewards.joint_deviation_l1(env).item() == pytest.approx(4.0)
    assert rewards.joint_deviation_l1(env, _Cfg(joint_ids=[1])).item() == pytest.approx(
        0.0
    )


def test_the_remaining_regularisers():
    robot = _Entity(
        root_link_lin_vel_b=torch.tensor([[0.0, 0.0, 0.4]]),
        projected_gravity_b=torch.tensor([[0.3, 0.4, -0.9]]),
    )
    env = _Env(
        entities={"robot": robot},
        action=torch.tensor([[1.0, 2.0]]),
        prev_action=torch.tensor([[0.0, 0.0]]),
        terminated=torch.tensor([True, False]),
    )
    assert rewards.lin_vel_z_l2(env).item() == pytest.approx(0.16)
    assert rewards.flat_orientation_l2(env).item() == pytest.approx(0.25)
    assert rewards.action_rate_l2(env).item() == pytest.approx(5.0)
    assert rewards.is_terminated(env).tolist() == [1.0, 0.0]
    assert observations.last_action(env).tolist() == [[1.0, 2.0]]


@pytest.mark.parametrize("attr", ["raw_actions", "raw_action"])
def test_last_action_finds_the_raw_input_under_either_spelling(attr: str):
    """One character apart between the engines, which is why it is duck-typed rather than guessed."""
    term = type("Term", (), {attr: torch.tensor([[3.0, 4.0]])})()
    env = _Env(action=torch.zeros(1, 2), prev_action=torch.zeros(1, 2))
    env.action_manager.get_term = lambda name: term
    assert observations.last_action(env, "joint_pos").tolist() == [[3.0, 4.0]]


def test_last_action_reports_an_unrecognised_action_term_rather_than_guessing():
    env = _Env(action=torch.zeros(1, 2), prev_action=torch.zeros(1, 2))
    env.action_manager.get_term = lambda name: object()
    with pytest.raises(PortabilityError, match="raw_actions nor raw_action"):
        observations.last_action(env, "joint_pos")


def test_time_out_fires_at_the_limit_and_not_before():
    assert terminations.time_out(_Env()).tolist() == [False, True]


def test_illegal_contact_asks_the_sensor_rather_than_thresholding_a_force():
    """The signature has no ``threshold``, and that is the point; see the term's docstring."""
    sensor_ref = ContactSensorRef(name="body", elements=("torso",))
    sensor = _Sensor(
        body_names=["torso"], current_contact_time=torch.tensor([[0.0], [0.4]])
    )
    env = _Env(sensors={"body": sensor})
    assert terminations.illegal_contact(env, sensor_ref).tolist() == [False, True]
    assert "threshold" not in inspect.signature(terminations.illegal_contact).parameters


def test_terrain_levels_vel_promotes_and_demotes():
    """Walked past half a tile: up. Walked less than half the commanded distance: down."""

    class _Generator:
        size = (8.0, 8.0)

    class _Terrain:
        def __init__(self):
            self.cfg = type("Cfg", (), {"terrain_generator": _Generator()})()
            self.terrain_levels = torch.tensor([3.0, 3.0, 3.0])
            self.calls: list = []

        def update_env_origins(self, env_ids, move_up, move_down):
            self.calls.append((env_ids.clone(), move_up.clone(), move_down.clone()))

    robot = _Entity(
        root_link_pos_w=torch.tensor(
            [[5.0, 0.0, 0.8], [0.2, 0.0, 0.8], [0.0, 0.0, 0.8]]
        )
    )
    env = _Env(
        entities={"robot": robot},
        commands={
            "base_velocity": torch.tensor(
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            )
        },
    )
    env.scene.terrain = _Terrain()
    env.scene.env_origins = torch.zeros(3, 3)
    env.max_episode_length_s = 20.0

    mean = curriculums.terrain_levels_vel(env, torch.tensor([0, 1, 2]), "base_velocity")
    assert mean.item() == pytest.approx(3.0)
    env_ids, move_up, move_down = env.scene.terrain.calls[0]
    assert move_up.tolist() == [True, False, False]
    assert move_down.tolist() == [False, True, False]


def test_terrain_levels_vel_refuses_a_plane():
    env = _Env()
    env.scene.terrain = type(
        "Plane", (), {"cfg": type("Cfg", (), {"terrain_generator": None})()}
    )()
    with pytest.raises(RuntimeError, match="no generator"):
        curriculums.terrain_levels_vel(env, torch.tensor([0]), "base_velocity")


@pytest.mark.parametrize(
    ("name", "compat_read"),
    [
        ("joint_acc_l2", "compat_robot.joint_acceleration"),
        ("joint_torques_l2", "compat_robot.joint_applied_torque"),
        ("contact_slide", "compat_robot.body_linear_velocity_w"),
    ],
)
def test_native_quantity_rewards_keep_the_formula_in_the_task(
    name: str, compat_read: str
):
    """Task policy owns the formula; compat selects only the native quantity."""
    source = inspect.getsource(getattr(rewards, name))
    assert compat_read in source


def test_obsolete_dof_reward_aliases_are_not_reintroduced():
    assert not hasattr(rewards, "dof_acc_l2")
    assert not hasattr(rewards, "dof_torques_l2")
