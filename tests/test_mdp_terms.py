"""Guard: the portable terms are portable, and compute what the golden computes.

Two questions, checked separately because they fail differently.

**Does each term only read things that mean the same on both engines?** Answered statically, by
scanning every ``asset.data.<attr>`` access in ``instinctlab/mdp/`` and confronting it with the
denylist, the legacy alias tables, and the two engines' own data classes. This is the check that
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

import instinctlab.mdp as mdp
from instinctlab.compat.denylist import DENYLIST, LEGACY_COM_ALIASES, LEGACY_LINK_ALIASES
from instinctlab.compat.vocab import HUB
from instinctlab.spec.sensor import ContactSensorRef

_ENGINE_ROOTS = frozenset({"isaaclab", "isaacsim", "mjlab", "omni", "pxr", "carb", "mujoco", "warp"})

# Frame-free quantities the hub does not carry because there is nothing to disambiguate about them.
# Their presence on both engines is checked below rather than taken on trust.
_FRAME_FREE = frozenset({"default_joint_pos", "default_joint_vel", "soft_joint_pos_limits"})
_SENSOR_METADATA = frozenset({"validity"})


def _mdp_modules() -> list[pathlib.Path]:
    return sorted(pathlib.Path(mdp.__file__).parent.glob("*.py"))


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
    assert len(mdp.__all__) >= 15


def test_no_term_reads_a_denylisted_attribute():
    """The traps: same name on both engines, different meaning."""
    offenders = {attr: sorted(users) for attr, users in _data_attributes().items() if attr in DENYLIST}
    assert not offenders, f"portable terms read denylisted attributes: {offenders}"


def test_no_term_reads_an_isaac_legacy_alias():
    """``root_lin_vel_b`` reads like a link quantity, is a centre-of-mass one, and mjlab has none."""
    aliases = set(LEGACY_COM_ALIASES) | set(LEGACY_LINK_ALIASES)
    offenders = {attr: sorted(users) for attr, users in _data_attributes().items() if attr in aliases}
    assert not offenders, f"portable terms read Isaac Lab legacy aliases: {offenders}"


def test_every_attribute_read_is_either_in_the_hub_or_frame_free():
    unknown = {
        a: sorted(u)
        for a, u in _data_attributes().items()
        if a not in HUB and a not in _FRAME_FREE and a not in _SENSOR_METADATA
    }
    assert not unknown, (
        f"attributes outside the signed vocabulary: {unknown}. Add a hub entry naming the frame and "
        "origin, or record why the quantity has no frame to name."
    )


@pytest.mark.parametrize("attr", sorted(_FRAME_FREE))
def test_the_frame_free_attributes_exist_on_both_engines(attr: str):
    """They sit outside the hub, so nothing else would notice if an engine renamed one."""
    isaaclab = pytest.importorskip("isaaclab")
    source = pathlib.Path(isaaclab.__file__).parent / "assets/articulation/articulation_data.py"
    isaac_names = {
        node.name if isinstance(node, ast.FunctionDef) else node.target.id
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.FunctionDef) or (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name))
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
    source = pathlib.Path(isaaclab.__file__).parent / "assets/articulation/articulation_data.py"
    class_def = next(
        node
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.ClassDef) and node.name == "ArticulationData"
    )
    body = next(n for n in class_def.body if isinstance(n, ast.FunctionDef) and n.name == "root_link_vel_w")
    text = ast.unparse(body)
    assert "self.root_com_vel_w.clone()" in text
    assert "vel[:, :3] +=" in text
    assert "vel[:, 3:] +=" not in text and "vel[:, 3:6] +=" not in text


def test_linear_velocity_is_the_one_that_differs():
    """The counterpart: the correction that makes ``base_lin_vel`` a whitelist entry."""
    isaaclab = pytest.importorskip("isaaclab")
    source = pathlib.Path(isaaclab.__file__).parent / "assets/articulation/articulation_data.py"
    class_def = next(
        node
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.ClassDef) and node.name == "ArticulationData"
    )
    body = next(n for n in class_def.body if isinstance(n, ast.FunctionDef) and n.name == "root_link_vel_w")
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
    def __init__(self, *, entities=None, sensors=None, commands=None, action=None, prev_action=None, terminated=None):
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
        self.data = _Data(current_air_time=current_air_time, current_contact_time=current_contact_time)


def test_observations_read_the_link_quantities():
    robot = _Entity(
        root_link_ang_vel_b=torch.tensor([[1.0, 2.0, 3.0]]),
        root_link_lin_vel_b=torch.tensor([[4.0, 5.0, 6.0]]),
        projected_gravity_b=torch.tensor([[0.0, 0.1, -1.0]]),
    )
    env = _Env(entities={"robot": robot})
    assert torch.equal(mdp.base_ang_vel(env), robot.data.root_link_ang_vel_b)
    assert torch.equal(mdp.base_lin_vel(env), robot.data.root_link_lin_vel_b)
    assert torch.equal(mdp.projected_gravity(env), robot.data.projected_gravity_b)


def test_joint_observations_subtract_the_defaults_and_honour_the_selection():
    robot = _Entity(
        joint_pos=torch.tensor([[1.0, 2.0, 3.0]]),
        default_joint_pos=torch.tensor([[0.5, 0.5, 0.5]]),
        joint_vel=torch.tensor([[7.0, 8.0, 9.0]]),
        default_joint_vel=torch.zeros(1, 3),
    )
    env = _Env(entities={"robot": robot})
    assert torch.allclose(mdp.joint_pos_rel(env), torch.tensor([[0.5, 1.5, 2.5]]))
    assert torch.allclose(mdp.joint_pos_rel(env, _Cfg(joint_ids=[0, 2])), torch.tensor([[0.5, 2.5]]))
    assert torch.allclose(mdp.joint_vel(env, _Cfg(joint_ids=[1])), torch.tensor([[8.0]]))
    assert torch.allclose(mdp.joint_vel_rel(env), torch.tensor([[7.0, 8.0, 9.0]]))


def test_generated_commands_fails_loudly_when_the_command_is_absent():
    from instinctlab.compat.denylist import PortabilityError

    env = _Env(commands={"base_velocity": torch.ones(1, 3)})
    assert torch.equal(mdp.generated_commands(env, "base_velocity"), torch.ones(1, 3))
    with pytest.raises(PortabilityError):
        mdp.generated_commands(_Env(), "base_velocity")


def test_velocity_tracking_measures_in_the_yaw_frame():
    """A robot yawed 90 degrees, moving along world +x, is moving along its own -y."""
    half = torch.tensor(torch.pi / 4)
    robot = _Entity(
        root_link_quat_w=torch.tensor([[torch.cos(half), 0.0, 0.0, torch.sin(half)]]),
        root_link_lin_vel_w=torch.tensor([[1.0, 0.0, 0.0]]),
    )
    env = _Env(entities={"robot": robot}, commands={"base_velocity": torch.tensor([[0.0, -1.0, 0.0]])})
    assert mdp.track_lin_vel_xy_yaw_frame_exp(env, std=0.5, command_name="base_velocity").item() == pytest.approx(1.0)

    env = _Env(entities={"robot": robot}, commands={"base_velocity": torch.tensor([[1.0, 0.0, 0.0]])})
    reward = mdp.track_lin_vel_xy_yaw_frame_exp(env, std=0.5, command_name="base_velocity").item()
    assert reward == pytest.approx(torch.exp(torch.tensor(-2.0 / 0.25)).item())


def test_velocity_tracking_ignores_pitch_and_roll():
    """The ``yaw_quat`` is load-bearing: a robot leaning forward is still tracking its command.

    Without it the whole orientation would be projected out, and the tracking reward would fall off
    whenever the base pitched -- which for a walking humanoid is continuously.
    """
    pitch = torch.tensor(0.4)
    robot = _Entity(
        root_link_quat_w=torch.tensor([[torch.cos(pitch / 2), 0.0, torch.sin(pitch / 2), 0.0]]),
        root_link_lin_vel_w=torch.tensor([[1.0, 0.0, 0.0]]),
    )
    env = _Env(entities={"robot": robot}, commands={"base_velocity": torch.tensor([[1.0, 0.0, 0.0]])})
    assert mdp.track_lin_vel_xy_yaw_frame_exp(env, std=0.5, command_name="base_velocity").item() == pytest.approx(1.0)


def test_yaw_rate_tracking_is_a_plain_world_frame_error():
    robot = _Entity(root_link_ang_vel_w=torch.tensor([[0.0, 0.0, 0.3]]))
    env = _Env(entities={"robot": robot}, commands={"base_velocity": torch.tensor([[0.0, 0.0, 0.5]])})
    reward = mdp.track_ang_vel_z_world_exp(env, command_name="base_velocity", std=0.5).item()
    assert reward == pytest.approx(torch.exp(torch.tensor(-(0.2**2) / 0.25)).item())


def test_feet_air_time_rewards_single_stance_only():
    sensor_ref = ContactSensorRef(name="feet", elements=(".*foot",), track_air_time=True)
    sensor = _Sensor(
        body_names=["left_foot", "right_foot"],
        current_air_time=torch.tensor([[0.3, 0.0], [0.3, 0.4], [0.3, 0.0]]),
        current_contact_time=torch.tensor([[0.0, 0.2], [0.0, 0.0], [0.9, 0.8]]),
    )
    env = _Env(
        sensors={"feet": sensor},
        commands={"base_velocity": torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])},
    )
    reward = mdp.feet_air_time_positive_biped(env, "base_velocity", threshold=0.5, sensor=sensor_ref)
    # Row 0 is single stance: min(air 0.3, contact 0.2). Rows 1 and 2 are flight and double stance.
    assert reward.tolist() == pytest.approx([0.2, 0.0, 0.0])


def test_feet_air_time_pays_nothing_for_a_near_zero_command():
    sensor_ref = ContactSensorRef(name="feet", elements=(".*foot",), track_air_time=True)
    sensor = _Sensor(
        body_names=["left_foot", "right_foot"],
        current_air_time=torch.tensor([[0.3, 0.0]]),
        current_contact_time=torch.tensor([[0.0, 0.2]]),
    )
    env = _Env(sensors={"feet": sensor}, commands={"base_velocity": torch.tensor([[0.05, 0.0, 0.0]])})
    assert mdp.feet_air_time_positive_biped(env, "base_velocity", 0.5, sensor_ref).item() == 0.0


def test_feet_air_time_is_capped_at_the_threshold():
    sensor_ref = ContactSensorRef(name="feet", elements=(".*foot",), track_air_time=True)
    sensor = _Sensor(
        body_names=["left_foot", "right_foot"],
        current_air_time=torch.tensor([[9.0, 0.0]]),
        current_contact_time=torch.tensor([[0.0, 3.0]]),
    )
    env = _Env(sensors={"feet": sensor}, commands={"base_velocity": torch.tensor([[1.0, 0.0, 0.0]])})
    assert mdp.feet_air_time_positive_biped(env, "base_velocity", 0.5, sensor_ref).item() == pytest.approx(0.5)


def test_stand_still_only_bites_when_the_command_is_near_zero():
    robot = _Entity(joint_pos=torch.tensor([[1.0, 1.0]] * 2), default_joint_pos=torch.zeros(2, 2))
    env = _Env(entities={"robot": robot}, commands={"base_velocity": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])})
    assert mdp.stand_still(env, "base_velocity").tolist() == pytest.approx([2.0, 0.0])


def test_joint_limit_and_deviation_penalties():
    robot = _Entity(
        joint_pos=torch.tensor([[-1.5, 0.0, 2.5]]),
        default_joint_pos=torch.tensor([[0.0, 0.0, 0.0]]),
        soft_joint_pos_limits=torch.tensor([[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]]),
    )
    env = _Env(entities={"robot": robot})
    assert mdp.joint_pos_limits(env).item() == pytest.approx(0.5 + 1.5)
    assert mdp.joint_deviation_l1(env).item() == pytest.approx(4.0)
    assert mdp.joint_deviation_l1(env, _Cfg(joint_ids=[1])).item() == pytest.approx(0.0)


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
    assert mdp.lin_vel_z_l2(env).item() == pytest.approx(0.16)
    assert mdp.flat_orientation_l2(env).item() == pytest.approx(0.25)
    assert mdp.action_rate_l2(env).item() == pytest.approx(5.0)
    assert mdp.is_terminated(env).tolist() == [1.0, 0.0]
    assert mdp.last_action(env).tolist() == [[1.0, 2.0]]


@pytest.mark.parametrize("attr", ["raw_actions", "raw_action"])
def test_last_action_finds_the_raw_input_under_either_spelling(attr: str):
    """One character apart between the engines, which is why it is duck-typed rather than guessed."""
    term = type("Term", (), {attr: torch.tensor([[3.0, 4.0]])})()
    env = _Env(action=torch.zeros(1, 2), prev_action=torch.zeros(1, 2))
    env.action_manager.get_term = lambda name: term
    assert mdp.last_action(env, "joint_pos").tolist() == [[3.0, 4.0]]


def test_last_action_reports_an_unrecognised_action_term_rather_than_guessing():
    from instinctlab.compat.denylist import PortabilityError

    env = _Env(action=torch.zeros(1, 2), prev_action=torch.zeros(1, 2))
    env.action_manager.get_term = lambda name: object()
    with pytest.raises(PortabilityError, match="raw_actions nor raw_action"):
        mdp.last_action(env, "joint_pos")


def test_time_out_fires_at_the_limit_and_not_before():
    assert mdp.time_out(_Env()).tolist() == [False, True]


def test_illegal_contact_asks_the_sensor_rather_than_thresholding_a_force():
    """The signature has no ``threshold``, and that is the point; see the term's docstring."""
    sensor_ref = ContactSensorRef(name="body", elements=("torso",))
    sensor = _Sensor(body_names=["torso"], current_contact_time=torch.tensor([[0.0], [0.4]]))
    env = _Env(sensors={"body": sensor})
    assert mdp.illegal_contact(env, sensor_ref).tolist() == [False, True]
    assert "threshold" not in inspect.signature(mdp.illegal_contact).parameters


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

    robot = _Entity(root_link_pos_w=torch.tensor([[5.0, 0.0, 0.8], [0.2, 0.0, 0.8], [0.0, 0.0, 0.8]]))
    env = _Env(
        entities={"robot": robot},
        commands={"base_velocity": torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])},
    )
    env.scene.terrain = _Terrain()
    env.scene.env_origins = torch.zeros(3, 3)
    env.max_episode_length_s = 20.0

    mean = mdp.terrain_levels_vel(env, torch.tensor([0, 1, 2]), "base_velocity")
    assert mean.item() == pytest.approx(3.0)
    env_ids, move_up, move_down = env.scene.terrain.calls[0]
    assert move_up.tolist() == [True, False, False]
    assert move_down.tolist() == [False, True, False]


def test_terrain_levels_vel_refuses_a_plane():
    env = _Env()
    env.scene.terrain = type("Plane", (), {"cfg": type("Cfg", (), {"terrain_generator": None})()})()
    with pytest.raises(RuntimeError, match="no generator"):
        mdp.terrain_levels_vel(env, torch.tensor([0]), "base_velocity")


"""
The terms that are deliberately absent.
"""


@pytest.mark.parametrize(
    "name",
    ["joint_acc_l2", "dof_acc_l2", "joint_torques_l2", "dof_torques_l2", "contact_slide"],
)
def test_the_non_portable_rewards_are_not_offered_here(name: str):
    """Each reads a quantity the engines disagree about; they belong in per-engine registries."""
    assert not hasattr(mdp, name), f"{name} cannot be portable -- see the rewards module docstring"
