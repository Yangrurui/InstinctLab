"""Pin known BackendMetadata tokens. Do not put these in production cfg.

The two tests that used to sit here went with the unified stack: one read the CLI of the retired
``train_unified`` / ``play_unified`` / ``profile_backend`` scripts, and one checked that the unified
environment refuses a backend whose joint-acceleration source the task did not accept. That refusal
lived in the environment's constructor, so nothing is left to raise it -- the surviving half is the
vocabulary itself, pinned below.
"""

from __future__ import annotations

import ast
from pathlib import Path

from instinctlab.backends.mjlab.simulator import MjlabBackend
from instinctlab.backends.mock import MockSimulatorBackend
from instinctlab.sim.backend import JOINT_ACC_SOURCES
from instinctlab.verify.scene import locomotion_flat_scene

_ISAAC_BACKEND = Path(__file__).resolve().parents[1] / "source/instinctlab/instinctlab/backends/isaacsim/backend.py"


def _isaac_joint_acc_source() -> str:
    """Read the token out of Isaac's backend without importing it, which would need Isaac Sim.

    Parsed rather than searched for as text: a substring check also matches the token inside a
    comment, so commenting the declaration out left this passing while the backend declared nothing.
    """
    keywords = [
        keyword
        for node in ast.walk(ast.parse(_ISAAC_BACKEND.read_text()))
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "joint_acc_source"
    ]
    assert (
        len(keywords) == 1
    ), f"expected one joint_acc_source declaration in {_ISAAC_BACKEND.name}, found {len(keywords)}"
    return ast.literal_eval(keywords[0].value)


def test_known_backend_joint_acc_tokens() -> None:
    assert MockSimulatorBackend.metadata.joint_acc_source == "fd_v1"
    assert MjlabBackend.metadata.joint_acc_source == "qacc_v1"
    assert _isaac_joint_acc_source() == "isaaclab_lazy_fd_v1"
    assert JOINT_ACC_SOURCES == frozenset({"qacc_v1", "isaaclab_lazy_fd_v1", "fd_v1"})
    assert locomotion_flat_scene(num_envs=2).requirements.accepted_joint_acc_sources == JOINT_ACC_SOURCES
    assert "joint_acc_source" not in MjlabBackend.metadata.physics
