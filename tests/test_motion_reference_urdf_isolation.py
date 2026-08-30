"""URDF motion-reference FK must not import the unselected MuJoCo SDK."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_urdf_fk_is_sdk_isolated_and_preserves_joint_semantics(tmp_path: Path) -> None:
    urdf = tmp_path / "two_joint.urdf"
    urdf.write_text(
        """<robot name="two_joint">
  <link name="base"/>
  <link name="rotated"/>
  <link name="sliding"/>
  <joint name="turn" type="revolute">
    <parent link="base"/>
    <child link="rotated"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
  <joint name="slide" type="prismatic">
    <parent link="rotated"/>
    <child link="sliding"/>
    <origin xyz="0 1 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
  </joint>
</robot>
"""
    )
    code = f"""
import json
import math
import sys
import torch
from instinctlab_engine.motion_reference.clip import build_kinematics_chain
chain = build_kinematics_chain({str(urdf)!r})
poses = chain.forward_kinematics(torch.tensor([[math.pi / 2.0, 2.0]]))
print(json.dumps({{
    "joints": chain.get_joint_parameter_names(),
    "sliding_position": poses["sliding"].get_matrix()[0, :3, 3].tolist(),
    "mujoco_imported": "mujoco" in sys.modules,
    "pytorch_kinematics_imported": "pytorch_kinematics" in sys.modules,
}}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "source" / "instinctlab")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    report = json.loads(completed.stdout)

    assert report["joints"] == ["turn", "slide"]
    assert report["sliding_position"] == pytest.approx([0.0, 2.0, 0.0], abs=1e-6)
    assert report["mujoco_imported"] is False
    assert report["pytorch_kinematics_imported"] is False
