"""Deterministic stateful fixture implementations with no simulator imports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StatefulActuatorCfgBase:
    joint_names_expr: tuple[str, ...] = ()
    joint_names: tuple[str, ...] = ()
    effort_limit: float = 3.0
    stiffness: float = 2.0
    damping: float = 0.1

    def __post_init__(self) -> None:
        self.joint_names_expr = tuple(self.joint_names_expr)
        self.joint_names = tuple(self.joint_names)

    def build(self, num_envs: int) -> "StatefulActuator":
        return StatefulActuator(
            num_envs=num_envs,
            stiffness=self.stiffness,
            effort_limit=self.effort_limit,
        )


class StatefulActuator:
    """One-step delayed proportional effort with partial reset."""

    def __init__(self, *, num_envs: int, stiffness: float, effort_limit: float):
        self.stiffness = float(stiffness)
        self.effort_limit = float(effort_limit)
        self._pending = [0.0] * num_envs
        self.applied_effort = [0.0] * num_envs

    def compute(self, commands) -> list[float]:
        if len(commands) != len(self._pending):
            raise ValueError("one command is required for every environment")
        self.applied_effort = list(self._pending)
        limit = self.effort_limit
        self._pending = [
            max(-limit, min(limit, self.stiffness * float(command)))
            for command in commands
        ]
        return list(self.applied_effort)

    def reset(self, env_ids) -> None:
        for env_id in env_ids:
            self._pending[env_id] = 0.0
            self.applied_effort[env_id] = 0.0


class RuntimeAdapter:
    def matches(self, actuator: object) -> bool:
        return isinstance(actuator, StatefulActuator)

    def stiffness_groups(self, actuator: StatefulActuator):
        return (actuator.stiffness,)

    def effort_limits(self, _env, _asset, actuator: StatefulActuator):
        return (actuator.effort_limit,)


class ImuRuntime:
    """Scalar stand-in that makes latency/history and partial reset observable."""

    def __init__(self, sensor, context) -> None:
        self.sensor = sensor
        self.context = context
        self._delay_steps = round(sensor.latency / sensor.update_period)
        self._queues = [[] for _ in range(context.num_envs)]
        self.timestamp = 0.0

    def tick(self, samples, timestamp: float) -> list[float]:
        if len(samples) != len(self._queues):
            raise ValueError("one sample is required for every environment")
        output = []
        for queue, sample in zip(self._queues, samples, strict=True):
            queue.append(float(sample))
            if len(queue) > self.sensor.history_length + self._delay_steps + 1:
                queue.pop(0)
            output.append(
                queue[-self._delay_steps - 1]
                if len(queue) > self._delay_steps
                else 0.0
            )
        self.timestamp = float(timestamp)
        return output

    def reset(self, env_ids) -> None:
        for env_id in env_ids:
            self._queues[env_id].clear()
