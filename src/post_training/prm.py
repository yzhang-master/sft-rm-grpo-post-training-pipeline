import random
import re
from dataclasses import dataclass

from .data import Task
from .rewards import math_reward


@dataclass(frozen=True, slots=True)
class StepExample:
    task: Task
    prefix: list[str]
    step: str
    label: float


def _arithmetic_values(task: Task) -> tuple[int, int, int]:
    if task.a is None or task.b is None or task.c is None:
        raise ValueError("arithmetic task requires a, b, and c")
    return task.a, task.b, task.c


def candidate_steps(task: Task, prefix: list[str]) -> list[str]:
    a, b, c = _arithmetic_values(task)
    if len(prefix) == 0:
        correct = a + b
        values = [correct, correct - 2, correct - 1, correct + 1, correct + 2]
        return [f"{a} + {b} = {value}" for value in values]
    if len(prefix) == 1:
        match = re.search(r"= (-?\d+)", prefix[0])
        if match is None:
            raise ValueError("invalid arithmetic prefix")
        previous = int(match.group(1))
        correct = previous * c
        values = [correct, correct - 2, correct - 1, correct + 1, correct + 2]
        return [f"{previous} * {c} = {value}" for value in values]
    if len(prefix) == 2:
        return []
    raise ValueError("arithmetic trajectories contain exactly two process steps")


def sample_completion(task: Task, prefix: list[str], rng: random.Random) -> str:
    steps = list(prefix)
    while len(steps) < 2:
        steps.append(rng.choice(candidate_steps(task, steps)))
    final = steps[-1].rsplit("=", 1)[-1].strip()
    return "\n".join([*steps, f"#### {final}"])


def monte_carlo_step_label(
    task: Task, prefix: list[str], step: str, rollouts: int, seed: int
) -> float:
    if rollouts < 1:
        raise ValueError("Monte Carlo labeling requires at least one rollout")
    if step not in candidate_steps(task, prefix):
        raise ValueError("step is not valid at this prefix")
    rng = random.Random(seed)
    extended = [*prefix, step]
    successes = 0
    for _ in range(rollouts):
        completion = sample_completion(task, extended, rng)
        successes += math_reward(completion, task.answer) > 0
    return successes / rollouts


def build_step_examples(tasks: list[Task], rollouts: int = 32, seed: int = 17) -> list[StepExample]:
    examples = []
    for task_index, task in enumerate(tasks):
        for first_index, first in enumerate(candidate_steps(task, [])):
            label = monte_carlo_step_label(
                task, [], first, rollouts, seed + task_index * 100 + first_index
            )
            examples.append(StepExample(task, [], first, label))
            for second_index, second in enumerate(candidate_steps(task, [first])):
                label = monte_carlo_step_label(
                    task,
                    [first],
                    second,
                    rollouts,
                    seed + task_index * 100 + first_index * 10 + second_index,
                )
                examples.append(StepExample(task, [first], second, label))
    return examples


def step_text(example: StepExample) -> str:
    return "\n".join([example.task.prompt, *example.prefix, example.step])
