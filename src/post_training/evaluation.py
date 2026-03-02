from typing import Any

from .data import Task
from .policy import TinyPolicy, Tokenizer
from .rewards import verifier_reward


def evaluate_policy(
    policy: TinyPolicy, tasks: list[Task], max_length: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tokenizer = Tokenizer()
    predictions = []
    rewards = []
    lengths = []
    for task in tasks:
        completion = policy.greedy(task.prompt, tokenizer, max_length)
        reward = verifier_reward(task, completion)
        rewards.append(reward)
        lengths.append(len(tokenizer.encode(completion)))
        predictions.append(
            {
                "task_id": task.task_id,
                "kind": task.kind,
                "prompt": task.prompt,
                "completion": completion,
                "expected": task.answer,
                "verifier_reward": reward,
            }
        )
    by_kind = {}
    for kind in ("math", "tool"):
        selected = [
            reward
            for task, reward in zip(tasks, rewards, strict=True)
            if task.kind == kind
        ]
        by_kind[kind] = {
            "tasks": len(selected),
            "accuracy": sum(reward > 0 for reward in selected) / len(selected),
            "mean_reward": sum(selected) / len(selected),
        }
    metrics = {
        "tasks": len(tasks),
        "accuracy": sum(reward > 0 for reward in rewards) / len(rewards),
        "mean_reward": sum(rewards) / len(rewards),
        "mean_length": sum(lengths) / len(lengths),
        "by_kind": by_kind,
    }
    return metrics, predictions
