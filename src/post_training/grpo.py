import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .config import GRPOConfig
from .data import Task
from .policy import TinyPolicy, Tokenizer
from .rewards import RewardBreakdown, RewardCalculator, TinyRewardModel


@dataclass(slots=True)
class Rollout:
    task: Task
    completion: str
    ids: list[int]
    old_log_probs: torch.Tensor
    reward: RewardBreakdown
    advantage: float = 0.0


class LocalRolloutWorker:
    """Samples actual policy tokens and scores each decoded completion locally."""

    def __init__(self, policy: TinyPolicy, rewards: RewardCalculator, max_length: int) -> None:
        self.policy = policy
        self.rewards = rewards
        self.max_length = max_length
        self.tokenizer = Tokenizer()

    def sample(self, task: Task, generator: torch.Generator) -> Rollout:
        with torch.no_grad():
            ids, log_probs = self.policy.rollout(
                task.prompt, self.tokenizer, self.max_length, generator
            )
        completion = self.tokenizer.decode(ids)
        return Rollout(task, completion, ids, log_probs, self.rewards(task, completion))

    def sample_group(self, task: Task, group_size: int, seed: int) -> list[Rollout]:
        generator = torch.Generator().manual_seed(seed)
        group = [self.sample(task, generator) for _ in range(group_size)]
        advantages = group_advantages(torch.tensor([item.reward.total for item in group]))
        for item, advantage in zip(group, advantages, strict=True):
            item.advantage = float(advantage)
        return group

    def sample_groups(self, tasks: list[Task], group_size: int, seed: int) -> list[Rollout]:
        return [
            rollout
            for task_index, task in enumerate(tasks)
            for rollout in self.sample_group(task, group_size, seed + 1009 * task_index)
        ]


def group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if rewards.ndim != 1 or len(rewards) < 2:
        raise ValueError("GRPO requires a one-dimensional group of at least two rewards")
    centered = rewards - rewards.mean()
    scale = rewards.std(unbiased=False)
    if scale < eps:
        return torch.zeros_like(rewards)
    return centered / (scale + eps)


def grpo_loss(
    policy: TinyPolicy,
    reference: TinyPolicy,
    rollouts: list[Rollout],
    clip: float = 0.2,
    beta: float = 0.02,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not rollouts:
        raise ValueError("GRPO loss requires rollouts")
    tokenizer = Tokenizer()
    objectives, sampled_kls, entropies, lengths = [], [], [], []
    for rollout in rollouts:
        new_logp, entropy, _ = policy.evaluate(rollout.task.prompt, rollout.ids, tokenizer)
        with torch.no_grad():
            ref_logp, _, _ = reference.evaluate(rollout.task.prompt, rollout.ids, tokenizer)
        old_logp = rollout.old_log_probs
        if len(old_logp) != len(new_logp):
            raise ValueError("behavior and training token lengths differ")
        ratio = (new_logp - old_logp).exp()
        advantage = torch.tensor(rollout.advantage)
        unclipped = ratio * advantage
        clipped = ratio.clamp(1 - clip, 1 + clip) * advantage
        objectives.append(-torch.minimum(unclipped, clipped))
        delta = ref_logp - new_logp
        sampled_kls.append(delta.exp() - delta - 1)
        entropies.append(entropy)
        lengths.append(len(new_logp))
    token_count = sum(lengths)
    objective = torch.cat(objectives).sum() / token_count
    sampled_kl = torch.cat(sampled_kls).sum() / token_count
    loss = objective + beta * sampled_kl
    with torch.no_grad():
        exact_kl = torch.tensor(0.0)
        for rollout in rollouts:
            _, _, policy_logp = policy.evaluate(rollout.task.prompt, rollout.ids, tokenizer)
            _, _, reference_logp = reference.evaluate(rollout.task.prompt, rollout.ids, tokenizer)
            exact_kl += (policy_logp.exp() * (policy_logp - reference_logp)).sum()
    metrics = {
        "reward": sum(item.reward.total for item in rollouts) / len(rollouts),
        "verifier_reward": sum(item.reward.verifier for item in rollouts) / len(rollouts),
        "accuracy": sum(item.reward.verifier > 0 for item in rollouts) / len(rollouts),
        "kl": float(exact_kl / token_count),
        "entropy": float(torch.cat(entropies).detach().mean()),
        "length": sum(lengths) / len(lengths),
        "advantage_sum": sum(item.advantage for item in rollouts),
        "objective": float(objective.detach()),
    }
    return loss, metrics


class InterruptedTraining(RuntimeError):
    pass


def train_grpo(
    initial_policy: TinyPolicy,
    tasks: list[Task],
    reward_model: TinyRewardModel,
    prm: TinyRewardModel,
    config: GRPOConfig,
    seed: int,
    resume_payload: dict[str, Any] | None = None,
    on_checkpoint: Callable[[dict[str, Any], list[dict[str, float]]], None] | None = None,
    interrupt_after: int | None = None,
) -> tuple[TinyPolicy, list[dict[str, float]]]:
    torch.manual_seed(seed + 1000)
    policy = copy.deepcopy(initial_policy)
    reference = copy.deepcopy(initial_policy).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    history: list[dict[str, float]] = []
    start_step = 0
    if resume_payload is not None:
        policy.load_state_dict(resume_payload["policy_state"])
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        torch.set_rng_state(resume_payload["torch_rng_state"])
        history = list(resume_payload["history"])
        start_step = int(resume_payload["completed_steps"])
    rewards = RewardCalculator(reward_model, prm, config.rm_weight, config.prm_weight)
    worker = LocalRolloutWorker(policy, rewards, config.max_length)
    for step in range(start_step, config.steps):
        policy.eval()
        rollouts = worker.sample_groups(tasks, config.group_size, seed + step)
        policy.train()
        loss, metrics = grpo_loss(
            policy, reference, rollouts, clip=config.clip, beta=config.kl_coefficient
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()
        metrics.update({"step": float(step + 1), "loss": float(loss.detach())})
        history.append(metrics)
        payload = {
            "policy_state": policy.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "completed_steps": step + 1,
            "history": history,
        }
        if on_checkpoint is not None:
            on_checkpoint(payload, history)
        if interrupt_after == step + 1:
            raise InterruptedTraining(f"interrupted after GRPO step {step + 1}")
    return policy, history
