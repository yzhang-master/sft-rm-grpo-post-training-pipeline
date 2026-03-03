import copy
from types import SimpleNamespace

import pytest
import torch

from post_training.grpo import LocalRolloutWorker, Rollout, group_advantages, grpo_loss
from post_training.policy import TinyPolicy, Tokenizer
from post_training.rewards import RewardBreakdown


def _rollout(policy, text, advantage):
    tokenizer = Tokenizer()
    ids = tokenizer.encode(text)
    old_logp, _, _ = policy.evaluate("question", ids, tokenizer)
    reward = RewardBreakdown(1.0, 0.0, 0.0, 1.0)
    task = SimpleNamespace(prompt="question")
    return Rollout(task, text, ids, old_logp.detach(), reward, advantage)


def test_group_relative_advantages_center_and_handle_zero_variance():
    rewards = torch.tensor([1.0, 3.0, 5.0, 7.0])
    expected = (rewards - 4) / (rewards.std(unbiased=False) + 1e-6)
    assert torch.allclose(group_advantages(rewards), expected)
    assert group_advantages(rewards).mean().abs() < 1e-6
    assert torch.equal(group_advantages(torch.ones(4)), torch.zeros(4))
    with pytest.raises(ValueError):
        group_advantages(torch.ones(1))


def test_clipped_grpo_has_real_gradient_and_exact_metrics():
    torch.manual_seed(8)
    policy = TinyPolicy(hidden_size=12)
    reference = copy.deepcopy(policy)
    rollouts = [_rollout(policy, "12", 1.0), _rollout(policy, "9", -1.0)]
    loss, metrics = grpo_loss(policy, reference, rollouts, beta=0.1)
    loss.backward()
    assert loss.isfinite()
    assert metrics["kl"] == 0.0
    assert metrics["entropy"] > 0
    assert metrics["length"] == 2.5
    assert metrics["advantage_sum"] == 0.0
    assert policy.head.weight.grad.abs().sum() > 0


def test_positive_advantage_ratio_is_clipped():
    torch.manual_seed(9)
    policy = TinyPolicy(hidden_size=12)
    reference = copy.deepcopy(policy)
    rollout = _rollout(policy, "12", 1.0)
    rollout.old_log_probs -= torch.log(torch.tensor(2.0))
    _, metrics = grpo_loss(policy, reference, [rollout], beta=0.0)
    assert metrics["objective"] == pytest.approx(-1.2)


def test_local_worker_generates_tokens_and_normalized_groups(dataset):
    torch.manual_seed(4)
    policy = TinyPolicy(hidden_size=12)
    reward = RewardBreakdown(0.0, 0.0, 0.0, 0.0)

    class FixedCalculator:
        def __call__(self, task, completion):
            return reward

    worker = LocalRolloutWorker(policy, FixedCalculator(), max_length=8)
    rollouts = worker.sample_groups(dataset.train[:1], group_size=4, seed=5)
    assert len(rollouts) == 4
    assert sum(item.advantage for item in rollouts) == pytest.approx(0.0, abs=1e-6)
    for rollout in rollouts:
        evaluated, _, _ = policy.evaluate(rollout.task.prompt, rollout.ids, Tokenizer())
        assert torch.allclose(rollout.old_log_probs, evaluated)
        assert len(evaluated) == len(rollout.ids) + 1
