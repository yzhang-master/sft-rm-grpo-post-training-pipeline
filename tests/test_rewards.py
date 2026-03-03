import pytest

from post_training.rewards import (
    RewardCalculator,
    math_reward,
    tool_reward,
    verifier_reward,
)


def test_math_verifier_uses_exact_final_rational_and_format_bonus():
    assert math_reward("work\n#### 1,024", "1024") == pytest.approx(1.2)
    assert math_reward("the answer is 0.5", "1/2") == 0.0
    assert math_reward("the answer is 0.5", "0.5") == 1.0
    assert math_reward("#### 12", "11") == 0.0
    assert math_reward("no number", "11") == 0.0


def test_tool_verifier_requires_exact_semantic_call():
    expected = '{"name":"weather","arguments":{"city":"paris","units":"c"}}'
    reordered = '{"arguments":{"units":"c","city":"paris"},"name":"weather"}'
    assert tool_reward(reordered, expected) == 1.0
    assert tool_reward('{"name":"weather","arguments":{"city":"london"}}', expected) == 0.0
    assert tool_reward("not json", expected) == 0.0
    assert tool_reward('[{"name":"weather"}]', expected) == 0.0


def test_task_dispatch_and_combined_reward(dataset):
    task = dataset.train[0]

    class FixedModel:
        def __init__(self, value):
            self.value = value

        def score(self, text):
            return self.value

    calculator = RewardCalculator(FixedModel(0.5), FixedModel(0.25), 0.2, 0.3)
    reward = calculator(task, task.response)
    assert reward.verifier == 1.2
    assert reward.total == pytest.approx(1.2 + 0.2 * 0.5 + 0.3 * 0.25)
    assert verifier_reward(dataset.train[8], dataset.train[8].response) == 1.0
