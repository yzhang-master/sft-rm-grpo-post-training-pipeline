from dataclasses import replace

import pytest

from post_training.config import validate_config


def test_dataset_has_disjoint_held_out_math_and_tool_tasks(dataset):
    assert len(dataset.train) == 14
    assert len(dataset.eval) == 6
    assert {task.kind for task in dataset.eval} == {"math", "tool"}
    assert {task.prompt for task in dataset.train}.isdisjoint(
        task.prompt for task in dataset.eval
    )


def test_config_rejects_invalid_grpo_and_reward_settings(small_config):
    with pytest.raises(ValueError, match="group_size"):
        validate_config(replace(small_config, grpo=replace(small_config.grpo, group_size=1)))
    with pytest.raises(ValueError, match="weights"):
        validate_config(replace(small_config, grpo=replace(small_config.grpo, rm_weight=-1)))
