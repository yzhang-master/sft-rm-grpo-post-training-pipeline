import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_training.config import (  # noqa: E402
    GRPOConfig,
    ModelConfig,
    PRMConfig,
    RewardConfig,
    SFTConfig,
    load_config,
)
from post_training.data import load_tasks  # noqa: E402


@pytest.fixture(scope="session")
def dataset():
    return load_tasks(ROOT / "data" / "tasks.json")


@pytest.fixture
def small_config(tmp_path):
    base = load_config(ROOT / "configs" / "tiny_cpu.json")
    return replace(
        base,
        data_path=str(ROOT / "data" / "tasks.json"),
        output_dir=str(tmp_path / "artifacts"),
        model=ModelConfig(hidden_size=12, max_length=64, feature_size=32),
        sft=SFTConfig(epochs=2, learning_rate=0.01, batch_size=7),
        reward_model=RewardConfig(epochs=3, learning_rate=0.02),
        prm=PRMConfig(epochs=2, learning_rate=0.02, monte_carlo_rollouts=4),
        grpo=GRPOConfig(
            steps=2,
            group_size=2,
            learning_rate=0.001,
            max_length=12,
            rm_weight=0.1,
            prm_weight=0.1,
        ),
    )
