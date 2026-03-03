import torch

from post_training.config import RewardConfig, SFTConfig
from post_training.policy import TinyPolicy
from post_training.training import train_reward_model, train_sft


def test_sft_updates_real_recurrent_policy_and_reduces_loss(dataset):
    torch.manual_seed(3)
    model = TinyPolicy(hidden_size=16)
    history = train_sft(
        model,
        dataset.train[:4],
        SFTConfig(epochs=6, learning_rate=0.02, batch_size=2),
        seed=3,
    )
    assert len(history) == 6
    assert history[-1]["loss"] < history[0]["loss"]


def test_pairwise_reward_training_learns_local_preferences(dataset):
    model, history = train_reward_model(
        dataset.train,
        RewardConfig(epochs=30, learning_rate=0.03),
        feature_size=48,
        hidden_size=16,
        seed=4,
    )
    assert history[-1]["loss"] < history[0]["loss"]
    assert history[-1]["pair_accuracy"] >= 0.9
    chosen = model.score(f"{dataset.train[0].prompt}\n{dataset.train[0].response}")
    rejected = model.score(f"{dataset.train[0].prompt}\n{dataset.train[0].wrong_response}")
    assert chosen > rejected
