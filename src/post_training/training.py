import random

import torch

from .config import PRMConfig, RewardConfig, SFTConfig
from .data import Task
from .policy import TinyPolicy, Tokenizer
from .prm import build_step_examples, step_text
from .rewards import TinyRewardModel, hashed_features


def train_sft(
    model: TinyPolicy,
    tasks: list[Task],
    config: SFTConfig,
    seed: int,
) -> list[dict[str, float]]:
    if not tasks:
        raise ValueError("SFT requires at least one task")
    tokenizer = Tokenizer()
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history = []
    for epoch in range(config.epochs):
        indices = list(range(len(tasks)))
        rng.shuffle(indices)
        epoch_loss = 0.0
        updates = 0
        for start in range(0, len(indices), config.batch_size):
            batch = [tasks[index] for index in indices[start : start + config.batch_size]]
            losses = [model.response_loss(task.prompt, task.response, tokenizer) for task in batch]
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach())
            updates += 1
        history.append({"epoch": float(epoch + 1), "loss": epoch_loss / updates})
    return history


def train_reward_model(
    tasks: list[Task], config: RewardConfig, feature_size: int, hidden_size: int, seed: int
) -> tuple[TinyRewardModel, list[dict[str, float]]]:
    if not tasks:
        raise ValueError("reward training requires preference pairs")
    torch.manual_seed(seed)
    model = TinyRewardModel(feature_size, hidden_size)
    chosen = torch.stack(
        [hashed_features(f"{task.prompt}\n{task.response}", feature_size) for task in tasks]
    )
    rejected = torch.stack(
        [hashed_features(f"{task.prompt}\n{task.wrong_response}", feature_size) for task in tasks]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    for epoch in range(config.epochs):
        chosen_logits = model.network(chosen).squeeze(-1)
        rejected_logits = model.network(rejected).squeeze(-1)
        loss = -torch.nn.functional.logsigmoid(chosen_logits - rejected_logits).mean()
        accuracy = (chosen_logits > rejected_logits).float().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": float(loss.detach()),
                "pair_accuracy": float(accuracy.detach()),
            }
        )
    return model, history


def train_prm(
    tasks: list[Task], config: PRMConfig, feature_size: int, hidden_size: int, seed: int
) -> tuple[TinyRewardModel, list[dict[str, float]], int]:
    examples = build_step_examples(tasks, config.monte_carlo_rollouts, seed)
    if not examples:
        raise ValueError("PRM training requires math tasks")
    torch.manual_seed(seed + 1)
    model = TinyRewardModel(feature_size, hidden_size)
    features = torch.stack([hashed_features(step_text(item), feature_size) for item in examples])
    labels = torch.tensor([item.label for item in examples], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    for epoch in range(config.epochs):
        logits = model.network(features).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append({"epoch": float(epoch + 1), "loss": float(loss.detach())})
    return model, history, len(examples)
