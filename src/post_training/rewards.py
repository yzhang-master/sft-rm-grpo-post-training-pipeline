import json
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import torch
from torch import nn

from .data import Task

NUMBER = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)")
FEATURE_TOKEN = re.compile(r"[a-z]+|-?\d+|[()+*_{}\[\]\":=]")


def numeric_value(text: str) -> Fraction | None:
    match = NUMBER.search(text.replace(",", "").strip())
    if match is None:
        return None
    try:
        return Fraction(match.group())
    except (ValueError, ZeroDivisionError):
        return None


def math_reward(completion: str, expected: str) -> float:
    marked = completion.rsplit("####", 1)
    candidate = marked[1] if len(marked) == 2 else completion
    matches = NUMBER.findall(candidate.replace(",", ""))
    target = numeric_value(expected)
    if not matches or target is None:
        return 0.0
    try:
        predicted = Fraction(matches[-1])
    except (ValueError, ZeroDivisionError):
        return 0.0
    if predicted != target:
        return 0.0
    return 1.2 if len(marked) == 2 else 1.0


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def tool_reward(completion: str, expected: str) -> float:
    predicted = _json_object(completion)
    target = _json_object(expected)
    if predicted is None or target is None:
        return 0.0
    return float(predicted == target)


def verifier_reward(task: Task, completion: str) -> float:
    if task.kind == "math":
        return math_reward(completion, task.answer)
    if task.kind == "tool":
        return tool_reward(completion, task.answer)
    raise ValueError(f"unsupported task kind: {task.kind}")


def hashed_features(text: str, size: int = 64) -> torch.Tensor:
    features = torch.zeros(size)
    tokens = FEATURE_TOKEN.findall(text.lower())
    for position, token in enumerate(tokens):
        digest = sum((index + 1) * byte for index, byte in enumerate(token.encode()))
        features[(digest + 37 * position) % size] += 1.0
    return features / max(1, len(tokens))


class TinyRewardModel(nn.Module):
    def __init__(self, feature_size: int = 64, hidden_size: int = 32) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.network = nn.Sequential(
            nn.Linear(feature_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

    def forward_texts(self, texts: list[str]) -> torch.Tensor:
        features = torch.stack([hashed_features(text, self.feature_size) for text in texts])
        return self.network(features).squeeze(-1)

    def score(self, text: str) -> float:
        with torch.no_grad():
            return float(self.forward_texts([text]).sigmoid()[0])


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    verifier: float
    learned_rm: float
    learned_prm: float
    total: float


class RewardCalculator:
    def __init__(
        self,
        reward_model: TinyRewardModel,
        prm: TinyRewardModel,
        rm_weight: float,
        prm_weight: float,
    ) -> None:
        self.reward_model = reward_model
        self.prm = prm
        self.rm_weight = rm_weight
        self.prm_weight = prm_weight

    def __call__(self, task: Task, completion: str) -> RewardBreakdown:
        verified = verifier_reward(task, completion)
        outcome = self.reward_model.score(f"{task.prompt}\n{completion}")
        steps = [line for line in completion.splitlines() if not line.startswith("####")]
        if not steps:
            steps = [completion]
        process = sum(
            self.prm.score(f"{task.prompt}\n{step}") for step in steps
        ) / len(steps)
        total = verified + self.rm_weight * outcome + self.prm_weight * process
        return RewardBreakdown(verified, outcome, process, total)
