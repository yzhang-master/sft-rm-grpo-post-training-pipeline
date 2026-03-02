import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelConfig:
    hidden_size: int = 32
    max_length: int = 64
    feature_size: int = 64


@dataclass(frozen=True, slots=True)
class SFTConfig:
    epochs: int = 30
    learning_rate: float = 0.02
    batch_size: int = 7


@dataclass(frozen=True, slots=True)
class RewardConfig:
    epochs: int = 50
    learning_rate: float = 0.02


@dataclass(frozen=True, slots=True)
class PRMConfig:
    epochs: int = 40
    learning_rate: float = 0.02
    monte_carlo_rollouts: int = 32


@dataclass(frozen=True, slots=True)
class GRPOConfig:
    steps: int = 2
    group_size: int = 4
    learning_rate: float = 0.001
    clip: float = 0.2
    kl_coefficient: float = 0.02
    max_length: int = 48
    rm_weight: float = 0.1
    prm_weight: float = 0.1


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    data_path: str
    output_dir: str
    seed: int
    model: ModelConfig
    sft: SFTConfig
    reward_model: RewardConfig
    prm: PRMConfig
    grpo: GRPOConfig
    base_dir: Path

    @property
    def resolved_data_path(self) -> Path:
        return (self.base_dir / self.data_path).resolve()

    @property
    def resolved_output_dir(self) -> Path:
        return (self.base_dir / self.output_dir).resolve()

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_path": self.data_path,
            "output_dir": self.output_dir,
            "seed": self.seed,
            "model": asdict(self.model),
            "sft": asdict(self.sft),
            "reward_model": asdict(self.reward_model),
            "prm": asdict(self.prm),
            "grpo": asdict(self.grpo),
        }

    @property
    def fingerprint(self) -> str:
        payload = canonical_json(self.to_dict()).encode()
        return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def validate_config(config: PipelineConfig) -> None:
    _positive("model.hidden_size", config.model.hidden_size)
    _positive("model.max_length", config.model.max_length)
    _positive("model.feature_size", config.model.feature_size)
    _positive("sft.epochs", config.sft.epochs)
    _positive("sft.learning_rate", config.sft.learning_rate)
    _positive("sft.batch_size", config.sft.batch_size)
    _positive("reward_model.epochs", config.reward_model.epochs)
    _positive("reward_model.learning_rate", config.reward_model.learning_rate)
    _positive("prm.epochs", config.prm.epochs)
    _positive("prm.learning_rate", config.prm.learning_rate)
    _positive("prm.monte_carlo_rollouts", config.prm.monte_carlo_rollouts)
    _positive("grpo.steps", config.grpo.steps)
    _positive("grpo.learning_rate", config.grpo.learning_rate)
    if config.grpo.group_size < 2:
        raise ValueError("grpo.group_size must be at least two")
    if config.grpo.max_length > config.model.max_length:
        raise ValueError("grpo.max_length cannot exceed model.max_length")
    if not 0 < config.grpo.clip < 1:
        raise ValueError("grpo.clip must be between zero and one")
    if config.grpo.kl_coefficient < 0:
        raise ValueError("grpo.kl_coefficient cannot be negative")
    if config.grpo.rm_weight < 0 or config.grpo.prm_weight < 0:
        raise ValueError("learned reward weights cannot be negative")


def load_config(path: str | Path) -> PipelineConfig:
    source = Path(path).resolve()
    with source.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    config = PipelineConfig(
        data_path=raw["data_path"],
        output_dir=raw["output_dir"],
        seed=int(raw.get("seed", 7)),
        model=ModelConfig(**raw.get("model", {})),
        sft=SFTConfig(**raw.get("sft", {})),
        reward_model=RewardConfig(**raw.get("reward_model", {})),
        prm=PRMConfig(**raw.get("prm", {})),
        grpo=GRPOConfig(**raw.get("grpo", {})),
        base_dir=source.parent,
    )
    validate_config(config)
    return config
