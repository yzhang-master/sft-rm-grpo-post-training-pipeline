import copy
from pathlib import Path
from typing import Any

import torch

from .artifacts import (
    load_checkpoint,
    manifest_hash,
    save_checkpoint,
    validate_manifest,
    write_json,
    write_manifest,
)
from .config import PipelineConfig
from .data import TaskDataset, load_tasks
from .evaluation import evaluate_policy
from .grpo import train_grpo
from .policy import TinyPolicy
from .rewards import TinyRewardModel
from .training import train_prm, train_reward_model, train_sft


def _dependency_hashes(root: Path, stages: list[str]) -> dict[str, str]:
    return {stage: manifest_hash(root / stage) for stage in stages}


def _artifact_status(
    root: Path,
    stage: str,
    config: PipelineConfig,
    data: TaskDataset,
    dependencies: dict[str, str],
    allow_running: bool = False,
) -> str | None:
    if not (root / stage / "manifest.json").exists():
        return None
    manifest = validate_manifest(
        root / stage,
        stage,
        config.fingerprint,
        data.source_hash,
        dependencies,
        allow_running=allow_running,
    )
    return manifest["status"]


def _new_policy(config: PipelineConfig) -> TinyPolicy:
    return TinyPolicy(config.model.hidden_size, config.model.max_length)


def _load_policy(path: Path, config: PipelineConfig) -> TinyPolicy:
    model = _new_policy(config)
    model.load_state_dict(load_checkpoint(path)["model_state"])
    return model


def _load_reward(path: Path, config: PipelineConfig) -> TinyRewardModel:
    model = TinyRewardModel(config.model.feature_size, config.model.hidden_size)
    model.load_state_dict(load_checkpoint(path)["model_state"])
    return model


def _finish_stage(
    directory: Path,
    stage: str,
    config: PipelineConfig,
    data: TaskDataset,
    dependencies: dict[str, str],
    files: list[str],
    metadata: dict[str, Any],
) -> None:
    write_manifest(
        directory,
        stage,
        "complete",
        config.fingerprint,
        data.source_hash,
        dependencies,
        files,
        metadata,
    )


def run_pipeline(
    config: PipelineConfig, resume: bool = True, interrupt_after_grpo_step: int | None = None
) -> dict[str, Any]:
    root = config.resolved_output_dir
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(f"output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    data = load_tasks(config.resolved_data_path)
    statuses: dict[str, str] = {}

    dependencies: dict[str, str] = {}
    status = _artifact_status(root, "sft", config, data, dependencies)
    if status == "complete":
        policy = _load_policy(root / "sft" / "checkpoint.pt", config)
        statuses["sft"] = "reused"
    else:
        torch.manual_seed(config.seed)
        policy = _new_policy(config)
        history = train_sft(policy, data.train, config.sft, config.seed)
        directory = root / "sft"
        save_checkpoint(directory / "checkpoint.pt", {"model_state": policy.state_dict()})
        write_json(directory / "metrics.json", {"history": history})
        _finish_stage(
            directory,
            "sft",
            config,
            data,
            dependencies,
            ["checkpoint.pt", "metrics.json"],
            {"epochs": len(history), "final_loss": history[-1]["loss"]},
        )
        statuses["sft"] = "ran"

    dependencies = _dependency_hashes(root, ["sft"])
    status = _artifact_status(root, "reward_model", config, data, dependencies)
    if status == "complete":
        reward_model = _load_reward(root / "reward_model" / "checkpoint.pt", config)
        statuses["reward_model"] = "reused"
    else:
        reward_model, history = train_reward_model(
            data.train,
            config.reward_model,
            config.model.feature_size,
            config.model.hidden_size,
            config.seed + 100,
        )
        directory = root / "reward_model"
        save_checkpoint(directory / "checkpoint.pt", {"model_state": reward_model.state_dict()})
        write_json(directory / "metrics.json", {"history": history})
        _finish_stage(
            directory,
            "reward_model",
            config,
            data,
            dependencies,
            ["checkpoint.pt", "metrics.json"],
            {"epochs": len(history), "final_loss": history[-1]["loss"]},
        )
        statuses["reward_model"] = "ran"

    dependencies = _dependency_hashes(root, ["sft", "reward_model"])
    status = _artifact_status(root, "prm", config, data, dependencies)
    if status == "complete":
        prm = _load_reward(root / "prm" / "checkpoint.pt", config)
        statuses["prm"] = "reused"
    else:
        prm, history, example_count = train_prm(
            data.math_train,
            config.prm,
            config.model.feature_size,
            config.model.hidden_size,
            config.seed + 200,
        )
        directory = root / "prm"
        save_checkpoint(directory / "checkpoint.pt", {"model_state": prm.state_dict()})
        write_json(directory / "metrics.json", {"history": history})
        _finish_stage(
            directory,
            "prm",
            config,
            data,
            dependencies,
            ["checkpoint.pt", "metrics.json"],
            {
                "epochs": len(history),
                "final_loss": history[-1]["loss"],
                "step_examples": example_count,
                "monte_carlo_rollouts": config.prm.monte_carlo_rollouts,
            },
        )
        statuses["prm"] = "ran"

    dependencies = _dependency_hashes(root, ["sft", "reward_model", "prm"])
    status = _artifact_status(root, "grpo", config, data, dependencies, allow_running=True)
    resume_payload = None
    if status == "running":
        resume_payload = load_checkpoint(root / "grpo" / "checkpoint.pt")
    initial_policy = copy.deepcopy(policy)
    if status == "complete":
        final_policy = _load_policy(root / "grpo" / "checkpoint.pt", config)
        statuses["grpo"] = "reused"
    else:
        directory = root / "grpo"

        def checkpoint_callback(
            payload: dict[str, Any], history: list[dict[str, float]]
        ) -> None:
            save_checkpoint(directory / "checkpoint.pt", payload)
            write_json(directory / "metrics.json", {"history": history})
            write_manifest(
                directory,
                "grpo",
                "running",
                config.fingerprint,
                data.source_hash,
                dependencies,
                ["checkpoint.pt", "metrics.json"],
                {"completed_steps": payload["completed_steps"]},
            )

        final_policy, history = train_grpo(
            initial_policy,
            data.train,
            reward_model,
            prm,
            config.grpo,
            config.seed,
            resume_payload=resume_payload,
            on_checkpoint=checkpoint_callback,
            interrupt_after=interrupt_after_grpo_step,
        )
        final_payload = {
            "model_state": final_policy.state_dict(),
            "policy_state": final_policy.state_dict(),
            "history": history,
            "completed_steps": config.grpo.steps,
        }
        save_checkpoint(directory / "checkpoint.pt", final_payload)
        write_json(directory / "metrics.json", {"history": history})
        _finish_stage(
            directory,
            "grpo",
            config,
            data,
            dependencies,
            ["checkpoint.pt", "metrics.json"],
            {"steps": len(history), **history[-1]},
        )
        statuses["grpo"] = "resumed" if resume_payload is not None else "ran"

    dependencies = _dependency_hashes(root, ["sft", "grpo"])
    status = _artifact_status(root, "evaluation", config, data, dependencies)
    if status == "complete":
        with (root / "evaluation" / "metrics.json").open(encoding="utf-8") as handle:
            import json

            evaluation = json.load(handle)
        statuses["evaluation"] = "reused"
    else:
        sft_metrics, sft_predictions = evaluate_policy(policy, data.eval, config.model.max_length)
        grpo_metrics, grpo_predictions = evaluate_policy(
            final_policy, data.eval, config.model.max_length
        )
        evaluation = {"sft": sft_metrics, "grpo": grpo_metrics}
        directory = root / "evaluation"
        write_json(directory / "metrics.json", evaluation)
        write_json(
            directory / "predictions.json", {"sft": sft_predictions, "grpo": grpo_predictions}
        )
        _finish_stage(
            directory,
            "evaluation",
            config,
            data,
            dependencies,
            ["metrics.json", "predictions.json"],
            {"eval_tasks": len(data.eval)},
        )
        statuses["evaluation"] = "ran"

    summary = {
        "schema_version": 1,
        "config_hash": config.fingerprint,
        "data_hash": data.source_hash,
        "stage_status": statuses,
        "evaluation": evaluation,
    }
    write_json(root / "run_summary.json", summary)
    return summary
