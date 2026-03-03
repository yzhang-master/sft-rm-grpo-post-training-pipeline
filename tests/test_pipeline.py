import json
from dataclasses import replace

import pytest

from post_training.grpo import InterruptedTraining
from post_training.pipeline import run_pipeline


def test_pipeline_resumes_interrupted_grpo_and_reuses_complete_stages(small_config):
    with pytest.raises(InterruptedTraining, match="step 1"):
        run_pipeline(small_config, interrupt_after_grpo_step=1)
    root = small_config.resolved_output_dir
    running = json.loads((root / "grpo" / "manifest.json").read_text())
    assert running["status"] == "running"
    assert running["metadata"]["completed_steps"] == 1

    summary = run_pipeline(small_config)
    assert summary["stage_status"] == {
        "sft": "reused",
        "reward_model": "reused",
        "prm": "reused",
        "grpo": "resumed",
        "evaluation": "ran",
    }
    completed = json.loads((root / "grpo" / "manifest.json").read_text())
    history = json.loads((root / "grpo" / "metrics.json").read_text())["history"]
    assert completed["status"] == "complete"
    assert [item["step"] for item in history] == [1.0, 2.0]
    assert all(item["entropy"] > 0 and item["length"] > 0 for item in history)
    assert all(item["kl"] >= 0 for item in history)
    assert summary["evaluation"]["grpo"]["tasks"] == 6

    reused = run_pipeline(small_config)
    assert set(reused["stage_status"].values()) == {"reused"}


def test_pipeline_rejects_config_change_and_tampered_checkpoint(small_config):
    run_pipeline(small_config)
    changed = replace(small_config, seed=small_config.seed + 1)
    with pytest.raises(ValueError, match="config_hash"):
        run_pipeline(changed)
    checkpoint = small_config.resolved_output_dir / "sft" / "checkpoint.pt"
    checkpoint.write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        run_pipeline(small_config)


def test_nonempty_output_requires_explicit_resume(small_config):
    small_config.resolved_output_dir.mkdir(parents=True)
    (small_config.resolved_output_dir / "file").write_text("do not overwrite")
    with pytest.raises(FileExistsError):
        run_pipeline(small_config, resume=False)
