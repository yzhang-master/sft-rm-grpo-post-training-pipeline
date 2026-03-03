import json

import pytest

from post_training.artifacts import (
    manifest_hash,
    validate_manifest,
    write_json,
    write_manifest,
)


def _stage(path, stage, dependencies=None, status="complete"):
    write_json(path / "metrics.json", {"value": 3})
    write_manifest(path, stage, status, "config", "data", dependencies or {}, ["metrics.json"])


def test_artifact_checksum_and_dependency_validation(tmp_path):
    _stage(tmp_path / "sft", "sft")
    dependency = manifest_hash(tmp_path / "sft")
    _stage(tmp_path / "grpo", "grpo", {"sft": dependency})
    manifest = validate_manifest(tmp_path / "grpo", "grpo", "config", "data", {"sft": dependency})
    assert manifest["status"] == "complete"
    with pytest.raises(ValueError, match="dependencies"):
        validate_manifest(tmp_path / "grpo", "grpo", "config", "data", {"sft": "wrong"})


def test_tampering_and_incomplete_non_grpo_stage_fail_closed(tmp_path):
    _stage(tmp_path / "sft", "sft")
    write_json(tmp_path / "sft" / "metrics.json", {"value": 4})
    with pytest.raises(ValueError, match="checksum"):
        validate_manifest(tmp_path / "sft", "sft", "config", "data", {})
    _stage(tmp_path / "other", "sft", status="running")
    with pytest.raises(ValueError, match="not resumable"):
        validate_manifest(tmp_path / "other", "sft", "config", "data", {})


def test_manifest_rejects_config_data_and_path_changes(tmp_path):
    _stage(tmp_path / "sft", "sft")
    with pytest.raises(ValueError, match="config_hash"):
        validate_manifest(tmp_path / "sft", "sft", "new", "data", {})
    with pytest.raises(ValueError, match="data_hash"):
        validate_manifest(tmp_path / "sft", "sft", "config", "new", {})
    path = tmp_path / "sft" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"] = {"../outside": "abc"}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="flat"):
        validate_manifest(tmp_path / "sft", "sft", "config", "data", {})
