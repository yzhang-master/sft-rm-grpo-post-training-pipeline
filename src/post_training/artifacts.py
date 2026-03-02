import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 1


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_json(path: str | Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n"
    _atomic_bytes(Path(path), payload)


def save_checkpoint(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(destination)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary")
    return payload


def manifest_hash(stage_dir: str | Path) -> str:
    return file_hash(Path(stage_dir) / "manifest.json")


def write_manifest(
    stage_dir: str | Path,
    stage: str,
    status: str,
    config_hash: str,
    data_hash: str,
    dependencies: dict[str, str],
    files: list[str],
    metadata: dict[str, Any] | None = None,
) -> None:
    directory = Path(stage_dir)
    checksums = {name: file_hash(directory / name) for name in files}
    write_json(
        directory / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "stage": stage,
            "status": status,
            "config_hash": config_hash,
            "data_hash": data_hash,
            "dependencies": dependencies,
            "files": checksums,
            "metadata": metadata or {},
        },
    )


def validate_manifest(
    stage_dir: str | Path,
    stage: str,
    config_hash: str,
    data_hash: str,
    dependencies: dict[str, str],
    allow_running: bool = False,
) -> dict[str, Any]:
    directory = Path(stage_dir)
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {stage} manifest: {path}")
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "config_hash": config_hash,
        "data_hash": data_hash,
        "dependencies": dependencies,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{stage} artifact has invalid {key}")
    allowed_statuses = {"complete", "running"} if allow_running else {"complete"}
    if manifest.get("status") not in allowed_statuses:
        raise ValueError(f"{stage} artifact status is not resumable")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{stage} artifact has no checked files")
    for name, expected_hash in files.items():
        if Path(name).name != name:
            raise ValueError("artifact file names must be relative and flat")
        file_path = directory / name
        if not file_path.is_file() or file_hash(file_path) != expected_hash:
            raise ValueError(f"{stage} artifact failed checksum validation: {name}")
    return manifest
