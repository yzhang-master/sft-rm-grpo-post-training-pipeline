import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    split: str
    kind: str
    prompt: str
    answer: str
    response: str
    wrong_response: str
    a: int | None = None
    b: int | None = None
    c: int | None = None

    @property
    def steps(self) -> list[str]:
        return [line for line in self.response.splitlines() if not line.startswith("####")]


class TaskDataset:
    def __init__(self, tasks: list[Task], source_hash: str) -> None:
        self.tasks = tasks
        self.source_hash = source_hash
        self.train = [task for task in tasks if task.split == "train"]
        self.eval = [task for task in tasks if task.split == "eval"]
        if not self.train or not self.eval:
            raise ValueError("dataset requires nonempty train and eval splits")
        ids = [task.task_id for task in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task ids must be unique")
        train_prompts = {task.prompt for task in self.train}
        if any(task.prompt in train_prompts for task in self.eval):
            raise ValueError("held-out prompts cannot occur in training")

    @property
    def math_train(self) -> list[Task]:
        return [task for task in self.train if task.kind == "math"]


def _task(raw: dict[str, Any]) -> Task:
    task = Task(
        task_id=raw["id"],
        split=raw["split"],
        kind=raw["kind"],
        prompt=raw["prompt"],
        answer=raw["answer"],
        response=raw["response"],
        wrong_response=raw["wrong_response"],
        a=raw.get("a"),
        b=raw.get("b"),
        c=raw.get("c"),
    )
    if task.kind not in {"math", "tool"}:
        raise ValueError(f"unsupported task kind: {task.kind}")
    if task.kind == "math" and any(value is None for value in (task.a, task.b, task.c)):
        raise ValueError(f"math task {task.task_id} requires a, b, and c")
    return task


def load_tasks(path: str | Path) -> TaskDataset:
    source = Path(path)
    payload = source.read_bytes()
    raw = json.loads(payload)
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported task dataset schema")
    return TaskDataset(
        [_task(item) for item in raw["tasks"]],
        hashlib.sha256(payload).hexdigest(),
    )
