import argparse
import json
from pathlib import Path

from post_training.data import load_tasks
from post_training.rewards import verifier_reward


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/tasks.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    try:
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise SystemExit("install the optional vllm dependency on a supported GPU host") from error

    tasks = load_tasks(args.data).eval
    model = LLM(model=args.model)
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    outputs = model.generate([task.prompt for task in tasks], sampling)
    records = []
    for task, output in zip(tasks, outputs, strict=True):
        completion = output.outputs[0].text
        records.append(
            {
                "task_id": task.task_id,
                "kind": task.kind,
                "completion": completion,
                "verifier_reward": verifier_reward(task, completion),
            }
        )
    result = {
        "model": args.model,
        "backend": "vllm",
        "accuracy": sum(item["verifier_reward"] > 0 for item in records) / len(records),
        "records": records,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
