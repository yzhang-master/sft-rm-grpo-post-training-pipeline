import argparse
import json
import time
from pathlib import Path

from post_training.artifacts import write_json
from post_training.config import load_config
from post_training.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tiny_cpu.json")
    parser.add_argument("--output", default="results/tiny_cpu_benchmark.json")
    args = parser.parse_args()
    config = load_config(args.config)
    started = time.perf_counter()
    summary = run_pipeline(config)
    elapsed = time.perf_counter() - started
    root = config.resolved_output_dir

    def metrics(stage):
        return json.loads((root / stage / "metrics.json").read_text())["history"]

    result = {
        "schema_version": 1,
        "scope": "tiny CPU systems smoke; not a pretrained or 3B model run",
        "stage_status": summary["stage_status"],
        "wall_seconds": elapsed,
        "sft_final_loss": metrics("sft")[-1]["loss"],
        "reward_model_final": metrics("reward_model")[-1],
        "prm_final_loss": metrics("prm")[-1]["loss"],
        "grpo_history": metrics("grpo"),
        "held_out_evaluation": summary["evaluation"],
        "full_model_and_vllm": "pending; no full-model or vLLM training was run",
    }
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
