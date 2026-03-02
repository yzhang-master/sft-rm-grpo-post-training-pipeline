import argparse
import json

from post_training.config import load_config
from post_training.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tiny_cpu.json")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    summary = run_pipeline(load_config(args.config), resume=not args.no_resume)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
