import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MUTATIONS = [
    (
        "math verifier rejects wrong answers",
        "src/post_training/rewards.py",
        "if predicted != target:\n        return 0.0",
        "if False:\n        return 0.0",
        "tests/test_rewards.py::test_math_verifier_uses_exact_final_rational_and_format_bonus",
    ),
    (
        "combined reward includes learned components",
        "src/post_training/rewards.py",
        "total = verified + self.rm_weight * outcome + self.prm_weight * process",
        "total = verified",
        "tests/test_rewards.py::test_task_dispatch_and_combined_reward",
    ),
    (
        "GRPO centers each reward group",
        "src/post_training/grpo.py",
        "centered = rewards - rewards.mean()",
        "centered = rewards",
        "tests/test_grpo.py::test_group_relative_advantages_center_and_handle_zero_variance",
    ),
    (
        "stage dependency hashes are enforced",
        "src/post_training/artifacts.py",
        "if manifest.get(key) != value:",
        'if key != "dependencies" and manifest.get(key) != value:',
        "tests/test_artifacts.py::test_artifact_checksum_and_dependency_validation",
    ),
    (
        "resume starts after the saved GRPO step",
        "src/post_training/grpo.py",
        'start_step = int(resume_payload["completed_steps"])',
        "start_step = 0",
        "tests/test_pipeline.py::test_pipeline_resumes_interrupted_grpo_and_reuses_complete_stages",
    ),
]


def main() -> None:
    scratch_root = ROOT / ".mutation-tmp"
    scratch_root.mkdir(exist_ok=True)
    failures = []
    try:
        for name, relative, original, mutated, test in MUTATIONS:
            with tempfile.TemporaryDirectory(dir=scratch_root) as temporary:
                copy = Path(temporary)
                for directory in ("src", "tests", "data", "configs"):
                    shutil.copytree(ROOT / directory, copy / directory)
                target = copy / relative
                content = target.read_text()
                if content.count(original) != 1:
                    raise RuntimeError(f"mutation anchor is not unique: {name}")
                target.write_text(content.replace(original, mutated))
                environment = os.environ.copy()
                environment["PYTHONPATH"] = str(copy / "src")
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", test],
                    cwd=copy,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    failures.append(name)
                    print(f"SURVIVED: {name}")
                else:
                    print(f"KILLED: {name}")
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)
    if failures:
        raise SystemExit(f"mutation checks survived: {failures}")
    print(f"All {len(MUTATIONS)} mutation checks were killed.")


if __name__ == "__main__":
    main()
