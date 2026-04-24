# sft-rm-grpo-post-training-pipeline

> **Era:** 2025+ · Staff AI Engineer post-training systems. Techniques from 2021–2024, reimplemented in 2026 on current PyTorch.

## Context

This flagship combines the local function-calling SFT, preference/reward modeling, Monte Carlo PRM labeling, and verified GRPO techniques behind one configuration and one artifact lifecycle. It is an executable tiny CPU systems test, **not a pretrained-model or 3B training run**.

## Problem

Post-training pipelines often stop at disconnected training scripts. Reproducible operation also needs stage dependency checks, durable model/optimizer state, deterministic rollout boundaries, verifiable rewards, and held-out evaluation. Resuming from an incompatible or corrupted artifact should fail rather than silently train from the wrong state.

## Method

```text
configs/tiny_cpu.json + data/tasks.json
              |
              v
   SFT (tiny recurrent token policy)
              |
              v
   pairwise outcome reward model (RM)
              |
              v
   process reward model (PRM), Monte Carlo labels
              |
              v
   GRPO <- local autoregressive rollout worker
              |  <- exact math / JSON tool verifier
              |  <- learned RM and PRM shaping
              v
   held-out math and tool evaluation
```

### SFT and reward training

SFT teacher-forces the complete response, including end-of-sequence, with recurrent token log-probability loss. The RM trains on `(prompt, chosen, rejected)` triples with the pairwise logistic loss `-logsigmoid(score(chosen)-score(rejected))`.

For each arithmetic prefix and candidate process step, the PRM label is the empirical terminal success rate over sampled continuations. Terminal correct/wrong steps therefore have labels 1/0, while a correct first step has a soft success probability. The PRM uses binary cross-entropy against these labels. PRM generation currently covers math tasks; tool calls still receive exact outcome verification.

### Verified rewards and GRPO

Math verification compares exact rational values after the last `####`, falling back to the final number. Explicitly marked correct answers receive 1.2 and other correct answers 1.0. Tool verification parses JSON and requires exact semantic equality of the call name and arguments; it never executes generated code. GRPO optimizes the combined reward

```text
R = verifier_reward + rm_weight * sigmoid(RM) + prm_weight * mean_steps(sigmoid(PRM))
```

while accuracy uses only the verifier. For each prompt, the local worker actually samples a group of token completions from the policy. Advantages are

```text
A_i = (R_i - mean(R)) / (population_std(R) + 1e-6)
```

with zero advantages for a zero-variance group. The token-level loss uses the behavior-policy ratio, PPO clipping, and the sampled k3 KL estimator:

```text
L_pg = -min(ratio*A, clamp(ratio, 1-clip, 1+clip)*A)
L = mean_tokens(L_pg) + beta * mean_tokens(exp(ref-new) - (ref-new) - 1)
```

The initial SFT policy is the frozen KL reference. Each history row records total and verifier reward, verifier accuracy, exact categorical policy/reference KL, token entropy, completion length including EOS, objective, and loss.

### Artifacts and resume

Each stage writes `checkpoint.pt` as applicable, `metrics.json`, and `manifest.json`. Manifests bind the complete normalized config, source-data SHA-256, upstream manifest SHA-256 values, and per-file SHA-256 checksums. Writes use same-directory temporary files followed by atomic replacement.

- A completed stage is reused only after all checks pass.
- Changed config/data/dependencies or corrupted files fail closed.
- Every GRPO step saves model, optimizer, completed step, history, and PyTorch RNG state. A `running` GRPO artifact resumes at the next step with the same frozen SFT reference.
- Incomplete non-GRPO stages are restarted; they do not claim mid-epoch resume.
- Evaluation depends on both the SFT baseline and final GRPO policy and writes all greedy predictions for inspection.

## Results

The tracked `results/tiny_cpu_benchmark.json` is from a fresh run of every stage, not artifact reuse. It took **41.68820620200131 seconds** on this CPU environment. Exact floating-point histories are in that JSON.

| Stage/metric | Observed result |
|---|---:|
| SFT final training loss | 0.17838966846466064 |
| RM final training loss | 0.06514627486467361 |
| RM training pair accuracy | 1.0 (14/14 pairs) |
| PRM final training loss | 0.24220560491085052 |
| PRM training examples | 240, with 32 rollouts per first-step label |

| GRPO step | Total reward | Verifier reward | Accuracy | KL | Entropy | Length |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.2506914549529375 | 0.16428571428571428 | 0.14285714285714285 | 0.0 | 0.26951342821121216 | 37.089285714285715 |
| 2 | 0.2683748959164534 | 0.18214285714285713 | 0.16071428571428573 | 0.00012504002370405942 | 0.2862096428871155 | 35.75 |

Held-out greedy evaluation covers three unseen math prompts and three unseen tool prompts:

| Policy | Math accuracy | Tool accuracy | Overall accuracy | Mean reward | Mean length |
|---|---:|---:|---:|---:|---:|
| SFT | 0/3 | 0/3 | 0/6 | 0.0 | 35.0 |
| GRPO | 0/3 | 0/3 | 0/6 | 0.0 | 35.0 |

This is a negative held-out result, not evidence of reasoning improvement. Higher sampled rollout reward across two GRPO steps does not contradict the zero greedy held-out accuracy; those are different measurements.

Verification: **20 tests passed**, Ruff passed, and all **5 mutation checks were killed**. The mutations targeted wrong-answer acceptance, learned-reward omission, missing GRPO group centering, ignored stage dependency hashes, and restarting GRPO at step zero during resume.

Full-model and GPU results are pending:

| Evaluation/training | Status | Command |
|---|---|---|
| Qwen2.5-3B vLLM rollout evaluation on the same held-out tasks | **pending; not run** | `PYTHONPATH=src python scripts/vllm_rollouts.py --model Qwen/Qwen2.5-3B-Instruct --output results/qwen25_3b_vllm_eval.json` |
| Full-model SFT/RM/PRM/GRPO training and GSM8K/MATH evaluation | **pending; not run** | Full-model trainer integration is required; no tiny run is relabeled as that command or result |

The vLLM command needs the optional `vllm` extra and a supported GPU host. It evaluates real vLLM completions with the local verifiers; it is not a full-model training implementation.

## Reproduce

Use Python 3.12 and CPU PyTorch. From the repository root:

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
PYTHONPATH=src python scripts/mutation_check.py
PYTHONPATH=src python scripts/run_pipeline.py --config configs/tiny_cpu.json
PYTHONPATH=src python scripts/benchmark.py --config configs/tiny_cpu.json
```

The pipeline is resumable by default. A second invocation should report every stage as `reused`. `--no-resume` refuses a nonempty output directory rather than deleting it. To run a fresh benchmark, choose a new `output_dir` in a copied config or deliberately remove only the generated `results/tiny_cpu/` directory.

No dataset download, API key, GPU, or external service is used by the tiny pipeline. `data/tasks.json` contains original local arithmetic and typed-tool fixtures, not GSM8K/MATH test examples.

## Limitations

- The policy is a small hashed-prompt GRU with a fixed character vocabulary. It has no pretrained language knowledge, and the reward models are tiny hashed-feature networks.
- Held-out accuracy is zero. The 14 training tasks and six eval tasks are sufficient for a systems smoke test, not a learning or generalization benchmark.
- PRM Monte Carlo labels are implemented only for the two-step arithmetic family. Tool PRM labels and free-form process verification are not implemented.
- The tool verifier checks exact JSON call equality; it does not execute tools or establish real-world task success.
- vLLM support is an evaluation rollout script, not a drop-in distributed GRPO worker. Full-model training, GPU throughput, and GSM8K/MATH numbers remain pending.
- Only GRPO supports mid-stage checkpoint resume. Other interrupted stages restart safely from their validated dependencies.

## References

- J. Schulman et al. *Proximal Policy Optimization Algorithms.* 2017. arXiv:1707.06347.
- Z. Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* 2024. arXiv:2402.03300.
- K. Cobbe et al. *Training Verifiers to Solve Math Word Problems.* 2021. arXiv:2110.14168.
- H. Lightman et al. *Let's Verify Step by Step.* 2023. arXiv:2305.20050.
- L. Ouyang et al. *Training Language Models to Follow Instructions with Human Feedback.* 2022. arXiv:2203.02155.

## License

MIT
