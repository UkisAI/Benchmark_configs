# Source revisions and protocol boundaries

This catalog contains evaluation settings, upstream pointers, and Python execution code. It does not include benchmark questions, solutions, trajectories, results, weights, or credentials.

| Benchmark | Upstream source | Run-time revision |
|---|---|---|
| GPQA Diamond | [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa) | Supply the dataset revision with the dataset path. |
| IFBench | [IFBench](https://github.com/allenai/IFBench) | Supply the checkout revision to the scorer. |
| AIME 2024, AIME 2025, HMMT November 2025 | [MathArena](https://github.com/eth-sri/matharena) | Supply harness and dataset revisions to the launcher. |
| C-Eval validation | [C-Eval](https://huggingface.co/datasets/ceval/ceval-exam) | Supply the dataset revision to the runner. |
| MMLU-Pro | [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | Supply the dataset revision to the runner. |
| ERQA | [ERQA](https://github.com/embodiedreasoning/ERQA) | Supply the dataset path; verify locally before running. |
| LiveCodeBench | [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) | Supply harness and dataset revisions to the launcher. |
| Terminal-Bench 2.1 | [Terminal-Bench tasks](https://github.com/harbor-framework/terminal-bench-2-1) | Supply the dataset ref to the launcher. |
| DeepSWE v1.1 | [DeepSWE tasks](https://github.com/datacurve-ai/deep-swe) | Supply the task checkout revision to the launcher. |

No revision IDs or artifact digests are stored in this public repository. Record exact inputs in the private run manifest.

Terminal-Bench 2.1 has **two separate configurations**. Flash-Next runs eight trials concurrently and omits `override_timeout_sec`, leaving each task's native timeout in force. Qwen-27B runs 89 trials concurrently with `override_timeout_sec=18000`. These timeout policies produce different benchmark protocols. Concurrency itself is limited by available GPU memory and must be recorded per run. Both use Terminus 2 and five attempts per task.

DeepSWE v1.1 uses 113 tasks and three attempts per task. All 113 pinned `task.toml` files specify a separate verifier environment, one collection hook, a 1,800-second verifier timeout, a 300-second collection timeout, and a 10,800-second agent timeout. It uses mini-swe-agent through Pier; the saved configuration had zero retries and 16 concurrent trials per model.

The 100,000-token LiveCodeBench profile is a new evaluation setting. Full `release_v6` set has 1,055 questions; the incremental v6 subset has 175. AIME 2024 and 2025 use the 250,000-token, five-seed profile specified here; do not combine results from runs with different caps or trial counts.
