# Source revisions and protocol boundaries

This catalog contains evaluation settings and upstream pointers. It does not include benchmark questions, solutions, trajectories, results, weights, or credentials.

| Benchmark source | Repository | Pinned revision or version |
|---|---|---|
| GPQA Diamond | [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) | Dataset `633f5ee89ab8ad4522a9f850766b73f62147ffdd` |
| IFBench | [allenai/IFBench](https://github.com/allenai/IFBench) | Harness `1091c4c3de6c1f6ed12c012ed68f11ea450b0117`; 300-question JSONL SHA256 `a2bf7b8e7ed11d39c65b4ec781da8b99663ed0121f97033d70fcecb61f8a9d85` |
| AIME 2024, AIME 2025, HMMT November 2025 | [eth-sri/matharena](https://github.com/eth-sri/matharena) | Harness `a11194deff8c67a232974a383795e8a2776b4c6f`; HMMT dataset `118dbfb45c4c9467c672268ed55166642897aa46`; pin AIME dataset revisions before each run |
| C-Eval validation | [hkust-nlp/ceval](https://github.com/hkust-nlp/ceval), [dataset](https://huggingface.co/datasets/ceval/ceval-exam) | Harness `cba65ae93bcf189149ced9f66ae0c958201faed9`; pin selected dataset revision before each run |
| MMLU-Pro | [TIGER-AI-Lab/MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) | Harness `f418b116db00b065c2aea046518d8fcf74d39872`; dataset `b189ec765aa7ed75c8acfea42df31fdae71f97be` |
| ERQA | [embodiedreasoning/ERQA](https://github.com/embodiedreasoning/ERQA) | Harness `5bba488252c5b345a43335bfb9239b6bf12bd91c`; prepared 400-question manifest SHA256 `17513e83e4727dba9b96b13a4cd2704af0c9201d6c61a8573a394573ef33d5bc` |
| LiveCodeBench | [LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench), [dataset](https://huggingface.co/datasets/livecodebench/code_generation_lite) | Harness `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`; dataset `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`; full 1,055-question set SHA256 `b5baada8fb536948cba9fbc3b13dd8cf8a132badeb82b9548dc5aadada2152c0` |
| Terminal-Bench 2.1 | [tasks](https://github.com/harbor-framework/terminal-bench-2-1), [Harbor](https://github.com/harbor-framework/harbor) | Dataset `sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a`; Harbor 0.20.0; Terminus 2.0.0 |
| DeepSWE v1.1 | [datacurve-ai/deep-swe](https://github.com/datacurve-ai/deep-swe) | Tasks `0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea`; Pier 0.3.1; mini-swe-agent 2.4.6 |

Terminal-Bench 2.1 has **two separate configurations**. Flash-Next runs eight trials concurrently and omits `override_timeout_sec`, leaving each task's native timeout in force. Qwen-27B runs 89 trials concurrently with `override_timeout_sec=18000`. These timeout policies produce different benchmark protocols. Concurrency itself is limited by available GPU memory and must be recorded per run. Both use Terminus 2 and five attempts per task.

DeepSWE v1.1 uses 113 tasks and three attempts per task. All 113 pinned `task.toml` files specify a separate verifier environment, one collection hook, a 1,800-second verifier timeout, a 300-second collection timeout, and a 10,800-second agent timeout. It uses mini-swe-agent through Pier; the saved configuration had zero retries and 16 concurrent trials per model.

The 100,000-token LiveCodeBench profile is a new evaluation setting. Full `release_v6` set has 1,055 questions; the incremental v6 subset has 175. AIME 2024 and 2025 use the 250,000-token, five-seed profile specified here; do not combine results from runs with different caps or trial counts.
