# Benchmark_configs

Configuration catalog for UkisAI model evaluations. This repository stores protocol settings and pointers to upstream question sets. It contains no questions, responses, model weights, API keys, or benchmark results.

## Profiles

`protocol.json` defines the shared single-turn Qwen3.8 BF16/vLLM profile: vLLM 0.27.1, Qwen3 reasoning parser, 262,144-token server context, thinking enabled with `reasoning_effort=xhigh`, temperature 1, top-p 0.95, top-k 20, min-p 0, presence penalty 0, repetition penalty 1, and explicit seeds 0–4. Per-benchmark files in `benchmarks/` fix the dataset, split, output cap, prompt/scorer, and expected sample count.

The agent benchmarks have separate protocols. Terminal-Bench 2.1 uses Harbor 0.20.0, Terminus 2, JSON text actions, top-p 1.0, and five attempts. Its Flash-Next and Qwen-27B protocols are separate files with different concurrency and timeout rules. DeepSWE v1.1 uses mini-swe-agent, the 113-task pinned corpus, three attempts, and a separate verifier environment. Their configs live in `agents/`.

| Benchmark | Questions | Trials/question | Max output tokens | Status |
|---|---:|---:|---:|---|
| GPQA Diamond | 198 | 5 | 100,000 | Existing five-seed profile |
| IFBench | 300 | 5 | 81,920 | Existing five-seed profile; strict and loose scores share responses |
| AIME 2024 I+II | 30 | 5 | 250,000 | Five-seed, 250,000-token profile |
| AIME 2025 | 30 | 5 | 250,000 | Matching proposed profile; verify run-specific history before comparing |
| C-Eval validation | 1,346 | 5 | 16,384 | Existing five-seed profile; 6,730 responses |
| HMMT November 2025 | 30 | 5 | 250,000 | Existing five-seed profile |
| ERQA | 400 | 5 | 100,000 | Existing five-seed profile; image input |
| MMLU-Pro | 12,032 | 5 | 100,000 | Existing five-seed profile; official five-shot CoT |
| LiveCodeBench release_v6 | 1,055 | 5 | 100,000 | Increased cap requested; earlier 32,768-cap runs are separate evidence |
| DeepSWE v1.1 | 113 | 3 | Agent/task limits | Pinned historical profile; Pier 0.3.1 and mini-swe-agent 2.4.6 from local rack2 backup |
| Terminal-Bench 2.1 | 89 | 5 | Agent/task limits | Flash-Next: 8 parallel, native task timeouts. Qwen-27B: 89 parallel, 18,000-second override. |

The five-seed profile is a policy for future matched comparisons. Do not silently merge results from the older three-repeat GPQA sweeps, the four-trial AIME 2024 development checks, the incremental-only 175-question LCB runs, or agent benchmarks into this profile.

## Reproduction rules

1. Pin the model and the upstream dataset/harness revisions in each run manifest. Never resolve a floating `main` or `latest` during a reported run.
2. Freeze selected question IDs and hashes before generation. Use the same IDs and seeds for every arm. Record failures and cap hits; do not silently omit them from the denominator.
3. Keep server context and completion budget distinct. Check prompt length plus allowed output against model context before launch. The 250,000-token mathematics cap leaves little room for the prompt in a 262,144-token context.
4. Use final answer content for IFBench scoring, excluding the reasoning field. Report strict and loose from the same response bank.
5. For LiveCodeBench, name the scope explicitly: full `release_v6` (1,055) or incremental v6 only (175). The 100,000-cap profile requires a fresh matched run; it does not change older 32,768-cap scores.
6. For Terminal-Bench and DeepSWE, verify the agent implementation and verifier isolation in the actual environment. The config file alone cannot prove a package patch was installed.

For Terminal-Bench 2.1, use `agents/terminal_bench_2_1.flash_next_native.harbor.json` for Flash-Next. It has `n_concurrent_trials=8` and no `override_timeout_sec`, so every task keeps its native timeout. `agents/terminal_bench_2_1.qwen27b_fixed5h.harbor.json` is a different Qwen-27B protocol with `n_concurrent_trials=89` and `override_timeout_sec=18000`. Concurrency is a hardware scheduling choice and should be recorded with the available GPU memory; the timeout difference changes the benchmark protocol.

`PROVENANCE.md` lists the Tesla 1 source files and known limitations. Template fields such as `REPLACE_MODEL` and `REPLACE_API_BASE` must be filled for a new run. They are intentionally not model-specific.
