---
name: ukisai-benchmarks
description: Run, monitor, and report UkisAI model benchmarks from the Benchmark_configs repo — GPQA Diamond, MMLU-Pro, C-Eval, IFBench, LiveCodeBench, ERQA, AIME 2024/2025, HMMT, Terminal-Bench 2.1 (TB2.1, Harbor), and DeepSWE. Use when starting, resuming, babysitting, debugging, speeding up, or summarizing any benchmark or eval run, when doing a progress / paranoid / improve check-in, or when comparing a model's scores or thinking length against a base model.
---

# UkisAI benchmarks

Repo: `https://github.com/UkisAI/Benchmark_configs`. Benchmarks, sizes, limits and flags: `benchmarks/README.md`.

## Run

```bash
git clone https://github.com/UkisAI/Benchmark_configs && cd Benchmark_configs
cp benchmarks/pins.example.env benchmarks/pins.env   # exact revisions from the private run manifest
bash benchmarks/setup.sh                             # once
huggingface-cli login                                # GPQA is gated
python benchmarks/gpqa.py --limit 2 --seeds 0        # smoke test, then drop the flags
```

The model must already be served. Scripts default to the one model at `127.0.0.1:8000/v1`; change it with `--model` and `--api-base`. Run in `tmux`. Rerunning resumes.

## Monitor

Right after starting a run: `bash monitors/start.sh` (cron; `AGENT=claude` for Claude Code), or in a Claude Code session `/loop 10m`, `20m`, `30m` with the check-ins below. They stop once every run is scored.

Look at `benchmarks/runs/*/status.json` (done/total, errors, rate, ETA), `benchmarks/runs/*/*.log`, `nvidia-smi`, and vLLM `/metrics` (`vllm:kv_cache_usage_perc`, `vllm:generation_tokens_total`, `vllm:num_requests_waiting`, `vllm:num_preemptions_total`).

- **progress** (10 min): check in on progress. Report only.
- **paranoid** (20 min): assume something in the process is broken and stuck; diagnose fast so we don't lose time, and fix it if the fix is safe. Stuck = `status.json` `at` or the log hasn't moved in 15 minutes while requests are pending, the process is gone before `done == total`, the server doesn't answer `/v1/models`, or errors keep climbing. Restart a dead script (it resumes). Never restart one that is still making progress.
- **improve** (30 min): the pipeline is probably not optimized, slow, and underuses the resources; improve it without breaking it. Look for idle GPUs, low KV use with nothing waiting (raise `--conc`), or preemptions (lower it). Restart only resumable single-turn scripts. Never change the protocol: seeds, tokens, sampling, timeouts, prompts.

Every check-in reads `benchmarks/runs/NOTES.md` first and appends 2-4 lines: UTC time, check-in name, what you saw, what you changed.

## Report

Every progress update includes:

- **Progress**: `done/total` naming the benchmark. Unknown total → `x/unknown`; never invent a denominator.
- **KV cache**: current usage with source and unit (e.g. vLLM occupancy %). No server → `N/A (no serving)`.
- **Throughput**: `t/s`, saying whether it counts output or all tokens. `unavailable` if not exposed.

Keep it short. Never present partial scores as final.

Once results are final (`score_*.json` in each run folder), show one table, one row per benchmark:

| Benchmark | Base accuracy | Prior model accuracy | Current model accuracy | Current vs base | Mean thinking vs base | Median thinking vs base |
|---|---:|---:|---:|---:|---:|---:|
| Benchmark name | 0.00% | 0.00% | 0.00% | +0.00 pp | -0.00% | -0.00% |

- `Current vs base` = current − base, in percentage points.
- Thinking change = `(current / base − 1) × 100%` on the same token metric; negative means less thinking.
- Use `—` with a short reason for pending, invalid, or missing values (keep the Prior column even if empty).
- Replace the placeholders with real model names. Name the run folders and configs next to the table.

## Rules

- Compare only runs with the same questions, seeds, token cap, sampling, timeouts, and scorer. A 32k-cap LCB run is not comparable to a 100k one; TB2.1 `5h` is not comparable to `native`.
- Errors and timeouts count as wrong and stay in the denominator. A rerun retries request errors, never wrong answers.
- IFBench scores the visible answer only, never the thinking.
- Record any `--conc` or hardware change in `benchmarks/runs/NOTES.md`; it changes speed, not the protocol.
- Don't edit the benchmark scripts mid-run to "fix" a score.
- Never commit revision IDs or digests; they belong in `benchmarks/pins.env` and the private run manifest.
