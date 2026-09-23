---
name: ukisai-benchmarks
description: Run, monitor, and report UkisAI model benchmarks from the Benchmark_configs repo — GPQA Diamond, MMLU-Pro, C-Eval, IFBench, LiveCodeBench, ERQA, AIME 2024/2025, HMMT, Terminal-Bench 2.1 (TB2.1, Harbor), and DeepSWE. Use when starting, resuming, babysitting, debugging, speeding up, or summarizing any benchmark or eval run, when setting up check-in loops or cron monitors for a run, or when comparing a model's scores or thinking length against a base model.
---

# UkisAI benchmarks

Repo: `https://github.com/UkisAI/Benchmark_configs`. Scripts in `benchmarks/`, check-ins in `monitors/`.

## Setup

```bash
git clone https://github.com/UkisAI/Benchmark_configs && cd Benchmark_configs
bash benchmarks/setup.sh        # pinned graders into benchmarks/upstream, light pip + uv installs
huggingface-cli login           # GPQA is gated
```

The model must already be served behind an OpenAI-compatible endpoint (default) or Anthropic-compatible one (`--api anthropic`). Key goes in `API_KEY`.

## Run

Every script takes `--model` and `--api-base` (or `MODEL` / `API_BASE`), resumes when rerun, and writes to `benchmarks/runs/<benchmark>-<model>/`.
Run them from `benchmarks/` in `tmux` so they survive disconnects.

| Benchmark | Command | Size | Limit |
|---|---|---|---|
| GPQA Diamond | `python gpqa.py` | 198 × 5 | 100k tokens |
| MMLU-Pro | `python mmlu_pro.py` | 12,032 × 5 | 100k tokens |
| C-Eval val | `python ceval.py` | 1,346 × 5 | 16k tokens |
| IFBench | `python ifbench.py` | 300 × 5 | 81,920 tokens |
| LiveCodeBench v6 | `python lcb.py` | 1,055 × 5 | 100k tokens |
| ERQA | `python erqa.py` | 400 × 5 | 100k tokens |
| AIME 2024 / 2025 | `python aime_2024.py` / `aime_2025.py` | 30 × 5 | 250k tokens |
| HMMT Nov 2025 | `python hmmt.py` | 30 × 5 | 250k tokens |
| Terminal-Bench 2.1 | `python tb21.py --variant 5h` or `native` | 89 × 5 | 5h/task or native |
| DeepSWE v1.1 | `python deepswe.py` | 113 × 3 | 3h/task |

Smoke test first: `--limit 2 --seeds 0`. Then the full run.

## Monitor

Right after starting a run, turn on the three check-ins:

- **Claude Code**: run these three in the session.
  - `/loop 10m check-in on progress`
  - `/loop 20m paranoid check-in: assume something in the process is broken and stuck, diagnose the problem fast so we don't lose time`
  - `/loop 30m improve check-in: the current pipeline is probably not optimized, slow, and underuses the resources; improve it without breaking it`
- **Codex or anything headless**: `bash monitors/install.sh` (cron). `AGENT=claude bash monitors/install.sh` uses Claude Code instead. Stop with `bash monitors/uninstall.sh`.

Where to look: `benchmarks/runs/*/status.json` (done/total, errors, rate, ETA), `benchmarks/runs/*/*.log`, `nvidia-smi`, and vLLM `/metrics` (`vllm:kv_cache_usage_perc`, `vllm:generation_tokens_total`, `vllm:num_requests_waiting`, `vllm:num_preemptions_total`).

What each check-in may do:

- **checkin**: report only.
- **paranoid**: a run is stuck if `status.json` `at` or the log hasn't moved in 15 minutes while requests are pending, the process is gone before `done == total`, the server doesn't answer `/v1/models`, or errors keep climbing. Restart a dead script (it resumes). Never restart one that is still making progress.
- **improve**: look for idle GPUs, low KV use with nothing waiting (raise `--conc` on the next launch), or preemptions (lower it). Apply changes by restarting a script only when it's a resumable single-turn one. Never change the protocol: seeds, tokens, sampling, timeouts, prompts.

## Report

Every progress update includes:

- **Progress**: `done/total` naming the benchmark. Unknown total → `x/unknown`; never invent a denominator.
- **KV cache**: current usage with source and unit (e.g. vLLM occupancy %). No server → `N/A (no serving)`.
- **Throughput**: `t/s`, saying whether it counts output or all tokens. `unavailable` if not exposed.

Keep it short. Never present partial scores as final.

Once results are final, show one table, one row per benchmark:

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
