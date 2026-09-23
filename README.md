# Benchmark_configs

UkisAI's benchmark suite: one script per benchmark, cron check-ins that keep runs alive, and one agent skill that knows both.

| Folder | What's inside |
|---|---|
| [`benchmarks/`](benchmarks) | One script per benchmark. Pinned datasets, prompts, scoring. |
| [`monitors/`](monitors) | Check-ins every 10 / 20 / 30 min so a run never stalls unnoticed. |
| [`skills/`](skills) | The `ukisai-benchmarks` skill for Claude Code and Codex. |

## Quick start

```bash
bash benchmarks/setup.sh                                          # once
cd benchmarks && python gpqa.py --model my-model --api-base http://127.0.0.1:8000/v1
bash monitors/install.sh                                          # turn on check-ins
```

No questions, answers, weights, keys, or results are stored here. Runs write to `benchmarks/runs/` (git-ignored).
