# Monitors

Watch benchmark runs with your agent: track progress, catch crashes, propose fixes. Made for maximum efficiency and control.

```bash
bash monitors/start.sh                # Codex check-ins every 10 / 20 / 30 min
AGENT=claude bash monitors/start.sh   # same, with Claude Code
bash monitors/stop.sh                 # stop early; otherwise they stop once every run is scored
```

| Every | Check-in | Job |
|---|---|---|
| 10 min | `progress` | Report progress, KV cache, tokens/s. |
| 20 min | `paranoid` | Assume it's stuck; diagnose and fix fast. |
| 30 min | `improve` | Find idle GPUs or slow settings; speed up without breaking. |

The check-ins are defined once, in the Monitor section of [`SKILL.md`](../skills/ukisai-benchmarks/SKILL.md). Each one reads and appends to `benchmarks/runs/NOTES.md`. Logs go to `monitors/logs/`.

Already in a Claude Code session? Skip cron:

```
/loop 10m do the ukisai-benchmarks progress check-in
/loop 20m do the ukisai-benchmarks paranoid check-in
/loop 30m do the ukisai-benchmarks improve check-in
```
