# Monitors

Cron check-ins that watch a benchmark run so nothing stalls or crashes unnoticed.

| Every | Script | Job |
|---|---|---|
| 10 min | `checkin.sh` | Report progress, KV cache, tokens/s. |
| 20 min | `paranoid.sh` | Assume it's stuck; diagnose and fix fast. |
| 30 min | `improve.sh` | Find idle GPUs or slow settings; speed up without breaking. |

## Claude Code

Paste in the session running the benchmark:

```
/loop 10m check-in on progress
/loop 20m paranoid check-in: assume something in the process is broken and stuck, diagnose the problem fast so we don't lose time
/loop 30m improve check-in: the current pipeline is probably not optimized, slow, and underuses the resources; improve it without breaking it
```

## Codex (cron)

```bash
bash monitors/install.sh                  # Codex every 10/20/30 min
AGENT=claude bash monitors/install.sh     # same, with Claude Code
bash monitors/uninstall.sh                # stop
```

Each check-in reads and appends to `benchmarks/runs/NOTES.md`, so the next one knows what happened. Logs go to `monitors/logs/`. The crons remove themselves once every run is scored.
