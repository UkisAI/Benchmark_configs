#!/usr/bin/env bash
# Runs one headless agent check-in on benchmarks/runs. Usage: agent.sh <name> "<prompt>"
# AGENT=codex (default) or AGENT=claude.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p benchmarks/runs monitors/logs

# One check-in at a time: a slow 30m check must not overlap the next 10m one.
LOCK=monitors/logs/.lock
mkdir "$LOCK" 2>/dev/null || { echo "$(date -u +%FT%TZ) $1 skipped: another check-in is running"; exit 0; }
trap 'rmdir "$LOCK"' EXIT

PROMPT="$2

Context: benchmark runs live in $ROOT/benchmarks/runs/<benchmark>-<model>/ (status.json, run.log or *.log, raw/).
Follow the Monitor and Report sections of $ROOT/skills/ukisai-benchmarks/SKILL.md.
You remember nothing between check-ins: read benchmarks/runs/NOTES.md first, then append 2-4 lines
(UTC time, '$1', what you saw, what you changed).
If every run is finished and scored, run monitors/uninstall.sh."

echo "$(date -u +%FT%TZ) $1 start (${AGENT:-codex})"
case "${AGENT:-codex}" in
  codex)  codex exec --sandbox danger-full-access "$PROMPT" ;;
  claude) claude -p "$PROMPT" --allowedTools "Bash,Read,Edit,Write,Glob,Grep" ;;
  *) echo "AGENT must be codex or claude"; exit 1 ;;
esac
echo "$(date -u +%FT%TZ) $1 done"
