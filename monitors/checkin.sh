#!/usr/bin/env bash
# One agent check-in on benchmarks/runs. Usage: checkin.sh progress|paranoid|improve. AGENT=codex (default) or claude.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p benchmarks/runs monitors/logs

# Every run scored: turn the check-ins off.
if ls -d benchmarks/runs/*/ >/dev/null 2>&1 && [ -z "$(for r in benchmarks/runs/*/; do ls "$r"score_*.json >/dev/null 2>&1 || echo "$r"; done)" ]; then
  echo "$(date -u +%FT%TZ) every run is scored"; exec monitors/stop.sh
fi

# One check-in at a time: a slow 30m check must not overlap the next 10m one.
mkdir monitors/logs/.lock 2>/dev/null || { echo "$(date -u +%FT%TZ) $1 skipped: another check-in is running"; exit 0; }
trap 'rmdir monitors/logs/.lock' EXIT

PROMPT="Do the '$1' check-in from the Monitor section of $PWD/skills/ukisai-benchmarks/SKILL.md on $PWD/benchmarks/runs.
You remember nothing between check-ins: read benchmarks/runs/NOTES.md first, then append 2-4 lines (UTC time, '$1', what you saw, what you changed)."

echo "$(date -u +%FT%TZ) $1 start (${AGENT:-codex})"
case "${AGENT:-codex}" in
  codex)  codex exec --skip-git-repo-check --sandbox danger-full-access "$PROMPT" </dev/null ;;
  claude) claude -p "$PROMPT" --allowedTools "Bash,Read,Edit,Write,Glob,Grep" </dev/null ;;
  *) echo "AGENT must be codex or claude"; exit 1 ;;
esac
echo "$(date -u +%FT%TZ) $1 done"
