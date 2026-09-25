#!/usr/bin/env bash
# Start the three check-in cron jobs. AGENT=codex (default) or claude. They stop themselves once every run is scored.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DIR/logs"
ENV="PATH=$PATH AGENT=${AGENT:-codex}"
{ crontab -l 2>/dev/null | grep -v "$DIR/" || true
  echo "*/10 * * * * $ENV $DIR/checkin.sh progress >> $DIR/logs/progress.log 2>&1"
  echo "*/20 * * * * $ENV $DIR/checkin.sh paranoid >> $DIR/logs/paranoid.log 2>&1"
  echo "*/30 * * * * $ENV $DIR/checkin.sh improve  >> $DIR/logs/improve.log 2>&1"
} | crontab -
echo "check-ins on (${AGENT:-codex}); logs in $DIR/logs; stop with $DIR/stop.sh"
