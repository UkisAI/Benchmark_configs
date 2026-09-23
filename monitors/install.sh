#!/usr/bin/env bash
# Adds the three check-ins to crontab (10m / 20m / 30m). Safe to run twice.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DIR/logs"
ENV="PATH=$PATH AGENT=${AGENT:-codex}"
{ crontab -l 2>/dev/null | grep -v "$DIR/" || true
  echo "*/10 * * * * $ENV $DIR/checkin.sh  >> $DIR/logs/checkin.log 2>&1"
  echo "*/20 * * * * $ENV $DIR/paranoid.sh >> $DIR/logs/paranoid.log 2>&1"
  echo "*/30 * * * * $ENV $DIR/improve.sh  >> $DIR/logs/improve.log 2>&1"
} | crontab -
echo "check-ins on (${AGENT:-codex}); logs in $DIR/logs; stop with $DIR/uninstall.sh"
