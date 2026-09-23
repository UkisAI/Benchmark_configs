#!/usr/bin/env bash
# Removes the check-ins from crontab.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
{ crontab -l 2>/dev/null | grep -v "$DIR/" || true; } | crontab -
echo "check-ins off"
