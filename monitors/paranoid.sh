#!/usr/bin/env bash
# Every 20 minutes: assume it's broken and prove it isn't.
exec "$(dirname "$0")/agent.sh" paranoid "Paranoid check-in: assume something in the process is broken and stuck. Diagnose the problem fast so we don't lose time, and fix it if the fix is safe."
