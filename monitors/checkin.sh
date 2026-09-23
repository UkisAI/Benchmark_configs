#!/usr/bin/env bash
# Every 10 minutes: report progress.
exec "$(dirname "$0")/agent.sh" checkin "Check-in on progress."
