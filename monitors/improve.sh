#!/usr/bin/env bash
# Every 30 minutes: find speed left on the table.
exec "$(dirname "$0")/agent.sh" improve "Improve check-in: the current pipeline is probably not optimized, slow, and underuses the available resources. Improve it without breaking it."
