"""Install the ukisai-benchmarks skill for Claude Code and/or Codex. Usage: python skills/install.py [--claude] [--codex]"""
import shutil, sys
from pathlib import Path

SKILL = Path(__file__).parent / "ukisai-benchmarks"
TARGETS = {"--claude": Path.home() / ".claude/skills", "--codex": Path.home() / ".codex/skills"}

picked = [f for f in TARGETS if f in sys.argv] or [f for f in TARGETS if shutil.which(f[2:])]
if not picked:
    sys.exit("Neither claude nor codex found on PATH; pass --claude or --codex.")
for flag in picked:
    dest = TARGETS[flag] / SKILL.name
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(SKILL, dest)
    print(f"{flag[2:]}: installed {dest}")
