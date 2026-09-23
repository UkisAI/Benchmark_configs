#!/usr/bin/env bash
# One-time setup: pinned graders in ./upstream, light pip packages, and the agent CLIs. Needs git, pip, uv.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p upstream

pin() {  # pin <dir> <git url> <commit>
  [ -d "upstream/$1" ] || git clone -q "$2" "upstream/$1"
  git -C "upstream/$1" fetch -q origin "$3" 2>/dev/null || true
  git -C "upstream/$1" checkout -q "$3"
  echo "upstream/$1 @ ${3:0:12}"
}
pin IFBench         https://github.com/allenai/IFBench                  1091c4c3de6c1f6ed12c012ed68f11ea450b0117
pin ceval           https://github.com/hkust-nlp/ceval                  cba65ae93bcf189149ced9f66ae0c958201faed9
pin ERQA            https://github.com/embodiedreasoning/ERQA           5bba488252c5b345a43335bfb9239b6bf12bd91c
pin LiveCodeBench   https://github.com/LiveCodeBench/LiveCodeBench      28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24
pin deep-swe        https://github.com/datacurve-ai/deep-swe            0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea

# Scripts only call an OpenAI/Anthropic-compatible endpoint; these are for data loading and grading.
pip install -q datasets huggingface_hub numpy tqdm sympy antlr4-python3-runtime==4.11 -r upstream/IFBench/requirements.txt

# Agent CLIs get their own isolated installs (they conflict with each other).
uv tool install -q harbor==0.20.0
uv tool install -q datacurve-pier==0.3.1 --with mini-swe-agent==2.4.6
echo "setup done"
