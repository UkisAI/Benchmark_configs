#!/usr/bin/env bash
# One-time setup: graders in ./upstream (at pins.env revisions), light pip packages, and the agent CLIs. Needs git, pip, uv.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p upstream

# Exact revisions come from pins.env (git-ignored); empty = latest upstream.
[ -f pins.env ] && set -a && . ./pins.env && set +a

pin() {  # pin <dir> <git url> [commit]
  [ -d "upstream/$1" ] || git clone -q "$2" "upstream/$1"
  [ -z "${3:-}" ] || { git -C "upstream/$1" fetch -q origin; git -C "upstream/$1" checkout -q "$3"; }
  echo "upstream/$1 @ $(git -C "upstream/$1" rev-parse --short HEAD)"
}
pin IFBench         https://github.com/allenai/IFBench                 "${IFBENCH_COMMIT:-}"
pin ceval           https://github.com/hkust-nlp/ceval                 "${CEVAL_COMMIT:-}"
pin ERQA            https://github.com/embodiedreasoning/ERQA          "${ERQA_COMMIT:-}"
pin LiveCodeBench   https://github.com/LiveCodeBench/LiveCodeBench     "${LCB_COMMIT:-}"
pin deep-swe        https://github.com/datacurve-ai/deep-swe           "${DEEPSWE_COMMIT:-}"

# Scripts only call an OpenAI/Anthropic-compatible endpoint; these are for data loading and grading (anthropic==0.42.0: LCB imports its old prompt constants).
pip install -q datasets huggingface_hub numpy tqdm sympy antlr4-python3-runtime==4.11 anthropic==0.42.0 -r upstream/IFBench/requirements.txt

# Agent CLIs get their own isolated installs (they conflict with each other).
uv tool install -q harbor==0.20.0
uv tool install -q datacurve-pier==0.3.1 --with mini-swe-agent==2.4.6
echo "setup done"
