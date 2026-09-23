# Benchmarks

One file per benchmark. Point it at any OpenAI- or Anthropic-compatible endpoint and run it.

```bash
bash setup.sh                                                  # once: pinned harnesses + packages
python gpqa.py --model my-model --api-base http://127.0.0.1:8000/v1
```

| Benchmark | Run | Size | Limit | Scored by |
|---|---|---|---|---|
| GPQA Diamond | `python gpqa.py` | 198 × 5 seeds | 100k tokens | last `Answer: X` |
| MMLU-Pro | `python mmlu_pro.py` | 12,032 × 5 | 100k tokens | official 5-shot CoT extractor |
| C-Eval (val) | `python ceval.py` | 1,346 × 5 | 16k tokens | last `答案：X` |
| IFBench | `python ifbench.py` | 300 × 5 | 81,920 tokens | official `run_eval.py`, strict + loose |
| LiveCodeBench v6 | `python lcb.py` | 1,055 × 5 | 100k tokens | official code execution |
| ERQA | `python erqa.py` | 400 × 5 | 100k tokens | exact letter match |
| AIME 2024 | `python aime_2024.py` | 30 × 5 | 250k tokens | last `\boxed{}` vs gold |
| AIME 2025 | `python aime_2025.py` | 30 × 5 | 250k tokens | last `\boxed{}` vs gold |
| HMMT Nov 2025 | `python hmmt.py` | 30 × 5 | 250k tokens | last `\boxed{}` vs gold |
| Terminal-Bench 2.1 | `python tb21.py --variant 5h` | 89 × 5 | 5h per task, 89 parallel | Harbor 0.20.0 + Terminus 2 |
| | `python tb21.py --variant native` | 89 × 5 | each task's own timeout, 8 parallel | |
| DeepSWE v1.1 | `python deepswe.py` | 113 × 3 | 3h per task | Pier 0.3.1 + mini-swe-agent |

## Flags

| Flag | Meaning |
|---|---|
| `--model`, `--api-base` | Served model name and URL (or `MODEL`, `API_BASE`). Key from `API_KEY`. |
| `--api anthropic` | Use an Anthropic-compatible `/messages` endpoint (default `openai`). |
| `--limit 2 --seeds 0` | Smoke test on 2 questions. |
| `--conc N` | Parallel requests. Changes speed, not results. |
| `--out DIR` | Output folder (default `runs/<benchmark>-<model>`). |

## Output

- `runs/<benchmark>-<model>/status.json` shows live progress: done/total, errors, rate, ETA.
- `raw/` holds one JSON per answer. Rerunning resumes and retries only request errors.
- `score_<benchmark>.json` has per-seed accuracy, the mean, and output token stats.

## Protocol

- **Sampling:** Qwen3.8 thinking profile, temperature 1, top-p 0.95, top-k 20, `reasoning_effort=xhigh`, seeds 0–4.
- **Serving:** any compatible endpoint; published numbers used vLLM 0.27.1, BF16, 262,144 context.
- **Errors:** errors, timeouts and truncations count as wrong and stay in the denominator.
- **Pins:** every dataset and grader is pinned to a commit in `setup.sh` or the script. Compare only runs with the same settings.
- **Math:** AIME/HMMT use MathArena's prompt and data with a built-in `\boxed{}` checker (no MathArena install).
