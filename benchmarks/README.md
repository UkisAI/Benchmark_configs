# Benchmarks

One script per benchmark. Each one benchmarks the model you're serving, resumes when rerun, and writes a score file.

```bash
bash setup.sh                     # once, inside a venv: graders + packages (pins.env for exact revisions)
python gpqa.py                    # model served at 127.0.0.1:8000
```

No model flag needed: the script asks your server which model it serves and uses that. Otherwise:

| Your model is | Add |
|---|---|
| On another port | `--api-base http://127.0.0.1:8001/v1` |
| On another machine | `--api-base http://10.0.0.5:8000/v1` |
| One of several on the server | `--model <name>` |
| Behind a key | `API_KEY=... ` before `python` |

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
| `--model` | Served model name. Default: the one model at `--api-base`. |
| `--api-base` | Default `http://127.0.0.1:8000/v1`. Key from `API_KEY`. |
| `--api anthropic` | Anthropic-compatible `/messages` endpoint (default `openai`). |
| `--limit N` | First N questions only. |
| `--seeds 0,1,2` | Which seeds to run (default `0,1,2,3,4`). |
| `--max-tokens N` | Token limit per answer (default in the table above). |
| `--timeout S` | Seconds per request. |
| `--conc N` | Parallel requests. Changes speed, not results. |
| `--out DIR` | Default `runs/<benchmark>-<model>`. |

Changing `--limit`, `--seeds`, `--max-tokens` or `--timeout` changes the protocol, so the score can't be compared to published numbers. TB2.1 and DeepSWE take only `--model`, `--api-base`, `--conc`, `--out` (and `--variant` for TB2.1).

## Output

- `status.json`: live done/total, errors, rate, ETA.
- `raw/`: one JSON per answer. Rerun to resume; only request errors are retried.
- `score_<benchmark>.json`: per-seed and mean accuracy, output tokens. Agent benchmarks: mean reward plus Harbor's own metrics.

## Protocol

- **Sampling:** Qwen3.8 thinking profile, temperature 1, top-p 0.95, top-k 20, `reasoning_effort=xhigh`, seeds 0–4.
- **Serving:** any compatible endpoint; published numbers used vLLM 0.27.1, BF16, 262,144 context.
- **Errors:** errors, timeouts and truncations count as wrong and stay in the denominator.
- **Pins:** exact revisions go in `pins.env` (git-ignored; copy `pins.example.env`). Empty = latest, fine for smoke tests only.
- **Math:** AIME/HMMT use MathArena's prompt and data with a built-in `\boxed{}` checker.
