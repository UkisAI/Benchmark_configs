# Benchmark_configs

Every benchmark UkisAI uses to score its models. Point it at a served model and run.

| Folder | Use it to |
|---|---|
| [`benchmarks/`](benchmarks) | Run a benchmark. One script each, official prompts and graders. |
| [`monitors/`](monitors) | Watch runs with your agent: track progress, catch crashes, propose fixes. |
| [`skills/`](skills) | Give your agent the whole workflow in one skill. |

## Start

Serve your model on `127.0.0.1:8000`, then:

```bash
bash benchmarks/setup.sh          # once, inside a venv
python benchmarks/gpqa.py         # benchmarks the model you're serving
bash monitors/start.sh            # agent check-ins until it's scored
```

No model flag needed: the script asks your server which model it serves and uses that. Otherwise:

| Your model is | Add |
|---|---|
| On another port | `--api-base http://127.0.0.1:8001/v1` |
| On another machine | `--api-base http://10.0.0.5:8000/v1` |
| One of several on the server | `--model <name>` |
| Behind a key | `API_KEY=... ` before `python` |

Smoke test first with `--limit 2 --seeds 0`. Run in `tmux`.

Runs go to `benchmarks/runs/` (git-ignored). No questions, answers, weights, keys or results live here.
