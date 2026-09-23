"""AIME / HMMT through the official MathArena harness: one process per seed, 250k tokens. Needs setup.sh."""
import json, subprocess
import zstandard
from common import flags, report, UPSTREAM

MA = UPSTREAM / "matharena"
MODEL_YAML = """model: {model}
api: custom
base_url: {api_base}
human_readable_id: {model} seed{seed}
max_tokens: {max_tokens}
temperature: 1.0
top_p: 0.95
presence_penalty: 0.0
seed: {seed}
extra_body:
  top_k: 20
  min_p: 0.0
  repetition_penalty: 1.0
  chat_template_kwargs: {{enable_thinking: true, reasoning_effort: xhigh}}
concurrent_requests: {conc}
timeout: {timeout}
max_retries_inner: 2
sleep_after_request: 0
read_cost: 0
write_cost: 0
"""


def main(name, comps):
    a = flags(name, max_tokens=250_000, conc=30)
    cfg, out = (a.out / "models").resolve(), (a.out / "outputs").resolve()
    cfg.mkdir(parents=True, exist_ok=True)
    for s in a.seeds:
        (cfg / f"s{s}.yaml").write_text(MODEL_YAML.format(seed=s, **vars(a)))
    procs = [subprocess.Popen([MA / ".venv/bin/python", "scripts/run.py", "--comp", comp, "--models", f"s{s}", "--n", "1",
                               "--comp-configs-dir", "configs/competitions", "--model-configs-dir", str(cfg), "--output-dir", str(out)],
                              cwd=MA, stdout=open(a.out / f"{comp.replace('/', '_')}_s{s}.log", "a"), stderr=subprocess.STDOUT)
             for comp in comps for s in a.seeds]
    print(f"{name}: {len(procs)} MathArena processes running, logs in {a.out}", flush=True)
    assert all(p.wait() == 0 for p in procs), "a MathArena process failed; rerun resumes"
    rows = []
    for f in out.glob("*/*/s*/*.json.zst"):
        x = json.loads(zstandard.ZstdDecompressor().stream_reader(f.open("rb")).read())
        error = bool(x.get("error") or x.get("terminal_failure") or x["detailed_costs"][0].get("error"))
        rows.append(dict(seed=int(f.parent.name[1:]), correct=bool(x["correct"][0]) and not error, error=error))
    report(a, rows)
