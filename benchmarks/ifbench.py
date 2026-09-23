"""IFBench: 300 prompts x 5 seeds, 81,920 tokens, strict + loose scored on the visible answer only. Needs setup.sh."""
import json, subprocess
from common import flags, chat, run, report, ok, UPSTREAM

IFB = UPSTREAM / "IFBench"
a = flags("ifbench", max_tokens=81_920, conc=192)
rows = [json.loads(l) for l in open(IFB / "data/IFBench_test.jsonl") if l.strip()][: a.limit or None]


def ask(job):
    seed, r = job
    # Published runs used the server's default thinking settings here (no reasoning_effort).
    return dict(chat(a, [{"role": "user", "content": r["prompt"]}], seed, thinking=False), seed=seed, prompt=r["prompt"])


out = run(a, {f"s{s}_{i:03}": (s, r) for s in a.seeds for i, r in enumerate(rows)}, ask)
scores = {"strict": [], "loose": []}
for s in a.seeds:  # official run_eval.py; errors stay in the denominator with an empty answer
    d = (a.out / f"eval_s{s}").resolve()
    d.mkdir(exist_ok=True)
    mine = [x for x in out if x["seed"] == s]
    (d / "responses.jsonl").write_text("".join(json.dumps(dict(prompt=x["prompt"], response=x["content"]), ensure_ascii=False) + "\n" for x in mine))
    subprocess.run([IFB / ".venv/bin/python", "run_eval.py", f"--input_data={IFB / 'data/IFBench_test.jsonl'}",
                    f"--input_response_data={d / 'responses.jsonl'}", f"--output_dir={d}"], cwd=IFB, check=True)
    for mode in scores:
        follow = {g["prompt"]: g["follow_all_instructions"] for g in map(json.loads, next(d.glob(f"*eval_results_{mode}.jsonl")).open())}
        scores[mode] += [dict(x, correct=ok(x) and follow[x["prompt"]]) for x in mine]
for mode, graded in scores.items():
    report(a, graded, f"ifbench_{mode}")
