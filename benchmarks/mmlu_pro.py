"""MMLU-Pro: 12,032 questions x 5 seeds, official 5-shot CoT prompt, 100k tokens."""
import re
from datasets import load_dataset
from common import flags, chat, run, report, last

a = flags("mmlu_pro", max_tokens=100_000, conc=96)
ds = load_dataset("TIGER-Lab/MMLU-Pro", revision="b189ec765aa7ed75c8acfea42df31fdae71f97be")
clean = lambda r: dict(r, options=[o for o in r["options"] if o != "N/A"])
tests = sorted(map(clean, ds["test"]), key=lambda r: int(r["question_id"]))[: a.limit or None]
shots = {}
for r in map(clean, ds["validation"]):
    shots.setdefault(r["category"], []).append(r)


def example(q, options, cot=""):
    cot = (cot or "Let's think step by step.").removeprefix("A: ")
    return f"Question: {q}\nOptions: " + "".join(f"{'ABCDEFGHIJ'[i]}. {o}\n" for i, o in enumerate(options)) + "Answer: " + cot + "\n\n"


def prompt(r):
    head = (f"The following are multiple choice questions (with answers) about {r['category']}. Think step by step and then "
            'output the answer in the format of "The answer is (X)" at the end.\n\n')
    return head + "".join(example(s["question"], s["options"], s["cot_content"]) for s in shots[r["category"]]) + example(r["question"], r["options"])


def extract(t):  # official TIGER extractor: first "answer is (X)", else "Answer: X", else last lone letter
    m = re.search(r"answer is \(?([A-J])\)?", t) or re.search(r".*[aA]nswer:\s*([A-J])", t)
    return m.group(1) if m else last(r"\b([A-J])\b", t)


def ask(job):
    seed, r = job
    x = chat(a, [{"role": "user", "content": prompt(r)}], seed)
    return dict(x, seed=seed, gold=r["answer"], pred=extract(x["content"]))


out = run(a, {f"s{s}_{r['question_id']}": (s, r) for s in a.seeds for r in tests}, ask)
report(a, [dict(x, correct=not x.get("error") and x["pred"] == x["gold"]) for x in out])
