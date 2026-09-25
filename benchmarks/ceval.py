"""C-Eval validation: 1,346 questions x 5 seeds, 5-shot per subject, 16k tokens. Needs setup.sh (subject names)."""
import json
from datasets import load_dataset
from common import flags, chat, run, report, last, UPSTREAM

a = flags("ceval", max_tokens=16_384, conc=72, timeout=3600)
subjects = json.load(open(UPSTREAM / "ceval/subject_mapping.json"))  # {subject: [english, chinese, category]}
tests, shots = [], {}
for sub in sorted(subjects):
    d = load_dataset("ceval/ceval-exam", sub)
    shots[sub] = list(d["dev"])
    tests += [dict(r, subject=sub, qid=f"{sub}-{r['id']}") for r in d["val"]]
assert len(tests) == 1346, len(tests)
tests = tests[: a.limit or None]


def options(r):
    return "\n".join(f"{c}. {r[c]}" for c in "ABCD")


def prompt(r):
    few = "\n\n".join(f"问题：{s['question']}\n选项：\n{options(s)}\n解析：{s.get('explanation', '')}\n答案：{s['answer']}" for s in shots[r["subject"]])
    return (f"以下是一些示例问题：\n\n{few}\n\n\n以下是中国关于{subjects[r['subject']][1]}的单项选择题，请选出其中的正确答案。"
            '你的回答的最后一行应该是这样的格式："答案：[LETTER]"（不带引号），其中 [LETTER] 是 A、B、C、D 中的一个。\n\n'
            f"问题：{r['question']}\n选项：\n{options(r)}\n")


def ask(job):
    seed, r = job
    x = chat(a, [{"role": "user", "content": prompt(r)}], seed)
    return dict(x, seed=seed, gold=r["answer"], pred=last(r"答案：([A-D])", x["content"]))


out = run(a, {f"s{s}_{r['qid']}": (s, r) for s in a.seeds for r in tests}, ask)
report(a, [dict(x, correct=not x.get("error") and x["pred"] == x["gold"]) for x in out])
