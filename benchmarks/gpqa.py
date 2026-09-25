"""GPQA Diamond: 198 questions x 5 seeds, 100k tokens. Needs HF access to the gated Idavidrein/gpqa."""
import csv, random
from huggingface_hub import hf_hub_download
from common import flags, chat, run, report, ok, last, PIN

a = flags("gpqa", max_tokens=100_000, conc=192)
path = hf_hub_download("Idavidrein/gpqa", "gpqa_diamond.csv", repo_type="dataset", revision=PIN("GPQA_REVISION"))
rows = list(csv.DictReader(open(path, encoding="utf-8")))[: a.limit or None]
PROMPT = ("Answer the following multiple choice question. The last line of your response must be exactly:\n"
          "Answer: $LETTER\nwhere $LETTER is one of A, B, C, D.\n\n{q}\n\nA) {a}\nB) {b}\nC) {c}\nD) {d}")


def ask(job):
    seed, i = job
    r = rows[i]
    opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
    order = [0, 1, 2, 3]
    random.Random(i * 1000).shuffle(order)  # same option order for every seed, as in the published runs
    shown = [opts[j].strip() for j in order]
    x = chat(a, [{"role": "user", "content": PROMPT.format(q=r["Question"].strip(), a=shown[0], b=shown[1], c=shown[2], d=shown[3])}], seed)
    pred = last(r"Answer:\s*\**\s*([ABCD])\b", x["content"])  # final answer only, never the thinking
    return dict(x, seed=seed, gold="ABCD"[order.index(0)], pred=pred)


out = run(a, {f"s{s}_{i:03}": (s, i) for s in a.seeds for i in range(len(rows))}, ask)
report(a, [dict(x, correct=ok(x) and x["pred"] == x["gold"]) for x in out])
