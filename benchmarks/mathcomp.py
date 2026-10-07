"""AIME / HMMT with MathArena's prompt and data: final answer in \\boxed{}, compared to the gold answer."""
import re
from datasets import load_dataset
from sympy import simplify
from sympy.parsing.latex import parse_latex
from common import flags, chat, run, report, ok

PROMPT = "Put your final answer within \\boxed{{}}.{extra}\n\n{problem}"
AIME = "\nThe answer is an integer between 0 and 999 inclusive."


def boxed(text):
    """Contents of the last \\boxed{...}, braces balanced."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    depth, j = 0, i + 7
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth < 0:
            return text[j:k].strip()
    return None


def same(pred, gold):
    if pred is None:
        return False
    # \sqrt3 -> \sqrt{3}, as MathArena's parser does: parse_latex rejects an unbraced root or silently stops before it
    # ("6\sqrt3+5\pi" parses as 6), so equal answers scored wrong and some different answers could score right.
    norm = lambda s: re.sub(r"\\sqrt(\d+)", r"\\sqrt{\1}",
                            re.sub(r"\\[,!; ]|\\left|\\right|\s|\$", "", str(s)).replace("dfrac", "frac").replace("tfrac", "frac"))
    if norm(pred) == norm(gold):
        return True
    try:
        return simplify(parse_latex(norm(pred)) - parse_latex(norm(gold))) == 0
    except Exception:
        return False


def main(name, datasets, integer):
    a = flags(name, max_tokens=250_000, conc=30)
    rows = [r for d in datasets for r in load_dataset(d, split="train")][: a.limit or None]

    def ask(job):
        seed, r = job
        x = chat(a, [{"role": "user", "content": PROMPT.format(extra=AIME if integer else "", problem=r["problem"])}], seed)
        return dict(x, seed=seed, gold=str(r["answer"]), pred=boxed(x["content"]))

    out = run(a, {f"s{s}_{i:02}": (s, r) for s in a.seeds for i, r in enumerate(rows)}, ask)
    check = (lambda p, g: p is not None and p.strip().lstrip("0") == g.lstrip("0")) if integer else same
    report(a, [dict(x, correct=ok(x) and check(x["pred"], x["gold"])) for x in out])
