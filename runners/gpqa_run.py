#!/usr/bin/env python3
"""GPQA Diamond generation and scoring, adapted from the Tesla 1 runner."""

import csv, json, os, random, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API = os.environ.get("API_BASE", "http://localhost:8000/v1")
ROOT = API[:-3] if API.endswith("/v1") else API
MODEL = os.environ.get("MODEL", "qwen3.8-27b")
SEEDS = [int(value) for value in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
CONC = int(os.environ.get("CONC", "48"))
MAXTOK = int(os.environ.get("MAX_TOKENS", "100000"))
OUT = os.environ["OUT"]
TAG = os.environ.get("TAG", "baseline")
REQ_TIMEOUT = int(os.environ.get("REQ_TIMEOUT", "2400"))
DATA = os.environ["GPQA_DATA"]

LETTERS = ["A", "B", "C", "D"]
TEMPLATE = (
    "Answer the following multiple choice question. The last line of your "
    "response must be exactly:\nAnswer: $LETTER\nwhere $LETTER is one of A, B, C, D.\n\n"
    "{q}\n\nA) {a}\nB) {b}\nC) {c}\nD) {d}"
)


def post(path, payload, timeout=3600, root=False):
    req = urllib.request.Request(
        (ROOT if root else API) + path,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-noauth",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ntok(text):
    """Exact token count via vLLM /tokenize."""
    if not text:
        return 0
    try:
        return len(
            post("/tokenize", {"model": MODEL, "prompt": text}, root=True)["tokens"]
        )
    except Exception:
        return -1


def load():
    with open(DATA, encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 198:
        raise ValueError(f"expected 198 GPQA Diamond questions, got {len(rows)}")
    return rows


def extract(text):
    if not text:
        return None
    m = re.findall(r"Answer:\s*\**\s*([ABCD])\b", text)
    if m:
        return m[-1]
    m = re.findall(r"\b([ABCD])\b", text[-260:])  # fallback: last standalone letter
    return m[-1] if m else None


def run_one(args):
    idx, row, seed = args
    # Deterministic per (question, repeat) option shuffle -> reproducible.
    opts = [
        row["Correct Answer"].strip(),
        row["Incorrect Answer 1"].strip(),
        row["Incorrect Answer 2"].strip(),
        row["Incorrect Answer 3"].strip(),
    ]
    rng = random.Random(idx * 1000 + seed)
    order = [0, 1, 2, 3]
    rng.shuffle(order)
    shown = [opts[i] for i in order]
    gold = LETTERS[order.index(0)]
    prompt = TEMPLATE.format(
        q=row["Question"].strip(), a=shown[0], b=shown[1], c=shown[2], d=shown[3]
    )
    t0 = time.time()
    try:
        d = post(
            "/chat/completions",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "repetition_penalty": 1.0,
                "max_tokens": MAXTOK,
                "seed": seed,
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "reasoning_effort": "xhigh",
                },
            },
            timeout=REQ_TIMEOUT,
        )
    except Exception as e:
        return {
            "idx": idx,
            "task_id": "gpqa_diamond/%03d" % idx,
            "seed": seed,
            "error": str(e)[:300],
            "tag": TAG,
        }
    dt = time.time() - t0
    ch = d["choices"][0]
    msg = ch["message"]
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    pred = extract(content)
    return {
        "idx": idx,
        "seed": seed,
        "tag": TAG,
        "task_id": f"gpqa_diamond/{idx:03d}",
        "subdomain": row.get("Subdomain", ""),
        "domain": row.get("High-level domain", ""),
        "gold": gold,
        "pred": pred,
        "correct": pred == gold,
        "finish_reason": ch.get("finish_reason"),
        "truncated": ch.get("finish_reason") == "length",
        "prompt_tokens": d["usage"]["prompt_tokens"],
        "completion_tokens": d["usage"]["completion_tokens"],
        "reasoning_tokens": ntok(reasoning),
        "answer_tokens": ntok(content),
        "latency_s": round(dt, 2),
        "reasoning": reasoning,
        "content": content,
    }


def main():
    if SEEDS != [0, 1, 2, 3, 4] or MAXTOK != 100000:
        raise ValueError("GPQA profile requires seeds 0..4 and a 100000-token cap")
    rows = load()
    if os.path.exists(OUT):
        raise FileExistsError(f"GPQA output already exists: {OUT}")
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    jobs = [(i, r, seed) for seed in SEEDS for i, r in enumerate(rows)]

    print(
        f"[{TAG}] {len(rows)} questions x {len(SEEDS)} seeds = {len(jobs)} calls, conc={CONC}",
        flush=True,
    )
    res, done = [], 0
    t_start = time.time()
    with open(OUT, "x", encoding="utf-8") as fh, ThreadPoolExecutor(
        max_workers=CONC
    ) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            res.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            done += 1
            if done % 25 == 0:
                sc = [x for x in res if "correct" in x]
                ok = sum(1 for x in sc if x["correct"])
                el = time.time() - t_start
                rate = done / max(el, 1) * 60
                eta = (len(jobs) - done) / max(rate, 0.01)
                print(
                    f"  {done}/{len(jobs)} acc={100*ok/max(1,len(sc)):.1f}% "
                    f"{rate:.1f}/min eta={eta:.0f}min",
                    flush=True,
                )
    ok = [r for r in res if "correct" in r]
    err = [r for r in res if "error" in r]
    tr = [r for r in ok if r.get("truncated")]
    per = {}
    for r in ok:
        per.setdefault(r["seed"], []).append(r["correct"])
    print("\n" + "=" * 58)
    print(f"TAG              : {TAG}")
    print(f"scored           : {len(ok)}   errors: {len(err)}   TRUNCATED: {len(tr)}")
    for k in sorted(per):
        print(
            f"  seed {k}      : {100*sum(per[k])/len(per[k]):.2f}%  (n={len(per[k])})"
        )
    print(
        f"ACCURACY (errors count wrong): {100 * sum(r['correct'] for r in ok) / len(res):.2f}%"
    )
    if ok:
        import statistics as st

        print(
            f"mean completion  : {st.mean([r['completion_tokens'] for r in ok]):.0f} tok"
        )
        print(
            f"mean reasoning   : {st.mean([r['reasoning_tokens'] for r in ok if r['reasoning_tokens']>=0]):.0f} tok"
        )
        print(f"mean latency     : {st.mean([r['latency_s'] for r in ok]):.1f}s")
    if err:
        print("first error:", err[0]["error"][:200])
    print(f"WALL CLOCK       : {(time.time()-t_start)/60:.1f} min  @conc={CONC}")
    print("=" * 58)
    if err:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
