#!/usr/bin/env python3
"""Generate IFBench responses for official strict and loose scoring.

Adapted from Tesla 1's ifbench_gen.py. Only message.content is scored.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API = os.environ.get("API_BASE", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("MODEL", "qwen3.8-27b")
DATA = os.environ["IFB_DATA"]
OUTDIR = os.environ["OUTDIR"]
TAG = os.environ.get("TAG", "base")
K = int(os.environ.get("K", "5"))
CONC = int(os.environ.get("CONC", "192"))
MAXTOK = int(os.environ.get("MAX_TOKENS", "81920"))
REQ_TIMEOUT = int(os.environ.get("REQ_TIMEOUT", "3600"))
ATTEMPTS = int(os.environ.get("ATTEMPTS", "3"))


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def post(payload, timeout):
    req = urllib.request.Request(
        API + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-noauth",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.loads(fh.read().decode())


def done_keys(path):
    seen = set()
    if not os.path.exists(path):
        return seen
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("response") is not None:
                seen.add((d.get("key"), d.get("sample_index")))
    return seen


def one(row, si):
    t0 = time.time()
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": row["prompt"]}],
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": MAXTOK,
        "seed": si,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"},
    }
    last = None
    for _ in range(ATTEMPTS):
        try:
            d = post(body, REQ_TIMEOUT)
            ch = d["choices"][0]
            msg = ch["message"]
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
            content = msg.get("content") or ""
            us = d.get("usage") or {}
            return {
                "key": row["key"],
                "sample_index": si,
                "tag": TAG,
                "prompt": row["prompt"],
                "response": content,  # content ONLY -- see module docstring
                "reasoning": reasoning,
                "instruction_id_list": row.get("instruction_id_list"),
                "finish_reason": ch.get("finish_reason"),
                "truncated": ch.get("finish_reason") == "length",
                "prompt_tokens": us.get("prompt_tokens", 0),
                "completion_tokens": us.get("completion_tokens", 0),
                "seconds": round(time.time() - t0, 2),
            }
        except Exception as exc:  # noqa: BLE001
            last = "%s: %s" % (type(exc).__name__, str(exc)[:200])
            time.sleep(2)
    return {
        "key": row["key"],
        "sample_index": si,
        "tag": TAG,
        "prompt": row["prompt"],
        "response": None,
        "error": last,
        "seconds": round(time.time() - t0, 2),
    }


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    raw = os.path.join(OUTDIR, "%s_raw.jsonl" % TAG)
    rows = load_rows(DATA)
    if len(rows) != 300 or K != 5 or MAXTOK != 81920:
        raise ValueError(
            f"IFBench profile requires 300 prompts, five seeds, and 81920-token cap; got {len(rows)}, {K}, {MAXTOK}"
        )
    seen = done_keys(raw)
    jobs = [(r, si) for r in rows for si in range(K) if (r["key"], si) not in seen]
    print(
        "IFBench %s: %d prompts x k=%d = %d samples; %d already done, %d to run"
        % (TAG, len(rows), K, len(rows) * K, len(seen), len(jobs)),
        flush=True,
    )
    if not jobs:
        print("nothing to do", flush=True)
    t0 = time.time()
    n = 0
    with open(raw, "a", encoding="utf-8") as out, ThreadPoolExecutor(CONC) as ex:
        futs = [ex.submit(one, r, si) for r, si in jobs]
        for f in as_completed(futs):
            out.write(json.dumps(f.result(), ensure_ascii=False) + "\n")
            out.flush()
            n += 1
            if n % 50 == 0:
                el = time.time() - t0
                print(
                    "  %d/%d  %.1f/s  elapsed %.1f min"
                    % (n, len(jobs), n / el, el / 60),
                    flush=True,
                )

    # ---- emit IFBench-shaped response files, one per sample index ----
    recs = [json.loads(x) for x in open(raw, encoding="utf-8")]
    completed = {
        (r["key"], r["sample_index"]): r for r in recs if r.get("response") is not None
    }
    errs = len(rows) * K - len(completed)
    trunc = sum(1 for r in completed.values() if r.get("truncated"))
    for si in range(K):
        p = os.path.join(OUTDIR, "%s_responses_s%d.jsonl" % (TAG, si))
        with open(p, "w", encoding="utf-8") as fh:
            for r in completed.values():
                if r.get("sample_index") == si and r.get("response") is not None:
                    fh.write(
                        json.dumps(
                            {"prompt": r["prompt"], "response": r["response"]},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        print(
            "  wrote %s (%d rows)"
            % (
                p,
                sum(
                    1
                    for r in completed.values()
                    if r.get("sample_index") == si and r.get("response") is not None
                ),
            )
        )
    print(
        "TOTAL records=%d errors=%d truncated=%d (%.2f%%)"
        % (len(completed), errs, trunc, 100.0 * trunc / max(1, len(completed)))
    )
    if trunc > 0.005 * max(1, len(completed)):
        print(
            "  !! TRUNCATION ABOVE 0.5%% -- raise MAX_TOKENS before trusting any delta"
        )
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
