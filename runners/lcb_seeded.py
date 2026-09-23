#!/usr/bin/env python3
"""Resumable seeded LiveCodeBench generation against an OpenAI-compatible endpoint.

Generalisation of lcb_v6_seeded.py: LCB_VERSION selects the dataset tag
(e.g. v6, v1_v5, release_v6) and LCB_EXPECTED pins its cardinality.
"""

import concurrent.futures as cf
import json
import os
import pathlib
import tempfile
import time
import urllib.request
import sys

sys.path.insert(0, os.environ["LCB_CHECKOUT"])

from datasets import load_dataset as hf_load_dataset
from lcb_runner.benchmarks import code_generation as codegen
from lcb_runner.prompts.code_generation import (
    PromptConstants,
    get_generic_question_template_answer,
)
from lcb_runner.utils.extraction_utils import extract_code
from lcb_runner.lm_styles import LMStyle

REVISION = os.environ["LCB_DATASET_REVISION"]


def pinned_load_dataset(*args, **kwargs):
    kwargs["revision"] = REVISION
    return hf_load_dataset(*args, **kwargs)


codegen.load_dataset = pinned_load_dataset
load_code_generation_dataset = codegen.load_code_generation_dataset

API = os.environ["API_BASE"].rstrip("/")
MODEL = os.environ["MODEL"]
ARM = os.environ["ARM"]
OUT = pathlib.Path(os.environ["OUT"])
CONC = int(os.environ.get("CONC", "72"))
MAXTOK = int(os.environ.get("MAX_TOKENS", "100000"))
TIMEOUT = int(os.environ.get("REQ_TIMEOUT", "2400"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
VERSION = os.environ.get("LCB_VERSION", "release_v6")
EXPECTED = int(os.environ.get("LCB_EXPECTED", "1055"))


def post(payload):
    req = urllib.request.Request(
        API + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-noauth",
        },
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
        return json.loads(response.read())


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def artifact(seed, question_id):
    safe = str(question_id).replace("/", "_")
    return OUT / ARM / f"seed{seed}" / f"{safe}.json"


def valid(path):
    try:
        row = json.loads(path.read_text())
        return (
            row.get("arm") == ARM
            and row.get("content") is not None
            and not row.get("error")
        )
    except Exception:
        return False


def run_one(job):
    seed, problem = job
    path = artifact(seed, problem.question_id)
    if path.exists() and valid(path):
        return "cached", seed, problem.question_id
    messages = [
        {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
        {"role": "user", "content": get_generic_question_template_answer(problem)},
    ]
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": MAXTOK,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"},
    }
    started = time.time()
    try:
        response = post(payload)
        choice = response["choices"][0]
        message = choice["message"]
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        content = message.get("content") or ""
        row = {
            "arm": ARM,
            "seed": seed,
            "question_id": problem.question_id,
            "contest_date": problem.contest_date.isoformat(),
            "difficulty": problem.difficulty.value,
            "platform": problem.platform.value,
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": response.get("usage", {}).get("prompt_tokens"),
            "completion_tokens": response.get("usage", {}).get("completion_tokens"),
            "request_seconds": time.time() - started,
            "reasoning": reasoning,
            "content": content,
            "code": extract_code(content, LMStyle.OpenAIChat),
        }
        atomic_json(path, row)
        return "done", seed, problem.question_id
    except Exception as exc:
        return "error", seed, f"{problem.question_id}: {type(exc).__name__}: {exc}"


def main():
    problems = sorted(
        load_code_generation_dataset(VERSION), key=lambda x: str(x.question_id)
    )
    if len(problems) != EXPECTED:
        raise RuntimeError(
            f"{VERSION} cardinality changed: {len(problems)} != {EXPECTED}"
        )
    limit = int(os.environ.get("LCB_LIMIT", "0"))
    if limit:
        problems = problems[:limit]
    jobs = [(seed, problem) for seed in SEEDS for problem in problems]
    pending = [
        (s, p)
        for s, p in jobs
        if not (
            artifact(s, p.question_id).exists() and valid(artifact(s, p.question_id))
        )
    ]
    print(
        f"[{ARM}] version={VERSION} total={len(jobs)} cached={len(jobs)-len(pending)} pending={len(pending)} conc={CONC}",
        flush=True,
    )
    counts = {"done": 0, "cached": len(jobs) - len(pending), "error": 0}
    errors = []
    start = time.time()
    with cf.ThreadPoolExecutor(max_workers=CONC) as pool:
        futures = [pool.submit(run_one, job) for job in pending]
        for future in cf.as_completed(futures):
            status, seed, detail = future.result()
            counts[status] += 1
            if status == "error":
                errors.append((seed, detail))
            finished = counts["done"] + counts["error"]
            if finished % 25 == 0 or finished == len(pending):
                elapsed = time.time() - start
                rate = finished / max(elapsed, 1) * 60
                print(
                    f"[{ARM}] {finished}/{len(pending)} new; errors={counts['error']}; {rate:.1f}/min; elapsed={elapsed:.0f}s",
                    flush=True,
                )
    if errors:
        for item in errors[:20]:
            print("ERROR", item, flush=True)
        raise RuntimeError(
            f"{len(errors)} request errors; rerun resumes completed artifacts"
        )
    expected = len(problems)
    for seed in SEEDS:
        n = sum(valid(p) for p in (OUT / ARM / f"seed{seed}").glob("*.json"))
        if n < expected:
            raise RuntimeError(f"{ARM} seed{seed}: {n} valid artifacts < {expected}")
    print(
        f"[{ARM}] COMPLETE {len(jobs)}/{len(jobs)} wall_seconds={time.time()-start:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
