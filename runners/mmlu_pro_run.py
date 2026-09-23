#!/usr/bin/env python3
"""Resumable five-seed MMLU-Pro generation using the official TIGER prompt."""

import concurrent.futures as cf
import glob
import json
import os
import pathlib
import tempfile
import time
import urllib.request

import pyarrow.parquet as pq

API = os.environ["API_BASE"].rstrip("/")
MODEL = os.environ["MODEL"]
ARM = os.environ["ARM"]
OUT = pathlib.Path(os.environ["OUT"])
CONC = int(os.environ.get("CONC", "72"))
MAXTOK = int(os.environ.get("MAX_TOKENS", "100000"))
TIMEOUT = int(os.environ.get("REQ_TIMEOUT", "3600"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
DATA_ROOT = os.environ["MMLU_DATA_ROOT"]
DATASET_REVISION = os.environ["MMLU_DATASET_REVISION"]
CHOICES = "ABCDEFGHIJ"


def parquet(split):
    matches = glob.glob(f"{DATA_ROOT}/{DATASET_REVISION}/data/{split}-*.parquet")
    if len(matches) != 1:
        raise RuntimeError(f"expected one cached {split} parquet, found {matches}")
    return pq.read_table(matches[0]).to_pylist()


def preprocess(rows):
    result = []
    for original in rows:
        row = dict(original)
        row["options"] = [option for option in row["options"] if option != "N/A"]
        result.append(row)
    return result


def format_example(question, options, cot_content=""):
    if cot_content == "":
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = f"Question: {question}\nOptions: "
    for index, option in enumerate(options):
        example += f"{CHOICES[index]}. {option}\n"
    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example


def prompt_for(row, examples):
    instruction = (
        "The following are multiple choice questions (with answers) about "
        f"{row['category']}. Think step by step and then output the answer in the "
        'format of "The answer is (X)" at the end.\n\n'
    )
    for example in examples[row["category"]]:
        instruction += format_example(
            example["question"], example["options"], example["cot_content"]
        )
    return instruction + format_example(row["question"], row["options"])


def post(payload):
    request = urllib.request.Request(
        API + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-noauth",
        },
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read())


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


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
    seed, row, examples = job
    path = artifact(seed, row["question_id"])
    if path.exists() and valid(path):
        return "cached", seed, row["question_id"]
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt_for(row, examples)}],
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
        atomic_json(
            path,
            {
                "arm": ARM,
                "seed": seed,
                "question_id": row["question_id"],
                "category": row["category"],
                "answer": row["answer"],
                "answer_index": row["answer_index"],
                "finish_reason": choice.get("finish_reason"),
                "prompt_tokens": response.get("usage", {}).get("prompt_tokens"),
                "completion_tokens": response.get("usage", {}).get("completion_tokens"),
                "request_seconds": time.time() - started,
                "reasoning": message.get("reasoning")
                or message.get("reasoning_content")
                or "",
                "content": message.get("content") or "",
            },
        )
        return "done", seed, row["question_id"]
    except Exception as exc:
        return "error", seed, f"{row['question_id']}: {type(exc).__name__}: {exc}"


def main():
    tests = sorted(preprocess(parquet("test")), key=lambda row: int(row["question_id"]))
    validations = preprocess(parquet("validation"))
    if len(tests) != 12032 or len(validations) != 70:
        raise RuntimeError(
            f"dataset cardinality changed: test={len(tests)} validation={len(validations)}"
        )
    examples = {}
    for row in validations:
        examples.setdefault(row["category"], []).append(row)
    if set(map(len, examples.values())) != {5} or len(examples) != 14:
        raise RuntimeError(
            "expected exactly five validation examples in each of 14 categories"
        )
    limit = int(os.environ.get("MMLU_LIMIT", "0"))
    if limit:
        tests = tests[:limit]
    jobs = [(seed, row, examples) for seed in SEEDS for row in tests]
    pending = [
        job for job in jobs if not valid(artifact(job[0], job[1]["question_id"]))
    ]
    print(
        f"[{ARM}] total={len(jobs)} cached={len(jobs)-len(pending)} pending={len(pending)} conc={CONC}",
        flush=True,
    )
    done = errors = 0
    details = []
    started = time.time()
    with cf.ThreadPoolExecutor(max_workers=CONC) as pool:
        for future in cf.as_completed([pool.submit(run_one, job) for job in pending]):
            status, seed, detail = future.result()
            if status == "error":
                errors += 1
                details.append((seed, detail))
            else:
                done += 1
            finished = done + errors
            if finished % 100 == 0 or finished == len(pending):
                rate = finished / max(time.time() - started, 1) * 60
                print(
                    f"[{ARM}] {finished}/{len(pending)} new; errors={errors}; {rate:.1f}/min",
                    flush=True,
                )
    if errors:
        for detail in details[:20]:
            print("ERROR", detail, flush=True)
        raise RuntimeError(
            f"{errors} request errors; rerun resumes completed artifacts"
        )
    expected = len(tests)
    for seed in SEEDS:
        count = sum(valid(path) for path in (OUT / ARM / f"seed{seed}").glob("*.json"))
        if count < expected:
            raise RuntimeError(
                f"{ARM} seed{seed}: {count} valid artifacts < {expected}"
            )
    print(
        f"[{ARM}] COMPLETE {len(jobs)}/{len(jobs)} wall_seconds={time.time()-started:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
