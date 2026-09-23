#!/usr/bin/env python3
"""Resumable five-seed official C-Eval validation generation (5-shot)."""

import ast
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
MAXTOK = int(os.environ.get("MAX_TOKENS", "16384"))
TIMEOUT = int(os.environ.get("REQ_TIMEOUT", "3600"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
DATA_ROOT = pathlib.Path(os.environ["CEVAL_DATA_ROOT"])
ADAPTER = pathlib.Path(os.environ["CEVAL_ADAPTER"])
LETTERS = "ABCD"


def subject_mapping():
    tree = (
        DATA_ROOT
        / ".cache"
        / "huggingface"
        / "trees"
        / f"{os.environ['CEVAL_DATASET_REVISION']}.json"
    )
    if not tree.is_file():
        raise RuntimeError(f"C-Eval dataset revision metadata missing: {tree}")
    tree = ast.parse(ADAPTER.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SUBJECT_MAPPING" for t in node.targets
        ):
            value = ast.literal_eval(node.value)
            if len(value) != 52:
                raise RuntimeError(f"expected 52 C-Eval subjects, got {len(value)}")
            return value
    raise RuntimeError("SUBJECT_MAPPING not found in pinned EvalScope adapter")


def read_split(subject, split):
    matches = glob.glob(str(DATA_ROOT / subject / f"{split}-*.parquet"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {subject}/{split} parquet, found {matches}")
    return pq.read_table(matches[0]).to_pylist()


def choices(row):
    return [row[x] for x in LETTERS]


def sample_to_fewshot(row):
    opts = "\n".join(f"{LETTERS[i]}. {choice}" for i, choice in enumerate(choices(row)))
    return "\n".join(
        [
            f"问题：{row['question']}",
            f"选项：\n{opts}",
            f"解析：{row.get('explanation', '')}",
            f"答案：{row['answer']}",
        ]
    )


def user_prompt(subject_cn, row):
    opts = "\n".join(f"{LETTERS[i]}. {choice}" for i, choice in enumerate(choices(row)))
    return (
        f"以下是中国关于{subject_cn}的单项选择题，请选出其中的正确答案。你的回答的最后一行应该是这样的格式："
        '"答案：[LETTER]"（不带引号），其中 [LETTER] 是 A、B、C、D 中的一个。\n\n'
        f"问题：{row['question']}\n选项：\n{opts}\n"
    )


def prompt_for(subject_cn, row, dev_rows):
    fewshot = "\n\n".join(sample_to_fewshot(x) for x in dev_rows)
    return f"以下是一些示例问题：\n\n{fewshot}\n\n\n" + user_prompt(subject_cn, row)


def dataset():
    mapping = subject_mapping()
    tests, devs = [], {}
    for subject in sorted(mapping):
        dev = read_split(subject, "dev")
        val = read_split(subject, "val")
        if len(dev) != 5:
            raise RuntimeError(
                f"{subject}: expected exactly 5 dev examples, got {len(dev)}"
            )
        devs[subject] = dev
        for row in val:
            tests.append(
                {
                    "question_id": f"ceval-{subject}-#{row['id']}",
                    "subject": subject,
                    "subject_cn": mapping[subject][1],
                    "row": row,
                    "answer": row["answer"],
                }
            )
    if len(tests) != 1346:
        raise RuntimeError(
            f"C-Eval validation cardinality changed: {len(tests)} != 1346"
        )
    return tests, devs


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


def artifact(seed, qid):
    return OUT / ARM / f"seed{seed}" / f"{qid.replace('/', '_')}.json"


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
    seed, item, devs = job
    path = artifact(seed, item["question_id"])
    if path.exists() and valid(path):
        return "cached", seed, item["question_id"]
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt_for(
                    item["subject_cn"], item["row"], devs[item["subject"]]
                ),
            }
        ],
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
                "question_id": item["question_id"],
                "subject": item["subject"],
                "answer": item["answer"],
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
        return "done", seed, item["question_id"]
    except Exception as exc:
        return "error", seed, f"{item['question_id']}: {type(exc).__name__}: {exc}"


def main():
    tests, devs = dataset()
    limit = int(os.environ.get("CEVAL_LIMIT", "0"))
    if limit:
        tests = tests[:limit]
    jobs = [(seed, item, devs) for seed in SEEDS for item in tests]
    pending = [j for j in jobs if not valid(artifact(j[0], j[1]["question_id"]))]
    print(
        f"[{ARM}] total={len(jobs)} cached={len(jobs)-len(pending)} pending={len(pending)} conc={CONC}",
        flush=True,
    )
    done = errors = 0
    details = []
    started = time.time()
    with cf.ThreadPoolExecutor(max_workers=CONC) as pool:
        for future in cf.as_completed([pool.submit(run_one, j) for j in pending]):
            status, seed, detail = future.result()
            if status == "error":
                errors += 1
                details.append((seed, detail))
            else:
                done += 1
            finished = done + errors
            if finished % 100 == 0 or finished == len(pending):
                print(
                    f"[{ARM}] {finished}/{len(pending)} new; errors={errors}; {finished/max(time.time()-started,1)*60:.1f}/min",
                    flush=True,
                )
    if errors:
        for detail in details[:20]:
            print("ERROR", detail, flush=True)
        raise RuntimeError(
            f"{errors} request errors; rerun resumes completed artifacts"
        )
    for seed in SEEDS:
        count = sum(valid(p) for p in (OUT / ARM / f"seed{seed}").glob("*.json"))
        if count < len(tests):
            raise RuntimeError(f"{ARM} seed{seed}: {count} valid < {len(tests)}")
    print(
        f"[{ARM}] COMPLETE {len(jobs)}/{len(jobs)} wall_seconds={time.time()-started:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
