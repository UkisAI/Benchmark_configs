#!/usr/bin/env python3
"""Resumable five-seed ERQA multimodal generation. Interleaving replicates embodiedreasoning/ERQA eval_harness.py (visual_indices = character offsets)."""

import base64, concurrent.futures as cf, json, os, pathlib, tempfile, time, urllib.request

API = os.environ["API_BASE"].rstrip("/")
MODEL = os.environ["MODEL"]
ARM = os.environ["ARM"]
OUT = pathlib.Path(os.environ["OUT"])
CONC = int(os.environ.get("CONC", "24"))
MAXTOK = int(os.environ.get("MAX_TOKENS", "100000"))
TIMEOUT = int(os.environ.get("REQ_TIMEOUT", "3600"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
MANIFEST = pathlib.Path(os.environ["ERQA_MANIFEST"])


def dataset():
    with MANIFEST.open() as f:
        rows = [json.loads(x) for x in f]
    if len(rows) != 400 or len({r["question_id"] for r in rows}) != 400:
        raise RuntimeError("invalid ERQA manifest")
    return rows


def interleave(row):
    """Exact port of eval_harness.py content assembly: returns list of str | image-dict."""
    q = row["question"]
    imgs = row["images"]
    vis = row["visual_indices"]
    pairs = sorted(zip(imgs, vis), key=lambda x: x[1]) if vis else []
    contents = []
    if len(vis) == 0:
        contents = [*imgs, q]
    elif all(i == 0 for i in vis):
        contents = [im for im, _ in pairs] + [q]
    else:
        last = 0
        for im, idx in pairs:
            if idx == 0:
                contents.append(im)
            elif idx <= len(q):
                seg = q[last:idx]
                if seg:
                    contents.append(seg)
                contents.append(im)
                last = idx
            else:
                contents.append(im)
        if last < len(q):
            contents.append(q[last:])
        if not contents:
            contents = [q] + [im for im, _ in pairs]
    return contents


def to_openai(contents):
    out = []
    for item in contents:
        if isinstance(item, str):
            out.append({"type": "text", "text": item})
        else:
            b = pathlib.Path(item["path"]).read_bytes()
            out.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{item['mime_type']};base64,"
                        + base64.b64encode(b).decode()
                    },
                }
            )
    return out


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
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def artifact(seed, qid):
    return OUT / ARM / f"seed{seed}" / f"{int(qid):04d}.json"


def valid(path):
    try:
        r = json.loads(path.read_text())
        return (
            r.get("arm") == ARM and r.get("content") is not None and not r.get("error")
        )
    except Exception:
        return False


def run_one(job):
    seed, row = job
    path = artifact(seed, row["question_id"])
    if path.exists() and valid(path):
        return "cached", seed, row["question_id"]
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": to_openai(interleave(row))}],
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": MAXTOK,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"},
        "mm_processor_kwargs": {"min_pixels": 602112, "max_pixels": 4014080},
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
                "answer": row["answer"],
                "question_type": row["question_type"],
                "num_images": len(row["images"]),
                "image_sha256": [i["sha256"] for i in row["images"]],
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
    tests = dataset()
    limit = int(os.environ.get("ERQA_LIMIT", "0"))
    tests = tests[:limit] if limit else tests
    jobs = [(s, r) for s in SEEDS for r in tests]
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
            if finished % 50 == 0 or finished == len(pending):
                print(
                    f"[{ARM}] {finished}/{len(pending)} new; errors={errors}; {finished/max(time.time()-started,1)*60:.1f}/min",
                    flush=True,
                )
    if errors:
        for x in details[:20]:
            print("ERROR", x, flush=True)
        raise RuntimeError(f"{errors} request errors; rerun resumes artifacts")
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
