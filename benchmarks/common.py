"""Shared by every single-turn benchmark: flags, one chat call, a resumable run loop, a score summary."""
import argparse, json, os, re, statistics, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Qwen3.8 thinking profile used for every published number.
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0)
THINKING = {"enable_thinking": True, "reasoning_effort": "xhigh"}
UPSTREAM = Path(__file__).parent / "upstream"

# Exact upstream revisions live in pins.env (git-ignored), never in this repo. See pins.example.env.
for line in (Path(__file__).parent / "pins.env").read_text().splitlines() if (Path(__file__).parent / "pins.env").exists() else []:
    if "=" in line and not line.startswith("#"):
        os.environ.setdefault(*line.split("=", 1))
PIN = lambda name: os.environ.get(name) or None


def flags(name, max_tokens, conc=64, timeout=10800):
    p = argparse.ArgumentParser(description=f"Run {name}.")
    p.add_argument("--api-base", default=os.environ.get("API_BASE", "http://127.0.0.1:8000/v1"))
    p.add_argument("--api", choices=["openai", "anthropic"], default=os.environ.get("API", "openai"), help="endpoint style")
    p.add_argument("--model", default=os.environ.get("MODEL"), required=not os.environ.get("MODEL"))
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--conc", type=int, default=conc, help="parallel requests")
    p.add_argument("--max-tokens", type=int, default=max_tokens)
    p.add_argument("--timeout", type=int, default=timeout, help="seconds per request")
    p.add_argument("--limit", type=int, default=0, help="first N questions only (smoke test)")
    p.add_argument("--out", help="default: benchmarks/runs/<benchmark>-<model>")
    a = p.parse_args()
    a.name, a.seeds = name, [int(s) for s in a.seeds.split(",")]
    a.out = Path(a.out or Path(__file__).parent / f"runs/{name}-{a.model}")
    return a


def chat(a, messages, seed, thinking=True, **extra):
    """One request to an OpenAI- or Anthropic-compatible endpoint. Returns content, reasoning, tokens, or error."""
    t = time.time()
    try:
        return dict((anthropic if a.api == "anthropic" else openai)(a, messages, seed, thinking, **extra), seconds=time.time() - t)
    except Exception as e:  # kept and scored wrong; a rerun asks again
        return dict(content="", reasoning="", finish_reason="error", error=f"{type(e).__name__}: {e}", seconds=time.time() - t)


def post(a, path, body, headers):
    req = urllib.request.Request(a.api_base.rstrip("/") + path, json.dumps(body).encode(), {"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=a.timeout) as f:
        return json.load(f)


def openai(a, messages, seed, thinking, **extra):
    body = dict(model=a.model, messages=messages, max_tokens=a.max_tokens, seed=seed, **SAMPLING, **extra)
    if thinking:
        body["chat_template_kwargs"] = THINKING
    r = post(a, "/chat/completions", body, {"Authorization": "Bearer " + os.environ.get("API_KEY", "none")})
    c, u = r["choices"][0], r.get("usage") or {}
    return dict(content=c["message"].get("content") or "", reasoning=c["message"].get("reasoning") or c["message"].get("reasoning_content") or "",
                finish_reason=c.get("finish_reason"), prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"))


def anthropic(a, messages, seed, thinking, **extra):
    def block(b):  # OpenAI image_url data URL -> Anthropic base64 image
        if b.get("type") != "image_url":
            return b
        head, data = b["image_url"]["url"].split(",", 1)
        return {"type": "image", "source": {"type": "base64", "media_type": head[5:].split(";")[0], "data": data}}
    msgs = [dict(m, content=[block(b) for b in m["content"]]) if isinstance(m["content"], list) else m for m in messages if m["role"] != "system"]
    body = dict(model=a.model, messages=msgs, max_tokens=a.max_tokens, temperature=1.0)
    system = [m["content"] for m in messages if m["role"] == "system"]
    if system:
        body["system"] = system[0]
    if thinking:
        body["thinking"] = {"type": "enabled", "budget_tokens": a.max_tokens - 1024}
    r = post(a, "/messages", body, {"x-api-key": os.environ.get("API_KEY", "none"), "anthropic-version": "2023-06-01"})
    u = r.get("usage") or {}
    return dict(content="".join(b.get("text", "") for b in r["content"] if b["type"] == "text"),
                reasoning="".join(b.get("thinking", "") for b in r["content"] if b["type"] == "thinking"),
                finish_reason="length" if r.get("stop_reason") == "max_tokens" else r.get("stop_reason"),
                prompt_tokens=u.get("input_tokens"), completion_tokens=u.get("output_tokens"))


def run(a, jobs, ask):
    """jobs: {key: job}. ask(job) -> dict. One JSON per key under out/raw.
    Rerun to resume: finished answers are kept, request errors (timeouts, crashes) are asked again."""
    raw = a.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    todo = [k for k in jobs if not (raw / f"{k}.json").exists() or "error" in json.loads((raw / f"{k}.json").read_text())]
    done, errors, start = len(jobs) - len(todo), 0, time.time()
    print(f"{a.name}: {len(jobs)} total, {done} cached, {len(todo)} to run, conc={a.conc}", flush=True)

    def one(k):
        x = dict(key=k, **ask(jobs[k]))
        tmp = raw / f"{k}.tmp"
        tmp.write_text(json.dumps(x, ensure_ascii=False))
        tmp.replace(raw / f"{k}.json")
        return x

    with ThreadPoolExecutor(a.conc) as pool:
        for n, f in enumerate(as_completed(pool.submit(one, k) for k in todo), 1):
            errors += "error" in f.result()
            rate = n / max(time.time() - start, 1) * 60
            status = dict(benchmark=a.name, done=done + n, total=len(jobs), errors=errors,
                          per_min=round(rate, 1), eta_min=round((len(todo) - n) / max(rate, 0.01)), at=time.time())
            (a.out / "status.json").write_text(json.dumps(status))
            if n % 25 == 0 or n == len(todo):
                print(json.dumps(status), flush=True)
    return [json.loads((raw / f"{k}.json").read_text()) for k in jobs]


def ok(x):
    """A response only counts if it finished cleanly with visible content."""
    return not x.get("error") and x.get("finish_reason") != "length" and x.get("content", "").strip() != ""


def report(a, rows, label=None):
    """rows need seed + correct. Prints and saves per-seed and mean accuracy."""
    label = label or a.name
    seeds = sorted({x["seed"] for x in rows})
    per = {s: 100 * sum(x["correct"] for x in rows if x["seed"] == s) / max(1, sum(x["seed"] == s for x in rows)) for s in seeds}
    score = dict(benchmark=label, model=a.model, mean=sum(per.values()) / len(per), per_seed=per, n=len(rows),
                 errors=sum(bool(x.get("error")) for x in rows), truncated=sum(x.get("finish_reason") == "length" for x in rows))
    tokens = [x["completion_tokens"] for x in rows if x.get("completion_tokens")]
    if tokens:  # thinking + answer tokens, for the "thinking vs base" columns
        score.update(mean_output_tokens=statistics.mean(tokens), median_output_tokens=statistics.median(tokens))
    (a.out / f"score_{label}.json").write_text(json.dumps(score, indent=2))
    print(json.dumps(score, indent=2))


def last(pattern, text):
    m = re.findall(pattern, text or "")
    return m[-1] if m else None
