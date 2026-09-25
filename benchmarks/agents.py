"""Shared by the agent benchmarks: run a Harbor-style job config, keep status.json fresh, write the score."""
import json, subprocess, time
from common import parser, parse, KEY, PIN, THINKING  # noqa: F401  (re-exported for tb21.py / deepswe.py)


def flags(name, **variants):
    p = parser(name, conc=None)
    if variants:
        p.add_argument("--variant", choices=list(variants), default=next(iter(variants)),
                       help="; ".join(f"{k}: {v}" for k, v in variants.items()))
    return parse(p, name)


def launch(a, config, cli, total):
    """Write the job config, run it, write status.json every minute, then score_<benchmark>.json."""
    a.out.mkdir(parents=True, exist_ok=True)
    config.update(job_name=a.name, jobs_dir=str(a.out))
    (a.out / "config.json").write_text(json.dumps(config, indent=2))
    proc = subprocess.Popen(cli + [str(a.out / "config.json")], stdout=open(a.out / "run.log", "a"), stderr=subprocess.STDOUT)
    print(f"{a.name}: running, log {a.out / 'run.log'}", flush=True)
    while proc.poll() is None:
        done = len(list((a.out / a.name).glob("*/result.json")))
        (a.out / "status.json").write_text(json.dumps(dict(benchmark=a.name, done=done, total=total, pid=proc.pid, at=time.time())))
        time.sleep(60)
    print(f"{a.name}: exited {proc.returncode}; results in {a.out / a.name}", flush=True)
    score(a, total)
    raise SystemExit(proc.returncode)


def score(a, total):
    """Mean reward over all `total` trials (missing or crashed = 0), next to Harbor's own job metrics."""
    job = a.out / a.name
    trials = [json.loads(p.read_text()) for p in job.glob("*/result.json")]
    rewards = [((t.get("verifier_result") or {}).get("rewards") or {}).get("reward", 0) for t in trials]
    stats = json.loads((job / "result.json").read_text()).get("stats", {}) if (job / "result.json").exists() else {}
    s = dict(benchmark=a.name, model=a.model, mean=100 * sum(rewards) / total, n=len(trials), total=total,
             errors=sum(bool(t.get("exception_info")) for t in trials),
             harbor_metrics={k: v.get("metrics") for k, v in stats.get("evals", {}).items()})
    (a.out / f"score_{a.name}.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))
