"""Shared by the agent benchmarks: flags, then run a Harbor-style job config and keep status.json fresh."""
import argparse, json, os, subprocess, time
from pathlib import Path


def flags(name, **variants):
    p = argparse.ArgumentParser(description=f"Run {name}.")
    p.add_argument("--api-base", default=os.environ.get("API_BASE", "http://127.0.0.1:8000/v1"))
    p.add_argument("--model", default=os.environ.get("MODEL"), required=not os.environ.get("MODEL"))
    p.add_argument("--api-key", default=os.environ.get("API_KEY", "none"))
    p.add_argument("--conc", type=int, help="parallel trials (default depends on variant)")
    p.add_argument("--out", help="default: benchmarks/runs/<benchmark>-<model>")
    if variants:
        p.add_argument("--variant", choices=list(variants), default=next(iter(variants)),
                       help="; ".join(f"{k}: {v}" for k, v in variants.items()))
    a = p.parse_args()
    a.name, a.out = name, Path(a.out or Path(__file__).parent / f"runs/{name}-{a.model}").resolve()
    return a


def launch(a, config, cli, total):
    """Write the job config, run it, and write status.json every minute until it exits."""
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
    raise SystemExit(proc.returncode)
