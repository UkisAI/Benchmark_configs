#!/usr/bin/env python3
"""Render and launch one of the pinned Terminal-Bench 2.1 Harbor profiles."""

import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "flash_next_native": "terminal_bench_2_1.flash_next_native.harbor.json",
    "qwen27b_fixed5h": "terminal_bench_2_1.qwen27b_fixed5h.harbor.json",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--harbor", type=Path, required=True)
    parser.add_argument("--model", required=True, help="served model ID")
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--dataset-ref", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.harbor.is_file():
        parser.error(f"Harbor executable missing: {args.harbor}")
    source = ROOT / "agents" / PROFILES[args.profile]
    config = json.loads(source.read_text(encoding="utf-8"))
    meta = json.loads(
        source.with_name(source.name.replace(".harbor.json", ".meta.json")).read_text(
            encoding="utf-8"
        )
    )
    if config["n_attempts"] != 5 or meta["task_count"] != 89:
        raise RuntimeError("Terminal-Bench profile count mismatch")
    if config["datasets"][0]["ref"] != meta["dataset"]["ref"]:
        raise RuntimeError("Terminal-Bench dataset ref mismatch")
    config["datasets"][0]["ref"] = args.dataset_ref
    agent = config["agents"][0]
    if args.profile == "flash_next_native" and "override_timeout_sec" in agent:
        raise RuntimeError("Flash-Next must use native task timeouts")
    if args.profile == "qwen27b_fixed5h" and agent.get("override_timeout_sec") != 18000:
        raise RuntimeError("Qwen-27B profile requires the 18000-second override")
    config["job_name"] = args.job_name
    config["jobs_dir"] = str(args.out.resolve() / "jobs")
    agent["model_name"] = "openai/" + args.model
    agent["kwargs"]["api_base"] = args.api_base
    agent["kwargs"]["api_key"] = os.environ.get(
        "OPENAI_API_KEY", "local-evaluation-no-auth"
    )
    print(
        f"Terminal-Bench 2.1 {args.profile}: 89 tasks x 5 attempts; concurrency {config['n_concurrent_trials']}"
    )
    if args.dry_run:
        return
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    target = out / "harbor_job.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != config:
        raise RuntimeError(f"existing job config differs: {target}")
    target.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    subprocess.run(
        [str(args.harbor), "run", "--config", str(target), "--yes"], check=True
    )


if __name__ == "__main__":
    main()
