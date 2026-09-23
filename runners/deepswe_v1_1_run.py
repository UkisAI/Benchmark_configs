#!/usr/bin/env python3
"""Validate pinned DeepSWE tasks, then run the three-attempt Pier job."""

import argparse
import hashlib
import json
import os
import subprocess
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_tasks(tasks_dir, revision, checksums):
    task_files = sorted(tasks_dir.glob("*/task.toml"))
    if len(task_files) != 113:
        raise RuntimeError(f"expected 113 DeepSWE tasks, found {len(task_files)}")
    for path in task_files:
        task = tomllib.loads(path.read_text(encoding="utf-8"))
        verifier = task["verifier"]
        if (
            verifier.get("environment_mode") != "separate"
            or verifier.get("timeout_sec") != 1800
        ):
            raise RuntimeError(f"verifier isolation/timeout mismatch: {path}")
        if (
            len(verifier.get("collect", [])) != 1
            or verifier["collect"][0].get("timeout_sec") != 300
        ):
            raise RuntimeError(f"verifier collection mismatch: {path}")
        if task["agent"].get("timeout_sec") != 10800:
            raise RuntimeError(f"agent timeout mismatch: {path}")
    if checksums:
        expected = json.loads(checksums.read_text(encoding="utf-8"))
        files = {
            name: item["sha256"]
            for name, item in expected.items()
            if name.startswith("deep-swe/tasks/")
        }
        if not files:
            raise RuntimeError("no task file hashes in checksum manifest")
        for name, digest in files.items():
            path = tasks_dir / name.removeprefix("deep-swe/tasks/")
            if not path.is_file() or sha256(path) != digest:
                raise RuntimeError(f"task checksum mismatch: {path}")
    else:
        checkout = tasks_dir.parent
        head = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        if head != revision:
            raise RuntimeError(f"DeepSWE checkout is {head}; expected {revision}")
    return len(task_files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pier", type=Path, required=True, help="Pier 0.3.1 executable"
    )
    parser.add_argument(
        "--tasks-dir", type=Path, required=True, help="pinned deep-swe/tasks directory"
    )
    parser.add_argument(
        "--checksums",
        type=Path,
        help="source_checksums.json for an archive without .git",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--task-revision", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(
        (ROOT / "agents" / "deepswe_v1_1.harbor.template.json").read_text(
            encoding="utf-8"
        )
    )
    if not args.pier.is_file():
        parser.error(f"Pier executable missing: {args.pier}")
    count = verify_tasks(args.tasks_dir, args.task_revision, args.checksums)
    python = args.pier.with_name("python")
    versions = subprocess.check_output(
        [
            str(python),
            "-c",
            "import importlib.metadata as m; print(m.version('datacurve-pier')); print(m.version('mini-swe-agent'))",
        ],
        text=True,
    ).splitlines()
    if versions != ["0.3.1", "2.4.6"]:
        raise RuntimeError(f"unexpected Pier/mini-swe-agent versions: {versions}")
    if config["n_attempts"] != 3 or config["n_concurrent_trials"] != 16:
        raise RuntimeError("DeepSWE profile mismatch")
    config["job_name"] = args.job_name
    config["jobs_dir"] = str(args.out.resolve() / "jobs")
    config["datasets"][0]["path"] = str(args.tasks_dir.resolve())
    agent = config["agents"][0]
    agent["model_name"] = "openai/" + args.model
    agent["kwargs"]["version"] = "2.4.6"
    agent["env"] = {
        "OPENAI_API_BASE": args.api_base,
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "local-evaluation-no-auth"),
    }
    print(
        f"DeepSWE v1.1: {count} tasks x 3 attempts; Pier {versions[0]}, mini-swe-agent {versions[1]}"
    )
    if args.dry_run:
        return
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    target = out / "pier_job.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != config:
        raise RuntimeError(f"existing job config differs: {target}")
    target.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    subprocess.run(
        [str(args.pier), "run", "--config", str(target), "--yes"], check=True
    )


if __name__ == "__main__":
    main()
