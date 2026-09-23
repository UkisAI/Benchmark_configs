#!/usr/bin/env python3
"""Generate and officially grade the full 1,055-question LCB release_v6."""

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = 1055
SEEDS = range(5)


def checked_head(checkout, expected):
    head = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != expected:
        raise RuntimeError(f"LiveCodeBench checkout is {head}; expected {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lcb", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=72)
    parser.add_argument("--grader-processes", type=int, default=24)
    parser.add_argument("--harness-revision", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(
        (ROOT / "benchmarks" / "livecodebench_v6_full.json").read_text(encoding="utf-8")
    )
    checked_head(args.lcb, args.harness_revision)
    if (
        config["expected_questions"] != EXPECTED
        or config["max_output_tokens"] != 100000
    ):
        raise RuntimeError("LiveCodeBench profile mismatch")
    if args.concurrency < 1 or args.grader_processes < 1:
        parser.error("concurrency and grader processes must be positive")
    print(
        f"LCB release_v6 full: {EXPECTED} questions x 5 seeds; cap 100000; arm {args.arm}"
    )
    if args.dry_run:
        return

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "run_config.json"
    manifest = {
        "benchmark": config["id"],
        "model": args.model,
        "api_base": args.api_base,
        "arm": args.arm,
        "concurrency": args.concurrency,
        "harness_revision": args.harness_revision,
        "dataset_revision": args.dataset_revision,
        "config": config,
    }
    if (
        manifest_path.exists()
        and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest
    ):
        raise RuntimeError(f"run configuration changed: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    python = args.lcb / ".venv" / "bin" / "python"
    env = dict(
        os.environ,
        API_BASE=args.api_base,
        MODEL=args.model,
        ARM=args.arm,
        OUT=str(out),
        CONC=str(args.concurrency),
        MAX_TOKENS="100000",
        SEEDS="0,1,2,3,4",
        LCB_VERSION="release_v6",
        LCB_EXPECTED="1055",
        LCB_CHECKOUT=str(args.lcb),
        LCB_DATASET_REVISION=args.dataset_revision,
    )
    with (out / f"{args.arm}_generation.log").open("a", encoding="utf-8") as log:
        subprocess.run(
            [str(python), str(Path(__file__).with_name("lcb_seeded.py"))],
            cwd=args.lcb,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )

    grade_dir = out / "official_grades"
    grade_dir.mkdir(exist_ok=True)
    scores = {}
    completion = []
    for seed in SEEDS:
        paths = sorted((out / args.arm / f"seed{seed}").glob("*.json"))
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        ids = {str(row["question_id"]) for row in rows}
        if (
            len(rows) != EXPECTED
            or len(ids) != EXPECTED
            or any(row.get("error") for row in rows)
        ):
            raise RuntimeError(
                f"seed{seed}: expected {EXPECTED} unique complete responses; got {len(rows)}"
            )
        custom = grade_dir / f"{args.arm}_seed{seed}.json"
        custom.write_text(
            json.dumps(
                [
                    {
                        "question_id": row["question_id"],
                        "code_list": [row.get("code") or ""],
                    }
                    for row in rows
                ]
            ),
            encoding="utf-8",
        )
        graded_path = custom.with_name(
            custom.stem + "_codegeneration_output_eval_all.json"
        )
        if not graded_path.exists():
            command = [
                str(python),
                str(Path(__file__).with_name("lcb_official_grader.py")),
                "--model",
                "Qwen/Qwen3-235B-A22B",
                "--scenario",
                "codegeneration",
                "--release_version",
                "release_v6",
                "--custom_output_file",
                str(custom),
                "--num_process_evaluate",
                str(args.grader_processes),
                "--timeout",
                "10",
            ]
            with (grade_dir / f"{args.arm}_seed{seed}.log").open(
                "w", encoding="utf-8"
            ) as log:
                subprocess.run(
                    command,
                    cwd=args.lcb,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        graded = json.loads(graded_path.read_text(encoding="utf-8"))
        if (
            len(graded) != EXPECTED
            or {str(row["question_id"]) for row in graded} != ids
        ):
            raise RuntimeError(f"seed{seed}: official grade count/IDs mismatch")
        scores[str(seed)] = (
            sum(bool(row["graded_list"][0]) for row in graded) / EXPECTED
        )
        completion.extend(int(row["completion_tokens"]) for row in rows)
    summary = {
        "benchmark": config["id"],
        "arm": args.arm,
        "questions": EXPECTED,
        "generations": EXPECTED * len(SEEDS),
        "per_seed_accuracy": scores,
        "accuracy": statistics.mean(scores),
        "mean_completion_tokens": statistics.mean(completion),
    }
    (out / f"{args.arm}_score.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
