#!/usr/bin/env python3
"""Score a complete IFBench response bank with the pinned official evaluator."""

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ifbench", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--harness-revision", required=True)
    args = parser.parse_args()
    config = json.loads(
        (ROOT / "benchmarks" / "ifbench.json").read_text(encoding="utf-8")
    )
    head = subprocess.check_output(
        ["git", "-C", str(args.ifbench), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != args.harness_revision:
        raise RuntimeError(
            "IFBench checkout revision does not match --harness-revision"
        )
    data = args.ifbench / config["dataset"]["file"]
    raw = args.outdir / f"{args.tag}_raw.jsonl"
    rows = [
        json.loads(line)
        for line in raw.read_text(encoding="utf-8").splitlines()
        if line
    ]
    completed = {
        (row["key"], row["sample_index"]): row
        for row in rows
        if row.get("response") is not None
    }
    if len(completed) != 1500:
        raise RuntimeError(
            f"IFBench bank incomplete: {len(completed)}/1500 completed responses"
        )
    score_dir = args.outdir / f"{args.tag}_official_scores"
    score_dir.mkdir(exist_ok=True)
    for seed in range(5):
        responses = args.outdir / f"{args.tag}_responses_s{seed}.jsonl"
        if (
            sum(
                bool(line)
                for line in responses.read_text(encoding="utf-8").splitlines()
            )
            != 300
        ):
            raise RuntimeError(f"incomplete IFBench response file: {responses}")
        subprocess.run(
            [
                str(args.python),
                str(args.ifbench / "run_eval.py"),
                "--input_data",
                str(data),
                "--input_response_data",
                str(responses),
                "--output_dir",
                str(score_dir),
            ],
            cwd=args.ifbench,
            check=True,
        )
    summary = {"benchmark": "ifbench", "tag": args.tag, "questions": 300, "seeds": 5}
    for mode in ("strict", "loose"):
        files = sorted(
            score_dir.glob(f"{args.tag}_responses_s*-eval_results_{mode}.jsonl")
        )
        if len(files) != 5:
            raise RuntimeError(f"missing official {mode} files: {files}")
        scores = []
        for path in files:
            graded = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            if len(graded) != 300:
                raise RuntimeError(f"{path}: expected 300 grades, found {len(graded)}")
            scores.append(
                sum(bool(row["follow_all_instructions"]) for row in graded) / 300
            )
        summary[mode] = sum(scores) / len(scores)
        summary[f"{mode}_per_seed"] = scores
    target = args.outdir / f"{args.tag}_score.json"
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
