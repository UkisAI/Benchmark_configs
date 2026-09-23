#!/usr/bin/env python3
"""Score one complete C-Eval, MMLU-Pro, or ERQA response bank."""

import argparse
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = {"ceval_validation", "mmlu_pro", "erqa"}


def predicted(benchmark, content):
    if benchmark == "ceval_validation":
        matches = re.findall(r"答案\s*[:：]\s*([A-D])\b", content)
        return matches[-1] if matches else None
    if benchmark == "mmlu_pro":
        matches = re.findall(
            r"the answer is\s*\(?([A-J])\)?", content, flags=re.IGNORECASE
        )
        if matches:
            return matches[-1].upper()
        matches = re.findall(r"answer\s*:\s*([A-J])\b", content, flags=re.IGNORECASE)
        return matches[-1].upper() if matches else None
    return content.replace(".", "").strip().lower()


def gold_answer(benchmark, row):
    answer = str(row["answer"])
    return answer.lower() if benchmark == "erqa" else answer.upper()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), required=True)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="run directory holding arm/seedN artifacts",
    )
    parser.add_argument("--arm", required=True)
    parser.add_argument(
        "--tokenizer", required=True, help="the served model's tokenizer path or ID"
    )
    args = parser.parse_args()
    config = json.loads(
        (ROOT / "benchmarks" / f"{args.benchmark}.json").read_text(encoding="utf-8")
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    expected = config["expected_questions"]
    seeds = config["seed_policy"]["seeds"]
    per_seed = {}
    thinking = []
    completion = []
    cap_hits = 0
    for seed in seeds:
        paths = sorted((args.out / args.arm / f"seed{seed}").glob("*.json"))
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        ids = {str(row["question_id"]) for row in rows}
        if len(rows) != expected or len(ids) != expected:
            raise RuntimeError(
                f"{args.arm} seed{seed}: {len(rows)} rows, {len(ids)} unique IDs; expected {expected}"
            )
        if any(
            row.get("error") or row.get("arm") != args.arm or row.get("seed") != seed
            for row in rows
        ):
            raise RuntimeError(f"{args.arm} seed{seed}: error or mismatched artifact")
        correct = 0
        for row in rows:
            content = row.get("content") or ""
            correct += predicted(args.benchmark, content) == gold_answer(
                args.benchmark, row
            )
            thinking.append(
                len(
                    tokenizer.encode(
                        row.get("reasoning") or "", add_special_tokens=False
                    )
                )
            )
            completion.append(int(row.get("completion_tokens") or 0))
            cap_hits += row.get("finish_reason") == "length"
        per_seed[str(seed)] = {
            "correct": correct,
            "total": expected,
            "accuracy": correct / expected,
        }
    summary = {
        "benchmark": args.benchmark,
        "arm": args.arm,
        "questions": expected,
        "generations": expected * len(seeds),
        "per_seed": per_seed,
        "accuracy": statistics.mean(item["accuracy"] for item in per_seed.values()),
        "mean_thinking_tokens": statistics.mean(thinking),
        "median_thinking_tokens": statistics.median(thinking),
        "mean_completion_tokens": statistics.mean(completion),
        "cap_hits": cap_hits,
        "tokenizer": args.tokenizer,
    }
    target = args.out / f"{args.arm}_{args.benchmark}_score.json"
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
