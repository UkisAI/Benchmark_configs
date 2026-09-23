#!/usr/bin/env python3
"""Launch the pinned MathArena AIME/HMMT profile used on Tesla 1."""

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPETITIONS = {
    "aime_2024": (
        ("aime/aime_2024_I", "MathArena/aime_2024_I"),
        ("aime/aime_2024_II", "MathArena/aime_2024_II"),
    ),
    "aime_2025": (("aime/aime_2025", "MathArena/aime_2025"),),
    "hmmt_nov_2025": (("hmmt/hmmt_nov_2025", "MathArena/hmmt_nov_2025"),),
}


def checked_head(path, expected):
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise RuntimeError(f"MathArena checkout is {actual}; expected {expected}")


def model_yaml(model, api_base, seed, cap, concurrency, arm):
    # JSON-quoted scalars are valid YAML. All other values match Tesla 1's HMMT configs.
    return "\n".join(
        [
            f"model: {json.dumps(model)}",
            "api: custom",
            f"base_url: {json.dumps(api_base)}",
            f"human_readable_id: {json.dumps(f'{arm} seed{seed}')}",
            f"max_tokens: {cap}",
            "temperature: 1.0",
            "top_p: 0.95",
            "presence_penalty: 0.0",
            f"seed: {seed}",
            "extra_body:",
            "  top_k: 20",
            "  min_p: 0.0",
            "  repetition_penalty: 1.0",
            "  chat_template_kwargs:",
            "    enable_thinking: true",
            "    reasoning_effort: xhigh",
            f"concurrent_requests: {concurrency}",
            "timeout: 10800",
            "max_retries_inner: 2",
            "sleep_after_request: 0",
            "read_cost: 0",
            "write_cost: 0",
            "",
        ]
    )


def main(benchmark):
    parser = argparse.ArgumentParser(
        description=f"Run {benchmark} through pinned MathArena"
    )
    parser.add_argument("--matharena", type=Path, required=True)
    parser.add_argument(
        "--model", required=True, help="served model ID from /v1/models"
    )
    parser.add_argument("--api-base", required=True, help="OpenAI-compatible /v1 URL")
    parser.add_argument("--arm", required=True, help="output label, e.g. base")
    parser.add_argument(
        "--out", type=Path, required=True, help="dedicated run directory"
    )
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--harness-revision", required=True)
    parser.add_argument(
        "--dataset-revision",
        action="append",
        required=True,
        help="dataset=revision; repeat for each competition dataset",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(
        (ROOT / "benchmarks" / f"{benchmark}.json").read_text(encoding="utf-8")
    )
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    if config["seed_policy"]["seeds"] != protocol["seeds"] or args.concurrency < 1:
        parser.error("seed policy mismatch or invalid concurrency")
    revisions = dict(item.split("=", 1) for item in args.dataset_revision)
    if any(not revision for revision in revisions.values()):
        parser.error("all dataset revisions must be supplied")
    specs = COMPETITIONS[benchmark]
    if set(revisions) != {dataset for _, dataset in specs}:
        parser.error("dataset revision list does not match competition list")
    checked_head(args.matharena, args.harness_revision)
    print(
        f"{benchmark}: {config['expected_questions']} questions x 5 seeds; output cap {config['max_output_tokens']}"
    )
    for competition, dataset in specs:
        print(f"  {competition} <- {dataset}@{revisions[dataset]}")
    if args.dry_run:
        return

    from huggingface_hub import snapshot_download

    out = args.out.resolve()
    manifest_path = out / "run_config.json"
    manifest = {
        "benchmark": benchmark,
        "model": args.model,
        "api_base": args.api_base,
        "arm": args.arm,
        "concurrency": args.concurrency,
        "harness_commit": args.harness_revision,
        "dataset_revisions": revisions,
        "config": config,
    }
    if (
        manifest_path.exists()
        and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest
    ):
        raise RuntimeError(f"run configuration changed: {manifest_path}")
    model_dir = out / "configs" / "models"
    comp_dir = out / "configs" / "competitions"
    model_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for seed in protocol["seeds"]:
        name = f"{args.arm}_s{seed}"
        names.append(name)
        (model_dir / f"{name}.yaml").write_text(
            model_yaml(
                args.model,
                args.api_base,
                seed,
                config["max_output_tokens"],
                args.concurrency,
                args.arm,
            ),
            encoding="utf-8",
        )
    for competition, dataset in specs:
        snapshot = Path(
            snapshot_download(
                repo_id=dataset, repo_type="dataset", revision=revisions[dataset]
            )
        )
        original = (
            args.matharena / "configs" / "competitions" / f"{competition}.yaml"
        ).read_text(encoding="utf-8")
        lines = original.splitlines()
        if sum(line.startswith("dataset_path:") for line in lines) != 1:
            raise RuntimeError(f"expected one dataset_path in {competition}")
        lines = [
            f"dataset_path: {json.dumps(str(snapshot))}"
            if line.startswith("dataset_path:")
            else line
            for line in lines
        ]
        target = comp_dir / f"{competition}.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    command = [
        str(args.matharena / ".venv" / "bin" / "python"),
        "scripts/run.py",
        "--comp",
        *(competition for competition, _ in specs),
        "--models",
        *names,
        "--n",
        "1",
        "--comp-configs-dir",
        str(comp_dir),
        "--model-configs-dir",
        str(model_dir),
        "--output-dir",
        str(out / "matharena_outputs"),
    ]
    subprocess.run(command, cwd=args.matharena, check=True)


if __name__ == "__main__":
    raise SystemExit("use aime_2024_run.py, aime_2025_run.py, or hmmt_nov_2025_run.py")
