"""LiveCodeBench release_v6: 1,055 problems x 5 seeds, 100k tokens, graded by the official evaluator. Needs setup.sh."""
import json, os, subprocess, sys
from pathlib import Path
from common import flags, chat, run, report, UPSTREAM

LCB = UPSTREAM / "LiveCodeBench"
if Path(sys.prefix).resolve() != (LCB / ".venv").resolve():  # lcb_runner only imports inside its own venv
    os.execv(LCB / ".venv/bin/python", [str(LCB / ".venv/bin/python"), __file__, *sys.argv[1:]])
sys.path.insert(0, str(LCB))
from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.prompts.code_generation import PromptConstants, get_generic_question_template_answer
from lcb_runner.utils.extraction_utils import extract_code
from lcb_runner.lm_styles import LMStyle

a = flags("lcb", max_tokens=100_000, conc=96, timeout=7200)
problems = sorted(load_code_generation_dataset("release_v6"), key=lambda p: str(p.question_id))
assert len(problems) == 1055, len(problems)
problems = problems[: a.limit or None]


def ask(job):
    seed, p = job
    x = chat(a, [{"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
                 {"role": "user", "content": get_generic_question_template_answer(p)}], seed)
    return dict(x, seed=seed, qid=str(p.question_id), code=extract_code(x["content"], LMStyle.OpenAIChat))


out = run(a, {f"s{s}_{p.question_id}": (s, p) for s in a.seeds for p in problems}, ask)
grades = {}
for s in a.seeds:  # official execution-based grading, one file per seed
    f = (a.out / f"codes_s{s}.json").resolve()
    f.write_text(json.dumps([{"question_id": x["qid"], "code_list": [x.get("code") or ""]} for x in out if x["seed"] == s]))
    subprocess.run([sys.executable, "-m", "lcb_runner.runner.custom_evaluator", "--model", "Qwen/Qwen3-235B-A22B",
                    "--scenario", "codegeneration", "--release_version", "release_v6", "--custom_output_file", str(f),
                    "--num_process_evaluate", "4", "--timeout", "10"], cwd=LCB, check=True)
    for g in json.loads(f.with_name(f.stem + "_codegeneration_output_eval_all.json").read_text()):
        grades[s, str(g["question_id"])] = bool(g["graded_list"][0])
report(a, [dict(x, correct=not x.get("error") and grades[x["seed"], x["qid"]]) for x in out])
