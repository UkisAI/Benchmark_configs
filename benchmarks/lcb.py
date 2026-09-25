"""LiveCodeBench release_v6: 1,055 problems x 5 seeds, 100k tokens, graded by the official test runner. Needs setup.sh."""
import json, os, sys
from huggingface_hub import hf_hub_download
from common import flags, chat, run, report, UPSTREAM

sys.path.insert(0, str(UPSTREAM / "LiveCodeBench"))  # only the light parts: data, prompt, code extraction, test runner
os.chdir(UPSTREAM / "LiveCodeBench")  # lcb_runner reads its prompt files relative to the working directory
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
from lcb_runner.prompts.code_generation import PromptConstants, get_generic_question_template_answer
from lcb_runner.utils.extraction_utils import extract_code
from lcb_runner.lm_styles import LMStyle
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
from lcb_runner.evaluation.pass_k_utils import extract_instance_results

if __name__ == "__main__":  # the grader starts worker processes that re-import this file
    a = flags("lcb", max_tokens=100_000, conc=96, timeout=7200)
    # release_v6 = the files the dataset's own loading script reads (datasets 4 no longer runs loading scripts)
    files = [hf_hub_download("livecodebench/code_generation_lite", f, repo_type="dataset") for f in ["test.jsonl"] + [f"test{i}.jsonl" for i in range(2, 7)]]
    problems = sorted((CodeGenerationProblem(**json.loads(l)) for f in files for l in open(f) if l.strip()), key=lambda p: str(p.question_id))
    assert len(problems) == 1055, len(problems)
    problems = problems[: a.limit or None]

    def ask(job):
        seed, i = job
        p = problems[i]
        x = chat(a, [{"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
                     {"role": "user", "content": get_generic_question_template_answer(p)}], seed)
        return dict(x, seed=seed, i=i, code=extract_code(x["content"], LMStyle.OpenAIChat))

    out = run(a, {f"s{s}_{p.question_id}": (s, i) for s in a.seeds for i, p in enumerate(problems)}, ask)
    rows = []
    for s in a.seeds:  # official execution-based grading, 10s per test
        mine = [x for x in out if x["seed"] == s]
        graded = extract_instance_results(codegen_metrics([problems[x["i"]].get_evaluation_sample() for x in mine],
                                                          [[x.get("code") or ""] for x in mine], num_process_evaluate=4, timeout=10)[1])
        rows += [dict(x, correct=not x.get("error") and g[0]) for x, g in zip(mine, graded)]
    report(a, rows)
