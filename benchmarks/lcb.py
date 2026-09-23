"""LiveCodeBench release_v6: 1,055 problems x 5 seeds, 100k tokens, graded by the official test runner. Needs setup.sh."""
import sys
from common import flags, chat, run, report, UPSTREAM

sys.path.insert(0, str(UPSTREAM / "LiveCodeBench"))  # only the light parts: data, prompt, code extraction, test runner
from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.prompts.code_generation import PromptConstants, get_generic_question_template_answer
from lcb_runner.utils.extraction_utils import extract_code
from lcb_runner.lm_styles import LMStyle
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
from lcb_runner.evaluation.pass_k_utils import extract_instance_results

if __name__ == "__main__":
    a = flags("lcb", max_tokens=100_000, conc=96, timeout=7200)
    problems = sorted(load_code_generation_dataset("release_v6"), key=lambda p: str(p.question_id))
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
