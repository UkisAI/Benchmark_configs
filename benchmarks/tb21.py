"""Terminal-Bench 2.1: 89 tasks x 5 attempts with Harbor 0.20.0 + Terminus 2."""
from agents import flags, launch, KEY, PIN

a = flags("tb21", **{"5h": "Qwen-27B protocol: 5h per task, 89 parallel",
                     "native": "Flash-Next protocol: each task's own timeout, 8 parallel"})
conc = a.conc or (89 if a.variant == "5h" else 8)
agent = {
    "name": "terminus-2",
    "model_name": f"openai/{a.model}",
    "n_concurrent": conc,
    "kwargs": {
        "api_base": a.api_base, "parser_name": "json",
        "temperature": 1.0, "interleaved_thinking": True, "store_all_messages": True,
        "model_info": {"max_input_tokens": 131072, "max_output_tokens": 131072},
        "llm_kwargs": {"timeout": 3600, "api_key": KEY}, "llm_call_kwargs": {"top_p": 1.0},
    },
}
if a.variant == "5h":
    agent["override_timeout_sec"] = 18000
launch(a, {
    "n_attempts": 5,
    "n_concurrent_trials": conc,
    "agents": [agent],
    "datasets": [{"name": "terminal-bench/terminal-bench-2-1", **({"ref": PIN("TB21_REF")} if PIN("TB21_REF") else {})}],
}, ["harbor", "run", "-c"], total=89 * 5)
