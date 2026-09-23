"""DeepSWE v1.1: 113 tasks x 3 attempts with Pier 0.3.1 + mini-swe-agent 2.4.6, separate verifier. Needs setup.sh."""
from common import UPSTREAM
from agents import flags, launch

a = flags("deepswe")
launch(a, {
    "n_attempts": 3,
    "timeout_multiplier": 1,
    "n_concurrent_trials": a.conc or 16,
    "quiet": True,
    "retry": {"max_retries": 0},
    "environment": {"type": "docker", "force_build": False, "delete": True},
    "verifier": {"disable": False},
    "agents": [{
        "name": "mini-swe-agent",
        "model_name": f"openai/{a.model}",
        "kwargs": {"version": "2.4.6", "model_class": "litellm", "cost_limit": 0, "model_kwargs": {
            "temperature": 1, "top_p": 0.95, "timeout": 3600,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"}}}},
        "env": {"OPENAI_API_BASE": a.api_base, "OPENAI_API_KEY": a.api_key},
    }],
    "datasets": [{"path": str((UPSTREAM / "deep-swe/tasks").resolve())}],
}, ["pier", "run", "-y", "-c"], total=113 * 3)
