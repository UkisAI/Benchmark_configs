"""DeepSWE v1.1: 113 tasks x 3 attempts with Pier 0.3.1 + mini-swe-agent 2.4.6, separate verifier. Needs setup.sh."""
import subprocess, time
from urllib.parse import urlparse
from common import UPSTREAM
from agents import flags, launch, KEY, THINKING

a = flags("deepswe")

# The agent runs in Docker behind Pier's egress proxy, which only allows ports 80/443 and can't see the host's 127.0.0.1.
# For the length of the run, a socat container serves --api-base on port 80 of a private Docker network's host address.
u, docker = urlparse(a.api_base), lambda *args: subprocess.run(["docker", *args], check=True, capture_output=True, text=True).stdout.strip()
subprocess.run(["docker", "rm", "-f", "deepswe-model-proxy"], capture_output=True)
subprocess.run(["docker", "network", "create", "deepswe-model"], capture_output=True)
bridge = docker("network", "inspect", "deepswe-model", "-f", "{{(index .IPAM.Config 0).Gateway}}")
docker("run", "-d", "--name", "deepswe-model-proxy", "--network", "host", "alpine/socat",
       f"TCP-LISTEN:80,fork,reuseaddr,bind={bridge}", f"TCP:{u.hostname}:{u.port or 80}")
time.sleep(2)
assert docker("inspect", "-f", "{{.State.Running}}", "deepswe-model-proxy") == "true", docker("logs", "deepswe-model-proxy")
try:
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
                "extra_body": {"chat_template_kwargs": THINKING}}},
            "env": {"OPENAI_API_BASE": f"http://{bridge}{u.path}", "OPENAI_API_KEY": KEY},
        }],
        "datasets": [{"path": str((UPSTREAM / "deep-swe/tasks").resolve())}],
    }, ["pier", "run", "-y", "-c"], total=113 * 3)
finally:
    docker("rm", "-f", "deepswe-model-proxy")
    docker("network", "rm", "deepswe-model")
