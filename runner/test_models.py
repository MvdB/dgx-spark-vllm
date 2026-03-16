#!/usr/bin/env python3
"""
Smoke-test runner for all compatible vLLM models on DGX Spark.

For each model:
  1. Starts vllm_spark.sh (docker container)
  2. Waits until the API is ready
  3. Sends a few test queries
  4. Stops the container
  5. Writes results to runner/model_status.md and pushes to git

Usage:
  python3 test_models.py              # test all compatible models
  python3 test_models.py qwen3.5-9b   # test one model by pattern
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
REPO_DIR      = Path(__file__).parent.parent
RUNNER_DIR    = Path(__file__).parent
VLLM_SCRIPT   = Path.home() / "vllm_spark.sh"        # production copy
HF_MODELS_DIR = Path(os.environ.get("HF_MODELS_DIR", Path.home() / "hf_models"))
STATUS_FILE   = RUNNER_DIR / "model_status.md"

# ── Runtime config ─────────────────────────────────────────────────────────
HOST_PORT        = int(os.environ.get("HOST_PORT", "8000"))
BASE_URL         = f"http://127.0.0.1:{HOST_PORT}"
CONTAINER        = os.environ.get("CONTAINER_NAME", "vllm-server")
VLLM_TAG         = os.environ.get("DEFAULT_VLLM_TAG", "v0.17.1")
STARTUP_TIMEOUT  = int(os.environ.get("STARTUP_TIMEOUT", "600"))   # 10 min
POLL_INTERVAL    = 10   # seconds between readiness checks
QUERY_TIMEOUT    = 120  # seconds per query

# ── Test queries ───────────────────────────────────────────────────────────
TEST_QUERIES = [
    {
        "label": "sanity",
        "messages": [{"role": "user", "content": "Reply with exactly one word: Ready"}],
        "max_tokens": 10,
    },
    {
        "label": "math",
        "messages": [{"role": "user", "content": "What is 7 × 8? Reply with just the number."}],
        "max_tokens": 10,
    },
    {
        "label": "short-text",
        "messages": [{"role": "user", "content": "Name the capital of Japan in one word."}],
        "max_tokens": 10,
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_compatible_models() -> list[str]:
    models = []
    for d in sorted(HF_MODELS_DIR.iterdir()):
        if not d.is_dir() or not (d / "config.json").exists():
            continue
        profile = d / "vllm_profile.conf"
        if not profile.exists():
            continue
        compat = "1"
        for line in profile.read_text().splitlines():
            if line.startswith("PROFILE_VLLM_COMPATIBLE="):
                compat = line.split("=", 1)[1].strip().strip("'\"")
        if compat == "0":
            continue
        models.append(d.name)
    return models


def docker_rm() -> None:
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)


def start_model(model_name: str) -> tuple[bool, str]:
    """Start vllm_spark.sh for the given model. Returns (ok, stderr)."""
    docker_rm()
    proc = subprocess.run(
        [str(VLLM_SCRIPT), "--model", model_name, "--skip-pull"],
        capture_output=True, text=True, timeout=120,
    )
    return proc.returncode == 0, (proc.stderr + proc.stdout)[-2000:]


def wait_ready(timeout: int = STARTUP_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(POLL_INTERVAL)
    return False


def get_model_id() -> str | None:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/v1/models", timeout=5) as r:
            return json.loads(r.read())["data"][0]["id"]
    except Exception:
        return None


def send_query(model_id: str, messages: list, max_tokens: int = 10) -> tuple[str | None, float | None]:
    payload = json.dumps({
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    t0 = time.monotonic()
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=QUERY_TIMEOUT) as r:
            data = json.loads(r.read())
        elapsed = time.monotonic() - t0
        text = data["choices"][0]["message"]["content"].strip()
        return text, round(elapsed, 1)
    except Exception as e:
        return None, None


def last_container_error() -> str:
    try:
        logs = subprocess.run(
            ["docker", "logs", "--tail", "30", CONTAINER],
            capture_output=True, text=True,
        )
        combined = logs.stdout + logs.stderr
        for line in reversed(combined.splitlines()):
            l = line.strip()
            if l and ("ERROR" in l or "Error" in l or "FATAL" in l):
                return l[-200:]
    except Exception:
        pass
    return "unknown error"


# ── Test a single model ────────────────────────────────────────────────────

def test_model(model_name: str) -> dict:
    result: dict = {
        "model":       model_name,
        "started":     False,
        "ready":       False,
        "startup_s":   None,
        "responses":   [],
        "error":       None,
    }

    log(f"{'─'*56}")
    log(f"Model: {model_name}")

    ok, output = start_model(model_name)
    if not ok:
        result["error"] = f"vllm_spark.sh exited non-zero"
        log(f"  ERR  Container start failed:\n{output[-500:]}")
        return result
    result["started"] = True

    log(f"  ..   Waiting for API (up to {STARTUP_TIMEOUT}s) …")
    t0 = time.monotonic()
    if not wait_ready():
        result["error"] = last_container_error()
        log(f"  ERR  Timeout / crash: {result['error']}")
        docker_rm()
        return result

    result["startup_s"] = round(time.monotonic() - t0)
    result["ready"] = True
    log(f"  OK   Ready in {result['startup_s']}s")

    model_id = get_model_id()
    if not model_id:
        result["error"] = "Could not resolve model id from /v1/models"
        docker_rm()
        return result

    for q in TEST_QUERIES:
        text, elapsed = send_query(model_id, q["messages"], q["max_tokens"])
        resp = {"label": q["label"], "reply": text, "latency_s": elapsed}
        result["responses"].append(resp)
        if text is not None:
            log(f"  OK   [{q['label']}] → '{text}'  ({elapsed}s)")
        else:
            log(f"  ERR  [{q['label']}] query failed")

    docker_rm()
    return result


# ── Markdown report ────────────────────────────────────────────────────────

def write_status_md(results: list[dict], total: int) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    done = len(results)

    lines = [
        "# Model Status",
        "",
        f"**Hardware:** DGX Spark · GB10 · 128 GB unified memory  ",
        f"**vLLM image:** `vllm/vllm-openai:{VLLM_TAG}`  ",
        f"**Last run:** {now} ({done}/{total} models tested)",
        "",
        "| Model | Status | Startup | sanity | math | short-text |",
        "|---|:---:|---:|---|---|---|",
    ]

    for r in results:
        name = r["model"]

        if not r["started"]:
            status = "❌ start failed"
            startup = q1 = q2 = q3 = "—"
        elif not r["ready"]:
            err = (r["error"] or "")[:80]
            status = f"❌ `{err}`"
            startup = q1 = q2 = q3 = "—"
        else:
            all_ok = all(resp["reply"] is not None for resp in r["responses"])
            status = "✅" if all_ok else "⚠️ partial"
            startup = f"{r['startup_s']}s"
            q1 = q2 = q3 = "—"
            for resp in r["responses"]:
                cell = f"`{resp['reply']}`&nbsp;{resp['latency_s']}s" if resp["reply"] else "❌"
                if resp["label"] == "sanity":
                    q1 = cell
                elif resp["label"] == "math":
                    q2 = cell
                elif resp["label"] == "short-text":
                    q3 = cell

        lines.append(f"| `{name}` | {status} | {startup} | {q1} | {q2} | {q3} |")

    lines += [
        "",
        "> Auto-generated by [`runner/test_models.py`](test_models.py). "
        "Do not edit manually.",
        "",
    ]

    STATUS_FILE.write_text("\n".join(lines), encoding="utf-8")
    log(f"  Status written → {STATUS_FILE}")


def git_push(message: str) -> None:
    try:
        subprocess.run(["git", "-C", str(REPO_DIR), "add", str(STATUS_FILE)],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "commit", "-m", message],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "push"],
                       check=True, capture_output=True)
        log(f"  Pushed to GitHub.")
    except subprocess.CalledProcessError as e:
        log(f"  git push failed: {e.stderr.decode()[:200] if e.stderr else e}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    pattern = sys.argv[1].lower() if len(sys.argv) > 1 else None
    models  = get_compatible_models()

    if pattern:
        models = [m for m in models if pattern in m.lower()]

    if not models:
        log("No matching compatible models found.")
        sys.exit(1)

    log(f"Models to test ({len(models)}):")
    for m in models:
        log(f"    {m}")

    results: list[dict] = []
    for model_name in models:
        r = test_model(model_name)
        results.append(r)
        write_status_md(results, total=len(models))

        # push intermediate results so progress is visible on GitHub
        done = len(results)
        remaining = len(models) - done
        git_push(
            f"ci: model status update ({done}/{len(models)} tested, {remaining} remaining)\n\n"
            f"Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
        )

    log(f"{'═'*56}")
    ok  = sum(1 for r in results if r["ready"])
    err = len(results) - ok
    log(f"Done. {ok}/{len(results)} models OK, {err} failed.")

    git_push(
        f"ci: model status – final ({ok}/{len(results)} OK)\n\n"
        f"Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    )


if __name__ == "__main__":
    main()
