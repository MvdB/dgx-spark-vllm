#!/usr/bin/env python3
"""
Smoke-test + benchmark runner for all compatible vLLM models on DGX Spark.

Per model:
  1. Start vllm_spark.sh
  2. Wait until API is ready  → startup_s
  3. Warmup query             → warmup_s  (first request triggers CUDA JIT)
  4. Correctness queries      → sanity / math / short-text
  5. Streaming benchmark      → TTFT, throughput (tok/s), total bench time
  6. Stop container
  7. Write runner/model_status.md and push to git

Usage:
  python3 test_models.py              # test all compatible models
  python3 test_models.py qwen3.5-9b   # single model by pattern
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

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_DIR      = Path(__file__).parent.parent
RUNNER_DIR    = Path(__file__).parent
VLLM_SCRIPT   = Path.home() / "vllm_spark.sh"
HF_MODELS_DIR = Path(os.environ.get("HF_MODELS_DIR", Path.home() / "hf_models"))
STATUS_FILE   = RUNNER_DIR / "model_status.md"

# ── Runtime config ─────────────────────────────────────────────────────────
HOST_PORT       = int(os.environ.get("HOST_PORT",        "8000"))
BASE_URL        = f"http://127.0.0.1:{HOST_PORT}"
CONTAINER       = os.environ.get("CONTAINER_NAME",       "vllm-server")
VLLM_TAG        = os.environ.get("DEFAULT_VLLM_TAG",     "v0.17.1")
STARTUP_TIMEOUT = int(os.environ.get("STARTUP_TIMEOUT",  "600"))   # seconds
POLL_INTERVAL   = 10
QUERY_TIMEOUT   = 180   # per query (generous for large models)

# Benchmark generation length – long enough for stable tok/s, short enough to be quick
BENCH_MAX_TOKENS = 150
BENCH_PROMPT     = (
    "Explain in detail how a transformer neural network processes text. "
    "Cover tokenization, embeddings, attention heads, and output generation."
)


# ── Correctness queries ────────────────────────────────────────────────────
CORRECTNESS_QUERIES = [
    {
        "label":    "sanity",
        "messages": [{"role": "user", "content": "Reply with exactly one word: Ready"}],
        "max_tokens": 10,
    },
    {
        "label":    "math",
        "messages": [{"role": "user", "content": "What is 7 × 8? Reply with just the number."}],
        "max_tokens": 10,
    },
    {
        "label":    "capitals",
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
    """Non-streaming query. Returns (text, latency_s)."""
    payload = json.dumps({
        "model":      model_id,
        "messages":   messages,
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
        return data["choices"][0]["message"]["content"].strip(), round(time.monotonic() - t0, 1)
    except Exception:
        return None, None


def bench_stream(model_id: str) -> dict:
    """
    Streaming benchmark query.
    Returns:
      ttft_s       – time to first output token (seconds)
      throughput_s – output tokens / second (generation phase only)
      output_tokens – number of output tokens (from usage if available, else estimated)
      total_s      – wall-clock time for entire response
      error        – None or error string
    """
    payload = json.dumps({
        "model":          model_id,
        "messages":       [{"role": "user", "content": BENCH_PROMPT}],
        "max_tokens":     BENCH_MAX_TOKENS,
        "temperature":    0,
        "stream":         True,
        "stream_options": {"include_usage": True},
    }).encode()

    result = {
        "ttft_s":       None,
        "throughput_s": None,
        "output_tokens": None,
        "total_s":      None,
        "error":        None,
    }

    t0 = time.monotonic()
    ttft: float | None = None
    char_count = 0
    usage_tokens: int | None = None

    try:
        req = urllib.request.Request(
            f"{BASE_URL}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=QUERY_TIMEOUT) as r:
            for raw_line in r:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Usage appears in last chunk when include_usage=True
                if chunk.get("usage"):
                    usage_tokens = chunk["usage"].get("completion_tokens")

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                content = choices[0].get("delta", {}).get("content", "")
                if content:
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    char_count += len(content)

        total = time.monotonic() - t0

        # Token count: prefer usage field, fall back to rough char estimate (÷4)
        out_tokens = usage_tokens if usage_tokens is not None else max(1, char_count // 4)

        gen_time = total - (ttft or 0)
        throughput = round(out_tokens / gen_time, 1) if gen_time > 0 else None

        result.update({
            "ttft_s":        round(ttft, 2) if ttft is not None else None,
            "throughput_s":  throughput,
            "output_tokens": out_tokens,
            "total_s":       round(total, 1),
        })

    except Exception as e:
        result["error"] = str(e)[:120]

    return result


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
        "warmup_s":    None,
        "correctness": [],
        "bench":       {},
        "error":       None,
    }

    log(f"{'─'*56}")
    log(f"Model: {model_name}")

    ok, output = start_model(model_name)
    if not ok:
        result["error"] = "vllm_spark.sh exited non-zero"
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
        result["error"] = "Could not resolve model id"
        docker_rm()
        return result

    # ── Warmup (first request triggers CUDA JIT; discard for benchmarking) ──
    log(f"  ..   Warmup query …")
    _, warmup_s = send_query(model_id,
                             [{"role": "user", "content": "Hi"}],
                             max_tokens=5)
    result["warmup_s"] = warmup_s
    log(f"  OK   Warmup done in {warmup_s}s")

    # ── Correctness queries ────────────────────────────────────────────────
    for q in CORRECTNESS_QUERIES:
        text, latency = send_query(model_id, q["messages"], q["max_tokens"])
        result["correctness"].append({"label": q["label"], "reply": text, "latency_s": latency})
        if text is not None:
            log(f"  OK   [{q['label']}] → '{text}'  ({latency}s)")
        else:
            log(f"  ERR  [{q['label']}] failed")

    # ── Streaming benchmark ────────────────────────────────────────────────
    log(f"  ..   Streaming benchmark ({BENCH_MAX_TOKENS} tok max) …")
    bench = bench_stream(model_id)
    result["bench"] = bench
    if bench["error"]:
        log(f"  ERR  Bench: {bench['error']}")
    else:
        log(f"  OK   TTFT {bench['ttft_s']}s  |  "
            f"{bench['throughput_s']} tok/s  |  "
            f"{bench['output_tokens']} tokens  |  "
            f"{bench['total_s']}s total")

    docker_rm()
    return result


# ── Markdown report ────────────────────────────────────────────────────────

def _cell(val, unit: str = "", decimals: int = 1) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.{decimals}f}{unit}"
    return f"{val}{unit}"


def write_status_md(results: list[dict], total: int) -> None:
    now  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    done = len(results)

    lines = [
        "# Model Status",
        "",
        f"**Hardware:** DGX Spark · GB10 · 128 GB unified memory  ",
        f"**vLLM image:** `vllm/vllm-openai:{VLLM_TAG}`  ",
        f"**Last run:** {now} ({done}/{total} models tested)",
        "",
        "## Results",
        "",
        "| Model | Status | Startup | Warmup | TTFT | Throughput | Out tokens | Bench total |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in results:
        name = f"`{r['model']}`"

        if not r["started"]:
            lines.append(f"| {name} | ❌ start failed | — | — | — | — | — | — |")
            continue

        if not r["ready"]:
            err = (r["error"] or "unknown")[:60]
            lines.append(f"| {name} | ❌ `{err}` | — | — | — | — | — | — |")
            continue

        all_ok   = all(c["reply"] is not None for c in r["correctness"])
        bench_ok = not r["bench"].get("error") and r["bench"].get("ttft_s") is not None
        if all_ok and bench_ok:
            status = "✅"
        elif all_ok:
            status = "⚠️ bench failed"
        else:
            status = "⚠️ partial"

        b = r["bench"]
        lines.append(
            f"| {name} | {status} "
            f"| {_cell(r['startup_s'], 's', 0)} "
            f"| {_cell(r['warmup_s'], 's', 1)} "
            f"| {_cell(b.get('ttft_s'), 's', 2)} "
            f"| {_cell(b.get('throughput_s'), ' tok/s', 1)} "
            f"| {_cell(b.get('output_tokens'), '', 0)} "
            f"| {_cell(b.get('total_s'), 's', 1)} |"
        )

    # Correctness detail
    lines += ["", "## Correctness", ""]
    lines += ["| Model | sanity | math | capitals |", "|---|---|---|---|"]
    for r in results:
        if not r.get("ready"):
            continue
        name = f"`{r['model']}`"
        cells = {}
        for c in r.get("correctness", []):
            v = f"`{c['reply']}`&nbsp;{c['latency_s']}s" if c["reply"] else "❌"
            cells[c["label"]] = v
        lines.append(
            f"| {name} "
            f"| {cells.get('sanity', '—')} "
            f"| {cells.get('math', '—')} "
            f"| {cells.get('capitals', '—')} |"
        )

    lines += [
        "",
        "---",
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

        done      = len(results)
        remaining = len(models) - done
        git_push(
            f"ci: model status {done}/{len(models)} "
            f"({'done' if remaining == 0 else f'{remaining} remaining'})\n\n"
            f"Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
        )

    log(f"{'═'*56}")
    ok  = sum(1 for r in results if r["ready"])
    log(f"Done. {ok}/{len(results)} models OK.")


if __name__ == "__main__":
    main()
