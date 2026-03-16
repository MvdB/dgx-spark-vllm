#!/usr/bin/env python3
"""
Smoke-test + benchmark runner for all compatible vLLM models on DGX Spark.

Per model:
  1. Start vllm_spark.sh
  2. Wait until API is ready  → startup_s
  3. Warmup query             → warmup_s  (first request triggers CUDA JIT)
  4. Correctness queries      → sanity / math / capitals
  5. Streaming benchmark      → TTFT, throughput (tok/s), total bench time
  6. Stop container
  7. Write runner/README.md and push to git

Usage:
  python3 test_models.py              # test all compatible models
  python3 test_models.py qwen3.5-9b   # single model by pattern
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_DIR      = Path(__file__).parent.parent
RUNNER_DIR    = Path(__file__).parent
VLLM_SCRIPT   = Path.home() / "vllm_spark.sh"
HF_MODELS_DIR = Path(os.environ.get("HF_MODELS_DIR", Path.home() / "hf_models"))
STATUS_FILE   = RUNNER_DIR / "README.md"

# ── Runtime config ─────────────────────────────────────────────────────────
HOST_PORT       = int(os.environ.get("HOST_PORT",        "8000"))
BASE_URL        = f"http://127.0.0.1:{HOST_PORT}"
CONTAINER       = os.environ.get("CONTAINER_NAME",       "vllm-server")
VLLM_TAG        = os.environ.get("DEFAULT_VLLM_TAG",     "v0.17.1")
STARTUP_TIMEOUT = int(os.environ.get("STARTUP_TIMEOUT",  "600"))
POLL_INTERVAL   = 10
QUERY_TIMEOUT   = 180

# Benchmark: long enough for stable tok/s, short enough to be quick
BENCH_MAX_TOKENS = 150
BENCH_PROMPT     = (
    "Explain how a transformer neural network processes text. "
    "Cover tokenization, embeddings, attention, and output generation."
)

# System message used for all correctness queries.
# Disables extended thinking on Qwen3 / reasoning models so replies are concise.
CONCISE_SYSTEM = {
    "role": "system",
    "content": "Reply as briefly as possible. No reasoning steps, no explanation.",
}


# ── Correctness queries ────────────────────────────────────────────────────
CORRECTNESS_QUERIES = [
    {
        "label":    "sanity",
        "messages": [CONCISE_SYSTEM,
                     {"role": "user", "content": "Reply with exactly one word: Ready"}],
        "max_tokens": 15,
    },
    {
        "label":    "math",
        "messages": [CONCISE_SYSTEM,
                     {"role": "user", "content": "What is 7 × 8? Reply with just the number."}],
        "max_tokens": 15,
    },
    {
        "label":    "capitals",
        "messages": [CONCISE_SYSTEM,
                     {"role": "user", "content": "Capital of Japan? One word."}],
        "max_tokens": 15,
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


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


def send_query(model_id: str, messages: list, max_tokens: int = 15) -> tuple[str | None, float | None]:
    payload = json.dumps({
        "model":       model_id,
        "messages":    messages,
        "max_tokens":  max_tokens,
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
        text = data["choices"][0]["message"]["content"].strip()
        return text, round(time.monotonic() - t0, 1)
    except Exception:
        return None, None


def bench_stream(model_id: str) -> dict:
    """
    Streaming benchmark. Measures:
      ttft_s        time to first output token
      throughput_s  output tok/s (generation phase only, i.e. after first token)
      output_tokens from vLLM usage field if available, else char-count estimate
      total_s       wall-clock for whole response
    """
    payload = json.dumps({
        "model":          model_id,
        "messages":       [{"role": "user", "content": BENCH_PROMPT}],
        "max_tokens":     BENCH_MAX_TOKENS,
        "temperature":    0,
        "stream":         True,
        "stream_options": {"include_usage": True},
    }).encode()

    result: dict = {
        "ttft_s": None, "throughput_s": None,
        "output_tokens": None, "total_s": None, "error": None,
    }

    t0 = time.monotonic()
    ttft: float | None = None
    first_chunk_t: float | None = None   # time of any first content-carrying chunk
    char_count = 0
    usage_tokens: int | None = None

    try:
        req = urllib.request.Request(
            f"{BASE_URL}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=QUERY_TIMEOUT) as r:
            for raw in r:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if chunk.get("usage"):
                    usage_tokens = chunk["usage"].get("completion_tokens")

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta   = choices[0].get("delta", {})
                content = delta.get("content") or ""

                # Track time of first chunk that has ANY delta (even empty content)
                if first_chunk_t is None and delta:
                    first_chunk_t = time.monotonic() - t0

                if content:
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    char_count += len(content)

        total = time.monotonic() - t0

        # If ttft wasn't captured (model streams in one chunk), fall back to first chunk time
        if ttft is None and first_chunk_t is not None:
            ttft = first_chunk_t

        out_tokens = usage_tokens if usage_tokens is not None else max(1, char_count // 4)
        gen_time   = max(0.01, total - (ttft or 0))
        throughput = round(out_tokens / gen_time, 1)

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
        "model": model_name, "started": False, "ready": False,
        "startup_s": None, "warmup_s": None,
        "correctness": [], "bench": {}, "error": None,
    }

    log(f"{'─'*56}")
    log(f"Model: {model_name}")

    ok, output = start_model(model_name)
    if not ok:
        result["error"] = "vllm_spark.sh exited non-zero"
        log(f"  ERR  {output[-300:]}")
        return result
    result["started"] = True

    log(f"  ..   Waiting for API (up to {STARTUP_TIMEOUT}s) …")
    t0 = time.monotonic()
    if not wait_ready():
        result["error"] = last_container_error()
        log(f"  ERR  {result['error']}")
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

    # Warmup
    log(f"  ..   Warmup …")
    _, warmup_s = send_query(model_id, [{"role": "user", "content": "Hi"}], max_tokens=5)
    result["warmup_s"] = warmup_s
    log(f"  OK   Warmup {warmup_s}s")

    # Correctness
    for q in CORRECTNESS_QUERIES:
        text, latency = send_query(model_id, q["messages"], q["max_tokens"])
        result["correctness"].append({"label": q["label"], "reply": text, "latency_s": latency})
        if text is not None:
            log(f"  OK   [{q['label']}] → '{text[:40]}'  ({latency}s)")
        else:
            log(f"  ERR  [{q['label']}] failed")

    # Benchmark
    log(f"  ..   Streaming benchmark ({BENCH_MAX_TOKENS} tok) …")
    bench = bench_stream(model_id)
    result["bench"] = bench
    if bench["error"]:
        log(f"  ERR  Bench: {bench['error']}")
    else:
        log(f"  OK   TTFT {bench['ttft_s']}s | {bench['throughput_s']} tok/s | "
            f"{bench['output_tokens']} tok | {bench['total_s']}s total")

    docker_rm()
    return result


# ── Markdown helpers ───────────────────────────────────────────────────────

def _fmt(val, unit: str = "", decimals: int = 1) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.{decimals}f}{unit}"
    return f"{val}{unit}"


def _cell(reply: str | None, latency: float | None) -> str:
    """Safe markdown table cell for a correctness reply."""
    if reply is None:
        return "❌"
    # Sanitize: strip newlines, escape pipes, truncate
    text = reply.replace("\r", "").replace("\n", " ").replace("|", "\\|").strip()
    if len(text) > 35:
        text = text[:34] + "…"
    lat = f"&nbsp;{latency}s" if latency is not None else ""
    return f"`{text}`{lat}"


# ── Write README.md ────────────────────────────────────────────────────────

def write_readme(results: list[dict], total: int) -> None:
    now  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    done = len(results)

    lines = [
        "# runner",
        "",
        "Scripts for starting vLLM on DGX Spark and benchmarking local models.",
        "",
        "| File | Purpose |",
        "|---|---|",
        "| `vllm_spark.sh` | Interactive model picker + vLLM container runner |",
        "| `vllm_spark_profiler.py` | Auto-generates per-model vLLM parameter profiles |",
        "| `test_models.py` | Smoke-test + benchmark runner (writes this file) |",
        "",
        "See the [main README](../README.md) for full documentation.",
        "",
        "To re-run the benchmark:",
        "```bash",
        "python3 test_models.py              # all models",
        "python3 test_models.py qwen3.5-9b   # single model by pattern",
        "```",
        "",
        "---",
        "",
        "## Benchmark Results",
        "",
        f"**Hardware:** DGX Spark · GB10 · 128 GB unified memory  ",
        f"**vLLM image:** `vllm/vllm-openai:{VLLM_TAG}`  ",
        f"**Last run:** {now} ({done}/{total} models tested)",
        "",
        "| Model | Status | Startup | Warmup | TTFT | Throughput | Out tokens | Bench total |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in results:
        name = f"`{r['model']}`"
        if not r["started"]:
            lines.append(f"| {name} | ❌ start&nbsp;failed | — | — | — | — | — | — |")
            continue
        if not r["ready"]:
            err = (r["error"] or "unknown")[:50].replace("|", "\\|")
            lines.append(f"| {name} | ❌ `{err}` | — | — | — | — | — | — |")
            continue

        all_ok   = all(c["reply"] is not None for c in r["correctness"])
        b        = r["bench"]
        bench_ok = not b.get("error") and b.get("ttft_s") is not None
        status   = "✅" if (all_ok and bench_ok) else ("⚠️" if all_ok else "⚠️&nbsp;partial")

        lines.append(
            f"| {name} | {status} "
            f"| {_fmt(r['startup_s'], 's', 0)} "
            f"| {_fmt(r['warmup_s'], 's', 1)} "
            f"| {_fmt(b.get('ttft_s'), 's', 2)} "
            f"| {_fmt(b.get('throughput_s'), '&nbsp;tok/s', 1)} "
            f"| {_fmt(b.get('output_tokens'), '', 0)} "
            f"| {_fmt(b.get('total_s'), 's', 1)} |"
        )

    lines += [
        "",
        "## Correctness",
        "",
        "| Model | sanity | math | capitals |",
        "|---|---|---|---|",
    ]

    for r in results:
        if not r.get("ready"):
            continue
        name  = f"`{r['model']}`"
        cells = {c["label"]: _cell(c["reply"], c["latency_s"]) for c in r.get("correctness", [])}
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
        "> Auto-generated by [`test_models.py`](test_models.py). Do not edit manually.",
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
        results.append(test_model(model_name))
        write_readme(results, total=len(models))
        done = len(results)
        git_push(
            f"ci: benchmark {done}/{len(models)}"
            f"{' – done' if done == len(models) else ''}\n\n"
            f"Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
        )

    log(f"{'═'*56}")
    ok = sum(1 for r in results if r["ready"])
    log(f"Done. {ok}/{len(results)} models OK.")


if __name__ == "__main__":
    main()
