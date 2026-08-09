#!/usr/bin/env python3
"""Kurzer Perf-Pass (TTFT + Tok/s) für die SaaS-Kohorte über den LiteLLM-Proxy.

Der SaaS-Lauf fuhr nur 4 Playbooks (kein 06_performance). Dieses Tool misst per
Streaming pro Modell TTFT (Time-to-First-Token) + Durchsatz (completion_tokens /
Generierungszeit) über wenige short/medium/long-Prompts und ergänzt ein
06_performance-Playbook im jeweiligen SaaS-Report (Format wie der lokale
perf_benchmark, damit make_public_site/build_site es rendern).

ACHTUNG: misst Cloud-Modell + Proxy + Netz — NICHT lokale Hardware. Entsprechend
mit source='litellm-proxy (cloud)' markiert.

  python saas_perf.py                 # alle 16
  python saas_perf.py --models GPT-5.6-luna,Claude-Sonnet-5
"""
from __future__ import annotations
import argparse, json, os, statistics, sys, time
from pathlib import Path
from openai import OpenAI

TP = Path(__file__).resolve().parent
RUN = TP / "reports" / "2026-08-07_saas"

COHORT = [
    ("Claude-Haiku-4.5", "claude-haiku-4-5"), ("Claude-Sonnet-5", "claude-sonnet-5"),
    ("Claude-Opus-5", "claude-opus-5"), ("Claude-Fable-5", "claude-fable-5"),
    ("GPT-5.6-luna", "gpt-5.6-luna"), ("GPT-5.6-sol", "gpt-5.6-sol"),
    ("GPT-5.6-terra", "gpt-5.6-terra"), ("Gemini-3.1-Pro", "gemini/gemini-3.1-pro-preview"),
    ("Gemini-3.6-Flash", "gemini/gemini-3.6-flash"),
    ("Gemini-3.5-Flash-Lite", "gemini/gemini-3.5-flash-lite"),
    ("Grok-4.5", "xai/grok-4.5"), ("Grok-4.1-Fast", "xai/grok-4-1-fast-reasoning"),
    ("Mistral-Large", "mistral/mistral-large-latest"),
    ("Mistral-Medium", "mistral/mistral-medium-latest"),
    ("Magistral-Medium", "mistral/magistral-medium-latest"),
    ("Ministral-8B", "mistral/ministral-8b-latest"),
]
# (label, prompt, max_tokens) — kurz gehalten
PROMPTS = [
    ("short", "Nenne drei Vorteile erneuerbarer Energien.", 150),
    ("short", "Was ist der Unterschied zwischen RAM und SSD? Kurz.", 150),
    ("short", "Gib ein Beispiel für eine Metapher.", 150),
    ("medium", "Erkläre in einem Absatz, wie HTTPS funktioniert.", 320),
    ("medium", "Beschreibe die Grundidee der Relativitätstheorie.", 320),
    ("long", "Schreibe eine kurze Zusammenfassung der Französischen Revolution.", 520),
    ("long", "Erkläre die Funktionsweise eines Transformers (ML) verständlich.", 520),
]


def _client() -> OpenAI:
    for line in (TP / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); os.environ.setdefault(k, v)
    host = os.environ.get("JUDGE_HOST", "10.0.0.6"); port = os.environ.get("JUDGE_PORT", "4000")
    return OpenAI(base_url=f"http://{host}:{port}/v1", api_key=os.environ["JUDGE_API_KEY"])


def measure(client, model_id, prompt, max_tokens):
    """→ (ttft_ms, tok_s) oder (None, None) bei Fehler."""
    t0 = time.perf_counter(); ttft = None; t_last = t0; comp = None
    try:
        stream = client.chat.completions.create(
            model=model_id, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, stream=True, stream_options={"include_usage": True})
        for ch in stream:
            now = time.perf_counter()
            if ch.choices:
                dl = ch.choices[0].delta
                if dl and (getattr(dl, "content", None) or getattr(dl, "reasoning_content", None)):
                    if ttft is None:
                        ttft = now - t0
                    t_last = now
            if getattr(ch, "usage", None):
                comp = ch.usage.completion_tokens
    except Exception as e:
        print(f"    ! Fehler: {type(e).__name__}: {str(e)[:80]}", flush=True)
        return None, None
    if ttft is None or not comp:
        return None, None
    gen = max(t_last - (t0 + ttft), 1e-3)
    return ttft * 1000.0, comp / gen


def pct(vals, p):
    if not vals:
        return None
    vals = sorted(vals); k = (len(vals) - 1) * p / 100.0
    lo = int(k); hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def run_model(client, name, model_id) -> bool:
    rp = RUN / f"{name}.json"
    if not rp.exists():
        print(f"  {name}: kein Report — skip"); return False
    print(f"  {name} ({model_id}) …", flush=True)
    ttfts, by = [], {"short": [], "medium": [], "long": []}
    errs = 0
    for kind, prompt, mt in PROMPTS:
        ttft, toks = measure(client, model_id, prompt, mt)
        if ttft is None:
            errs += 1; continue
        ttfts.append(ttft); by[kind].append(toks)
    all_toks = [t for v in by.values() for t in v]
    if not all_toks:
        print(f"    → keine validen Messungen"); return False
    metrics = {
        "model": model_id, "n_measurements": len(all_toks), "n_errors": errs,
        "ttft_p50_ms": pct(ttfts, 50), "ttft_p95_ms": pct(ttfts, 95), "ttft_p99_ms": pct(ttfts, 99),
        "throughput_mean_tok_s": statistics.mean(all_toks),
        "throughput_median_tok_s": statistics.median(all_toks),
        "throughput_by_type": {k: (round(statistics.mean(v), 1) if v else None) for k, v in by.items()},
        "source": "litellm-proxy (cloud+proxy-latenz, nicht lokale hardware)",
    }
    d = json.loads(rp.read_text(encoding="utf-8"))
    d.setdefault("playbooks", {})["06_performance"] = {
        "total": 1, "passed": 1, "failed": 0, "pass_rate": 1.0, "mean_score": 1.0,
        "duration_seconds": 0.0, "knockouts": [],
        "results": [{"test_id": "perf_benchmark", "model": model_id, "evaluator": "performance",
                     "verdict": "pass", "score": 1.0, "response_type": "answer",
                     "response": json.dumps(metrics, ensure_ascii=False)}],
    }
    rp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    → TTFT p50 {metrics['ttft_p50_ms']:.0f}ms · {metrics['throughput_median_tok_s']:.1f} tok/s "
          f"(n={len(all_toks)}, err={errs})")
    return True


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models"); args = ap.parse_args()
    client = _client()
    cohort = COHORT
    if args.models:
        want = {x.strip() for x in args.models.split(",")}
        cohort = [c for c in COHORT if c[0] in want]
    print(f"SaaS-Perf (kurz): {len(cohort)} Modelle über Proxy\n")
    ok = sum(run_model(client, n, mid) for n, mid in cohort)
    print(f"\n✓ {ok}/{len(cohort)} SaaS-Reports um 06_performance ergänzt")


if __name__ == "__main__":
    main()
