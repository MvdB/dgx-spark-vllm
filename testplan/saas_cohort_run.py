#!/usr/bin/env python3
"""SaaS-Referenzkohorte gegen den LiteLLM-Proxy (OpenAI-kompatibel).

Wiederverwendet die vorhandenen Evaluatoren (Quality/Bias/Code) 1:1 — die Rubriken
stecken in den Evaluator-Klassen, nicht in den Playbook-YAMLs. Zwei Anpassungen
per Monkeypatch, weil Frontier-Modelle ueber den Proxy andere Sampling-Regeln haben:

  * query_target: sendet KEIN temperature (OpenAI GPT-5.x lehnt !=1 mit HTTP 400 ab)
    und keine chat_template_kwargs (kein vLLM dahinter).
  * query_judge : reasoning_effort=minimal (senkt Sonnet-5-Output nachweislich,
    "none" wird nicht honoriert) und ebenfalls kein temperature.

Schreibt pro Modell reports/<run_id>/<Name>.json im exakt gleichen Schema wie der
Orchestrator (reporter._write_json) — damit build_site.py/Dashboard es lesen.

Nur die vier nicht-adversarialen Playbooks: 01_quality, 02_german_language,
03_bias, 05_code. 04_security wird bewusst NIE an Dritt-APIs geschickt.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TP = Path(__file__).resolve().parent
sys.path.insert(0, str(TP))

from openai import OpenAI  # noqa: E402

from lib.config import TestplanConfig, ModelConfig  # noqa: E402
from lib.testdata import TestDataLoader  # noqa: E402
from evaluators.base import BaseEvaluator, EvalResult, PlaybookResult, Verdict  # noqa: E402
from evaluators.quality import QualityEvaluator  # noqa: E402
from evaluators.bias import BiasEvaluator  # noqa: E402
from evaluators.code_eval import CodeEvaluator  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Kohorte: (Report-Name, Proxy-Modell-ID, Preis $/1M in, $/1M out)  — Preise grob,
# nur fuer die Kostenschaetzung. Judge-Kosten werden separat exakt gemessen.
# ─────────────────────────────────────────────────────────────────────────────
# Proxy-Modell-IDs exakt wie vom kohorten-spezifischen LiteLLM-Key exponiert
# (Anthropic/OpenAI ohne Provider-Prefix; Google/xAI/Mistral mit). Preise via
# /model/info gefuellt (fill_prices), Startwerte hier grob als Fallback.
COHORT: list[tuple[str, str, float, float]] = [
    ("Claude-Haiku-4.5",   "claude-haiku-4-5",                 1.00,  5.00),
    ("Claude-Sonnet-5",    "claude-sonnet-5",                  3.00, 15.00),
    ("Claude-Opus-5",      "claude-opus-5",                    5.00, 25.00),
    ("Claude-Fable-5",     "claude-fable-5",                  10.00, 50.00),
    ("GPT-5.6-luna",       "gpt-5.6-luna",                     1.25, 10.00),
    ("GPT-5.6-sol",        "gpt-5.6-sol",                      1.25, 10.00),
    ("GPT-5.6-terra",      "gpt-5.6-terra",                    1.25, 10.00),
    ("Gemini-3.1-Pro",     "gemini/gemini-3.1-pro-preview",    1.25, 10.00),
    ("Gemini-3.6-Flash",   "gemini/gemini-3.6-flash",          0.30,  2.50),
    ("Gemini-3.5-Flash-Lite", "gemini/gemini-3.5-flash-lite",  0.10,  0.40),
    ("Grok-4.5",           "xai/grok-4.5",                     3.00, 15.00),
    ("Grok-4.1-Fast",      "xai/grok-4-1-fast-reasoning",      0.20,  0.50),
    ("Mistral-Large",      "mistral/mistral-large-latest",     2.00,  6.00),
    ("Mistral-Medium",     "mistral/mistral-medium-latest",    0.40,  2.00),
    ("Magistral-Medium",   "mistral/magistral-medium-latest",  2.00,  5.00),
    ("Ministral-8B",       "mistral/ministral-8b-latest",      0.10,  0.10),
    # --- Erweiterung 2026-08: weitere Frontier-Chat-Modelle aus dem LiteLLM-Proxy ---
    ("DeepSeek-V4-Pro",    "deepseek/deepseek-v4-pro",         1.00,  3.00),
    ("GLM-5.2",            "z-ai/glm-5.2",                     0.50,  2.00),
    ("Kimi-K3",            "moonshotai/kimi-k3",               1.00,  3.00),
    ("MiniMax-M3",         "minimax/minimax-m3",               0.50,  2.00),
    ("Qwen3.8-Max",        "qwen/qwen3.8-max",                 1.00,  4.00),
    ("Qwen3.7-Plus",       "qwen/qwen3.7-plus",                0.50,  2.00),
    ("Step-3.7-Flash",     "stepfun/step-3.7-flash",           0.30,  1.00),
    ("Hunyuan-3",          "tencent/hy3",                      0.50,  2.00),
    ("MiMo-v2.5-Pro",      "xiaomi/mimo-v2.5-pro",             0.50,  2.00),
    ("MiMo-v2.5",          "xiaomi/mimo-v2.5",                 0.30,  1.00),
    ("Nemotron-3-Ultra-550B", "nvidia/nemotron-3-ultra-550b-a55b:free", 0.00, 0.00),
]

JUDGE_MODEL = "claude-sonnet-5"
JUDGE_PRICE_IN, JUDGE_PRICE_OUT = 2.00, 10.00  # Intro-Preis bis 2026-08-31

PLAYBOOKS = ["01_quality", "02_german_language", "03_bias", "05_code"]

# Usage-Zaehler (pro Modell Subject; global Judge) + exakte Kosten aus
# x-litellm-response-cost-Header (LiteLLM rechnet serverseitig ab).
TALLY = {"subj_in": 0, "subj_out": 0, "judge_in": 0, "judge_out": 0,
         "subj_cost": 0.0, "judge_cost": 0.0}
SUBJ_MAX_TOKENS = 4096  # Deckel gegen durchdrehende Reasoning-Modelle


BUDGET = {"spend": None, "max": None, "margin": 3.0, "stop": False}


class BudgetExceeded(RuntimeError):
    pass


def _create_with_cost(client, **kwargs):
    """chat.completions.create, gibt (completion, cost) zurueck. cost aus dem
    LiteLLM-Header x-litellm-response-cost. Aktualisiert den Key-Budget-Stand und
    stoppt den Lauf knapp vor dem Cap — einmal ausgeloest, feuert kein Call mehr
    (Kostenbremse), da evaluate_batch Exceptions pro Fall schluckt."""
    if BUDGET["stop"]:
        raise BudgetExceeded("Budget-Stop aktiv — kein weiterer Call.")
    try:
        raw = client.chat.completions.with_raw_response.create(**kwargs)
    except Exception as e:
        # Budget-Cap (429 budget_exceeded) ist terminal: sofort stoppen, sonst
        # bekommen alle restlichen Fälle 429 (der x-litellm-key-spend-Header
        # läuft nach und triggert die proaktive Bremse zu spät).
        if "budget has been exceeded" in str(e).lower() or "budget_exceeded" in str(e).lower():
            BUDGET["stop"] = True
            raise BudgetExceeded(f"Budget-Cap erreicht: {str(e)[:160]}")
        raise
    h = raw.headers

    def _f(name):
        try:
            return float(h.get(name))
        except (TypeError, ValueError):
            return None
    BUDGET["spend"] = _f("x-litellm-key-spend")
    BUDGET["max"] = _f("x-litellm-key-max-budget")
    if BUDGET["spend"] is not None and BUDGET["max"] is not None:
        if BUDGET["spend"] >= BUDGET["max"] - BUDGET["margin"]:
            BUDGET["stop"] = True
            raise BudgetExceeded(
                f"Key-Spend ${BUDGET['spend']:.2f} nahe Cap ${BUDGET['max']:.2f} "
                f"(Marge ${BUDGET['margin']:.0f}) — Lauf gestoppt.")
    cost = _f("x-litellm-response-cost")
    return raw.parse(), cost


def _load_env() -> None:
    """.env in os.environ spiegeln — die config.yaml referenziert ${JUDGE_PORT} etc.,
    die sonst nur beim Shell-Sourcing gesetzt sind."""
    for ln in (TP / ".env").read_text().splitlines():
        ln = ln.strip()
        if "=" in ln and not ln.startswith("#"):
            k, _, v = ln.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def _proxy_client() -> OpenAI:
    host = os.environ.get("JUDGE_HOST", "10.0.0.6")
    port = os.environ.get("JUDGE_PORT", "4000")
    key = os.environ["JUDGE_API_KEY"]
    return OpenAI(base_url=f"http://{host}:{port}/v1", api_key=key)


# ── Monkeypatches ────────────────────────────────────────────────────────────
def _saas_query_target(self, prompt, system_prompt="", max_tokens=SUBJ_MAX_TOKENS,
                       temperature=0.1, timeout=900, _degenerate_retry=False):
    messages = []
    eff_sys = system_prompt or self.default_system_prompt
    if eff_sys:
        messages.append({"role": "system", "content": eff_sys})
    messages.append({"role": "user", "content": prompt})
    start = time.monotonic()
    completion, cost = _create_with_cost(
        self.target_client, model=self.target_model, messages=messages,
        max_tokens=SUBJ_MAX_TOKENS, timeout=timeout,  # KEIN temperature
    )
    latency_ms = (time.monotonic() - start) * 1000
    msg = completion.choices[0].message
    content = msg.content or ""
    thinking = getattr(msg, "reasoning_content", None) or ""
    if not thinking and "<think>" in content:
        m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if m:
            thinking = m.group(1).strip()
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if not content and thinking:
        content = thinking
    tokens = completion.usage.completion_tokens if completion.usage else 0
    if completion.usage:
        TALLY["subj_in"] += completion.usage.prompt_tokens or 0
        TALLY["subj_out"] += completion.usage.completion_tokens or 0
    if cost is not None:
        TALLY["subj_cost"] += cost
    self.last_response_degenerate = False
    return content, thinking, latency_ms, tokens, False


def _saas_query_judge(self, prompt, system_prompt="", max_tokens=1024, temperature=0.0):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    completion, cost = _create_with_cost(
        self.judge_client, model=self.judge_model, messages=messages,
        max_tokens=max_tokens, reasoning_effort="minimal",  # KEIN temperature
    )
    if completion.usage:
        TALLY["judge_in"] += completion.usage.prompt_tokens or 0
        TALLY["judge_out"] += completion.usage.completion_tokens or 0
    if cost is not None:
        TALLY["judge_cost"] += cost
    return completion.choices[0].message.content or ""


BaseEvaluator.query_target = _saas_query_target
BaseEvaluator.query_judge = _saas_query_judge


# ── Playbook-Ausfuehrung (Spiegel von orchestrator._run_playbook, 4 Zweige) ──
def run_playbook(pb: str, model: ModelConfig, loader: TestDataLoader,
                 target_client, target_model, judge_client) -> list[EvalResult]:
    kw = dict(target_client=target_client, target_model=target_model,
              judge_client=judge_client, judge_model=JUDGE_MODEL,
              default_system_prompt=model.system_prompt,
              sampling=model.sampling, chat_template_kwargs=model.chat_template_kwargs)
    if pb == "01_quality":
        ev = QualityEvaluator(**kw)
        cases = loader.load_category("quality") + loader.load_category("long_context")
        return ev.evaluate_batch(cases)
    if pb == "02_german_language":
        ev = QualityEvaluator(**kw)
        cases = loader.load_category("german_language")
        cases += loader.filter_cases(loader.load_category("quality"),
                                     language="de", subcategory="german_quality")
        return ev.evaluate_batch(cases)
    if pb == "03_bias":
        ev = BiasEvaluator(**kw)
        cases = loader.load_category("bias")
        results = ev.evaluate_batch(cases)
        sig = BiasEvaluator.aggregate_significance(results)
        if sig["significant"]:
            results.append(EvalResult(
                test_id="bias_aggregate", model=target_model,
                evaluator="bias.significance", verdict=Verdict.KNOCKOUT,
                score=0.0, response="",
                reasoning=(f"Statistisch signifikanter Bias: χ²={sig['chi2']:.2f}, "
                           f"p={sig['p_value']:.4f}, Pass-Rate={sig['pass_rate']:.2%}"),
            ))
        return results
    if pb == "05_code":
        ev = CodeEvaluator(**kw)
        return ev.evaluate_batch(loader.load_category("code"))
    raise ValueError(f"unbekanntes Playbook {pb}")


def model_summary(pb_results: list[PlaybookResult], min_quality_pass_rate: float) -> dict:
    total = sum(p.total for p in pb_results)
    passed = sum(p.passed for p in pb_results)
    knockouts = sum(len(p.knockouts) for p in pb_results)
    rate = passed / total if total else 0.0
    q = next((p for p in pb_results if p.playbook == "01_quality"), None)
    if knockouts or (q and q.pass_rate < min_quality_pass_rate):
        overall = "K.O."
    elif rate >= 0.85:
        overall = "PASS"
    elif rate >= 0.75:
        overall = "WARN"
    else:
        overall = "FAIL"
    return {"overall": overall, "total_tests": total, "passed": passed,
            "pass_rate": f"{rate * 100:.0f}", "knockouts": knockouts}


def write_report(run_dir: Path, name: str, proxy_id: str, pb_results: list[PlaybookResult],
                 thresholds) -> Path:
    data = {
        "meta": {
            "run": run_dir.name, "model": name, "profile": proxy_id,
            "judge": JUDGE_MODEL,
            "thresholds": {
                "hallucination_rate": thresholds.hallucination_rate,
                "factual_accuracy_target": thresholds.factual_accuracy_target,
            },
            "source": "saas_proxy",  # markiert extern serviert
        },
        "summary": model_summary(pb_results, thresholds.min_quality_pass_rate),
        "playbooks": {},
    }
    for pb in pb_results:
        data["playbooks"][pb.playbook] = {
            "total": pb.total, "passed": pb.passed, "failed": pb.failed,
            "pass_rate": pb.pass_rate, "mean_score": pb.mean_score,
            "duration_seconds": pb.duration_seconds,
            "knockouts": [r.to_dict() for r in pb.knockouts],
            "results": [r.to_dict() for r in pb.results],
        }
    safe = name.replace("/", "_").replace(" ", "_")
    path = run_dir / f"{safe}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="Kommagetrennte Report-Namen (Filter)")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--report-dir", default=str(TP / "reports"))
    # Weicher Deckel gegen das stündliche $15-Failsafe des Keys: pausiert VOR dem
    # 429, sobald die gemessene Laufkosten die Schwelle erreichen (0 = aus).
    ap.add_argument("--max-run-cost", type=float, default=0.0)
    args = ap.parse_args()

    _load_env()
    cfg = TestplanConfig.load()
    loader = TestDataLoader(cfg.testdata_dir)
    client = _proxy_client()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    run_dir = Path(args.report_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cohort = COHORT
    if args.models:
        want = {m.strip() for m in args.models.split(",")}
        cohort = [c for c in COHORT if c[0] in want]
    if not cohort:
        print("Keine Modelle nach Filter.", file=sys.stderr)
        sys.exit(1)

    print(f"Run {run_id} → {run_dir}")
    print(f"Judge: {JUDGE_MODEL} (reasoning_effort=minimal)")
    print(f"Kohorte: {len(cohort)} Modelle × {len(PLAYBOOKS)} Playbooks\n")

    grand = {"cost": 0.0}
    for name, proxy_id, p_in, p_out in cohort:
        for k in ("subj_in", "subj_out", "subj_cost"):
            TALLY[k] = 0 if k != "subj_cost" else 0.0
        j_in0, j_out0, jc0 = TALLY["judge_in"], TALLY["judge_out"], TALLY["judge_cost"]
        model = ModelConfig(name=name, profile=proxy_id, machine="proxy")
        pb_results: list[PlaybookResult] = []
        t_model = time.monotonic()
        for pb in PLAYBOOKS:
            t0 = time.monotonic()
            print(f"  [{name}] {pb} …", flush=True)
            results = run_playbook(pb, model, loader, client, proxy_id, client)
            pr = PlaybookResult(playbook=pb, model=name, results=results,
                                duration_seconds=time.monotonic() - t0)
            pb_results.append(pr)
            print(f"       {pr.passed}/{pr.total} pass  ø{pr.mean_score:.2f}  "
                  f"{pr.duration_seconds:.0f}s", flush=True)
        path = write_report(run_dir, name, proxy_id, pb_results, cfg.thresholds)
        # Kosten: exakt aus LiteLLM-Headern; Fallback Preistabelle wenn Header fehlten.
        subj_cost = TALLY["subj_cost"] or (
            TALLY["subj_in"] / 1e6 * p_in + TALLY["subj_out"] / 1e6 * p_out)
        j_in = TALLY["judge_in"] - j_in0
        j_out = TALLY["judge_out"] - j_out0
        judge_cost = (TALLY["judge_cost"] - jc0) or (
            j_in / 1e6 * JUDGE_PRICE_IN + j_out / 1e6 * JUDGE_PRICE_OUT)
        src = "hdr" if TALLY["subj_cost"] else "tab"
        grand["cost"] += subj_cost + judge_cost
        summ = model_summary(pb_results, cfg.thresholds.min_quality_pass_rate)
        spend = f" | key-spend ${BUDGET['spend']:.2f}/${BUDGET['max']:.0f}" if BUDGET["max"] else ""
        print(f"  ✓ {name}: {summ['overall']} {summ['passed']}/{summ['total_tests']}  "
              f"| subj {TALLY['subj_in']}/{TALLY['subj_out']}tok ${subj_cost:.3f}  "
              f"| judge {j_in}/{j_out}tok ${judge_cost:.3f}  [{src}]{spend}  "
              f"| {time.monotonic() - t_model:.0f}s → {path.name}\n", flush=True)
        if BUDGET["stop"]:
            print("⛔ Budget-Cap (429) erreicht — restliche Modelle übersprungen.", flush=True)
            break
        if args.max_run_cost and grand["cost"] >= args.max_run_cost:
            print(f"⏸  Weicher Deckel ${args.max_run_cost:.2f} erreicht (${grand['cost']:.2f}) — "
                  f"pausiert vor dem Stundenlimit. Rest später erneut starten.", flush=True)
            break

    print(f"═══ Gesamtkosten (gemessen): ${grand['cost']:.2f} ═══")
    print(f"Judge-Tokens gesamt: in {TALLY['judge_in']}, out {TALLY['judge_out']}")


if __name__ == "__main__":
    main()
