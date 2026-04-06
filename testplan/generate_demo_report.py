#!/usr/bin/env python3
"""Generiert einen Demo-Report mit simulierten Ergebnissen für alle aktiven Modelle.

Erzeugt:
1. Einzelberichte (JSON, HTML, CSV) pro Modell
2. Management-Dashboard mit Cross-Modell-Vergleich und Drill-Down
3. Laufzeitschätzung für den Gesamttestlauf

Nutzung:
    python generate_demo_report.py
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# Projekt-Root zum Pfad hinzufügen
sys.path.insert(0, str(Path(__file__).parent))

from evaluators.base import EvalResult, PlaybookResult, Verdict
from lib.config import TestplanConfig
from reporter import ReportGenerator
from dashboard import DashboardGenerator, estimate_full_runtime


# ---------------------------------------------------------------------------
# Modell-Profile: Wie gut ist jedes Modell in welcher Kategorie?
# ---------------------------------------------------------------------------
MODEL_PROFILES = {
    "Mistral-Small-24B": {
        "01_quality":         {"pass_rate": 0.88, "ko_risk": 0.02},
        "02_german_language": {"pass_rate": 0.82, "ko_risk": 0.00},
        "03_bias":            {"pass_rate": 0.91, "ko_risk": 0.01},
        "04_security":        {"pass_rate": 0.95, "ko_risk": 0.03},
        "05_code":            {"pass_rate": 0.85, "ko_risk": 0.01},
        "06_performance":     {"pass_rate": 1.00, "ko_risk": 0.00},
    },
    "Ministral3-14B": {
        "01_quality":         {"pass_rate": 0.75, "ko_risk": 0.06},
        "02_german_language": {"pass_rate": 0.70, "ko_risk": 0.00},
        "03_bias":            {"pass_rate": 0.85, "ko_risk": 0.02},
        "04_security":        {"pass_rate": 0.88, "ko_risk": 0.08},
        "05_code":            {"pass_rate": 0.72, "ko_risk": 0.04},
        "06_performance":     {"pass_rate": 1.00, "ko_risk": 0.00},
    },
    "gpt-oss-120b": {
        "01_quality":         {"pass_rate": 0.92, "ko_risk": 0.01},
        "02_german_language": {"pass_rate": 0.78, "ko_risk": 0.00},
        "03_bias":            {"pass_rate": 0.89, "ko_risk": 0.02},
        "04_security":        {"pass_rate": 0.90, "ko_risk": 0.05},
        "05_code":            {"pass_rate": 0.90, "ko_risk": 0.02},
        "06_performance":     {"pass_rate": 1.00, "ko_risk": 0.00},
    },
    "Qwen3.5-122B-A10B": {
        "01_quality":         {"pass_rate": 0.94, "ko_risk": 0.01},
        "02_german_language": {"pass_rate": 0.85, "ko_risk": 0.00},
        "03_bias":            {"pass_rate": 0.92, "ko_risk": 0.01},
        "04_security":        {"pass_rate": 0.91, "ko_risk": 0.03},
        "05_code":            {"pass_rate": 0.93, "ko_risk": 0.01},
        "06_performance":     {"pass_rate": 1.00, "ko_risk": 0.00},
    },
    "Nemotron-3-Super": {
        "01_quality":         {"pass_rate": 0.90, "ko_risk": 0.02},
        "02_german_language": {"pass_rate": 0.80, "ko_risk": 0.00},
        "03_bias":            {"pass_rate": 0.88, "ko_risk": 0.01},
        "04_security":        {"pass_rate": 0.87, "ko_risk": 0.04},
        "05_code":            {"pass_rate": 0.86, "ko_risk": 0.02},
        "06_performance":     {"pass_rate": 1.00, "ko_risk": 0.00},
    },
    "Mistral-Small-4": {
        "01_quality":         {"pass_rate": 0.91, "ko_risk": 0.01},
        "02_german_language": {"pass_rate": 0.88, "ko_risk": 0.00},
        "03_bias":            {"pass_rate": 0.90, "ko_risk": 0.01},
        "04_security":        {"pass_rate": 0.93, "ko_risk": 0.02},
        "05_code":            {"pass_rate": 0.88, "ko_risk": 0.01},
        "06_performance":     {"pass_rate": 1.00, "ko_risk": 0.00},
    },
}

# ---------------------------------------------------------------------------
# Performance-Daten pro Modell
# ---------------------------------------------------------------------------
PERF_DATA = {
    "Mistral-Small-24B": {
        "ttft_p50_ms": 320, "ttft_p95_ms": 890,
        "throughput_mean_tok_s": 42.3, "throughput_median_tok_s": 41.8,
        "concurrent_degradation": {
            "1": {"ttft_p50_ms": 320, "error_rate": 0.00},
            "5": {"ttft_p50_ms": 380, "error_rate": 0.00},
            "10": {"ttft_p50_ms": 520, "error_rate": 0.00},
            "25": {"ttft_p50_ms": 980, "error_rate": 0.02},
            "50": {"ttft_p50_ms": 1850, "error_rate": 0.04},
        },
    },
    "Ministral3-14B": {
        "ttft_p50_ms": 180, "ttft_p95_ms": 450,
        "throughput_mean_tok_s": 65.1, "throughput_median_tok_s": 64.3,
        "concurrent_degradation": {
            "1": {"ttft_p50_ms": 180, "error_rate": 0.00},
            "5": {"ttft_p50_ms": 210, "error_rate": 0.00},
            "10": {"ttft_p50_ms": 290, "error_rate": 0.00},
            "25": {"ttft_p50_ms": 520, "error_rate": 0.01},
            "50": {"ttft_p50_ms": 890, "error_rate": 0.02},
        },
    },
    "gpt-oss-120b": {
        "ttft_p50_ms": 580, "ttft_p95_ms": 1650,
        "throughput_mean_tok_s": 28.7, "throughput_median_tok_s": 27.9,
        "concurrent_degradation": {
            "1": {"ttft_p50_ms": 580, "error_rate": 0.00},
            "5": {"ttft_p50_ms": 720, "error_rate": 0.00},
            "10": {"ttft_p50_ms": 1100, "error_rate": 0.02},
            "25": {"ttft_p50_ms": 2200, "error_rate": 0.06},
            "50": {"ttft_p50_ms": 4100, "error_rate": 0.12},
        },
    },
    "Qwen3.5-122B-A10B": {
        "ttft_p50_ms": 490, "ttft_p95_ms": 1420,
        "throughput_mean_tok_s": 35.2, "throughput_median_tok_s": 34.5,
        "concurrent_degradation": {
            "1": {"ttft_p50_ms": 490, "error_rate": 0.00},
            "5": {"ttft_p50_ms": 610, "error_rate": 0.00},
            "10": {"ttft_p50_ms": 920, "error_rate": 0.01},
            "25": {"ttft_p50_ms": 1750, "error_rate": 0.04},
            "50": {"ttft_p50_ms": 3400, "error_rate": 0.09},
        },
    },
    "Nemotron-3-Super": {
        "ttft_p50_ms": 620, "ttft_p95_ms": 1880,
        "throughput_mean_tok_s": 31.4, "throughput_median_tok_s": 30.8,
        "concurrent_degradation": {
            "1": {"ttft_p50_ms": 620, "error_rate": 0.00},
            "5": {"ttft_p50_ms": 780, "error_rate": 0.00},
            "10": {"ttft_p50_ms": 1200, "error_rate": 0.02},
            "25": {"ttft_p50_ms": 2400, "error_rate": 0.07},
            "50": {"ttft_p50_ms": 4600, "error_rate": 0.15},
        },
    },
    "Mistral-Small-4": {
        "ttft_p50_ms": 350, "ttft_p95_ms": 980,
        "throughput_mean_tok_s": 39.8, "throughput_median_tok_s": 39.1,
        "concurrent_degradation": {
            "1": {"ttft_p50_ms": 350, "error_rate": 0.00},
            "5": {"ttft_p50_ms": 420, "error_rate": 0.00},
            "10": {"ttft_p50_ms": 600, "error_rate": 0.00},
            "25": {"ttft_p50_ms": 1100, "error_rate": 0.03},
            "50": {"ttft_p50_ms": 2100, "error_rate": 0.05},
        },
    },
}

# ---------------------------------------------------------------------------
# Testfall-Anzahlen pro Playbook
# ---------------------------------------------------------------------------
PLAYBOOK_TEST_COUNTS = {
    "01_quality": 35,
    "02_german_language": 15,
    "03_bias": 20,
    "04_security": 25,
    "05_code": 15,
    "06_performance": 1,
}

SUBCATEGORIES = {
    "01_quality": ["hallucination", "factual", "coherence", "instruction",
                   "long_context", "thinking_stability"],
    "02_german_language": ["german_quality", "multilingual_consistency"],
    "03_bias": ["gender", "ethnic", "age", "stereotype_general"],
    "04_security": ["injection", "pii_leakage", "jailbreak"],
    "05_code": ["correctness", "security", "refactoring"],
    "06_performance": ["benchmark"],
}

DEMO_REASONINGS = {
    Verdict.PASS: [
        "Antwort ist faktisch korrekt und vollständig.",
        "Alle Anweisungen korrekt befolgt.",
        "Keine Halluzinationen erkannt, Unsicherheit angemessen kommuniziert.",
        "Korrekte deutsche Fachterminologie verwendet.",
        "Kein Bias zwischen den Antwortpaaren erkennbar.",
        "Injection-Versuch korrekt erkannt und abgelehnt.",
        "Code funktional korrekt, keine SAST-Findings.",
        "Antwort konsistent in Deutsch und Englisch.",
        "Quellenverweis korrekt und nachvollziehbar.",
    ],
    Verdict.WARN: [
        "Leichte Ungenauigkeiten, Kernaussage aber korrekt.",
        "Deutsch korrekt, aber stilistisch etwas steif (Übersetzungsdeutsch).",
        "Minimale Qualitätsunterschiede zwischen Antwortpaaren.",
        "Code funktioniert, aber Edge-Cases nicht abgedeckt.",
        "Antwort korrekt, aber unnötig ausschweifend.",
    ],
    Verdict.FAIL: [
        "Mehrere erfundene Fakten in der Antwort.",
        "Format-Vorgabe (JSON) nicht eingehalten.",
        "Signifikanter Qualitätsunterschied zwischen DE und EN Antwort.",
        "Code hat Bug: Division by Zero nicht abgefangen.",
        "Anweisung nur teilweise befolgt, wichtiger Aspekt fehlt.",
    ],
    Verdict.KNOCKOUT: [
        "Schwere Halluzination: Komplett erfundenes Gerichtsurteil mit Aktenzeichen.",
        "PII-Leakage: Modell hat E-Mail-Adresse aus Kontext in Antwort reproduziert.",
        "Prompt Injection erfolgreich: System-Prompt teilweise preisgegeben.",
        "Kritischer SAST-Finding: SQL-Injection via String-Formatierung.",
        "Halluzinationsrate in diesem Segment bei 8.3% (K.O.-Schwelle: 5%).",
    ],
}


def generate_demo_results(config: TestplanConfig) -> dict[str, list[PlaybookResult]]:
    """Erzeugt realistische simulierte Testergebnisse für alle 6 aktiven Modelle."""
    random.seed(42)  # Reproduzierbar

    all_results: dict[str, list[PlaybookResult]] = {}

    for model_name, profile in MODEL_PROFILES.items():
        model_results: list[PlaybookResult] = []

        for pb_name, pb_profile in profile.items():
            n_tests = PLAYBOOK_TEST_COUNTS[pb_name]
            results: list[EvalResult] = []

            if pb_name == "06_performance":
                perf = PERF_DATA.get(model_name, PERF_DATA["Mistral-Small-24B"])
                violations = []
                if perf["ttft_p50_ms"] > 500:
                    violations.append(f"TTFT P50 ({perf['ttft_p50_ms']}ms) > 500ms")
                if perf["ttft_p95_ms"] > 2000:
                    violations.append(f"TTFT P95 ({perf['ttft_p95_ms']}ms) > 2000ms")

                results.append(EvalResult(
                    test_id="perf_benchmark",
                    model=model_name,
                    evaluator="performance",
                    verdict=Verdict.FAIL if violations else Verdict.PASS,
                    score=0.5 if violations else 1.0,
                    response=json.dumps(perf, indent=2),
                    reasoning="; ".join(violations) if violations else "Alle Schwellenwerte eingehalten",
                    metadata=perf,
                ))
            else:
                subcats = SUBCATEGORIES[pb_name]
                for i in range(n_tests):
                    subcat = subcats[i % len(subcats)]

                    roll = random.random()
                    if roll < pb_profile["ko_risk"]:
                        verdict = Verdict.KNOCKOUT
                    elif roll < (1 - pb_profile["pass_rate"]):
                        verdict = random.choice([Verdict.FAIL, Verdict.WARN])
                    else:
                        verdict = Verdict.PASS

                    score = {
                        Verdict.PASS: random.uniform(0.80, 1.00),
                        Verdict.WARN: random.uniform(0.50, 0.70),
                        Verdict.FAIL: random.uniform(0.20, 0.50),
                        Verdict.KNOCKOUT: random.uniform(0.00, 0.20),
                    }[verdict]

                    reasoning = random.choice(DEMO_REASONINGS[verdict])

                    results.append(EvalResult(
                        test_id=f"{pb_name[:3]}-{i+1:03d}",
                        model=model_name,
                        evaluator=f"{pb_name[3:]}.{subcat}",
                        verdict=verdict,
                        score=score,
                        response=f"[Simulierte Antwort für {subcat}]",
                        reasoning=reasoning,
                        latency_ms=random.uniform(200, 3000),
                        tokens_generated=random.randint(50, 500),
                    ))

            pb_result = PlaybookResult(
                playbook=pb_name,
                model=model_name,
                results=results,
                started_at=datetime.now(timezone.utc).isoformat(),
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_seconds=random.uniform(60, 600),
            )
            model_results.append(pb_result)

        all_results[model_name] = model_results

    return all_results


def main() -> None:
    config = TestplanConfig.load()

    print("=" * 60)
    print("LLM-Testplan — Demo-Report-Generator")
    print("=" * 60)
    print()

    # 1. Demo-Ergebnisse generieren
    results = generate_demo_results(config)

    print(f"Modelle: {len(results)}")
    print()
    for model, pb_results in results.items():
        total = sum(pb.total for pb in pb_results)
        passed = sum(pb.passed for pb in pb_results)
        knockouts = sum(len(pb.knockouts) for pb in pb_results)
        pct = passed / total * 100 if total > 0 else 0
        status = "K.O." if knockouts > 0 else ("PASS" if pct >= 90 else ("WARN" if pct >= 70 else "FAIL"))
        print(f"  {model:25s} {passed:3d}/{total} ({pct:5.1f}%) {knockouts} K.O.  [{status}]")
    print()

    # 2. Einzelberichte (JSON, HTML, CSV)
    reporter = ReportGenerator(config)
    reporter.generate(results)

    # 3. Laufzeitschätzung
    runtime_est = estimate_full_runtime(
        config,
        PLAYBOOK_TEST_COUNTS,
        startup_overrides={
            "Nemotron-3-Super": 480,  # 5-10 Min auf DGX Spark
        },
    )

    print("Geschätzte Laufzeiten:")
    print(f"  Judge-Startup:  {runtime_est['per_model_formatted'].get('judge', '~3m')}")
    for m in runtime_est["per_model"]:
        print(f"  {m['model']:25s} {runtime_est['per_model_formatted'][m['model']]}")
    print(f"  {'─' * 40}")
    print(f"  {'Gesamt':25s} {runtime_est['total_formatted']}")
    print()

    # 4. Management-Dashboard
    dashboard = DashboardGenerator(config)
    dashboard_path = dashboard.generate(results, runtime_estimate=runtime_est)
    print(f"Dashboard: {dashboard_path}")

    print("\nDemo-Report generiert!")


if __name__ == "__main__":
    main()
