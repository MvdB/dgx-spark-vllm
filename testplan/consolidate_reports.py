"""Konsolidiert kanonische Einzel-Reports aller Kohorten in ein gemeinsames Verzeichnis.

Liest die besten verfügbaren JSON-Reports pro Modell, rekonstruiert PlaybookResult-Objekte
und generiert README.md + index.html über die bestehende Reporter-Infrastruktur.

Ausgabe: testplan/reports/<RUN_TIMESTAMP>/
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Sicherstellen, dass testplan/ im Python-Pfad liegt
sys.path.insert(0, str(Path(__file__).parent))

from evaluators.base import EvalResult, PlaybookResult, Verdict
from lib.config import ModelConfig, TestplanConfig
from reporter import ReportGenerator

REPORTS_DIR = Path(__file__).parent / "reports"

# ---------------------------------------------------------------------------
# Kanonisches Mapping: Modell → bester verfügbarer Run
# Priorität: letzter stabiler Lauf ohne Judge-Ausfall / Container-Crash
# ---------------------------------------------------------------------------
CANONICAL: dict[str, tuple[str, str, list[str], int]] = {
    # name → (run_dir, profile, tags, params_b)
    # --- Re-Run alle Kohorten (2026-04-15_1341) — 20 Hal-Tests, Safety-Refusal-Fix ---
    "Gemma-4-26B-A4B":          ("2026-04-15_1341", "google--gemma-4-26B-A4B-it",                          ["cohort_b", "moe", "instruct"],        26),
    "Gemma-4-31B":              ("2026-04-15_1341", "google--gemma-4-31B-it",                               ["cohort_b", "dense", "instruct"],      31),
    "Qwen3.5-35B-A3B":          ("2026-04-15_1341", "Qwen--Qwen3.5-35B-A3B-GPTQ-Int4",                    ["cohort_b", "moe", "instruct"],        35),
    "Qwen3.5-122B-A10B":        ("2026-04-15_1341", "Qwen--Qwen3.5-122B-A10B-GPTQ-Int4",                  ["cohort_b", "moe", "large"],           122),
    "Nemotron-3-Super":         ("2026-04-15_1341", "nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",     ["cohort_b", "mamba_moe"],             120),
    "Mistral-Small-4":          ("2026-04-15_1341", "mistralai--Mistral-Small-4-119B-2603-NVFP4",          ["cohort_b", "moe", "custom_vllm"],    119),
    "gpt-oss-120b":             ("2026-04-15_1341", "openai--gpt-oss-120b",                                ["cohort_b", "dense", "instruct"],     120),
    "Qwen3.5-27B":              ("2026-04-15_1341", "Qwen--Qwen3.5-27B-GPTQ-Int4",                        ["cohort_c", "dense", "instruct"],      27),
    "Nemotron-3-Nano-30B":      ("2026-04-15_1341", "nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",         ["cohort_c", "mamba_moe", "reasoning"], 30),
    "Ministral-3-14B-Reasoning":("2026-04-15_1341", "mistralai--Ministral-3-14B-Reasoning-2512",          ["cohort_c", "dense", "reasoning"],     14),
    "Ministral-3-14B-Instruct": ("2026-04-15_1341", "mistralai--Ministral-3-14B-Instruct-2512",           ["cohort_c", "dense", "instruct"],      14),
    "Mistral-Small-3.2-24B":    ("2026-04-15_1341", "mistralai--Mistral-Small-3.2-24B-Instruct-2506",     ["cohort_c", "dense", "instruct"],      24),
    "Gemma-4-E4B":              ("2026-04-15_1341", "google--gemma-4-E4B-it",                              ["cohort_c", "dense", "multimodal"],     4),
    "Gemma-4-E2B":              ("2026-04-15_1341", "google--gemma-4-E2B-it",                              ["cohort_c", "dense", "multimodal"],     2),
}


def load_playbook_results(json_path: Path, model_name: str) -> list[PlaybookResult]:
    """Rekonstruiert PlaybookResult-Liste aus einem gespeicherten JSON-Report."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    playbooks = data.get("playbooks", {})
    results: list[PlaybookResult] = []

    for pb_name, pb_data in playbooks.items():
        eval_results: list[EvalResult] = []
        for r in pb_data.get("results", []):
            try:
                verdict = Verdict(r.get("verdict", "error"))
            except ValueError:
                verdict = Verdict.ERROR
            eval_results.append(EvalResult(
                test_id=r.get("test_id", ""),
                model=r.get("model", model_name),
                evaluator=r.get("evaluator", ""),
                verdict=verdict,
                score=float(r.get("score", 0.0)),
                response=r.get("response", ""),
                reasoning=r.get("reasoning", ""),
                latency_ms=float(r.get("latency_ms", 0.0)),
                tokens_generated=int(r.get("tokens_generated", 0)),
                thinking=r.get("thinking", ""),
                response_type=r.get("response_type", "answer"),
                metadata=r.get("metadata", {}),
            ))

        results.append(PlaybookResult(
            playbook=pb_name,
            model=model_name,
            results=eval_results,
            duration_seconds=float(pb_data.get("duration_seconds", 0.0)),
        ))

    return results


def main() -> None:
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    out_dir = REPORTS_DIR / run_ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ausgabeverzeichnis: {out_dir}")

    # Minimale Config — nur für Reporter-Metadaten benötigt (keine Env-Vars erforderlich)
    from lib.config import (
        JudgeConfig, TargetConfig, Thresholds, PlaybookEntry,
    )
    config = TestplanConfig(
        judge=JudgeConfig(host="localhost", model="anthropic/claude-haiku-4-5"),
        target=TargetConfig(host="localhost"),
        models=[],
        thresholds=Thresholds(),
        playbooks=[
            PlaybookEntry(name="01_quality",        description="Halluzination, Faktentreue, Kohärenz, Instruktionsbefolgung"),
            PlaybookEntry(name="02_german_language", description="Deutsche Sprachqualität, Fachterminologie, Mehrsprachigkeit"),
            PlaybookEntry(name="03_bias",            description="Demografischer Bias, Stereotypen, Fairness"),
            PlaybookEntry(name="04_security",        description="Prompt Injection, PII-Leakage, Jailbreak-Resistenz"),
            PlaybookEntry(name="05_code",            description="Code-Generierung, Korrektheit, SAST-Prüfung"),
            PlaybookEntry(name="06_performance",     description="TTFT, Throughput, Latenz unter Last"),
        ],
        testdata_dir=REPORTS_DIR.parent / "testdata",
        report_dir=REPORTS_DIR,
        base_dir=REPORTS_DIR.parent,
    )

    reporter = ReportGenerator(config, run_timestamp=run_ts)

    all_results: dict[str, tuple[ModelConfig, list[PlaybookResult]]] = {}
    missing: list[str] = []

    for model_name, (run_dir, profile, tags, params_b) in CANONICAL.items():
        src_json = REPORTS_DIR / run_dir / f"{model_name}.json"
        if not src_json.exists():
            print(f"  ⚠️  Nicht gefunden: {src_json}")
            missing.append(model_name)
            continue

        print(f"  ✓  {model_name} ← {run_dir}")

        # JSON in Zielordner kopieren (Rohdaten immer erhalten)
        shutil.copy2(src_json, out_dir / src_json.name)

        # PlaybookResults rekonstruieren
        pb_results = load_playbook_results(src_json, model_name)

        model_cfg = ModelConfig(
            name=model_name,
            profile=profile,
            machine="machine_b",
            tags=tags,
            params_b=params_b,
        )

        # Einzel-Report (MD + HTML) aus Rohdaten regenerieren
        reporter.generate_single(model_cfg, pb_results)

        all_results[model_name] = (model_cfg, pb_results)

    # Dashboard (README.md + index.html) über alle Modelle
    reporter.update_dashboard(all_results)

    print(f"\nFertig: {len(all_results)} Modelle, {len(missing)} fehlend.")
    if missing:
        print(f"  Fehlend: {', '.join(missing)}")
    print(f"  README.md:  {out_dir / 'README.md'}")
    print(f"  index.html: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
