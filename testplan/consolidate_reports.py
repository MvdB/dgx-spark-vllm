"""Konsolidiert Einzel-Reports aller getesteten Modelle in ein gemeinsames Verzeichnis.

Erkennt automatisch den neuesten Run-Ordner pro Modell und generiert
README.md + index.html über die bestehende Reporter-Infrastruktur.

Ausgabe: testplan/reports/<RUN_TIMESTAMP>/
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluators.base import EvalResult, PlaybookResult, Verdict
from lib.config import ModelConfig, TestplanConfig
from reporter import ReportGenerator

REPORTS_DIR = Path(__file__).parent / "reports"

# ---------------------------------------------------------------------------
# Kanonisches Mapping: Modell → (profile, tags, params_b)
# Run-Verzeichnis wird automatisch ermittelt (neuester Lauf gewinnt).
# ---------------------------------------------------------------------------
CANONICAL: dict[str, tuple[str, list[str], int]] = {
    # name → (profile, tags, params_b)  — alphabetisch
    # Quant-Tags: "nvfp4" | "fp8" | "gptq_int4"  (kein Tag = BF16)
    "Gemma-4-26B-A4B":              ("google--gemma-4-26B-A4B-it",                            ["moe", "instruct"],                    26),
    "Gemma-4-31B":                  ("google--gemma-4-31B-it",                                 ["dense", "instruct"],                  31),
    "Gemma-4-31B-NVFP4":            ("RedHatAI--gemma-4-31B-it-NVFP4",                        ["dense", "instruct", "nvfp4"],         31),
    "Gemma-4-E2B":                  ("google--gemma-4-E2B-it",                                 ["dense", "multimodal"],                 2),
    "Gemma-4-E4B":                  ("google--gemma-4-E4B-it",                                 ["dense", "multimodal"],                 4),
    "LFM2.5-VL-450M":               ("LiquidAI--LFM2.5-VL-450M",                              ["dense", "multimodal"],                 0),
    "Ministral-3-14B-Instruct":     ("mistralai--Ministral-3-14B-Instruct-2512",               ["dense", "instruct"],                  14),
    "Ministral-3-14B-Reasoning":    ("mistralai--Ministral-3-14B-Reasoning-2512",              ["dense", "reasoning"],                 14),
    "Mistral-Small-3.2-24B":        ("mistralai--Mistral-Small-3.2-24B-Instruct-2506",         ["dense", "instruct"],                  24),
    "Mistral-Small-3.2-24B-NVFP4":  ("RedHatAI--Mistral-Small-3.2-24B-Instruct-2506-NVFP4",  ["dense", "instruct", "nvfp4"],         24),
    "Mistral-Small-4":              ("mistralai--Mistral-Small-4-119B-2603-NVFP4",             ["moe", "custom_vllm", "nvfp4"],       119),
    "Nemotron-3-Nano-30B":          ("nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",            ["mamba_moe", "reasoning", "fp8"],      30),
    "Nemotron-3-Super":             ("nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",        ["mamba_moe", "nvfp4"],                120),
    "Qwen3.5-0.8B":                 ("Qwen--Qwen3.5-0.8B",                                    ["dense", "instruct"],                   1),
    "Qwen3.5-2B":                   ("Qwen--Qwen3.5-2B",                                      ["dense", "instruct"],                   2),
    "Qwen3.5-4B":                   ("Qwen--Qwen3.5-4B",                                      ["dense", "instruct"],                   4),
    "Qwen3.5-9B":                   ("Qwen--Qwen3.5-9B",                                      ["dense", "instruct"],                   9),
    "Qwen3.5-27B":                  ("Qwen--Qwen3.5-27B-GPTQ-Int4",                           ["dense", "instruct", "gptq_int4"],     27),
    "Qwen3.5-35B-A3B":              ("Qwen--Qwen3.5-35B-A3B-GPTQ-Int4",                       ["moe", "instruct", "gptq_int4"],       35),
    "Qwen3.5-122B-A10B":            ("Qwen--Qwen3.5-122B-A10B-GPTQ-Int4",                     ["moe", "large", "gptq_int4"],         122),
    "Qwen3.6-35B-A3B-FP8":          ("Qwen--Qwen3.6-35B-A3B-FP8",                             ["moe", "instruct", "fp8"],             35),
    "gpt-oss-20b":                  ("openai--gpt-oss-20b",                                    ["dense", "instruct"],                  20),
    "gpt-oss-120b":                 ("openai--gpt-oss-120b",                                   ["dense", "instruct"],                 120),
    # --- Kohorte E ---
    "Devstral-Small-2-24B":         ("mistralai--Devstral-Small-2-24B-Instruct-2512",          ["dense", "instruct", "fp8"],           24),
    "Ministral-3-3B-Reasoning":     ("mistralai--Ministral-3-3B-Reasoning-2512",               ["dense", "reasoning"],                  3),
    "Ministral-3-8B-Instruct":      ("mistralai--Ministral-3-8B-Instruct-2512",                ["dense", "instruct", "fp8"],            8),
    "Ministral-3-8B-Reasoning":     ("mistralai--Ministral-3-8B-Reasoning-2512",               ["dense", "reasoning"],                  8),
    "Qwen3.5-27B-FP8":              ("Qwen--Qwen3.5-27B-FP8",                                  ["dense", "instruct", "fp8"],           27),
    "Qwen3.5-9B-GPTQ-Int4":         ("mssfj--Qwen3.5-9B-GPTQ-INT4",                            ["dense", "instruct", "gptq_int4"],     9),
}


def find_latest_run(model_name: str) -> Path | None:
    """Findet den neuesten Run-Ordner, der einen Report für model_name enthält."""
    if not REPORTS_DIR.exists():
        return None
    candidates = sorted(
        (d for d in REPORTS_DIR.iterdir() if d.is_dir() and (d / f"{model_name}.json").exists()),
        key=lambda d: d.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


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

    from lib.config import JudgeConfig, TargetConfig, Thresholds, PlaybookEntry
    config = TestplanConfig(
        judge=JudgeConfig(host="localhost", model="anthropic/claude-haiku-4-5"),
        target=TargetConfig(host="localhost"),
        models=[],
        thresholds=Thresholds(),
        playbooks=[
            PlaybookEntry(name="01_quality",         description="Halluzination, Faktentreue, Kohärenz, Instruktionsbefolgung"),
            PlaybookEntry(name="02_german_language",  description="Deutsche Sprachqualität, Fachterminologie, Mehrsprachigkeit"),
            PlaybookEntry(name="03_bias",             description="Demografischer Bias, Stereotypen, Fairness"),
            PlaybookEntry(name="04_security",         description="Prompt Injection, PII-Leakage, Jailbreak-Resistenz"),
            PlaybookEntry(name="05_code",             description="Code-Generierung, Korrektheit, SAST-Prüfung"),
            PlaybookEntry(name="06_performance",      description="TTFT, Throughput, Latenz unter Last"),
        ],
        testdata_dir=REPORTS_DIR.parent / "testdata",
        report_dir=REPORTS_DIR,
        base_dir=REPORTS_DIR.parent,
    )

    reporter = ReportGenerator(config, run_timestamp=run_ts)

    all_results: dict[str, tuple[ModelConfig, list[PlaybookResult]]] = {}
    missing: list[str] = []

    for model_name, (profile, tags, params_b) in CANONICAL.items():
        run_dir = find_latest_run(model_name)
        if run_dir is None:
            print(f"  -   {model_name} (kein Report gefunden)")
            missing.append(model_name)
            continue

        src_json = run_dir / f"{model_name}.json"
        print(f"  ✓  {model_name} ← {run_dir.name}")

        shutil.copy2(src_json, out_dir / src_json.name)

        pb_results = load_playbook_results(src_json, model_name)

        model_cfg = ModelConfig(
            name=model_name,
            profile=profile,
            machine="machine_b",
            tags=tags,
            params_b=params_b,
        )

        try:
            reporter.generate_single(model_cfg, pb_results)
        except Exception as e:
            print(f"  ⚠️  Einzel-Report {model_name} fehlgeschlagen: {e}")

        all_results[model_name] = (model_cfg, pb_results)

    try:
        reporter.update_dashboard(all_results)
    except Exception as e:
        print(f"  ⚠️  Dashboard fehlgeschlagen: {e}")

    print(f"\nFertig: {len(all_results)} Modelle, {len(missing)} noch ohne Report.")
    if missing:
        print(f"  Ausstehend: {', '.join(missing)}")
    print(f"  README.md:  {out_dir / 'README.md'}")
    print(f"  index.html: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
