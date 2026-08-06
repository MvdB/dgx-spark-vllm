#!/usr/bin/env python3
"""LLM-Testplan Orchestrator.

Haupteinstiegspunkt für den automatisierten Testlauf.

Ablauf:
1. Konfiguration laden
2. Judge-Modell auf Spark A starten/prüfen
3. Für jedes aktive Modell:
   a. Modell auf Spark B starten
   b. Alle aktivierten Playbooks durchlaufen
   c. K.O.-Kriterien nach jedem Playbook prüfen → ggf. Early-Abort
   d. Modell stoppen
4. Konsolidierten Report generieren

Nutzung:
    python orchestrator.py                          # Alle aktiven Modelle
    python orchestrator.py --tags cohort_a          # Nur Kohorte A
    python orchestrator.py --models "Mistral-Small-24B,gpt-oss-120b"
    python orchestrator.py --playbooks 01_quality,04_security
    python orchestrator.py --dry-run                # Nur Konfiguration prüfen
    python orchestrator.py --endpoint http://localhost:8000  # Gegen laufenden Endpoint
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# .env laden falls vorhanden (vor Config-Import, damit Env-Vars gesetzt sind)
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from lib.config import ModelConfig, TestplanConfig
from lib.testdata import TestDataLoader
from lib.vllm_control import VllmController, VllmInstance

from evaluators.base import EvalResult, PlaybookResult, Verdict
from evaluators.bias import BiasEvaluator
from evaluators.code_eval import CodeEvaluator
from evaluators.guard import GuardEvaluator
from evaluators.performance import HSFCalibrator, PerformanceEvaluator
from evaluators.quality import QualityEvaluator
from evaluators.security import PromptfooRunner, SecurityEvaluator

from reporter import ReportGenerator

logger = logging.getLogger("testplan")


def _query_model_id(client: object, fallback: str = "") -> str:
    """Frage die erste Modell-ID vom laufenden vLLM-Endpoint ab."""
    try:
        models = client.models.list()
        model_id = models.data[0].id
        logger.info("Modell-ID vom Endpoint: %s", model_id)
        return model_id
    except Exception as e:
        logger.warning("Konnte Modell-ID nicht abfragen, nutze Fallback '%s': %s", fallback, e)
        return fallback


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt)
    # Externe Libraries leiser stellen
    logging.getLogger("paramiko").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


class TestplanOrchestrator:
    """Koordiniert den gesamten Testlauf."""

    def __init__(self, config: TestplanConfig, args: argparse.Namespace):
        self.config = config
        self.args = args
        self.controller = VllmController(ssh_user=config.judge.ssh_user)
        self.loader = TestDataLoader(config.testdata_dir)
        run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
        self.reporter = ReportGenerator(config, run_timestamp=run_ts)
        self.all_results: dict[str, tuple[ModelConfig, list[PlaybookResult]]] = {}

    def run(self) -> int:
        """Haupteinstieg. Returns Exit-Code (0=OK, 1=Failures, 2=K.O.)."""
        logger.info("=" * 70)
        logger.info("LLM-Testplan gestartet: %s", datetime.now(timezone.utc).isoformat())
        logger.info("=" * 70)

        # Modelle bestimmen
        models = self._select_models()
        if not models:
            logger.error("Keine aktiven Modelle gefunden!")
            return 1

        logger.info("Zu testende Modelle: %s", [m.name for m in models])

        # Playbooks bestimmen. Ohne --playbooks: alle enabled. Mit --playbooks:
        # exakt die genannten — auch wenn sie per Default enabled:false sind
        # (explizit angefordert schlägt Default). So laufen Sonder-Playbooks wie
        # 08_guardrails oder 07_hsf_calibration nur auf ausdrücklichen Wunsch.
        if self.args.playbooks:
            wanted = [n.strip() for n in self.args.playbooks.split(",")]
            playbooks = [p for p in self.config.playbooks if p.name in wanted]
        else:
            playbooks = [p for p in self.config.playbooks if p.enabled]
        logger.info("Aktive Playbooks: %s", [p.name for p in playbooks])

        if self.args.dry_run:
            self._print_dry_run(models, playbooks)
            return 0

        # Testdaten laden und validieren
        errors = self.loader.validate()
        if errors:
            logger.warning("Testdaten-Validierung: %d Probleme", len(errors))
            for e in errors[:10]:
                logger.warning("  - %s", e)

        try:
            # Judge starten (falls nicht --endpoint Modus)
            judge_instance = None
            if not self.args.endpoint:
                judge_instance = self.controller.ensure_judge_running(self.config.judge)

            exit_code = 0

            for model in models:
                logger.info("-" * 70)
                logger.info("MODELL: %s", model.name)
                logger.info("-" * 70)

                try:
                    model_exit = self._test_model(model, playbooks, judge_instance)
                    exit_code = max(exit_code, model_exit)
                except Exception as e:
                    logger.error(
                        "Modell %s übersprungen — Fehler: %s", model.name, e, exc_info=True
                    )
                    exit_code = max(exit_code, 1)

                # Einzel-Report + Dashboard nach jedem Modell aktualisieren
                if model.name in self.all_results:
                    try:
                        self.reporter.generate_single(model, self.all_results[model.name][1])
                        self.reporter.update_dashboard(self.all_results)
                    except Exception as e:
                        logger.warning("Report für %s fehlgeschlagen: %s", model.name, e)

            # Finales Dashboard
            self.reporter.update_dashboard(self.all_results)

        finally:
            self.controller.close()

        logger.info("=" * 70)
        logger.info("Testplan abgeschlossen. Exit-Code: %d", exit_code)
        return exit_code

    def _select_models(self) -> list[ModelConfig]:
        """Wähle Modelle basierend auf CLI-Argumenten."""
        if self.args.models:
            names = [n.strip() for n in self.args.models.split(",")]
            by_name = {m.name: m for m in self.config.models}
            return [by_name[n] for n in names if n in by_name]
        if self.args.tags:
            tags = [t.strip() for t in self.args.tags.split(",")]
            return self.config.active_models(tags=tags)
        return self.config.active_models()

    def _test_model(
        self,
        model: ModelConfig,
        playbooks: list,
        judge_instance: VllmInstance | None,
    ) -> int:
        """Teste ein einzelnes Modell durch alle Playbooks.

        Returns:
            0 = alle bestanden, 1 = Warnungen/Failures, 2 = K.O.
        """
        target_instance = None
        model_results: list[PlaybookResult] = []

        try:
            # Modell starten (oder externen Endpoint nutzen)
            if self.args.endpoint:
                from openai import OpenAI
                target_client = OpenAI(
                    base_url=f"{self.args.endpoint}/v1",
                    api_key="not-needed",
                )
                target_model = _query_model_id(target_client, fallback=f"/hf_models/{model.profile}")
            else:
                target_instance = self.controller.start_model(
                    self.config.target, model,
                )
                target_client = target_instance.get_client()
                target_model = target_instance.resolve_model_id()

            # Judge-Client
            judge_client = None
            judge_model = None
            if judge_instance:
                judge_client = judge_instance.get_client()
                judge_model = self.config.judge.model

            exit_code = 0

            for pb in playbooks:
                logger.info("Playbook: %s — %s", pb.name, pb.description)
                started = datetime.now(timezone.utc).isoformat()
                t0 = time.monotonic()

                results = self._run_playbook(
                    pb.name, model, target_client, target_model,
                    judge_client, judge_model,
                    default_system_prompt=model.system_prompt,
                )

                pb_result = PlaybookResult(
                    playbook=pb.name,
                    model=model.name,
                    results=results,
                    started_at=started,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    duration_seconds=time.monotonic() - t0,
                )
                model_results.append(pb_result)

                # K.O.-Prüfung
                if pb_result.has_knockout:
                    logger.error(
                        "⛔ K.O.-KRITERIUM VERLETZT in %s für %s!",
                        pb.name, model.name,
                    )
                    for ko in pb_result.knockouts:
                        logger.error("  → %s: %s", ko.test_id, ko.reasoning[:200])
                    exit_code = 2

                    if not self.args.continue_after_ko:
                        logger.info("Abbruch für %s (--continue-after-ko nicht gesetzt)", model.name)
                        break

                elif pb_result.pass_rate < 0.8:
                    exit_code = max(exit_code, 1)

                logger.info(
                    "  → %s: %d/%d bestanden (%.0f%%), %d K.O.",
                    pb.name,
                    pb_result.passed,
                    pb_result.total,
                    pb_result.pass_rate * 100,
                    len(pb_result.knockouts),
                )

        finally:
            # Modell stoppen und Cooldown
            if target_instance:
                self.controller.stop_model(target_instance)
                logger.info(
                    "Cooldown: %ds...", self.config.target.cooldown_seconds
                )
                time.sleep(self.config.target.cooldown_seconds)

        self.all_results[model.name] = (model, model_results)
        return exit_code

    def _run_playbook(
        self,
        playbook_name: str,
        model: ModelConfig,
        target_client,
        target_model: str,
        judge_client,
        judge_model: str | None,
        default_system_prompt: str = "",
    ) -> list[EvalResult]:
        """Führe ein Playbook aus und gib Ergebnisse zurück.

        Mapping Playbooks → Testdaten-Verzeichnisse:
          01_quality         → quality/ + long_context/
          02_german_language → german_language/ + quality/ (DE-Filter)
          03_bias            → bias/
          04_security        → security/
          05_code            → code/
          06_performance     → performance/ (Prompts als Workload, keine Judge-Bewertung)
          07_hsf_calibration → performance/ (gleiche Prompts, andere Maschine)
        """
        results: list[EvalResult] = []

        if playbook_name == "01_quality":
            evaluator = QualityEvaluator(
                target_client=target_client,
                target_model=target_model,
                judge_client=judge_client,
                judge_model=judge_model,
                default_system_prompt=default_system_prompt,
                sampling=model.sampling,
                chat_template_kwargs=model.chat_template_kwargs,
            )
            # Qualitäts-Testfälle aus allen Subkategorien
            cases = self.loader.load_category("quality")
            # Long-Context-Tests gehören auch zur Qualitätsbewertung
            cases += self.loader.load_category("long_context")
            results = evaluator.evaluate_batch(cases)

        elif playbook_name == "02_german_language":
            evaluator = QualityEvaluator(
                target_client=target_client,
                target_model=target_model,
                judge_client=judge_client,
                judge_model=judge_model,
                default_system_prompt=default_system_prompt,
                sampling=model.sampling,
                chat_template_kwargs=model.chat_template_kwargs,
            )
            # Dedizierte deutsche Sprachtests
            cases = self.loader.load_category("german_language")
            # Plus deutsche Tests aus dem Quality-Pool
            quality_de = self.loader.filter_cases(
                self.loader.load_category("quality"),
                language="de",
                subcategory="german_quality",
            )
            cases += quality_de
            results = evaluator.evaluate_batch(cases)

        elif playbook_name == "03_bias":
            evaluator = BiasEvaluator(
                target_client=target_client,
                target_model=target_model,
                judge_client=judge_client,
                judge_model=judge_model,
                default_system_prompt=default_system_prompt,
                sampling=model.sampling,
                chat_template_kwargs=model.chat_template_kwargs,
            )
            cases = self.loader.load_category("bias")
            results = evaluator.evaluate_batch(cases)

            # Statistische Signifikanz prüfen
            sig = BiasEvaluator.aggregate_significance(results)
            if sig["significant"]:
                logger.warning(
                    "⚠ Statistisch signifikanter Bias erkannt (p=%.4f)",
                    sig["p_value"],
                )
                # K.O. markieren
                results.append(EvalResult(
                    test_id="bias_aggregate",
                    model=target_model,
                    evaluator="bias.significance",
                    verdict=Verdict.KNOCKOUT,
                    score=0.0,
                    response="",
                    reasoning=(
                        f"Statistisch signifikanter Bias: χ²={sig['chi2']:.2f}, "
                        f"p={sig['p_value']:.4f}, Pass-Rate={sig['pass_rate']:.2%}"
                    ),
                ))

        elif playbook_name == "04_security":
            evaluator = SecurityEvaluator(
                target_client=target_client,
                target_model=target_model,
                judge_client=judge_client,
                judge_model=judge_model,
                default_system_prompt=default_system_prompt,
                sampling=model.sampling,
                chat_template_kwargs=model.chat_template_kwargs,
            )
            cases = self.loader.load_category("security")
            results = evaluator.evaluate_batch(cases)

        elif playbook_name == "05_code":
            evaluator = CodeEvaluator(
                target_client=target_client,
                target_model=target_model,
                judge_client=judge_client,
                judge_model=judge_model,
                default_system_prompt=default_system_prompt,
                sampling=model.sampling,
                chat_template_kwargs=model.chat_template_kwargs,
            )
            cases = self.loader.load_category("code")
            results = evaluator.evaluate_batch(cases)

        elif playbook_name == "06_performance":
            perf = PerformanceEvaluator(
                base_url=f"http://{self.config.target.host}:{self.config.target.port}"
                if not self.args.endpoint else self.args.endpoint,
                model=target_model,
            )
            report = asyncio.run(perf.run_benchmark())
            violations = perf.check_thresholds(
                report, self.config.thresholds, model.params_b, model.tags
            )

            summary = report.summary()
            # Performance-Schwellenwerte sind Richtwerte für große Modelle auf DGX Spark —
            # Verletzungen werden als WARN gewertet, nicht als FAIL/K.O.
            verdict = Verdict.WARN if violations else Verdict.PASS
            tput_threshold = self.config.thresholds.throughput_for_model(
                model.params_b, model.tags
            )
            by_type = summary.get("throughput_by_type", {})
            tput_detail = "  ".join(
                f"{t}={v} tok/s" for t, v in by_type.items()
            )
            metrics_line = (
                f"TTFT P50={summary['ttft_p50_ms']:.0f}ms  "
                f"P95={summary['ttft_p95_ms']:.0f}ms  |  "
                f"Throughput {summary['throughput_median_tok_s']:.1f} tok/s Median "
                f"(Threshold {tput_threshold} tok/s)  |  "
                + (f"{tput_detail}  |  " if tput_detail else "")
                + f"Messungen={summary['n_measurements']}  Fehler={summary['n_errors']}"
            )
            if violations:
                reasoning = metrics_line + "  |  ⚠ " + "; ".join(violations)
            else:
                reasoning = metrics_line + "  |  ✓ Alle Schwellenwerte eingehalten"
            results.append(EvalResult(
                test_id="perf_benchmark",
                model=target_model,
                evaluator="performance",
                verdict=verdict,
                score=1.0 if not violations else 0.7,
                response=json.dumps(summary, indent=2),
                reasoning=reasoning,
                metadata=summary,
            ))

        elif playbook_name == "08_guardrails":
            protocol = model.guard_protocol
            if not protocol:
                logger.error(
                    "Modell '%s' hat kein guard_protocol gesetzt — 08_guardrails "
                    "übersprungen", model.name)
                return results
            th = self.config.thresholds
            evaluator = GuardEvaluator(
                target_client=target_client,
                target_model=target_model,
                judge_client=None,          # Guardrails braucht keinen Judge
                judge_model=None,
                default_system_prompt="",   # Guard-Modelle bringen ihr eigenes Format mit
                sampling=model.sampling,
                chat_template_kwargs=model.chat_template_kwargs,
                guard_protocol=protocol,
                threshold=th.guard_shieldstral_threshold,
                reasoning_effort=th.guard_safeguard_reasoning_effort,
                ko_max_false_negative_rate=th.guard_max_false_negative_rate,
                ko_max_false_positive_rate=th.guard_max_false_positive_rate,
            )
            cases = self.loader.load_category("guardrails")
            results = evaluator.evaluate_batch(cases)

        elif playbook_name == "07_hsf_calibration":
            logger.info("HSF-Kalibrierung — benötigt Zugang zu Produktionsmaschine")
            # Wird separat konfiguriert und ausgeführt
            pass

        else:
            logger.warning("Unbekanntes Playbook: %s", playbook_name)

        return results

    def _print_dry_run(self, models: list[ModelConfig], playbooks: list) -> None:
        """Zeige was passieren WÜRDE, ohne etwas auszuführen."""
        print("\n=== DRY RUN ===\n")
        print(f"Judge: {self.config.judge.model} auf {self.config.judge.host}")
        print(f"Target: {self.config.target.host}\n")

        print("Modelle:")
        for m in models:
            print(f"  - {m.name} ({m.profile}) → {m.machine}")
            if m.notes:
                print(f"    ⚠ {m.notes}")

        print(f"\nPlaybooks ({len(playbooks)}):")
        for p in playbooks:
            print(f"  - {p.name}: {p.description} ({p.timeout_minutes}min)")

        # Testdaten-Zusammenfassung
        all_cases = self.loader.load_all()
        total = sum(len(c) for c in all_cases.values())
        print(f"\nTestdaten: {total} Fälle")
        for cat, cases in all_cases.items():
            if cases:
                langs = {}
                for c in cases:
                    langs[c.language] = langs.get(c.language, 0) + 1
                print(f"  - {cat}: {len(cases)} ({langs})")

        errors = self.loader.validate()
        if errors:
            print(f"\n⚠ Validierungsprobleme: {len(errors)}")
            for e in errors[:5]:
                print(f"  - {e}")

        print("\n=== ENDE DRY RUN ===")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-Testplan Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Pfad zur testplan.yaml (Standard: config/testplan.yaml)",
    )
    parser.add_argument(
        "--models", "-m",
        default=None,
        help="Komma-separierte Modellnamen (Standard: alle aktiven)",
    )
    parser.add_argument(
        "--tags", "-t",
        default=None,
        help="Komma-separierte Tags zum Filtern (z.B. cohort_a,dense)",
    )
    parser.add_argument(
        "--playbooks", "-p",
        default=None,
        help="Komma-separierte Playbook-Namen (z.B. 01_quality,04_security)",
    )
    parser.add_argument(
        "--endpoint", "-e",
        default=None,
        help="Externer vLLM-Endpoint (überspringt automatisches Starten)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Nur Konfiguration anzeigen, nichts ausführen",
    )
    parser.add_argument(
        "--continue-after-ko",
        action="store_true",
        help="Nach K.O.-Kriterium weitertesten (Standard: Abbruch)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Debug-Logging aktivieren",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    config = TestplanConfig.load(args.config)
    orchestrator = TestplanOrchestrator(config, args)
    exit_code = orchestrator.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
