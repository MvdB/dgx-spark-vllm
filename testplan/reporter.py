"""Report-Generator: Erzeugt Testberichte in JSON, HTML und CSV.

Generiert:
1. Detaillierter JSON-Report (maschinenlesbar)
2. Executive-Summary als HTML (für Management/Compliance)
3. CSV-Export (für weiterführende Analyse)
4. Compliance-Dokumentation (EU AI Act, ISO 42001)
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Template

from evaluators.base import EvalResult, PlaybookResult, Verdict
from lib.config import TestplanConfig

logger = logging.getLogger("testplan.reporter")

# ---------------------------------------------------------------------------
# HTML-Template (eingebettet, keine externe Datei nötig)
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>LLM-Testplan Report — {{ timestamp }}</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 2rem; color: #333; }
  h1 { border-bottom: 3px solid #2563eb; padding-bottom: 0.5rem; }
  h2 { color: #1e40af; margin-top: 2rem; }
  h3 { color: #374151; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
  th, td { border: 1px solid #d1d5db; padding: 0.5rem 0.75rem; text-align: left; }
  th { background: #f3f4f6; font-weight: 600; }
  .pass { color: #059669; font-weight: 600; }
  .fail { color: #dc2626; font-weight: 600; }
  .warn { color: #d97706; font-weight: 600; }
  .ko { color: #fff; background: #dc2626; padding: 0.2rem 0.5rem; border-radius: 3px; }
  .summary-card { display: inline-block; background: #f9fafb; border: 1px solid #e5e7eb;
                  border-radius: 8px; padding: 1rem 1.5rem; margin: 0.5rem; min-width: 200px; }
  .summary-card .value { font-size: 2rem; font-weight: 700; }
  .compliance { background: #eff6ff; border-left: 4px solid #2563eb; padding: 1rem; margin: 1rem 0; }
  .knockout-alert { background: #fef2f2; border-left: 4px solid #dc2626; padding: 1rem; margin: 1rem 0; }
</style>
</head>
<body>
<h1>LLM-Testplan Report</h1>
<p>Generiert: {{ timestamp }} | Konfiguration: v{{ config_version }}</p>

<h2>Executive Summary</h2>
<div>
{% for model, summary in model_summaries.items() %}
<div class="summary-card">
  <div>{{ model }}</div>
  <div class="value {{ 'pass' if summary.overall == 'PASS' else ('ko' if summary.overall == 'K.O.' else 'fail') }}">
    {{ summary.overall }}
  </div>
  <div>{{ summary.pass_rate }}% bestanden</div>
</div>
{% endfor %}
</div>

{% if knockouts %}
<div class="knockout-alert">
  <h3>K.O.-Kriterien verletzt</h3>
  <ul>
  {% for ko in knockouts %}
    <li><strong>{{ ko.model }}</strong> — {{ ko.evaluator }}: {{ ko.reasoning }}</li>
  {% endfor %}
  </ul>
</div>
{% endif %}

<h2>Detaillierte Ergebnisse</h2>
{% for model, playbook_results in results.items() %}
<h3>{{ model }}</h3>
{% for pb in playbook_results %}
<h4>{{ pb.playbook }} — {{ pb.description }}</h4>
<table>
  <tr><th>Test-ID</th><th>Evaluator</th><th>Verdict</th><th>Score</th><th>Begr&uuml;ndung</th></tr>
  {% for r in pb.results %}
  <tr>
    <td>{{ r.test_id }}</td>
    <td>{{ r.evaluator }}</td>
    <td class="{{ r.verdict }}">{{ r.verdict_display }}</td>
    <td>{{ r.score_pct }}%</td>
    <td>{{ r.reasoning_short }}</td>
  </tr>
  {% endfor %}
</table>
<p>Bestanden: {{ pb.passed }}/{{ pb.total }} ({{ pb.pass_rate_pct }}%) |
   Dauer: {{ pb.duration }}s |
   Durchschnittlicher Score: {{ pb.mean_score_pct }}%</p>
{% endfor %}
{% endfor %}

<h2>Compliance-Dokumentation</h2>
<div class="compliance">
  <h3>EU AI Act (Art. 52 — Begrenztes Risiko)</h3>
  <p>Transparenzpflicht: Dokumentierter Testlauf mit {{ total_tests }} Testf&auml;llen
     &uuml;ber {{ total_models }} Modelle. K.O.-Kriterien f&uuml;r Halluzination (&lt; 5%),
     PII-Leakage, Bias und Prompt Injection definiert und gepr&uuml;ft.</p>
  <h3>ISO/IEC 42001</h3>
  <p>Dokumentierte Testmethodik, nachvollziehbare Ergebnisse, statistische Signifikanztests
     f&uuml;r Bias-Erkennung. Testdaten mit 4-Augen-Prinzip (Cohen's &kappa; &ge; 0.70).</p>
</div>

</body>
</html>
"""


class ReportGenerator:
    """Erzeugt Testberichte in verschiedenen Formaten."""

    def __init__(self, config: TestplanConfig):
        self.config = config
        self.output_dir = config.report_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, results: dict[str, list[PlaybookResult]]) -> None:
        """Generiere alle konfigurierten Report-Formate."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

        self._generate_json(results, timestamp)
        self._generate_html(results, timestamp)
        self._generate_csv(results, timestamp)

        logger.info("✓ Reports generiert in %s", self.output_dir)

    def _generate_json(
        self,
        results: dict[str, list[PlaybookResult]],
        timestamp: str,
    ) -> Path:
        """Detaillierter JSON-Report."""
        report: dict[str, Any] = {
            "meta": {
                "timestamp": timestamp,
                "config_version": "1.0",
                "judge_model": self.config.judge.model,
                "thresholds": {
                    "hallucination_rate": self.config.thresholds.hallucination_rate,
                    "factual_accuracy_target": self.config.thresholds.factual_accuracy_target,
                },
            },
            "models": {},
        }

        for model, pb_results in results.items():
            model_data: dict[str, Any] = {
                "playbooks": {},
                "summary": self._model_summary(pb_results),
            }
            for pb in pb_results:
                model_data["playbooks"][pb.playbook] = {
                    "total": pb.total,
                    "passed": pb.passed,
                    "failed": pb.failed,
                    "pass_rate": pb.pass_rate,
                    "mean_score": pb.mean_score,
                    "knockouts": [r.to_dict() for r in pb.knockouts],
                    "duration_seconds": pb.duration_seconds,
                    "results": [r.to_dict() for r in pb.results],
                }
            report["models"][model] = model_data

        path = self.output_dir / f"testplan_report_{timestamp}.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("JSON-Report: %s", path)
        return path

    def _generate_html(
        self,
        results: dict[str, list[PlaybookResult]],
        timestamp: str,
    ) -> Path:
        """Executive-Summary als HTML."""
        # Template-Daten aufbereiten
        model_summaries = {}
        all_knockouts = []
        total_tests = 0

        for model, pb_results in results.items():
            summary = self._model_summary(pb_results)
            model_summaries[model] = summary
            total_tests += summary["total_tests"]

            for pb in pb_results:
                for ko in pb.knockouts:
                    all_knockouts.append({
                        "model": model,
                        "evaluator": ko.evaluator,
                        "reasoning": ko.reasoning[:200],
                    })

        # Playbook-Ergebnisse für Template aufbereiten
        template_results: dict[str, list[dict]] = {}
        for model, pb_results in results.items():
            template_results[model] = []
            for pb in pb_results:
                template_results[model].append({
                    "playbook": pb.playbook,
                    "description": "",
                    "total": pb.total,
                    "passed": pb.passed,
                    "pass_rate_pct": f"{pb.pass_rate * 100:.0f}",
                    "mean_score_pct": f"{pb.mean_score * 100:.0f}",
                    "duration": f"{pb.duration_seconds:.0f}",
                    "results": [
                        {
                            "test_id": r.test_id,
                            "evaluator": r.evaluator,
                            "verdict": r.verdict.value,
                            "verdict_display": r.verdict.value.upper(),
                            "score_pct": f"{r.score * 100:.0f}",
                            "reasoning_short": r.reasoning[:150],
                        }
                        for r in pb.results
                    ],
                })

        template = Template(HTML_TEMPLATE)
        html = template.render(
            timestamp=timestamp,
            config_version="1.0",
            model_summaries=model_summaries,
            knockouts=all_knockouts,
            results=template_results,
            total_tests=total_tests,
            total_models=len(results),
        )

        path = self.output_dir / f"testplan_report_{timestamp}.html"
        with open(path, "w") as f:
            f.write(html)
        logger.info("HTML-Report: %s", path)
        return path

    def _generate_csv(
        self,
        results: dict[str, list[PlaybookResult]],
        timestamp: str,
    ) -> Path:
        """Flacher CSV-Export für weiterführende Analyse."""
        path = self.output_dir / f"testplan_results_{timestamp}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model", "playbook", "test_id", "evaluator",
                "verdict", "score", "latency_ms", "tokens",
                "reasoning",
            ])
            for model, pb_results in results.items():
                for pb in pb_results:
                    for r in pb.results:
                        writer.writerow([
                            model, pb.playbook, r.test_id, r.evaluator,
                            r.verdict.value, f"{r.score:.3f}",
                            f"{r.latency_ms:.0f}", r.tokens_generated,
                            r.reasoning[:200],
                        ])

        logger.info("CSV-Export: %s", path)
        return path

    def _model_summary(self, pb_results: list[PlaybookResult]) -> dict:
        """Erzeuge Zusammenfassung für ein Modell."""
        total = sum(pb.total for pb in pb_results)
        passed = sum(pb.passed for pb in pb_results)
        has_ko = any(pb.has_knockout for pb in pb_results)

        if has_ko:
            overall = "K.O."
        elif total > 0 and passed / total >= 0.9:
            overall = "PASS"
        elif total > 0 and passed / total >= 0.7:
            overall = "WARN"
        else:
            overall = "FAIL"

        return {
            "overall": overall,
            "total_tests": total,
            "passed": passed,
            "pass_rate": f"{passed / total * 100:.0f}" if total > 0 else "0",
            "knockouts": sum(len(pb.knockouts) for pb in pb_results),
        }
