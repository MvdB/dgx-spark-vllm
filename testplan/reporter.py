"""Report-Generator: Erzeugt Testberichte in Markdown, HTML und JSON.

Ausgabe-Struktur pro Run:
  reports/<YYYY-MM-DD_HHMM>/
    README.md              — Dashboard: alle Modelle, Scores, Links (Git-Primärdoku)
    <Modell>.md            — Detailbericht pro Modell (Freigabe-Doku)
    <Modell>.html          — Quick-Check im Browser
    <Modell>.json          — Rohdaten für Auswertungen

README.md und Einzel-Reports werden nach jedem Modell aktualisiert,
sodass Teilergebnisse auch bei Abbruch erhalten bleiben.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Template

from evaluators.base import EvalResult, PlaybookResult, Verdict
from lib.config import ModelConfig, TestplanConfig

logger = logging.getLogger("testplan.reporter")

# Emoji-Mapping für Verdicts
VERDICT_EMOJI = {
    "pass": "✅",
    "fail": "❌",
    "warn": "⚠️",
    "knockout": "🚫",
    "error": "💥",
}

OVERALL_EMOJI = {
    "PASS": "✅",
    "WARN": "⚠️",
    "FAIL": "❌",
    "K.O.": "🚫",
    "–": "⏳",
}

PLAYBOOK_SHORT = {
    "01_quality": "Quality",
    "02_german_language": "German",
    "03_bias": "Bias",
    "04_security": "Security",
    "05_code": "Code",
    "06_performance": "Perf",
    "07_hsf_calibration": "HSF",
}

# ---------------------------------------------------------------------------
# HTML-Template (pro Modell, Quick-Check)
# ---------------------------------------------------------------------------
HTML_MODEL_TEMPLATE = """\
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>{{ model_name }} — LLM-Testbericht {{ date }}</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 2rem; color: #333; }
  h1 { border-bottom: 3px solid #2563eb; padding-bottom: 0.5rem; }
  h2 { color: #1e40af; margin-top: 2rem; }
  h3 { color: #374151; }
  a { color: #2563eb; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }
  th, td { border: 1px solid #d1d5db; padding: 0.4rem 0.7rem; text-align: left; }
  th { background: #f3f4f6; font-weight: 600; }
  .pass  { color: #059669; font-weight: 600; }
  .fail  { color: #dc2626; font-weight: 600; }
  .warn  { color: #d97706; font-weight: 600; }
  .ko    { color: #fff; background: #dc2626; padding: 0.15rem 0.4rem; border-radius: 3px; }
  .error { color: #6b7280; }
  .meta  { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 6px;
           padding: 0.8rem 1.2rem; margin-bottom: 1.5rem; }
  .meta p { margin: 0.2rem 0; }
  .ko-box { background: #fef2f2; border-left: 4px solid #dc2626;
            padding: 0.8rem 1rem; margin: 0.8rem 0; border-radius: 0 4px 4px 0; }
  .summary-row td { font-weight: 600; background: #f9fafb; }
  .approval { background: #eff6ff; border-left: 4px solid #2563eb;
              padding: 0.8rem 1rem; margin-top: 2rem; }
</style>
</head>
<body>
<p><a href="index.html">← Zurück zum Dashboard</a></p>
<h1>{{ model_name }}</h1>

<div class="meta">
  <p><strong>Testdatum:</strong> {{ date }}</p>
  <p><strong>Profil:</strong> {{ profile }}</p>
  <p><strong>Judge:</strong> {{ judge }}</p>
  <p><strong>Gesamtstatus:</strong> <span class="{{ overall_class }}">{{ overall }}</span></p>
</div>

<h2>Playbook-Ergebnisse</h2>
<table>
  <tr><th>Playbook</th><th>Beschreibung</th><th>Bestanden</th><th>Score ⌀</th><th>K.O.</th><th>Dauer</th></tr>
  {% for pb in playbook_summary %}
  <tr class="summary-row">
    <td>{{ pb.name }}</td>
    <td>{{ pb.description }}</td>
    <td class="{{ pb.rate_class }}">{{ pb.passed }}/{{ pb.total }} ({{ pb.pass_rate }}%)</td>
    <td>{{ pb.mean_score }}%</td>
    <td>{% if pb.knockouts > 0 %}<span class="ko">{{ pb.knockouts }} K.O.</span>{% else %}–{% endif %}</td>
    <td>{{ pb.duration }}min</td>
  </tr>
  {% endfor %}
</table>

{% if all_knockouts %}
<h2>K.O.-Verletzungen</h2>
{% for ko in all_knockouts %}
<div class="ko-box">
  <strong>{{ ko.test_id }}</strong> ({{ ko.evaluator }})<br>
  {{ ko.reasoning }}
</div>
{% endfor %}
{% endif %}

<h2>Detailergebnisse</h2>
{% for pb in playbook_details %}
<h3>{{ pb.name }} — {{ pb.description }}</h3>
<table>
  <tr><th>Test-ID</th><th>Evaluator</th><th>Verdict</th><th>Score</th><th>Begründung</th></tr>
  {% for r in pb.results %}
  <tr>
    <td>{{ r.test_id }}</td>
    <td>{{ r.evaluator }}</td>
    <td class="{{ r.verdict_class }}">{{ r.verdict }}</td>
    <td>{{ r.score }}%</td>
    <td style="font-size:0.8rem">{{ r.reasoning }}</td>
  </tr>
  {% endfor %}
</table>
{% endfor %}

<div class="approval">
  <h2>Freigabe</h2>
  <p><strong>Status:</strong> ⬜ Ausstehend</p>
  <p><strong>Freigabe durch:</strong> ___________</p>
  <p><strong>Datum:</strong> ___________</p>
  <p><strong>Bemerkungen:</strong></p>
</div>

</body>
</html>
"""


class ReportGenerator:
    """Erzeugt Testberichte in Markdown, HTML und JSON."""

    def __init__(self, config: TestplanConfig, run_timestamp: str | None = None):
        self.config = config
        if run_timestamp is None:
            run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
        self.run_timestamp = run_timestamp
        self.run_dir = config.report_dir / run_timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Playbook-Beschreibungen für Lookups
        self._pb_descriptions: dict[str, str] = {
            p.name: p.description for p in config.playbooks
        }

    def generate_single(
        self,
        model: ModelConfig,
        pb_results: list[PlaybookResult],
    ) -> None:
        """Schreibe alle Reports für ein Modell. JSON immer zuerst (kritisch),
        HTML/MD fail-safe — ein Fehler bricht den Lauf nicht ab."""
        safe_name = model.name.replace("/", "_").replace(" ", "_")
        # JSON: Rohdaten — muss immer geschrieben werden
        self._write_json(safe_name, model, pb_results)
        # HTML + MD: Aufbereitung — Fehler loggen, nicht weiterwerfen
        for fmt, writer in [("html", self._write_html), ("md", self._write_markdown)]:
            try:
                writer(safe_name, model, pb_results)
            except Exception as e:
                logger.error("Report-Fehler (%s) für %s: %s", fmt, model.name, e, exc_info=True)
        logger.info("✓ Reports für %s in %s", model.name, self.run_dir)

    def update_dashboard(self, all_results: dict[str, tuple[ModelConfig, list[PlaybookResult]]]) -> None:
        """Schreibe/aktualisiere README.md mit allen bisherigen Modellen."""
        try:
            self._write_readme(all_results)
        except Exception as e:
            logger.error("Dashboard-Fehler: %s", e, exc_info=True)
        logger.info("✓ Dashboard aktualisiert: %s", self.run_dir / "README.md")

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def _write_markdown(
        self,
        safe_name: str,
        model: ModelConfig,
        pb_results: list[PlaybookResult],
    ) -> None:
        summary = self._model_summary(pb_results)
        date = self.run_timestamp.replace("_", " ")
        lines: list[str] = []

        lines += [
            f"# {model.name} — Testbericht",
            "",
            f"**Testdatum:** {date}  ",
            f"**Profil:** `{model.profile}`  ",
            f"**Judge:** {self.config.judge.model}  ",
            f"**Gesamtstatus:** {OVERALL_EMOJI.get(summary['overall'], '')} {summary['overall']}  ",
            f"**Testfälle:** {summary['total_tests']} ({summary['passed']} bestanden, "
            f"{summary['knockouts']} K.O.)",
            "",
            "---",
            "",
            "## Playbook-Ergebnisse",
            "",
            "| Playbook | Beschreibung | Bestanden | Score ⌀ | K.O. | Dauer |",
            "|----------|-------------|-----------|---------|------|-------|",
        ]

        for pb in pb_results:
            desc = self._pb_descriptions.get(pb.playbook, "")
            ko_str = f"🚫 {len(pb.knockouts)}" if pb.knockouts else "–"
            duration = f"{pb.duration_seconds / 60:.0f}min"
            lines.append(
                f"| {pb.playbook} | {desc} "
                f"| {pb.passed}/{pb.total} ({pb.pass_rate * 100:.0f}%) "
                f"| {pb.mean_score * 100:.0f}% "
                f"| {ko_str} "
                f"| {duration} |"
            )

        # K.O.-Details
        all_kos = [ko for pb in pb_results for ko in pb.knockouts]
        if all_kos:
            lines += ["", "---", "", "## K.O.-Verletzungen", ""]
            for ko in all_kos:
                lines += [
                    f"### 🚫 {ko.test_id} ({ko.evaluator})",
                    "",
                    f"> {str(ko.reasoning)[:500]}",
                    "",
                ]

        # Detailergebnisse
        lines += ["---", "", "## Detailergebnisse", ""]
        for pb in pb_results:
            desc = self._pb_descriptions.get(pb.playbook, "")
            lines += [
                f"### {pb.playbook} — {desc}",
                "",
                "| Test-ID | Verdict | Score | Begründung |",
                "|---------|---------|-------|------------|",
            ]
            for r in pb.results:
                emoji = VERDICT_EMOJI.get(r.verdict.value, "")
                reasoning = str(r.reasoning)[:200].replace("\n", " ").replace("|", "\\|")
                lines.append(
                    f"| {r.test_id} "
                    f"| {emoji} {r.verdict.value.upper()} "
                    f"| {r.score * 100:.0f}% "
                    f"| {reasoning} |"
                )
            lines.append("")

        # Freigabe-Sektion
        lines += [
            "---",
            "",
            "## Freigabe",
            "",
            "> **Status:** ⬜ Ausstehend  ",
            "> **Freigabe durch:** ___________  ",
            "> **Datum:** ___________  ",
            "> **Bemerkungen:**  ",
            "",
        ]

        path = self.run_dir / f"{safe_name}.md"
        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_readme(
        self,
        all_results: dict[str, tuple[ModelConfig, list[PlaybookResult]]],
    ) -> None:
        lines: list[str] = [
            "# LLM-Testplan — Auswertung",
            "",
            f"**Testlauf:** {self.run_timestamp.replace('_', ' ')} UTC  ",
            f"**Judge:** {self.config.judge.model}  ",
            f"**Modelle:** {len(all_results)}  ",
            "",
            "---",
            "",
            "## Übersicht",
            "",
        ]

        # Playbook-Spalten ermitteln (welche wurden getestet?)
        all_pb_names: list[str] = []
        for _, (_, pb_results) in all_results.items():
            for pb in pb_results:
                if pb.playbook not in all_pb_names:
                    all_pb_names.append(pb.playbook)

        header_cols = " | ".join(PLAYBOOK_SHORT.get(p, p) for p in all_pb_names)
        sep_cols = " | ".join("---" for _ in all_pb_names)
        lines.append(f"| Modell | Status | {header_cols} | Bericht |")
        lines.append(f"|--------|--------|{sep_cols}|---------|")

        for model_name, (model, pb_results) in all_results.items():
            summary = self._model_summary(pb_results)
            overall = f"{OVERALL_EMOJI.get(summary['overall'], '')} {summary['overall']}"
            pb_map = {pb.playbook: pb for pb in pb_results}
            pb_cells = []
            for pb_name in all_pb_names:
                if pb_name in pb_map:
                    pb = pb_map[pb_name]
                    ko = " 🚫" if pb.knockouts else ""
                    pb_cells.append(f"{pb.pass_rate * 100:.0f}%{ko}")
                else:
                    pb_cells.append("–")
            pb_str = " | ".join(pb_cells)
            safe_name = model_name.replace("/", "_").replace(" ", "_")
            lines.append(
                f"| {model_name} | {overall} | {pb_str} | [{model_name}]({safe_name}.md) |"
            )

        # Freigabenstatus-Tabelle
        lines += [
            "",
            "---",
            "",
            "## Freigabenstatus",
            "",
            "| Modell | Gesamtstatus | K.O. | Freigabe durch | Datum |",
            "|--------|-------------|------|----------------|-------|",
        ]
        for model_name, (model, pb_results) in all_results.items():
            summary = self._model_summary(pb_results)
            overall = f"{OVERALL_EMOJI.get(summary['overall'], '')} {summary['overall']}"
            ko_count = summary["knockouts"]
            lines.append(
                f"| {model_name} | {overall} | {ko_count} | | |"
            )

        lines += [
            "",
            "---",
            "",
            f"*Generiert: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC*",
            "",
        ]

        path = self.run_dir / "README.md"
        path.write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------
    # HTML (Quick-Check, pro Modell)
    # ------------------------------------------------------------------

    def _write_html(
        self,
        safe_name: str,
        model: ModelConfig,
        pb_results: list[PlaybookResult],
    ) -> None:
        summary = self._model_summary(pb_results)
        overall_class = {
            "PASS": "pass", "WARN": "warn", "FAIL": "fail", "K.O.": "ko"
        }.get(summary["overall"], "")

        playbook_summary = []
        for pb in pb_results:
            rate = pb.pass_rate * 100
            playbook_summary.append({
                "name": pb.playbook,
                "description": self._pb_descriptions.get(pb.playbook, ""),
                "passed": pb.passed,
                "total": pb.total,
                "pass_rate": f"{rate:.0f}",
                "mean_score": f"{pb.mean_score * 100:.0f}",
                "knockouts": len(pb.knockouts),
                "duration": f"{pb.duration_seconds / 60:.0f}",
                "rate_class": "pass" if rate >= 80 else ("warn" if rate >= 60 else "fail"),
            })

        all_knockouts = [
            {"test_id": ko.test_id, "evaluator": ko.evaluator, "reasoning": str(ko.reasoning)[:400]}
            for pb in pb_results for ko in pb.knockouts
        ]

        playbook_details = []
        for pb in pb_results:
            playbook_details.append({
                "name": pb.playbook,
                "description": self._pb_descriptions.get(pb.playbook, ""),
                "results": [
                    {
                        "test_id": r.test_id,
                        "evaluator": r.evaluator,
                        "verdict": r.verdict.value.upper(),
                        "verdict_class": r.verdict.value,
                        "score": f"{r.score * 100:.0f}",
                        "reasoning": str(r.reasoning)[:200],
                    }
                    for r in pb.results
                ],
            })

        html = Template(HTML_MODEL_TEMPLATE).render(
            model_name=model.name,
            profile=model.profile,
            judge=self.config.judge.model,
            date=self.run_timestamp.replace("_", " "),
            overall=f"{OVERALL_EMOJI.get(summary['overall'], '')} {summary['overall']}",
            overall_class=overall_class,
            playbook_summary=playbook_summary,
            all_knockouts=all_knockouts,
            playbook_details=playbook_details,
        )

        path = self.run_dir / f"{safe_name}.html"
        path.write_text(html, encoding="utf-8")

    # ------------------------------------------------------------------
    # JSON (Rohdaten, pro Modell)
    # ------------------------------------------------------------------

    def _write_json(
        self,
        safe_name: str,
        model: ModelConfig,
        pb_results: list[PlaybookResult],
    ) -> None:
        data: dict[str, Any] = {
            "meta": {
                "run": self.run_timestamp,
                "model": model.name,
                "profile": model.profile,
                "judge": self.config.judge.model,
                "thresholds": {
                    "hallucination_rate": self.config.thresholds.hallucination_rate,
                    "factual_accuracy_target": self.config.thresholds.factual_accuracy_target,
                },
            },
            "summary": self._model_summary(pb_results),
            "playbooks": {},
        }

        for pb in pb_results:
            data["playbooks"][pb.playbook] = {
                "total": pb.total,
                "passed": pb.passed,
                "failed": pb.failed,
                "pass_rate": pb.pass_rate,
                "mean_score": pb.mean_score,
                "duration_seconds": pb.duration_seconds,
                "knockouts": [r.to_dict() for r in pb.knockouts],
                "results": [r.to_dict() for r in pb.results],
            }

        path = self.run_dir / f"{safe_name}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def _model_summary(self, pb_results: list[PlaybookResult]) -> dict:
        total = sum(pb.total for pb in pb_results)
        passed = sum(pb.passed for pb in pb_results)
        knockouts = sum(len(pb.knockouts) for pb in pb_results)
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
            "knockouts": knockouts,
        }
