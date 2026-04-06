"""Management-Dashboard: Cross-Modell-Vergleich mit Drill-Down.

Erzeugt ein interaktives HTML-Dashboard, das alle getesteten Modelle
nebeneinander vergleicht. Optimiert für Management-Präsentationen und
Compliance-Dokumentation.

Features:
- Executive Summary mit Ampel-Status pro Modell
- Vergleichsmatrix: Modelle × Playbooks (Pass-Rate, Score, K.O.)
- Performance-Vergleich (TTFT, Throughput, Concurrency)
- Drill-Down per Klick auf Modell-Details
- Laufzeitübersicht und -schätzung
- Compliance-Status und K.O.-Alerts
- Export-freundlich (Print-CSS)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Template

from evaluators.base import PlaybookResult, Verdict
from lib.config import TestplanConfig

# ---------------------------------------------------------------------------
# Playbook-Anzeigenamen
# ---------------------------------------------------------------------------
PLAYBOOK_LABELS = {
    "01_quality": "Qualität",
    "02_german_language": "Deutsch",
    "03_bias": "Bias",
    "04_security": "Security",
    "05_code": "Code",
    "06_performance": "Performance",
    "07_hsf_calibration": "HSF",
}

# ---------------------------------------------------------------------------
# Laufzeitschätzung
# ---------------------------------------------------------------------------
# Geschätzte Zeiten basieren auf typischen Werten für DGX Spark + vLLM
DEFAULT_STARTUP_TIMES = {
    # Modellgröße → geschätzte Startzeit in Sekunden
    "dense_small": 120,   # <15B: ~2 Minuten
    "dense_medium": 180,  # 15-30B: ~3 Minuten
    "moe_medium": 240,    # MoE bis 50B: ~4 Minuten
    "moe_large": 420,     # MoE >100B: ~7 Minuten
    "mamba_hybrid": 480,  # Mamba/Hybrid: ~8 Minuten (langsamer wegen State-Init)
}

# Geschätzte Verarbeitungszeit pro Testfall (Sekunden)
EVAL_TIME_PER_CASE = {
    "01_quality": 15,        # Target + Judge Query
    "02_german_language": 15,
    "03_bias": 20,           # Paired: 2× Target + Judge
    "04_security": 12,       # Target + Pattern/Judge
    "05_code": 25,           # Target + Execution + SAST + Judge
    "06_performance": 0,     # Separat berechnet
    "07_hsf_calibration": 0,
}

# Performance-Benchmark: Grundzeit
PERF_BENCHMARK_SECONDS = 300  # Warmup + Iterationen + Concurrency-Stufen


def estimate_model_runtime(
    model_name: str,
    model_tags: list[str],
    test_counts: dict[str, int],
    startup_override: float | None = None,
) -> dict[str, Any]:
    """Schätze Laufzeit für ein einzelnes Modell.

    Returns:
        Dict mit startup, per_playbook, cooldown, total (alles in Sekunden).
    """
    # Startup-Zeit bestimmen
    if startup_override:
        startup = startup_override
    elif "mamba_moe" in model_tags:
        startup = DEFAULT_STARTUP_TIMES["mamba_hybrid"]
    elif "moe" in model_tags and "large" in model_tags:
        startup = DEFAULT_STARTUP_TIMES["moe_large"]
    elif "moe" in model_tags:
        startup = DEFAULT_STARTUP_TIMES["moe_medium"]
    elif "dense" in model_tags:
        # Grobe Unterscheidung nach Name
        if any(x in model_name.lower() for x in ["14b", "12b"]):
            startup = DEFAULT_STARTUP_TIMES["dense_small"]
        else:
            startup = DEFAULT_STARTUP_TIMES["dense_medium"]
    else:
        startup = DEFAULT_STARTUP_TIMES["dense_medium"]

    playbook_times = {}
    for pb_name, n_cases in test_counts.items():
        if pb_name == "06_performance":
            playbook_times[pb_name] = PERF_BENCHMARK_SECONDS
        elif pb_name == "07_hsf_calibration":
            playbook_times[pb_name] = 600  # ~10 Min Kalibrierung
        else:
            per_case = EVAL_TIME_PER_CASE.get(pb_name, 15)
            playbook_times[pb_name] = n_cases * per_case

    cooldown = 30  # Standard-Cooldown

    total = startup + sum(playbook_times.values()) + cooldown

    return {
        "model": model_name,
        "startup_seconds": startup,
        "playbook_seconds": playbook_times,
        "cooldown_seconds": cooldown,
        "total_seconds": total,
    }


def estimate_full_runtime(
    config: TestplanConfig,
    test_counts: dict[str, int],
    startup_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Schätze Gesamtlaufzeit für alle aktiven Modelle.

    Args:
        config: Testplan-Konfiguration
        test_counts: Playbook-Name → Anzahl Testfälle
        startup_overrides: Modellname → manuelle Startup-Zeit in Sekunden

    Returns:
        Dict mit per_model, judge_startup, total, formatted.
    """
    overrides = startup_overrides or {}
    active = config.active_models()
    enabled_playbooks = [p.name for p in config.playbooks if p.enabled]
    filtered_counts = {k: v for k, v in test_counts.items() if k in enabled_playbooks}

    judge_startup = 180  # ~3 Minuten für Mistral-Small-24B Judge

    per_model = []
    for model in active:
        est = estimate_model_runtime(
            model.name,
            model.tags,
            filtered_counts,
            startup_override=overrides.get(model.name),
        )
        per_model.append(est)

    total = judge_startup + sum(m["total_seconds"] for m in per_model)

    return {
        "judge_startup_seconds": judge_startup,
        "per_model": per_model,
        "total_seconds": total,
        "total_formatted": _format_duration(total),
        "per_model_formatted": {
            m["model"]: _format_duration(m["total_seconds"]) for m in per_model
        },
    }


def _format_duration(seconds: float) -> str:
    """Formatiere Sekunden als 'Xh Ym'."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m"


# ---------------------------------------------------------------------------
# Dashboard HTML-Template
# ---------------------------------------------------------------------------
DASHBOARD_TEMPLATE = """\
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM-Testplan Dashboard — {{ timestamp_display }}</title>
<style>
  :root {
    --green: #059669; --green-bg: #ecfdf5; --green-border: #a7f3d0;
    --yellow: #d97706; --yellow-bg: #fffbeb; --yellow-border: #fde68a;
    --red: #dc2626; --red-bg: #fef2f2; --red-border: #fecaca;
    --blue: #2563eb; --blue-bg: #eff6ff; --blue-border: #bfdbfe;
    --gray: #6b7280; --gray-bg: #f9fafb; --gray-border: #e5e7eb;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 1400px; margin: 0 auto; padding: 1.5rem 2rem;
    color: #1f2937; background: #fff; line-height: 1.5;
  }

  /* --- Header --- */
  .header { border-bottom: 3px solid var(--blue); padding-bottom: 1rem; margin-bottom: 1.5rem; }
  .header h1 { font-size: 1.75rem; color: #111827; }
  .header .meta { color: var(--gray); font-size: 0.9rem; margin-top: 0.25rem; }

  /* --- Status Badges --- */
  .badge {
    display: inline-block; padding: 0.2rem 0.7rem; border-radius: 999px;
    font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.02em;
  }
  .badge-pass { background: var(--green-bg); color: var(--green); border: 1px solid var(--green-border); }
  .badge-warn { background: var(--yellow-bg); color: var(--yellow); border: 1px solid var(--yellow-border); }
  .badge-fail { background: var(--red-bg); color: var(--red); border: 1px solid var(--red-border); }
  .badge-ko { background: var(--red); color: #fff; border: 1px solid var(--red); }

  /* --- Cards --- */
  .card-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1rem; margin: 1.5rem 0;
  }
  .card {
    background: var(--gray-bg); border: 1px solid var(--gray-border);
    border-radius: 10px; padding: 1.25rem; text-align: center;
    transition: box-shadow 0.2s; cursor: pointer;
  }
  .card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
  .card .model-name { font-weight: 600; font-size: 1rem; margin-bottom: 0.5rem; color: #111827; }
  .card .score-big { font-size: 2.25rem; font-weight: 700; margin: 0.25rem 0; }
  .card .score-big.pass { color: var(--green); }
  .card .score-big.warn { color: var(--yellow); }
  .card .score-big.fail { color: var(--red); }
  .card .score-big.ko { color: var(--red); }
  .card .sub { color: var(--gray); font-size: 0.85rem; }
  .card-pass { border-left: 4px solid var(--green); }
  .card-warn { border-left: 4px solid var(--yellow); }
  .card-fail { border-left: 4px solid var(--red); }
  .card-ko { border-left: 4px solid var(--red); background: var(--red-bg); }

  /* --- Sections --- */
  .section { margin: 2rem 0; }
  .section h2 {
    font-size: 1.25rem; color: #1e40af; margin-bottom: 1rem;
    padding-bottom: 0.5rem; border-bottom: 2px solid var(--blue-bg);
  }

  /* --- Comparison Matrix --- */
  .matrix { width: 100%; border-collapse: separate; border-spacing: 0; margin: 1rem 0; }
  .matrix th, .matrix td {
    padding: 0.6rem 0.8rem; text-align: center; border: 1px solid var(--gray-border);
  }
  .matrix th { background: #f3f4f6; font-weight: 600; font-size: 0.85rem; color: #374151; }
  .matrix th:first-child { text-align: left; min-width: 180px; }
  .matrix td:first-child { text-align: left; font-weight: 600; }
  .matrix thead th:first-child { border-top-left-radius: 8px; }
  .matrix thead th:last-child { border-top-right-radius: 8px; }

  .cell-pass { background: var(--green-bg); color: var(--green); font-weight: 600; }
  .cell-warn { background: var(--yellow-bg); color: var(--yellow); font-weight: 600; }
  .cell-fail { background: var(--red-bg); color: var(--red); font-weight: 600; }
  .cell-ko { background: var(--red); color: #fff; font-weight: 700; }
  .cell-na { background: #f9fafb; color: #9ca3af; }

  /* --- K.O. Alert --- */
  .ko-alert {
    background: var(--red-bg); border-left: 4px solid var(--red);
    padding: 1rem 1.25rem; margin: 1rem 0; border-radius: 0 8px 8px 0;
  }
  .ko-alert h3 { color: var(--red); margin-bottom: 0.5rem; }
  .ko-alert li { margin: 0.25rem 0; }

  /* --- Runtime --- */
  .runtime-table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
  .runtime-table th, .runtime-table td {
    padding: 0.5rem 0.75rem; border: 1px solid var(--gray-border); text-align: left;
  }
  .runtime-table th { background: #f3f4f6; font-weight: 600; }
  .runtime-bar {
    height: 8px; background: var(--blue); border-radius: 4px;
    display: inline-block; vertical-align: middle; margin-left: 0.5rem;
  }

  /* --- Drill-Down --- */
  .model-detail { display: none; margin: 1rem 0; padding: 1.5rem; background: var(--gray-bg); border-radius: 10px; }
  .model-detail.active { display: block; }
  .model-detail h3 { color: #111827; margin-bottom: 1rem; }
  .detail-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  .detail-table th, .detail-table td { padding: 0.4rem 0.6rem; border: 1px solid var(--gray-border); }
  .detail-table th { background: #e5e7eb; }

  /* --- Compliance --- */
  .compliance-box {
    background: var(--blue-bg); border-left: 4px solid var(--blue);
    padding: 1rem 1.25rem; margin: 1rem 0; border-radius: 0 8px 8px 0;
  }

  /* --- Empfehlung --- */
  .recommendation {
    background: #f0fdf4; border: 2px solid var(--green);
    padding: 1.25rem; border-radius: 10px; margin: 1.5rem 0;
  }
  .recommendation h3 { color: var(--green); margin-bottom: 0.5rem; }

  /* --- Print --- */
  @media print {
    body { max-width: 100%; padding: 0.5cm; font-size: 10pt; }
    .card:hover { box-shadow: none; }
    .model-detail { display: block !important; break-inside: avoid; }
    .section { break-inside: avoid; }
  }

  /* --- Toggle Button --- */
  .toggle-btn {
    background: var(--blue); color: #fff; border: none; padding: 0.4rem 1rem;
    border-radius: 6px; cursor: pointer; font-size: 0.85rem; margin: 0.25rem;
  }
  .toggle-btn:hover { background: #1d4ed8; }
  .toggle-btn.outline {
    background: #fff; color: var(--blue); border: 1px solid var(--blue);
  }
</style>
</head>
<body>

<div class="header">
  <h1>LLM-Testplan — Modellvergleich</h1>
  <div class="meta">
    Generiert: {{ timestamp_display }} |
    Modelle: {{ n_models }} |
    Testfälle: {{ total_tests }} |
    Judge: {{ judge_model }} |
    Konfiguration: v{{ config_version }}
  </div>
</div>

<!-- ======================== Executive Summary ======================== -->
<div class="section">
  <h2>Executive Summary</h2>
  <div class="card-grid">
  {% for m in models_sorted %}
    <div class="card card-{{ m.overall_class }}" onclick="toggleDetail('{{ m.name_id }}')">
      <div class="model-name">{{ m.name }}</div>
      <div class="score-big {{ m.overall_class }}">{{ m.pass_rate_pct }}%</div>
      <div>{{ m.passed }}/{{ m.total }} bestanden</div>
      <div class="sub">
        {% if m.knockouts > 0 %}<span class="badge badge-ko">{{ m.knockouts }} K.O.</span>{% endif %}
        <span class="badge badge-{{ m.overall_class }}">{{ m.overall_label }}</span>
      </div>
    </div>
  {% endfor %}
  </div>
</div>

<!-- ======================== K.O. Alerts ======================== -->
{% if all_knockouts %}
<div class="ko-alert">
  <h3>K.O.-Kriterien verletzt</h3>
  <ul>
  {% for ko in all_knockouts %}
    <li><strong>{{ ko.model }}</strong> — {{ ko.playbook }} / {{ ko.evaluator }}: {{ ko.reasoning }}</li>
  {% endfor %}
  </ul>
</div>
{% endif %}

<!-- ======================== Vergleichsmatrix ======================== -->
<div class="section">
  <h2>Vergleichsmatrix — Pass-Rate pro Playbook</h2>
  <table class="matrix">
    <thead>
      <tr>
        <th>Modell</th>
        {% for pb in playbook_names %}<th>{{ playbook_labels[pb] }}</th>{% endfor %}
        <th>Gesamt</th>
      </tr>
    </thead>
    <tbody>
    {% for m in models_sorted %}
      <tr>
        <td>{{ m.name }}</td>
        {% for pb in playbook_names %}
          {% set cell = m.playbook_cells.get(pb) %}
          {% if cell %}
            <td class="cell-{{ cell.class }}">
              {{ cell.pass_rate_pct }}%
              {% if cell.ko %}<br><small>K.O.</small>{% endif %}
            </td>
          {% else %}
            <td class="cell-na">—</td>
          {% endif %}
        {% endfor %}
        <td class="cell-{{ m.overall_class }}"><strong>{{ m.pass_rate_pct }}%</strong></td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</div>

<!-- ======================== Score-Matrix ======================== -->
<div class="section">
  <h2>Durchschnittlicher Score pro Playbook</h2>
  <table class="matrix">
    <thead>
      <tr>
        <th>Modell</th>
        {% for pb in playbook_names %}<th>{{ playbook_labels[pb] }}</th>{% endfor %}
        <th>Gesamt</th>
      </tr>
    </thead>
    <tbody>
    {% for m in models_sorted %}
      <tr>
        <td>{{ m.name }}</td>
        {% for pb in playbook_names %}
          {% set cell = m.playbook_cells.get(pb) %}
          {% if cell %}
            <td class="cell-{{ cell.score_class }}">{{ cell.mean_score_pct }}%</td>
          {% else %}
            <td class="cell-na">—</td>
          {% endif %}
        {% endfor %}
        <td class="cell-{{ m.overall_class }}"><strong>{{ m.mean_score_pct }}%</strong></td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</div>

<!-- ======================== Performance ======================== -->
{% if perf_data %}
<div class="section">
  <h2>Performance-Vergleich</h2>
  <table class="matrix">
    <thead>
      <tr>
        <th>Modell</th>
        <th>TTFT P50</th>
        <th>TTFT P95</th>
        <th>Throughput (tok/s)</th>
        <th>Max Concurrency</th>
      </tr>
    </thead>
    <tbody>
    {% for p in perf_data %}
      <tr>
        <td>{{ p.model }}</td>
        <td>{{ p.ttft_p50 }} ms</td>
        <td>{{ p.ttft_p95 }} ms</td>
        <td>{{ p.throughput }}</td>
        <td>{{ p.max_concurrency }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</div>
{% endif %}

<!-- ======================== Laufzeitschätzung ======================== -->
{% if runtime %}
<div class="section">
  <h2>Geschätzte Laufzeiten</h2>
  <p style="color: var(--gray); margin-bottom: 1rem;">
    Basierend auf typischen Werten für DGX Spark (GB10, 128 GB) mit vLLM.
    Judge-Startup: ~{{ runtime.judge_startup }}. Gesamtlaufzeit: <strong>{{ runtime.total }}</strong>.
  </p>
  <table class="runtime-table">
    <thead>
      <tr>
        <th>Modell</th>
        <th>Startup</th>
        <th>Tests</th>
        <th>Cooldown</th>
        <th>Gesamt</th>
        <th></th>
      </tr>
    </thead>
    <tbody>
    {% for r in runtime.models %}
      <tr>
        <td>{{ r.model }}</td>
        <td>{{ r.startup }}</td>
        <td>{{ r.tests }}</td>
        <td>{{ r.cooldown }}</td>
        <td><strong>{{ r.total }}</strong></td>
        <td><span class="runtime-bar" style="width: {{ r.bar_pct }}%;"></span></td>
      </tr>
    {% endfor %}
    <tr style="background: #f3f4f6; font-weight: 600;">
      <td>Gesamt (inkl. Judge)</td>
      <td colspan="3"></td>
      <td><strong>{{ runtime.total }}</strong></td>
      <td></td>
    </tr>
    </tbody>
  </table>
</div>
{% endif %}

<!-- ======================== Empfehlung ======================== -->
{% if recommendation %}
<div class="recommendation">
  <h3>Empfehlung</h3>
  <p>{{ recommendation }}</p>
</div>
{% endif %}

<!-- ======================== Drill-Down Details ======================== -->
<div class="section">
  <h2>Detailergebnisse pro Modell</h2>
  <div style="margin-bottom: 1rem;">
    <button class="toggle-btn" onclick="showAll()">Alle einblenden</button>
    <button class="toggle-btn outline" onclick="hideAll()">Alle ausblenden</button>
  </div>

  {% for m in models_sorted %}
  <div id="detail-{{ m.name_id }}" class="model-detail">
    <h3>{{ m.name }}
      <span class="badge badge-{{ m.overall_class }}">{{ m.overall_label }}</span>
      {% if m.knockouts > 0 %}<span class="badge badge-ko">{{ m.knockouts }} K.O.</span>{% endif %}
    </h3>

    {% for pb in m.detail_playbooks %}
    <h4 style="margin-top: 1rem; color: #374151;">{{ playbook_labels.get(pb.name, pb.name) }}
      — {{ pb.passed }}/{{ pb.total }} bestanden ({{ pb.pass_rate_pct }}%)</h4>
    <table class="detail-table">
      <thead>
        <tr><th>Test-ID</th><th>Evaluator</th><th>Verdict</th><th>Score</th><th>Begründung</th></tr>
      </thead>
      <tbody>
      {% for r in pb.results %}
        <tr>
          <td>{{ r.test_id }}</td>
          <td>{{ r.evaluator }}</td>
          <td class="cell-{{ r.verdict_class }}">{{ r.verdict_display }}</td>
          <td>{{ r.score_pct }}%</td>
          <td>{{ r.reasoning_short }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
    {% endfor %}
  </div>
  {% endfor %}
</div>

<!-- ======================== Compliance ======================== -->
<div class="section">
  <h2>Compliance-Dokumentation</h2>
  <div class="compliance-box">
    <h3>EU AI Act (Art. 52 — Begrenztes Risiko)</h3>
    <p>Transparenzpflicht: Dokumentierter Testlauf mit {{ total_tests }} Testfällen
       über {{ n_models }} Modelle. K.O.-Kriterien für Halluzination (> 5%),
       PII-Leakage, Bias und Prompt Injection definiert und automatisiert geprüft.
       Alle Ergebnisse sind maschinenlesbar (JSON, CSV) und revisionssicher archivierbar.</p>
  </div>
  <div class="compliance-box">
    <h3>ISO/IEC 42001 — AI Management System</h3>
    <p>Dokumentierte Testmethodik mit {{ n_models }} Modellen, {{ n_playbooks }} Testbereichen,
       automatisierter Auswertung via LLM-as-Judge ({{ judge_model }}). Statistische
       Signifikanztests für Bias-Erkennung (Chi², p &lt; 0.05). Testdaten mit
       4-Augen-Prinzip (Cohen's &kappa; &ge; 0.70). Hardware-Skalierungsfaktor (HSF)
       mit Bootstrap-Konfidenzintervallen dokumentiert.</p>
  </div>
</div>

<div class="header" style="margin-top: 2rem; border-bottom: none; border-top: 3px solid var(--blue); padding-top: 1rem;">
  <div class="meta">
    LLM-Testplan v1.0 | KI-Plattform On-Premise | {{ timestamp_display }}
  </div>
</div>

<script>
function toggleDetail(id) {
  var el = document.getElementById('detail-' + id);
  if (el) el.classList.toggle('active');
}
function showAll() {
  document.querySelectorAll('.model-detail').forEach(function(el) { el.classList.add('active'); });
}
function hideAll() {
  document.querySelectorAll('.model-detail').forEach(function(el) { el.classList.remove('active'); });
}
</script>
</body>
</html>
"""


class DashboardGenerator:
    """Erzeugt das Cross-Modell-Vergleichs-Dashboard."""

    def __init__(self, config: TestplanConfig):
        self.config = config

    def generate(
        self,
        results: dict[str, list[PlaybookResult]],
        runtime_estimate: dict[str, Any] | None = None,
        output_dir: Path | None = None,
    ) -> Path:
        """Erzeuge HTML-Dashboard.

        Args:
            results: Modellname → Liste von PlaybookResults
            runtime_estimate: Optionale Laufzeitschätzung (von estimate_full_runtime)
            output_dir: Ausgabeverzeichnis (Standard: config.report_dir)

        Returns:
            Pfad zur generierten HTML-Datei.
        """
        out = output_dir or self.config.report_dir
        out.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        timestamp_display = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")

        # --- Daten aufbereiten ---
        models_data = []
        all_knockouts = []
        all_playbook_names = set()
        perf_entries = []
        total_tests = 0

        for model_name, pb_results in results.items():
            m_total = sum(pb.total for pb in pb_results)
            m_passed = sum(pb.passed for pb in pb_results)
            m_knockouts = sum(len(pb.knockouts) for pb in pb_results)
            m_scores = []
            total_tests += m_total

            # Status
            has_ko = any(pb.has_knockout for pb in pb_results)
            if has_ko:
                overall = "K.O."
                overall_class = "ko"
            elif m_total > 0 and m_passed / m_total >= 0.90:
                overall = "Bestanden"
                overall_class = "pass"
            elif m_total > 0 and m_passed / m_total >= 0.70:
                overall = "Warnung"
                overall_class = "warn"
            else:
                overall = "Nicht bestanden"
                overall_class = "fail"

            # Pro-Playbook-Zellen
            playbook_cells = {}
            detail_playbooks = []

            for pb in pb_results:
                all_playbook_names.add(pb.playbook)
                pr = pb.pass_rate
                ms = pb.mean_score

                m_scores.append(ms)

                cell_class = self._rate_class(pr)
                if pb.has_knockout:
                    cell_class = "ko"
                score_class = self._rate_class(ms)

                playbook_cells[pb.playbook] = {
                    "pass_rate_pct": f"{pr * 100:.0f}",
                    "mean_score_pct": f"{ms * 100:.0f}",
                    "class": cell_class,
                    "score_class": score_class,
                    "ko": pb.has_knockout,
                }

                # K.O.-Details
                for ko in pb.knockouts:
                    all_knockouts.append({
                        "model": model_name,
                        "playbook": pb.playbook,
                        "evaluator": ko.evaluator,
                        "reasoning": ko.reasoning[:200],
                    })

                # Drill-Down-Daten
                detail_results = []
                for r in pb.results:
                    vc = r.verdict.value
                    detail_results.append({
                        "test_id": r.test_id,
                        "evaluator": r.evaluator,
                        "verdict_display": vc.upper(),
                        "verdict_class": "ko" if r.verdict == Verdict.KNOCKOUT else vc,
                        "score_pct": f"{r.score * 100:.0f}",
                        "reasoning_short": r.reasoning[:150] if r.reasoning else "",
                    })

                detail_playbooks.append({
                    "name": pb.playbook,
                    "total": pb.total,
                    "passed": pb.passed,
                    "pass_rate_pct": f"{pb.pass_rate * 100:.0f}",
                    "results": detail_results,
                })

                # Performance-Daten extrahieren
                if pb.playbook == "06_performance":
                    for r in pb.results:
                        if r.metadata:
                            perf_entries.append({
                                "model": model_name,
                                "ttft_p50": r.metadata.get("ttft_p50_ms", "—"),
                                "ttft_p95": r.metadata.get("ttft_p95_ms", "—"),
                                "throughput": f"{r.metadata.get('throughput_mean_tok_s', r.metadata.get('throughput_median_tok_s', '—'))}",
                                "max_concurrency": self._extract_max_concurrency(r.metadata),
                            })

            mean_score = sum(m_scores) / len(m_scores) if m_scores else 0

            name_id = model_name.lower().replace(" ", "-").replace(".", "-")

            models_data.append({
                "name": model_name,
                "name_id": name_id,
                "total": m_total,
                "passed": m_passed,
                "pass_rate_pct": f"{m_passed / m_total * 100:.0f}" if m_total > 0 else "0",
                "pass_rate": m_passed / m_total if m_total > 0 else 0,
                "mean_score_pct": f"{mean_score * 100:.0f}",
                "knockouts": m_knockouts,
                "overall_label": overall,
                "overall_class": overall_class,
                "playbook_cells": playbook_cells,
                "detail_playbooks": detail_playbooks,
            })

        # Sortierung: Bestanden → Warnung → K.O./Fail, innerhalb nach Pass-Rate
        sort_order = {"pass": 0, "warn": 1, "fail": 2, "ko": 3}
        models_data.sort(key=lambda m: (sort_order.get(m["overall_class"], 9), -m["pass_rate"]))

        playbook_names = sorted(all_playbook_names)

        # --- Runtime-Daten aufbereiten ---
        runtime_tmpl = None
        if runtime_estimate:
            max_total = max(m["total_seconds"] for m in runtime_estimate["per_model"]) if runtime_estimate["per_model"] else 1
            runtime_models = []
            for rm in runtime_estimate["per_model"]:
                test_sec = sum(rm["playbook_seconds"].values())
                runtime_models.append({
                    "model": rm["model"],
                    "startup": _format_duration(rm["startup_seconds"]),
                    "tests": _format_duration(test_sec),
                    "cooldown": f"{rm['cooldown_seconds']}s",
                    "total": _format_duration(rm["total_seconds"]),
                    "bar_pct": min(100, rm["total_seconds"] / max_total * 100),
                })
            runtime_tmpl = {
                "judge_startup": _format_duration(runtime_estimate["judge_startup_seconds"]),
                "total": runtime_estimate["total_formatted"],
                "models": runtime_models,
            }

        # --- Empfehlung generieren ---
        passed_models = [m for m in models_data if m["overall_class"] == "pass"]
        recommendation = None
        if passed_models:
            best = max(passed_models, key=lambda m: m["pass_rate"])
            recommendation = (
                f"{best['name']} erreicht mit {best['pass_rate_pct']}% die höchste "
                f"Bestehensquote ohne K.O.-Kriterien. "
                f"Insgesamt haben {len(passed_models)} von {len(models_data)} Modellen "
                f"den Testplan bestanden."
            )
        else:
            recommendation = (
                f"Kein Modell hat den Testplan vollständig bestanden. "
                f"Bitte die K.O.-Kriterien und Detailergebnisse prüfen."
            )

        # --- Rendern ---
        template = Template(DASHBOARD_TEMPLATE)
        html = template.render(
            timestamp_display=timestamp_display,
            config_version="1.0",
            n_models=len(models_data),
            n_playbooks=len(playbook_names),
            total_tests=total_tests,
            judge_model=self.config.judge.model.split("/")[-1],
            models_sorted=models_data,
            playbook_names=playbook_names,
            playbook_labels=PLAYBOOK_LABELS,
            all_knockouts=all_knockouts,
            perf_data=perf_entries,
            runtime=runtime_tmpl,
            recommendation=recommendation,
        )

        path = out / f"dashboard_{timestamp}.html"
        with open(path, "w") as f:
            f.write(html)

        # Auch stabile Kopie für schnellen Zugriff
        latest = out / "dashboard_latest.html"
        with open(latest, "w") as f:
            f.write(html)

        return path

    @staticmethod
    def _rate_class(rate: float) -> str:
        if rate >= 0.90:
            return "pass"
        if rate >= 0.70:
            return "warn"
        return "fail"

    @staticmethod
    def _extract_max_concurrency(metadata: dict) -> str:
        deg = metadata.get("concurrent_degradation", {})
        if not deg:
            return "—"
        # Höchste Concurrency-Stufe ohne Fehler >5%
        max_c = 1
        for level, data in sorted(deg.items(), key=lambda x: int(x[0])):
            if data.get("error_rate", 0) <= 0.05:
                max_c = int(level)
        return str(max_c)
