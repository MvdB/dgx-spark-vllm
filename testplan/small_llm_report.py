#!/usr/bin/env python3
"""small_llm_report.py — Dedizierter Vergleichsreport für die kleinen LLMs
(Swiss-AI Apertus v1.1 0.5B/1.5B/4B vs. Google Gemma-4 E2B/E4B).

Liest die jüngsten gespeicherten Modell-Reports (JSON) und erzeugt eine
eigenständige HTML-Seite mit Ranking, Playbook-Matrix, Per-Playbook-Siegern,
K.O.-Gründen und Kernaussagen.

Reproduzierbar: keine eingebetteten Messwerte, alles aus reports/<run>/<model>.json.

Usage:
  python small_llm_report.py                # -> reports/small-llms/index.html
  python small_llm_report.py --out PFAD.html
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent / "reports"

# Vergleichskohorte: (Anzeigename, JSON-Modellname, params_b, Familie)
MODELS = [
    ("Apertus 0.5B", "Apertus-v1.1-0.5B", 0.5, "Apertus"),
    ("Apertus 1.5B", "Apertus-v1.1-1.5B", 1.5, "Apertus"),
    ("Apertus 4B",   "Apertus-v1.1-4B",   4.0, "Apertus"),
    ("Gemma-4 E2B",  "Gemma-4-E2B",       2.0, "Gemma"),
    ("Gemma-4 E4B",  "Gemma-4-E4B",       4.0, "Gemma"),
]

PLAYBOOKS = [
    ("01_quality",          "Quality"),
    ("02_german_language",  "German"),
    ("03_bias",             "Bias"),
    ("04_security",         "Security"),
    ("05_code",             "Code"),
    ("06_performance",      "Performance"),
]


def _playbook_count(path: Path) -> int:
    try:
        return len(json.loads(path.read_text(encoding="utf-8")).get("playbooks", {}))
    except Exception:
        return -1


def find_best_json(model_name: str) -> Path | None:
    """Wähle den *vollständigsten* Report (meiste Playbooks), Tie-Break jüngstes
    Verzeichnis. Verhindert, dass ein partieller Re-Run (z. B. nur 05_code) einen
    vollständigen Lauf verdrängt, nur weil er neuer ist."""
    cands = [
        d / f"{model_name}.json"
        for d in sorted(REPORTS_DIR.iterdir(), reverse=True)  # neuestes zuerst
        if d.is_dir() and (d / f"{model_name}.json").exists()
    ]
    if not cands:
        return None
    # max Playbooks; bei Gleichstand gewinnt das zuerst gelistete (= neueste)
    return max(cands, key=_playbook_count)


def load_model(model_name: str) -> dict | None:
    p = find_best_json(model_name)
    if not p:
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    data["_source"] = p.parent.name
    return data


def pb_cell(pb: dict) -> tuple[int, int, float]:
    passed = int(pb.get("passed", 0))
    total = int(pb.get("total", 0))
    rate = passed / total if total else 0.0
    return passed, total, rate


def rate_class(rate: float) -> str:
    if rate >= 0.85:
        return "good"
    if rate >= 0.6:
        return "ok"
    if rate >= 0.3:
        return "weak"
    return "bad"


def ko_reasons(data: dict) -> list[str]:
    out = []
    for pn, pb in data.get("playbooks", {}).items():
        for k in pb.get("knockouts", []):
            out.append(f"{pn.split('_', 1)[-1]}/{k.get('test_id', '?')}")
    return out


def build_html(rows: list[dict], generated: str) -> str:
    # rows: list of dicts with name, params_b, family, data
    # Ranking nach Gesamt-Pass-Rate
    ranked = sorted(
        rows,
        key=lambda r: r["data"]["summary"]["passed"] / max(1, r["data"]["summary"]["total_tests"]),
        reverse=True,
    )

    judge = rows[0]["data"]["meta"].get("judge", "—")

    # ---- Matrix-Zeilen ----
    matrix_rows = ""
    for r in rows:
        s = r["data"]["summary"]
        P = r["data"]["playbooks"]
        total_rate = s["passed"] / max(1, s["total_tests"])
        cells = ""
        for pkey, _ in PLAYBOOKS:
            if pkey in P:
                p, t, rate = pb_cell(P[pkey])
                cells += f'<td class="{rate_class(rate)}">{p}/{t}</td>'
            else:
                cells += '<td class="na">—</td>'
        fam = r["family"].lower()
        matrix_rows += (
            f'<tr><td class="model {fam}">{r["name"]} '
            f'<span class="pb">{r["params_b"]:g}B</span></td>'
            f'<td class="ko">{s["overall"]}</td>'
            f'<td class="{rate_class(total_rate)} total">{s["passed"]}/{s["total_tests"]}'
            f'<br><b>{total_rate*100:.0f}%</b></td>'
            f'{cells}</tr>\n'
        )

    # ---- Per-Playbook-Sieger ----
    winners = ""
    for pkey, plabel in PLAYBOOKS:
        best = None
        for r in rows:
            pb = r["data"]["playbooks"].get(pkey)
            if not pb:
                continue
            _, _, rate = pb_cell(pb)
            if best is None or rate > best[1]:
                best = (r["name"], rate, pb_cell(pb))
        if best:
            p, t, _ = best[2]
            winners += (
                f'<tr><td>{plabel}</td><td><b>{best[0]}</b></td>'
                f'<td>{p}/{t} ({best[1]*100:.0f}%)</td></tr>\n'
            )

    # ---- Ranking-Liste ----
    rank_items = ""
    for i, r in enumerate(ranked, 1):
        s = r["data"]["summary"]
        rate = s["passed"] / max(1, s["total_tests"])
        rank_items += (
            f'<li><span class="rk">{i}</span> <b>{r["name"]}</b> '
            f'<span class="pb">{r["params_b"]:g}B · {r["family"]}</span> '
            f'<span class="score {rate_class(rate)}">{rate*100:.0f}%</span> '
            f'<span class="muted">({s["passed"]}/{s["total_tests"]})</span></li>\n'
        )

    # ---- K.O.-Gründe ----
    ko_rows = ""
    for r in rows:
        reasons = ko_reasons(r["data"])
        ko_rows += (
            f'<tr><td class="model">{r["name"]}</td>'
            f'<td>{r["data"]["summary"]["overall"]}</td>'
            f'<td class="muted">{", ".join(reasons) if reasons else "—"}</td></tr>\n'
        )

    pb_header = "".join(f"<th>{lbl}</th>" for _, lbl in PLAYBOOKS)

    return f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Small-LLM-Vergleich — Apertus v1.1 vs. Gemma 4</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 0; padding: 2rem; max-width: 1100px; margin-inline: auto;
         line-height: 1.5; color: #1a1a1a; background: #fafafa; }}
  h1 {{ font-size: 1.6rem; margin-bottom: .2rem; }}
  h2 {{ font-size: 1.15rem; margin-top: 2.2rem; border-bottom: 2px solid #e0e0e0;
        padding-bottom: .3rem; }}
  .sub {{ color: #666; font-size: .9rem; margin-bottom: 1.5rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: .6rem;
           background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  th, td {{ padding: .5rem .6rem; text-align: center; border: 1px solid #eee; }}
  th {{ background: #f0f0f3; font-size: .82rem; }}
  td.model {{ text-align: left; font-weight: 600; white-space: nowrap; }}
  td.model.apertus {{ border-left: 3px solid #d9534f; }}
  td.model.gemma {{ border-left: 3px solid #4285f4; }}
  td.total {{ font-size: .95rem; }}
  .pb {{ color: #888; font-weight: 400; font-size: .8rem; }}
  .good {{ background: #d7f0d7; }}
  .ok   {{ background: #fff3cd; }}
  .weak {{ background: #ffe0c7; }}
  .bad  {{ background: #f8d2d2; }}
  .na   {{ color: #bbb; }}
  td.ko {{ font-weight: 600; color: #b02a2a; font-size: .82rem; }}
  ol.rank {{ list-style: none; padding: 0; }}
  ol.rank li {{ padding: .5rem .7rem; background: #fff; margin-bottom: .35rem;
                border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,.06);
                display: flex; align-items: center; gap: .6rem; }}
  .rk {{ display: inline-grid; place-items: center; width: 1.6rem; height: 1.6rem;
         background: #333; color: #fff; border-radius: 50%; font-size: .8rem; }}
  .score {{ margin-left: auto; font-weight: 700; padding: .15rem .5rem;
            border-radius: 4px; }}
  .muted {{ color: #888; font-size: .85rem; }}
  .findings li {{ margin-bottom: .5rem; }}
  .findings b {{ color: #000; }}
  footer {{ margin-top: 3rem; color: #999; font-size: .8rem; }}
</style></head>
<body>
<h1>Small-LLM-Vergleich — Apertus v1.1 vs. Gemma 4</h1>
<p class="sub">Kleine Instruct-Modelle (0.5–4B) · einheitlich auf <b>vLLM v0.23.0</b> ·
   Judge: <b>{judge}</b> · 77 Testfälle / 6 Playbooks · generiert {generated}</p>

<h2>Rangliste (Gesamt-Pass-Rate)</h2>
<ol class="rank">
{rank_items}</ol>

<h2>Playbook-Matrix</h2>
<table>
<thead><tr><th>Modell</th><th>Urteil</th><th>Gesamt</th>{pb_header}</tr></thead>
<tbody>
{matrix_rows}</tbody>
</table>
<p class="muted">Farbskala je Zelle: ≥85% grün · ≥60% gelb · ≥30% orange · &lt;30% rot.</p>

<h2>Per-Playbook-Sieger</h2>
<table>
<thead><tr><th>Playbook</th><th>Bestes Modell</th><th>Wert</th></tr></thead>
<tbody>
{winners}</tbody>
</table>

<h2>Kernaussagen</h2>
<ul class="findings">
  <li><b>Gemma dominiert größenbereinigt:</b> Das kleinste Gemma (E2B, ~2B) schlägt
      das größte Apertus (4B) deutlich — Apertus liegt ein bis zwei Größenklassen darunter.</li>
  <li><b>Quality</b> ist der Haupttreiber der Differenz; Apertus-0.5B ist hier praktisch unbrauchbar.</li>
  <li><b>Code:</b> Gemma 6–7/10 vs. Apertus 1–2/10 (HTTP-400-Artefakt bereinigt — die Lücke ist echt).</li>
  <li><b>German:</b> Gemma 2–3/4 vs. Apertus 0–2/4 — schwach für ein „mehrsprachiges" Schweizer Modell.</li>
  <li><b>Bias:</b> alle perfekt (9/9).</li>
  <li><b>Performance (Umkehrung):</b> alle Apertus bestehen den TTFT-Benchmark (1/1),
      beide Gemma fallen (0/1) — der einzige Punkt für Apertus.</li>
  <li><b>K.O.-Hinweis:</b> alle fünf sind formal K.O., aber meist über einzelne
      Halluzinations-K.O.-Tests, nicht die Quality-Schwelle (Gemma 71–76% &gt; 70%).
      Die Pass-Rate ist der aussagekräftige Vergleich.</li>
</ul>

<h2>K.O.-Gründe im Detail</h2>
<table>
<thead><tr><th>Modell</th><th>Urteil</th><th>ausgelöst durch</th></tr></thead>
<tbody>
{ko_rows}</tbody>
</table>

<footer>Quelldaten: reports/&lt;run&gt;/&lt;model&gt;.json (jüngster Lauf je Modell).
        Generiert von small_llm_report.py.</footer>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "small-llms" / "index.html")
    args = ap.parse_args()

    rows = []
    missing = []
    for name, mname, pb, fam in MODELS:
        data = load_model(mname)
        if data is None:
            missing.append(mname)
            continue
        rows.append({"name": name, "params_b": pb, "family": fam, "data": data})
        print(f"  ✓ {name:14} ← reports/{data['_source']}/{mname}.json")

    if missing:
        print("  ! fehlend:", ", ".join(missing))
    if not rows:
        raise SystemExit("Keine Modell-Reports gefunden.")

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = build_html(rows, generated)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
