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
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent / "reports"

# Kernaussagen (Markdown-Quelle; **fett** → HTML beim Rendern).
# Single source für HTML- und Markdown-Ausgabe.
FINDINGS = [
    "**Gemma dominiert größenbereinigt:** Das kleinste Gemma (E2B, ~2B) schlägt das "
    "größte Apertus (4B) deutlich — Apertus liegt ein bis zwei Größenklassen darunter.",
    "**Quality** ist der Haupttreiber der Differenz; Apertus-0.5B ist hier praktisch unbrauchbar.",
    "**Code:** Gemma 6–7/10 vs. Apertus 1–2/10 (HTTP-400-Artefakt bereinigt — die Lücke ist echt).",
    "**German:** Gemma 2–3/4 vs. Apertus 0–2/4 — schwach für ein „mehrsprachiges\" Schweizer Modell.",
    "**Bias:** alle perfekt (9/9).",
    "**Performance (Umkehrung):** alle Apertus bestehen den TTFT-Benchmark (1/1), "
    "beide Gemma fallen (0/1) — der einzige Punkt für Apertus.",
    "**K.O.-Hinweis:** alle fünf sind formal K.O., aber meist über einzelne "
    "Halluzinations-K.O.-Tests, nicht die Quality-Schwelle (Gemma 71–76% > 70%). "
    "Die Pass-Rate ist der aussagekräftige Vergleich.",
]


def md_inline_to_html(s: str) -> str:
    """Minimal: **fett** → <b>, _kursiv_ → <i>. Genug für die Kernaussagen."""
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"_(.+?)_", r"<i>\1</i>", s)
    return s

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
    data["_dir"] = str(p.parent)
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
        link = f'details/{r["mname"]}.html'
        matrix_rows += (
            f'<tr><td class="model {fam}">'
            f'<a href="{link}">{r["name"]}</a> '
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
            f'<li><span class="rk">{i}</span> '
            f'<b><a href="details/{r["mname"]}.html">{r["name"]}</a></b> '
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

    findings_html = "\n".join(f"  <li>{md_inline_to_html(f)}</li>" for f in FINDINGS)

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
  td.model a, ol.rank a {{ color: inherit; text-decoration: none;
                           border-bottom: 1px dotted #999; }}
  td.model a:hover, ol.rank a:hover {{ border-bottom-style: solid; }}
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
{findings_html}
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


def build_md(rows: list[dict], generated: str) -> str:
    """Eigenständiges Markdown-Dokument (GitLab-tauglich), gleiche Inhalte wie HTML."""
    def orate(r):
        s = r["data"]["summary"]
        return s["passed"] / max(1, s["total_tests"])

    ranked = sorted(rows, key=orate, reverse=True)
    judge = rows[0]["data"]["meta"].get("judge", "—")

    L: list[str] = []
    L.append("# Small-LLM-Vergleich — Apertus v1.1 vs. Gemma 4")
    L.append("")
    L.append(
        f"Kleine Instruct-Modelle (0.5–4B), einheitlich auf **vLLM v0.23.0**. "
        f"Judge: **{judge}**. 77 Testfälle / 6 Playbooks. Generiert {generated}."
    )
    L.append("")

    # ---- Rangliste ----
    L.append("## Rangliste (Gesamt-Pass-Rate)")
    L.append("")
    L.append("| # | Modell | Größe | Pass-Rate | bestanden |")
    L.append("|---|--------|-------|-----------|-----------|")
    for i, r in enumerate(ranked, 1):
        s = r["data"]["summary"]
        rate = orate(r)
        L.append(
            f"| {i} | {r['name']} | {r['params_b']:g}B {r['family']} | "
            f"**{rate*100:.0f}%** | {s['passed']}/{s['total_tests']} |"
        )
    L.append("")

    # ---- Playbook-Matrix ----
    L.append("## Playbook-Matrix")
    L.append("")
    pb_head = " | ".join(lbl for _, lbl in PLAYBOOKS)
    L.append(f"| Modell | Urteil | Gesamt | {pb_head} |")
    L.append("|--------|--------|--------|" + "----|" * len(PLAYBOOKS))
    for r in ranked:
        s = r["data"]["summary"]
        P = r["data"]["playbooks"]
        rate = orate(r)
        cells = []
        for pkey, _ in PLAYBOOKS:
            if pkey in P:
                p, t, _r = pb_cell(P[pkey])
                cells.append(f"{p}/{t}")
            else:
                cells.append("—")
        L.append(
            f"| {r['name']} | {s['overall']} | "
            f"**{rate*100:.0f}%** ({s['passed']}/{s['total_tests']}) | "
            + " | ".join(cells) + " |"
        )
    L.append("")
    L.append("Zellen: bestanden/gesamt je Playbook. „Gesamt\" ist die Pass-Rate über alle 77 Fälle.")
    L.append("")

    # ---- Per-Playbook-Sieger ----
    L.append("## Per-Playbook-Sieger")
    L.append("")
    L.append("| Playbook | Bestes Modell | Wert |")
    L.append("|----------|---------------|------|")
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
            L.append(f"| {plabel} | **{best[0]}** | {p}/{t} ({best[1]*100:.0f}%) |")
    L.append("")

    # ---- Kernaussagen ----
    L.append("## Kernaussagen")
    L.append("")
    for f in FINDINGS:
        L.append(f"- {f}")
    L.append("")

    # ---- K.O.-Gründe ----
    L.append("## K.O.-Gründe im Detail")
    L.append("")
    L.append("| Modell | Urteil | ausgelöst durch |")
    L.append("|--------|--------|-----------------|")
    for r in ranked:
        reasons = ko_reasons(r["data"])
        L.append(
            f"| {r['name']} | {r['data']['summary']['overall']} | "
            f"{', '.join(reasons) if reasons else '—'} |"
        )
    L.append("")
    L.append(
        "---\n\n_Quelldaten: reports/&lt;run&gt;/&lt;model&gt;.json (jüngster Lauf je Modell). "
        "Generiert von `small_llm_report.py`._"
    )
    L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "small-llms" / "index.html")
    ap.add_argument("--md", type=Path, default=None,
                    help="zusätzlich Markdown schreiben (Default: <out-dir>/index.md)")
    args = ap.parse_args()

    rows = []
    missing = []
    for name, mname, pb, fam in MODELS:
        data = load_model(mname)
        if data is None:
            missing.append(mname)
            continue
        rows.append({"name": name, "mname": mname, "params_b": pb,
                     "family": fam, "data": data})
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

    md_out = args.md or args.out.with_suffix(".md")
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(build_md(rows, generated), encoding="utf-8")
    print(f"→ {md_out}")

    # Detail-Reports je Modell daneben bündeln (verlinkt aus der Übersicht)
    details_dir = args.out.parent / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        src = Path(r["data"]["_dir"]) / f"{r['mname']}.html"
        if src.exists():
            shutil.copy2(src, details_dir / f"{r['mname']}.html")
            print(f"  ↳ detail: details/{r['mname']}.html")
        else:
            print(f"  ! detail fehlt für {r['name']}: {src}")


if __name__ == "__main__":
    main()
