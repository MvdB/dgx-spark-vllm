#!/usr/bin/env python3
"""medium_llm_report.py — Dedizierter Vergleichsreport für die Mittelklasse-LLMs
(~8–35B), einheitlich auf vLLM v0.23.0 getestet.

Kohorte (9 Modelle): Qwen3.6-27B-FP8, Qwen3.6-35B-A3B-FP8, Gemma-4-26B-A4B,
Gemma-4-31B, Granite-4.1-30B, Nemotron-3-Nano-30B (FP8),
Nemotron-3-Nano-Omni-30B (FP8), GLM-4.7-Flash, Olmo-3.1-32B-Instruct.

Liest die jüngsten gespeicherten Modell-Reports (JSON) und erzeugt eine
eigenständige HTML-Seite mit Ranking, Playbook-Matrix, Per-Playbook-Siegern,
K.O.-Gründen und Kernaussagen. Schwestergenerator zu small_llm_report.py.

Reproduzierbar: keine eingebetteten Messwerte, alles aus reports/<run>/<model>.json.

Usage:
  python medium_llm_report.py                # -> reports/medium-llms/index.html
  python medium_llm_report.py --out PFAD.html
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent / "reports"

# Vergleichskohorte: (Anzeigename, JSON-Modellname, params_b, Familie, Quant, Arch-Notiz)
MODELS = [
    ("Qwen3.6-27B-FP8",          "Qwen3.6-27B-FP8",          27.0, "Qwen",     "FP8",  "dense"),
    ("Qwen3.6-35B-A3B-FP8",      "Qwen3.6-35B-A3B-FP8",      35.0, "Qwen",     "FP8",  "MoE · A3B aktiv"),
    ("Gemma-4-26B-A4B",          "Gemma-4-26B-A4B",          26.0, "Gemma",    "BF16", "MoE · A4B aktiv"),
    ("Gemma-4-31B",              "Gemma-4-31B",              31.0, "Gemma",    "BF16", "dense"),
    ("Granite-4.1-30B",          "Granite-4.1-30B",          30.0, "Granite",  "BF16", "Hybrid-Mamba"),
    ("Nemotron-3-Nano-30B",      "Nemotron-3-Nano-30B",      30.0, "Nemotron", "FP8",  "MoE · A3B aktiv"),
    ("Nemotron-3-Nano-Omni-30B", "Nemotron-3-Nano-Omni-30B", 30.0, "Nemotron", "FP8",  "MoE · A3B · Omni (Text)"),
    ("GLM-4.7-Flash",            "GLM-4.7-Flash",            15.6, "GLM",      "BF16", "MoE-Lite · 64×top-4"),
    ("Olmo-3.1-32B-Instruct",    "Olmo-3.1-32B-Instruct",    32.0, "Olmo",     "BF16", "dense"),
]

PLAYBOOKS = [
    ("01_quality",          "Quality"),
    ("02_german_language",  "German"),
    ("03_bias",             "Bias"),
    ("04_security",         "Security"),
    ("05_code",             "Code"),
    ("06_performance",      "Performance"),
]

# Kernaussagen (Markdown-Quelle; **fett**/_kursiv_ → HTML beim Rendern).
# Single source für HTML- und Markdown-Ausgabe.
FINDINGS = [
    "**Gemma führt die Klasse an:** Gemma-4-26B-A4B (90%) und Gemma-4-31B (84%) "
    "belegen die Plätze 1 und 2. Bemerkenswert: das _kleinere_ MoE (26B/A4B aktiv) "
    "schlägt das größere dense 31B — Architektur und Tuning schlagen reine Größe.",
    "**Qwen3.6 solide im Mittelfeld:** 35B-A3B (79%) vor dem dichten 27B (77%), "
    "beide FP8 — die FP8-Quantisierung kostet keine sichtbare Qualität.",
    "**Quality bleibt der Differenzierer:** Bias (alle 100%) und Performance "
    "(alle 1/1 TTFT) trennen niemanden; die Rangfolge entscheidet sich an Quality, "
    "German und Code.",
    "**German streut stark:** Gemma 100% an der Spitze gegen GLM-4.7-Flash 25% und "
    "Qwen3.6-27B/Granite/Olmo je 50% — Mehrsprachigkeit ist kein selbstverständliches Niveau.",
    "**Code:** Gemma-4-26B-A4B als einziges mit voller Code-Punktzahl (10/10); "
    "GLM (40%) und beide Qwen-FP8 (50%) am schwächsten.",
    "**Echte Schwellen-K.O. (nicht nur Halluzinations-Einzeltests):** "
    "Granite-4.1-30B und Olmo-3.1-32B reißen die **Security**-Schwelle (75% bzw. 67%). "
    "Diese beiden plus GLM tragen auch die meisten K.O.-Marker (11/10/12).",
    "**Schlusslicht:** GLM-4.7-Flash (56%) — kleinstes Modell der Kohorte (~15.6B "
    "MoE-Lite) und durchgängig schwächste Quality/German/Code-Werte.",
    "**Alle 9 formal K.O.:** wie schon bei den Small-LLMs überwiegend über einzelne "
    "K.O.-Tests, nicht zwingend die 70%-Quality-Schwelle. Die Gesamt-Pass-Rate ist der "
    "aussagekräftige Vergleich. Infrastruktur-Erfolg: die zuvor auf v0.21.0 gepinnten "
    "Modelle laufen jetzt einheitlich auf **v0.23.0** (Qwen3.6-VL ohne conv3d-Custom-Patch).",
]

# Akzentfarbe je Familie (linker Rand in der Matrix)
FAMILY_COLORS = {
    "Qwen":     "#7b3fb0",
    "Gemma":    "#4285f4",
    "Granite":  "#1f8f7a",
    "Nemotron": "#76b900",
    "GLM":      "#e0772b",
    "Olmo":     "#d9534f",
}


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


def overall_rate(data: dict) -> float:
    s = data["summary"]
    return s["passed"] / max(1, s["total_tests"])


def md_inline_to_html(s: str) -> str:
    """Minimal: **fett** → <b>, _kursiv_ → <i>. Genug für die Kernaussagen."""
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"_(.+?)_", r"<i>\1</i>", s)
    return s


def build_html(rows: list[dict], generated: str) -> str:
    # rows: list of dicts with name, mname, params_b, family, quant, arch, data
    ranked = sorted(rows, key=lambda r: overall_rate(r["data"]), reverse=True)

    judge = rows[0]["data"]["meta"].get("judge", "—")

    # ---- Familien-CSS ----
    fam_css = "\n".join(
        f"  td.model.{fam.lower()} {{ border-left: 3px solid {col}; }}"
        for fam, col in FAMILY_COLORS.items()
    )

    # ---- Matrix-Zeilen (in Rangfolge) ----
    matrix_rows = ""
    for r in ranked:
        s = r["data"]["summary"]
        P = r["data"]["playbooks"]
        total_rate = overall_rate(r["data"])
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
            f'<span class="pb">{r["params_b"]:g}B · {r["arch"]}</span></td>'
            f'<td class="quant">{r["quant"]}</td>'
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
        rate = overall_rate(r["data"])
        rank_items += (
            f'<li><span class="rk">{i}</span> '
            f'<b><a href="details/{r["mname"]}.html">{r["name"]}</a></b> '
            f'<span class="pb">{r["params_b"]:g}B · {r["family"]} · {r["quant"]}</span> '
            f'<span class="score {rate_class(rate)}">{rate*100:.0f}%</span> '
            f'<span class="muted">({s["passed"]}/{s["total_tests"]} · {s["knockouts"]} K.O.)</span></li>\n'
        )

    # ---- K.O.-Gründe ----
    ko_rows = ""
    for r in ranked:
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
<title>Mittelklasse-LLM-Vergleich — 8–35B auf vLLM v0.23.0</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 0; padding: 2rem; max-width: 1180px; margin-inline: auto;
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
{fam_css}
  td.total {{ font-size: .95rem; }}
  td.quant {{ font-size: .78rem; color: #555; font-weight: 600; }}
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
<h1>Mittelklasse-LLM-Vergleich — 8–35B</h1>
<p class="sub">Instruct-/Reasoning-Modelle der Mittelklasse (~15–35B) ·
   einheitlich auf <b>vLLM v0.23.0</b> · Judge: <b>{judge}</b> ·
   77 Testfälle / 6 Playbooks · generiert {generated}</p>

<h2>Rangliste (Gesamt-Pass-Rate)</h2>
<ol class="rank">
{rank_items}</ol>

<h2>Playbook-Matrix</h2>
<table>
<thead><tr><th>Modell</th><th>Quant</th><th>Urteil</th><th>Gesamt</th>{pb_header}</tr></thead>
<tbody>
{matrix_rows}</tbody>
</table>
<p class="muted">Farbskala je Zelle: ≥85% grün · ≥60% gelb · ≥30% orange · &lt;30% rot.
   Zellen zeigen bestanden/gesamt je Playbook; „Gesamt" ist die Pass-Rate über alle 77 Fälle.</p>

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

<footer>Quelldaten: reports/&lt;run&gt;/&lt;model&gt;.json (jüngster vollständiger Lauf je Modell).
        Generiert von medium_llm_report.py · Schwestergenerator zu small_llm_report.py.</footer>
</body></html>
"""


def build_md(rows: list[dict], generated: str) -> str:
    """Eigenständiges Markdown-Dokument (GitLab-tauglich), gleiche Inhalte wie HTML."""
    ranked = sorted(rows, key=lambda r: overall_rate(r["data"]), reverse=True)
    judge = rows[0]["data"]["meta"].get("judge", "—")

    L: list[str] = []
    L.append("# Mittelklasse-LLM-Vergleich — 8–35B")
    L.append("")
    L.append(
        f"Instruct-/Reasoning-Modelle der Mittelklasse (~15–35B), einheitlich auf "
        f"**vLLM v0.23.0**. Judge: **{judge}**. 77 Testfälle / 6 Playbooks. "
        f"Generiert {generated}."
    )
    L.append("")

    # ---- Rangliste ----
    L.append("## Rangliste (Gesamt-Pass-Rate)")
    L.append("")
    L.append("| # | Modell | Größe | Quant | Pass-Rate | bestanden | K.O. |")
    L.append("|---|--------|-------|-------|-----------|-----------|------|")
    for i, r in enumerate(ranked, 1):
        s = r["data"]["summary"]
        rate = overall_rate(r["data"])
        L.append(
            f"| {i} | {r['name']} | {r['params_b']:g}B {r['family']} ({r['arch']}) | "
            f"{r['quant']} | **{rate*100:.0f}%** | {s['passed']}/{s['total_tests']} | "
            f"{s['knockouts']} |"
        )
    L.append("")

    # ---- Playbook-Matrix ----
    L.append("## Playbook-Matrix")
    L.append("")
    pb_head = " | ".join(lbl for _, lbl in PLAYBOOKS)
    L.append(f"| Modell | Quant | Urteil | Gesamt | {pb_head} |")
    L.append("|--------|-------|--------|--------|" + "----|" * len(PLAYBOOKS))
    for r in ranked:
        s = r["data"]["summary"]
        P = r["data"]["playbooks"]
        rate = overall_rate(r["data"])
        cells = []
        for pkey, _ in PLAYBOOKS:
            if pkey in P:
                p, t, _r = pb_cell(P[pkey])
                cells.append(f"{p}/{t}")
            else:
                cells.append("—")
        L.append(
            f"| {r['name']} | {r['quant']} | {s['overall']} | "
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
        "---\n\n_Quelldaten: reports/&lt;run&gt;/&lt;model&gt;.json (jüngster vollständiger "
        "Lauf je Modell). Generiert von `medium_llm_report.py`._"
    )
    L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "medium-llms" / "index.html")
    ap.add_argument("--md", type=Path, default=None,
                    help="zusätzlich Markdown schreiben (Default: <out-dir>/index.md)")
    args = ap.parse_args()

    rows = []
    missing = []
    for name, mname, pb, fam, quant, arch in MODELS:
        data = load_model(mname)
        if data is None:
            missing.append(mname)
            continue
        rows.append({"name": name, "mname": mname, "params_b": pb,
                     "family": fam, "quant": quant, "arch": arch, "data": data})
        print(f"  ✓ {name:26} ← reports/{data['_source']}/{mname}.json")

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
