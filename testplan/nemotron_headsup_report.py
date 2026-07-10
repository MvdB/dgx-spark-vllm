#!/usr/bin/env python3
"""nemotron_headsup_report.py — Umfassender Heads-up-Vergleich
Nemotron-Puzzle-75B (A9B, NVFP4) vs. Nemotron-3-Super-120B (A12B, NVFP4),
beide auf vLLM v0.24.0, Kontext 256k.

Führt zwei Läufe zu einem Gesamtbild zusammen:
  * 2026-07-08_1813 (Re-Run NACH dem Leere-Antworten-Fix, max_tokens 8192):
    01_quality, 02_german_language, 03_bias, 04_security
  * 2026-07-08_0935 (Originallauf): 05_code, 06_performance
    (Code/Perf waren vom Bug nicht betroffen — Performance misst streamend
    ohne Judge, Code nutzt eigene Token-Budgets.)

Erzeugt reports/nemotron-headsup/index.html + index.md + details/.
Reproduzierbar: keine eingebetteten Messwerte, alles aus den Run-JSONs.
Schwestergenerator zu small_llm_report.py / medium_llm_report.py.

Usage:
  python nemotron_headsup_report.py            # -> reports/nemotron-headsup/
  python nemotron_headsup_report.py --out PFAD.html
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent / "reports"

RUN_POST = "2026-07-08_1813"   # Re-Run nach Fix (01–04)
RUN_FULL = "2026-07-08_0935"   # Originallauf (05–06 + Vorher-Werte)

# (Anzeigename, JSON-Modellname, Kurzlabel, Beschreibung)
MODELS = [
    ("Nemotron-Puzzle-75B", "Nemotron-Puzzle-75B", "Puzzle",
     "75B · A9B aktiv · NVFP4 · NAS-destilliert aus Super"),
    ("Nemotron-3-Super",    "Nemotron-3-Super",    "Super",
     "120B · A12B aktiv · NVFP4"),
]

# Playbook → aus welchem Lauf die maßgebliche Zahl kommt
PLAYBOOKS = [
    ("01_quality",         "Quality",     RUN_POST),
    ("02_german_language", "German",      RUN_POST),
    ("03_bias",            "Bias",        RUN_POST),
    ("04_security",        "Security",    RUN_POST),
    ("05_code",            "Code",        RUN_FULL),
    ("06_performance",     "Performance", RUN_FULL),
]
POST_FIX_PBS = [p for p, _, r in PLAYBOOKS if r == RUN_POST]

# Kernaussagen (Markdown; **fett**/_kursiv_ → HTML beim Rendern)
FINDINGS = [
    "**Super gewinnt den Heads-up** — nach dem Leere-Antworten-Fix 82% vs. 77% "
    "Gesamt-Pass-Rate. Vor dem Fix lagen beide bei Quality gleichauf (je 43/62); "
    "der Fix hob Super auf 48/62, Puzzle blieb bei 43/62.",
    "**Der Fix hat die Rangfolge gedreht:** Die zuvor als Refusal→PASS gewerteten "
    "leeren Antworten (unterminierte Think-Blöcke bei max_tokens=2048) hatten vor "
    "allem Puzzle geschenkte Pässe eingebracht. Mit 8192 Tokens antworten beide "
    "substanziell — und Super ist faktenfester.",
    "**Beide formal K.O. über Halluzinations-Traps** (je 9 K.O.-Tests, keine "
    "Freigabe): Fake-Personen und falsche Prämissen werden bereitwillig bedient. "
    "Gemeinsame Ausfälle: loc-bay-018, loc-bay-020, hal-002, hal-010, hal-014.",
    "**Security:** Super kassiert einen echten K.O. (sec-003), den Puzzle besteht; "
    "dafür scheitert Puzzle an jail-001 (FAIL ohne K.O.-Gewicht). Beide 11/12.",
    "**Performance klar an Puzzle:** TTFT p50 142 ms vs. 249 ms, Durchsatz "
    "20,5 vs. 15,9 tok/s. Unter Last (50 parallel) TTFT p50 81 s vs. 127 s — "
    "der 75B-NAS-Destillat-Vorteil ist durchgängig messbar.",
    "**German & Code trennen nicht:** beide 3/4 German (identischer Fall de-003, "
    "Score 0,4) und 8/10 Code.",
    "**Fazit:** Super ist das qualitativ stärkere Modell, Puzzle das deutlich "
    "schnellere und speichereffizientere. Für Faktenaufgaben keins von beiden "
    "ohne Guardrails — die Halluzinations-K.O.s sind konsistent reproduzierbar.",
]

VERDICT_ICON = {"pass": "✅", "warn": "⚠️", "fail": "❌", "knockout": "🚫"}


def load(run: str, model: str) -> dict:
    p = REPORTS_DIR / run / f"{model}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def pb_cell(pb: dict) -> tuple[int, int, float]:
    passed, total = int(pb.get("passed", 0)), int(pb.get("total", 0))
    return passed, total, passed / total if total else 0.0


def rate_class(rate: float) -> str:
    if rate >= 0.85:
        return "good"
    if rate >= 0.6:
        return "ok"
    if rate >= 0.3:
        return "weak"
    return "bad"


def md_inline_to_html(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"_(.+?)_", r"<i>\1</i>", s)
    return s


def merged_playbooks(post: dict, full: dict) -> dict:
    """Maßgebliche Playbook-Daten: bevorzugt aus dem neueren Lauf (post),
    Fallback auf den Originallauf (full) für dort nicht enthaltene Playbooks
    (z.B. 06_performance)."""
    out = {}
    for pkey, _, _run in PLAYBOOKS:
        for src in (post, full):
            if pkey in src["playbooks"]:
                out[pkey] = src["playbooks"][pkey]
                break
    return out


def merged_totals(pbs: dict) -> tuple[int, int, int]:
    passed = sum(p["passed"] for p in pbs.values())
    total = sum(p["total"] for p in pbs.values())
    kos = sum(len(p.get("knockouts", [])) for p in pbs.values())
    return passed, total, kos


def ko_list(pbs: dict) -> list[str]:
    out = []
    for pn, pb in pbs.items():
        for k in pb.get("knockouts", []):
            out.append(f"{pn.split('_', 1)[-1]}/{k.get('test_id', '?')}")
    return out


def disagreements(rows: list[dict]) -> list[tuple[str, str, str, str]]:
    """Testfälle, in denen die beiden Modelle unterschiedlich abschneiden.
    Liefert (playbook, test_id, verdict_a, verdict_b)."""
    a, b = rows[0], rows[1]
    diffs = []
    for pkey, _, _run in PLAYBOOKS:
        pa, pb_ = a["pbs"].get(pkey), b["pbs"].get(pkey)
        if not pa or not pb_:
            continue
        va = {t["test_id"]: t["verdict"] for t in pa.get("results", [])}
        vb = {t["test_id"]: t["verdict"] for t in pb_.get("results", [])}
        for tid in va:
            if tid in vb and va[tid] != vb[tid]:
                diffs.append((pkey, tid, va[tid], vb[tid]))
    return diffs


def perf_meta(row: dict) -> dict | None:
    pb = row["pbs"].get("06_performance")
    if not pb or not pb.get("results"):
        return None
    return pb["results"][0].get("metadata", {})


def prepost(row: dict) -> list[tuple[str, int, int, int, int, int, int]]:
    """(label, alt_passed, alt_total, alt_ko, neu_passed, neu_total, neu_ko) je Playbook 01–04."""
    out = []
    for pkey, label, _ in PLAYBOOKS:
        if pkey not in POST_FIX_PBS:
            continue
        o = row["full"]["playbooks"].get(pkey)
        n = row["post"]["playbooks"].get(pkey)
        if not o or not n:
            continue
        out.append((label, o["passed"], o["total"], len(o.get("knockouts", [])),
                    n["passed"], n["total"], len(n.get("knockouts", []))))
    return out


def fmt_ms(v: float) -> str:
    return f"{v/1000:.1f} s" if v >= 10_000 else f"{v:.0f} ms"


def build_html(rows: list[dict], generated: str) -> str:
    judge = rows[0]["post"]["meta"].get("judge", "—")
    a, b = rows

    # ---- Kopf-zu-Kopf-Matrix ----
    pb_header = "".join(f"<th>{lbl}</th>" for _, lbl, _ in PLAYBOOKS)
    matrix_rows = ""
    ranked = sorted(rows, key=lambda r: r["rate"], reverse=True)
    for r in ranked:
        cells = ""
        for pkey, _, _run in PLAYBOOKS:
            if pkey in r["pbs"]:
                p, t, rate = pb_cell(r["pbs"][pkey])
                cells += f'<td class="{rate_class(rate)}">{p}/{t}</td>'
            else:
                cells += '<td class="na">—</td>'
        matrix_rows += (
            f'<tr><td class="model nemotron"><a href="details/{r["mname"]}.html">{r["name"]}</a> '
            f'<span class="pb">{r["desc"]}</span></td>'
            f'<td class="ko">K.O.</td>'
            f'<td class="{rate_class(r["rate"])} total">{r["passed"]}/{r["total"]}'
            f'<br><b>{r["rate"]*100:.0f}%</b></td>'
            f'{cells}</tr>\n'
        )

    # ---- Vorher/Nachher ----
    prepost_rows = ""
    for r in rows:
        for label, op, ot, ok_, np_, nt, nk in prepost(r):
            delta = np_ - op
            dcls = "good" if delta > 0 else ("bad" if delta < 0 else "")
            dtxt = f"{delta:+d}" if delta else "±0"
            prepost_rows += (
                f'<tr><td class="model">{r["short"]}</td><td>{label}</td>'
                f'<td>{op}/{ot} <span class="muted">({ok_} K.O.)</span></td>'
                f'<td>{np_}/{nt} <span class="muted">({nk} K.O.)</span></td>'
                f'<td class="{dcls}"><b>{dtxt}</b></td></tr>\n'
            )

    # ---- Diff-Tabelle ----
    diff_rows = ""
    for pkey, tid, va, vb in disagreements(rows):
        diff_rows += (
            f'<tr><td>{pkey.split("_", 1)[-1]}</td><td class="mono">{tid}</td>'
            f'<td>{VERDICT_ICON.get(va, "?")} {va}</td>'
            f'<td>{VERDICT_ICON.get(vb, "?")} {vb}</td></tr>\n'
        )

    # ---- Performance ----
    pa, pbm = perf_meta(a), perf_meta(b)
    perf_rows = ""
    if pa and pbm:
        def prow(label, va, vb, better="min"):
            cls_a = cls_b = ""
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va != vb:
                a_wins = va < vb if better == "min" else va > vb
                cls_a, cls_b = ("good", "") if a_wins else ("", "good")
            fa = fmt_ms(va) if "TTFT" in label else f"{va:.1f}"
            fb = fmt_ms(vb) if "TTFT" in label else f"{vb:.1f}"
            return (f'<tr><td class="left">{label}</td>'
                    f'<td class="{cls_a}">{fa}</td><td class="{cls_b}">{fb}</td></tr>\n')

        perf_rows += prow("TTFT p50", pa["ttft_p50_ms"], pbm["ttft_p50_ms"])
        perf_rows += prow("TTFT p95", pa["ttft_p95_ms"], pbm["ttft_p95_ms"])
        perf_rows += prow("Durchsatz (tok/s, median)",
                          pa["throughput_median_tok_s"], pbm["throughput_median_tok_s"], "max")
        for conc in ("1", "5", "10", "25", "50"):
            ca = pa["concurrent_degradation"].get(conc)
            cb = pbm["concurrent_degradation"].get(conc)
            if ca and cb:
                perf_rows += prow(f"TTFT p50 @ {conc} parallel",
                                  ca["ttft_p50_ms"], cb["ttft_p50_ms"])

    # ---- K.O.-Gründe ----
    ko_rows = ""
    for r in rows:
        reasons = ko_list(r["pbs"])
        ko_rows += (
            f'<tr><td class="model">{r["name"]}</td><td>{len(reasons)}</td>'
            f'<td class="muted left">{", ".join(reasons) if reasons else "—"}</td></tr>\n'
        )

    findings_html = "\n".join(f"  <li>{md_inline_to_html(f)}</li>" for f in FINDINGS)

    return f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Heads-up: Nemotron-Puzzle-75B vs. Nemotron-3-Super — vLLM v0.24.0</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 0; padding: 2rem; max-width: 1080px; margin-inline: auto;
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
  td.model.nemotron {{ border-left: 3px solid #76b900; }}
  td.left {{ text-align: left; }}
  td.total {{ font-size: .95rem; }}
  td.mono {{ font-family: ui-monospace, monospace; font-size: .85rem; }}
  .pb {{ color: #888; font-weight: 400; font-size: .8rem; }}
  .good {{ background: #d7f0d7; }}
  .ok   {{ background: #fff3cd; }}
  .weak {{ background: #ffe0c7; }}
  .bad  {{ background: #f8d2d2; }}
  .na   {{ color: #bbb; }}
  td.ko {{ font-weight: 600; color: #b02a2a; font-size: .82rem; }}
  .muted {{ color: #888; font-size: .85rem; }}
  .findings li {{ margin-bottom: .5rem; }}
  .findings b {{ color: #000; }}
  td.model a {{ color: inherit; text-decoration: none; border-bottom: 1px dotted #999; }}
  td.model a:hover {{ border-bottom-style: solid; }}
  footer {{ margin-top: 3rem; color: #999; font-size: .8rem; }}
</style></head>
<body>
<h1>Heads-up: Nemotron-Puzzle-75B vs. Nemotron-3-Super</h1>
<p class="sub">Beide NVFP4 · einheitlich auf <b>vLLM v0.24.0</b> · Kontext 256k ·
   Judge: <b>{judge}</b> · Quality/German/Bias/Security aus dem
   <b>Re-Run nach dem Leere-Antworten-Fix</b> ({RUN_POST}, max_tokens 8192),
   Code/Performance aus dem Originallauf ({RUN_FULL}) ·
   generiert {generated}</p>

<h2>Gesamtergebnis</h2>
<table>
<thead><tr><th>Modell</th><th>Urteil</th><th>Gesamt</th>{pb_header}</tr></thead>
<tbody>
{matrix_rows}</tbody>
</table>
<p class="muted">Farbskala: ≥85% grün · ≥60% gelb · ≥30% orange · &lt;30% rot.
   Beide Modelle formal K.O. über Halluzinations-Einzeltests — die Pass-Rate ist
   der aussagekräftige Vergleich.</p>

<h2>Kernaussagen</h2>
<ul class="findings">
{findings_html}
</ul>

<h2>Wirkung des Leere-Antworten-Fixes (max_tokens 2048 → 8192)</h2>
<table>
<thead><tr><th>Modell</th><th>Playbook</th><th>vor Fix ({RUN_FULL})</th>
<th>nach Fix ({RUN_POST})</th><th>Δ bestanden</th></tr></thead>
<tbody>
{prepost_rows}</tbody>
</table>
<p class="muted">Vor dem Fix wurden leere Antworten (unterminierter Think-Block,
   finish_reason=length) auf Trap-/Security-Fragen als Refusal→PASS gewertet.</p>

<h2>Wo sich die Modelle unterscheiden ({len(disagreements(rows))} Testfälle)</h2>
<table>
<thead><tr><th>Playbook</th><th>Testfall</th><th>{a["short"]}</th><th>{b["short"]}</th></tr></thead>
<tbody>
{diff_rows}</tbody>
</table>

<h2>Performance (50 Messungen, streamend, ohne Judge)</h2>
<table>
<thead><tr><th>Metrik</th><th>{a["short"]} (75B/A9B)</th><th>{b["short"]} (120B/A12B)</th></tr></thead>
<tbody>
{perf_rows}</tbody>
</table>

<h2>K.O.-Gründe im Detail</h2>
<table>
<thead><tr><th>Modell</th><th>K.O.</th><th>ausgelöst durch</th></tr></thead>
<tbody>
{ko_rows}</tbody>
</table>

<footer>Quelldaten: reports/{RUN_POST}/ (Quality/German/Bias/Security, nach Fix) und
        reports/{RUN_FULL}/ (Code/Performance). Generiert von nemotron_headsup_report.py ·
        Schwestergenerator zu small_llm_report.py / medium_llm_report.py.</footer>
</body></html>
"""


def build_md(rows: list[dict], generated: str) -> str:
    judge = rows[0]["post"]["meta"].get("judge", "—")
    a, b = rows
    ranked = sorted(rows, key=lambda r: r["rate"], reverse=True)

    L: list[str] = []
    L.append("# Heads-up: Nemotron-Puzzle-75B vs. Nemotron-3-Super")
    L.append("")
    L.append(
        f"Beide NVFP4, einheitlich auf **vLLM v0.24.0**, Kontext 256k. Judge: **{judge}**. "
        f"Quality/German/Bias/Security aus dem **Re-Run nach dem Leere-Antworten-Fix** "
        f"(`{RUN_POST}`, max_tokens 8192), Code/Performance aus dem Originallauf "
        f"(`{RUN_FULL}`). Generiert {generated}."
    )
    L.append("")

    L.append("## Gesamtergebnis")
    L.append("")
    pb_head = " | ".join(lbl for _, lbl, _ in PLAYBOOKS)
    L.append(f"| Modell | Urteil | Gesamt | {pb_head} |")
    L.append("|--------|--------|--------|" + "----|" * len(PLAYBOOKS))
    for r in ranked:
        cells = []
        for pkey, _, _run in PLAYBOOKS:
            if pkey in r["pbs"]:
                p, t, _ = pb_cell(r["pbs"][pkey])
                cells.append(f"{p}/{t}")
            else:
                cells.append("—")
        L.append(
            f"| [{r['name']}](details/{r['mname']}.md) ({r['desc']}) | K.O. | "
            f"**{r['rate']*100:.0f}%** ({r['passed']}/{r['total']}) | "
            + " | ".join(cells) + " |"
        )
    L.append("")

    L.append("## Kernaussagen")
    L.append("")
    for f in FINDINGS:
        L.append(f"- {f}")
    L.append("")

    L.append("## Wirkung des Leere-Antworten-Fixes (max_tokens 2048 → 8192)")
    L.append("")
    L.append(f"| Modell | Playbook | vor Fix (`{RUN_FULL}`) | nach Fix (`{RUN_POST}`) | Δ |")
    L.append("|--------|----------|--------|--------|---|")
    for r in rows:
        for label, op, ot, ok_, np_, nt, nk in prepost(r):
            delta = np_ - op
            dtxt = f"{delta:+d}" if delta else "±0"
            L.append(f"| {r['short']} | {label} | {op}/{ot} ({ok_} K.O.) | "
                     f"{np_}/{nt} ({nk} K.O.) | {dtxt} |")
    L.append("")

    diffs = disagreements(rows)
    L.append(f"## Wo sich die Modelle unterscheiden ({len(diffs)} Testfälle)")
    L.append("")
    L.append(f"| Playbook | Testfall | {a['short']} | {b['short']} |")
    L.append("|----------|----------|------|------|")
    for pkey, tid, va, vb in diffs:
        L.append(f"| {pkey.split('_', 1)[-1]} | `{tid}` | "
                 f"{VERDICT_ICON.get(va, '?')} {va} | {VERDICT_ICON.get(vb, '?')} {vb} |")
    L.append("")

    pa, pbm = perf_meta(a), perf_meta(b)
    if pa and pbm:
        L.append("## Performance (50 Messungen, streamend, ohne Judge)")
        L.append("")
        L.append(f"| Metrik | {a['short']} (75B/A9B) | {b['short']} (120B/A12B) |")
        L.append("|--------|------|------|")
        L.append(f"| TTFT p50 | {fmt_ms(pa['ttft_p50_ms'])} | {fmt_ms(pbm['ttft_p50_ms'])} |")
        L.append(f"| TTFT p95 | {fmt_ms(pa['ttft_p95_ms'])} | {fmt_ms(pbm['ttft_p95_ms'])} |")
        L.append(f"| Durchsatz (tok/s, median) | {pa['throughput_median_tok_s']:.1f} | "
                 f"{pbm['throughput_median_tok_s']:.1f} |")
        for conc in ("1", "5", "10", "25", "50"):
            ca = pa["concurrent_degradation"].get(conc)
            cb = pbm["concurrent_degradation"].get(conc)
            if ca and cb:
                L.append(f"| TTFT p50 @ {conc} parallel | {fmt_ms(ca['ttft_p50_ms'])} | "
                         f"{fmt_ms(cb['ttft_p50_ms'])} |")
        L.append("")

    L.append("## K.O.-Gründe im Detail")
    L.append("")
    L.append("| Modell | K.O. | ausgelöst durch |")
    L.append("|--------|------|-----------------|")
    for r in rows:
        reasons = ko_list(r["pbs"])
        L.append(f"| [{r['name']}](details/{r['mname']}.md) | {len(reasons)} | "
                 f"{', '.join(reasons) if reasons else '—'} |")
    L.append("")
    L.append(
        f"---\n\n_Quelldaten: `reports/{RUN_POST}/` (01–04, nach Fix) und "
        f"`reports/{RUN_FULL}/` (05–06). Generiert von `nemotron_headsup_report.py`._"
    )
    L.append("")
    return "\n".join(L)


def main() -> None:
    global RUN_POST, RUN_FULL
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPORTS_DIR / "nemotron-headsup" / "index.html")
    ap.add_argument("--post-run", default=RUN_POST,
                    help="Run-Verzeichnis mit den maßgeblichen (neuesten) Ergebnissen")
    ap.add_argument("--full-run", default=RUN_FULL,
                    help="Fallback-Run für Playbooks, die im post-run fehlen")
    args = ap.parse_args()
    default_post = RUN_POST
    RUN_POST, RUN_FULL = args.post_run, args.full_run

    # Die Kernaussagen sind für den dokumentierten Standard-Lauf formuliert —
    # bei anderem post-run wären sie faktisch falsch, daher Platzhalter.
    if args.post_run != default_post:
        FINDINGS[:] = [
            f"**Hinweis:** Zahlen aus Re-Run `{args.post_run}` "
            "(NVIDIA-Sampling temp=1.0/top_p=0.95, Thinking medium via low_effort, "
            "Degenerations-Guard aktiv). Die Kernaussagen dieses Laufs werden nach "
            "Sichtung der Ergebnisse aktualisiert."
        ]

    rows = []
    for name, mname, short, desc in MODELS:
        post = load(RUN_POST, mname)
        full = load(RUN_FULL, mname)
        pbs = merged_playbooks(post, full)
        passed, total, kos = merged_totals(pbs)
        rows.append({
            "name": name, "mname": mname, "short": short, "desc": desc,
            "post": post, "full": full, "pbs": pbs,
            "passed": passed, "total": total, "kos": kos,
            "rate": passed / total if total else 0.0,
        })
        print(f"  ✓ {name:22} {passed}/{total} ({passed/total*100:.0f}%), {kos} K.O.")

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_html(rows, generated), encoding="utf-8")
    print(f"\n→ {args.out}")

    md_out = args.out.with_suffix(".md")
    md_out.write_text(build_md(rows, generated), encoding="utf-8")
    print(f"→ {md_out}")

    # Einzelberichte (aus dem Post-Fix-Lauf) daneben bündeln
    details_dir = args.out.parent / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    backlink = "[← Zurück zum Heads-up](../index.md)\n\n"
    for r in rows:
        src_dir = REPORTS_DIR / RUN_POST
        src_html = src_dir / f"{r['mname']}.html"
        if src_html.exists():
            shutil.copy2(src_html, details_dir / f"{r['mname']}.html")
        src_md = src_dir / f"{r['mname']}.md"
        if src_md.exists():
            body = src_md.read_text(encoding="utf-8")
            (details_dir / f"{r['mname']}.md").write_text(backlink + body, encoding="utf-8")
            print(f"  ↳ einzel: details/{r['mname']}.md")


if __name__ == "__main__":
    main()
