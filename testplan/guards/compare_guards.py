#!/usr/bin/env python3
"""Baut den Guard-Vergleichsreport aus den field_run.py-JSONs.

Aufruf:  compare_guards.py <results_dir> [out_md]

Liest alle *.json (je ein Guard-Modell), erzeugt eine Markdown-Vergleichstabelle
mit Konfusionsmatrix und den betriebsrelevanten Raten, eine Aufschlüsselung der
Fehler nach Fehlerklasse und eine Übereinstimmungsmatrix (Guard gegen Guard).
"""
import glob
import json
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
out_md = Path(sys.argv[2]) if len(sys.argv) > 2 else results_dir / "COMPARISON.md"

runs = []
for f in sorted(glob.glob(str(results_dir / "*.json"))):
    runs.append(json.loads(Path(f).read_text()))
if not runs:
    print("Keine Ergebnis-JSONs gefunden in", results_dir)
    sys.exit(1)


def pct(x):
    return "n/a" if x is None or x != x else f"{x*100:.1f}%"


lines: list[str] = []
lines.append("# Guard-Modelle im Vergleich\n")
lines.append(f"Testsatz: `testdata/guardrails/` (82 Fälle, 70 DE / 12 EN, "
             f"44 safe / 38 unsafe). Ein Guard-Modell pro Lauf, gleiches Set, "
             f"Schwelle/Effort wie angegeben. Kein LLM-Judge — Wahrheit ist das "
             f"Label.\n")

# --- Hauptmetriken -------------------------------------------------------
lines.append("## Kennzahlen\n")
lines.append("| Modell | Recall | FPR | **Trap-FPR** | Precision | F1 | Acc | "
             "Latenz Ø | K.O. |")
lines.append("|---|--:|--:|--:|--:|--:|--:|--:|:--|")
rows = []
for r in runs:
    m = r["metrics"]
    ko = ", ".join(k.replace("guard_ko_", "") for k in r["knockouts"]) or "—"
    lines.append(
        f"| {r['label']} | {pct(m['recall'])} | {pct(m['fpr'])} | "
        f"**{pct(m['trap_fpr'])}** | {pct(m['precision'])} | {pct(m['f1'])} | "
        f"{pct(m['accuracy'])} | {m['latency_ms_mean']:.0f} ms | {ko} |")
    rows.append(r)
lines.append("")

# --- Konfusionsmatrix ----------------------------------------------------
lines.append("## Konfusionsmatrix\n")
lines.append("| Modell | TP | TN | FP | FN | ERR |")
lines.append("|---|--:|--:|--:|--:|--:|")
for r in runs:
    c = r["metrics"]["confusion"]
    lines.append(f"| {r['label']} | {c['tp']} | {c['tn']} | {c['fp']} | "
                 f"{c['fn']} | {c['errors']} |")
lines.append("")
lines.append("- **FN** (Durchrutscher) = echter Schaden als safe eingestuft — "
             "das sicherheitskritische Versagen.")
lines.append("- **FP** (Fehlalarm) = harmloser Inhalt blockiert; **Trap-FPR** "
             "misst das nur auf den Fehlalarm-Fallen und ist der "
             "betriebsentscheidende Wert.\n")

# --- Fehler im Detail ----------------------------------------------------
lines.append("## Fehlklassifikationen im Detail\n")
for r in runs:
    errs = [c for c in r["per_case"] if c["outcome"] in ("FP", "FN", "ERROR")]
    if not errs:
        lines.append(f"### {r['label']} — keine Fehler\n")
        continue
    lines.append(f"### {r['label']} — {len(errs)} Fehler\n")
    lines.append("| Fall | Fehler | Wahrheit→Vorhersage | Kategorie |")
    lines.append("|---|:--|:--|:--|")
    for c in errs:
        tag = "Trap" if c["trap"] else (c["subcategory"] or "")
        harm = c["harm_category"] if c["harm_category"] not in (None, "none") else tag
        lines.append(f"| {c['id']} | {c['outcome']} | "
                     f"{c['truth']}→{c['prediction']} | {harm} |")
    lines.append("")

# --- DE/EN-Abfall --------------------------------------------------------
lines.append("## Mehrsprachigkeit (EN-Spiegel vs. DE)\n")
lines.append("Accuracy auf den 12 EN-Spiegelfällen gegen ihre DE-Zwillinge "
             "(gleiche IDs via `mirror:`-Tag im Testsatz).\n")
lines.append("| Modell | Acc gesamt | Acc DE | Acc EN |")
lines.append("|---|--:|--:|--:|")
for r in runs:
    de = [c for c in r["per_case"] if c["language"] == "de"
          and c["outcome"] in ("TP", "TN", "FP", "FN")]
    en = [c for c in r["per_case"] if c["language"] == "en"
          and c["outcome"] in ("TP", "TN", "FP", "FN")]

    def acc(cs):
        if not cs:
            return None
        return sum(1 for c in cs if c["outcome"] in ("TP", "TN")) / len(cs)
    lines.append(f"| {r['label']} | {pct(r['metrics']['accuracy'])} | "
                 f"{pct(acc(de))} | {pct(acc(en))} |")
lines.append("")

# --- Übereinstimmungsmatrix ---------------------------------------------
lines.append("## Übereinstimmung Guard gegen Guard\n")
lines.append("Anteil der Fälle, in denen zwei Guards dasselbe safe/unsafe-Urteil "
             "fällen (nur geparste Fälle beider).\n")
preds = {r["label"]: {c["id"]: c["prediction"] for c in r["per_case"]
                      if c["outcome"] in ("TP", "TN", "FP", "FN")} for r in runs}
labels = [r["label"] for r in runs]
header = "| | " + " | ".join(labels) + " |"
lines.append(header)
lines.append("|---|" + "|".join("--:" for _ in labels) + "|")
for a in labels:
    cells = []
    for b in labels:
        common = set(preds[a]) & set(preds[b])
        if not common:
            cells.append("—")
            continue
        agree = sum(1 for k in common if preds[a][k] == preds[b][k]) / len(common)
        cells.append(f"{agree*100:.0f}%")
    lines.append(f"| **{a}** | " + " | ".join(cells) + " |")
lines.append("")

lines.append("---\n")
lines.append("Erzeugt aus `guards/field_run.py`-JSONs via `guards/compare_guards.py`. "
             "Rohdaten (per-Fall-Urteile, Latenzen, Scores) liegen daneben als "
             "`*.json`.\n")

out_md.write_text("\n".join(lines))
print("Report:", out_md)
# kurze Konsolenzusammenfassung
print("\n" + "\n".join(lines[3:5]))
for r in runs:
    m = r["metrics"]
    print(f"  {r['label']:22s} Recall {pct(m['recall']):>6s}  "
          f"Trap-FPR {pct(m['trap_fpr']):>6s}  F1 {pct(m['f1']):>6s}")
