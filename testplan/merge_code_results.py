#!/usr/bin/env python3
"""merge_code_results.py — Patch 05_code-Ergebnisse aus einem Re-Run in einen
bestehenden Voll-Report.

Hintergrund: Der ursprüngliche 05_code-Lauf lieferte 0/10 für alle Apertus-
Modelle (HTTP-400-Artefakt durch zu großes max_tokens, gefixt in base.py).
Ein Code-only-Re-Run erzeugt ein eigenes Report-Verzeichnis mit NUR dem
05_code-Playbook. Dieses Skript übernimmt diesen Block in die vollständigen
JSON-Reports des ursprünglichen Laufs und berechnet den Summary-Block neu,
damit consolidate_reports.py das Dashboard korrekt aus den gepatchten JSONs
rekonstruiert.

Usage:
  python merge_code_results.py <ziel_run_dir> <neuer_code_run_dir> \
      --models "Apertus-v1.1-0.5B,Apertus-v1.1-1.5B,Apertus-v1.1-4B" \
      [--min-quality-pass-rate 0.70]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def recompute_summary(playbooks: dict, min_qpr: float) -> dict:
    total = sum(int(pb.get("total", 0)) for pb in playbooks.values())
    passed = sum(int(pb.get("passed", 0)) for pb in playbooks.values())
    knockouts = sum(len(pb.get("knockouts", [])) for pb in playbooks.values())
    has_ko = any(len(pb.get("knockouts", [])) > 0 for pb in playbooks.values())

    quality = playbooks.get("01_quality")
    if quality and int(quality.get("total", 0)) > 0:
        # pass_rate ist als Anteil (0..1) gespeichert; tolerant gegenüber Prozent
        qpr = float(quality.get("pass_rate", 0.0))
        if qpr > 1.0:
            qpr /= 100.0
        if qpr < min_qpr:
            has_ko = True
            knockouts += 1

    if has_ko:
        overall = "K.O."
    elif total > 0 and passed / total >= 0.85:
        overall = "PASS"
    elif total > 0 and passed / total >= 0.75:
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


def regenerate_html_md(target_dir: Path, models: list[str], min_qpr: float) -> None:
    """Erzeuge HTML/MD je Modell aus der (gepatchten) JSON neu, damit die
    Detailseiten zur korrigierten Übersicht passen. Fail-safe: ein Fehler hier
    darf den Merge nicht entwerten (die JSON sind bereits geschrieben)."""
    try:
        from lib.config import (TestplanConfig, JudgeConfig, TargetConfig,
                                 Thresholds, PlaybookEntry, ModelConfig)
        from reporter import ReportGenerator
        from consolidate_reports import load_playbook_results, REPORTS_DIR
    except Exception as e:  # pragma: no cover
        print(f"  ! HTML/MD-Regenerierung übersprungen (Import fehlgeschlagen: {e})")
        return

    config = TestplanConfig(
        judge=JudgeConfig(host="localhost", model="anthropic/claude-haiku-4-5"),
        target=TargetConfig(host="localhost"),
        models=[],
        thresholds=Thresholds(min_quality_pass_rate=min_qpr),
        playbooks=[PlaybookEntry(name=n, description="") for n in
                   ["01_quality", "02_german_language", "03_bias",
                    "04_security", "05_code", "06_performance"]],
        testdata_dir=REPORTS_DIR.parent / "testdata",
        report_dir=REPORTS_DIR,
        base_dir=REPORTS_DIR.parent,
    )
    reporter = ReportGenerator(config, run_timestamp=target_dir.name)

    for model in models:
        jp = target_dir / f"{model}.json"
        if not jp.exists():
            continue
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
            pb_results = load_playbook_results(jp, model)
            cfg = ModelConfig(name=model, profile=data["meta"].get("profile", ""),
                              machine="machine_b", tags=[], params_b=0)
            safe = model.replace("/", "_").replace(" ", "_")
            reporter._write_html(safe, cfg, pb_results)
            reporter._write_markdown(safe, cfg, pb_results)
            print(f"  ↻ {model}: HTML/MD neu erzeugt")
        except Exception as e:
            print(f"  ! {model}: HTML/MD-Regenerierung fehlgeschlagen: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target_dir", type=Path, help="Bestehender Voll-Report (Ziel)")
    ap.add_argument("code_dir", type=Path, help="Neuer Code-only-Report (Quelle)")
    ap.add_argument("--models", required=True, help="Komma-Liste der Modellnamen")
    ap.add_argument("--playbook", default="05_code", help="Zu mergendes Playbook")
    ap.add_argument("--min-quality-pass-rate", type=float, default=0.70)
    ap.add_argument("--no-regen-html", action="store_true",
                    help="HTML/MD-Detailseiten NICHT neu erzeugen")
    args = ap.parse_args()

    pb_key = args.playbook
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    for model in models:
        tgt = args.target_dir / f"{model}.json"
        src = args.code_dir / f"{model}.json"
        if not tgt.exists():
            print(f"  ! {model}: Ziel-JSON fehlt ({tgt}) — übersprungen")
            continue
        if not src.exists():
            print(f"  ! {model}: Code-JSON fehlt ({src}) — übersprungen")
            continue

        tgt_data = json.loads(tgt.read_text(encoding="utf-8"))
        src_data = json.loads(src.read_text(encoding="utf-8"))

        new_block = src_data.get("playbooks", {}).get(pb_key)
        if new_block is None:
            print(f"  ! {model}: {pb_key} im Code-Report nicht gefunden — übersprungen")
            continue

        old_block = tgt_data.get("playbooks", {}).get(pb_key, {})
        old_pf = f"{old_block.get('passed', '?')}/{old_block.get('total', '?')}"
        new_pf = f"{new_block.get('passed', '?')}/{new_block.get('total', '?')}"

        # Backup einmalig
        bak = tgt.with_suffix(".json.prepatch")
        if not bak.exists():
            shutil.copy2(tgt, bak)

        tgt_data.setdefault("playbooks", {})[pb_key] = new_block
        tgt_data["summary"] = recompute_summary(tgt_data["playbooks"], args.min_quality_pass_rate)

        tgt.write_text(json.dumps(tgt_data, indent=2, ensure_ascii=False), encoding="utf-8")
        s = tgt_data["summary"]
        print(f"  ✓ {model}: {pb_key} {old_pf} → {new_pf} | "
              f"overall={s['overall']} {s['passed']}/{s['total_tests']} "
              f"({s['pass_rate']}%) KO={s['knockouts']}")

    if not args.no_regen_html:
        print("--- HTML/MD-Detailseiten neu erzeugen ---")
        regenerate_html_md(args.target_dir, models, args.min_quality_pass_rate)


if __name__ == "__main__":
    main()
