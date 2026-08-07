#!/usr/bin/env python3
"""Vervollständigt den SaaS-Kohortenlauf trotz des stündlichen $15-Failsafes.

Wiederholt NUR die noch unvollständigen Modelle (Report fehlt oder ERROR-Rate
>30 %, z.B. nach Budget-429). Nach jedem Versuch, der nicht alles schafft, wird
das Stunden-Fenster ausgesessen (Default 30 min) und erneut probiert. Monatliches
Budget ($75) deckt den Rest (~$6) locker; die Bremse ist nur die Rate.
"""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

TP = Path(__file__).resolve().parent
sys.path.insert(0, str(TP))
from saas_cohort_run import COHORT  # noqa: E402

RUN = "2026-08-07_saas"
RUNDIR = TP / "reports" / RUN
PY = str(TP / ".venv" / "bin" / "python")
MAX_ATTEMPTS = 10
WAIT = 1800  # 30 min zwischen Versuchen (Stunden-Fenster aussitzen)


def err_rate(name: str) -> float:
    f = RUNDIR / f"{name}.json"
    if not f.exists():
        return 1.0
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return 1.0
    tot = err = 0
    for v in d.get("playbooks", {}).values():
        for r in v.get("results", []):
            tot += 1
            if r.get("verdict") == "error":
                err += 1
    return 1.0 if tot == 0 else err / tot


def incomplete() -> list[str]:
    return [name for (name, *_rest) in COHORT if err_rate(name) > 0.3]


def main() -> int:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        rem = incomplete()
        print(f"══ Versuch {attempt}/{MAX_ATTEMPTS} · offen: {len(rem)} → {rem}", flush=True)
        if not rem:
            print("✅ ALLE 16 MODELLE VOLLSTÄNDIG", flush=True)
            return 0
        subprocess.run([PY, "-u", str(TP / "saas_cohort_run.py"),
                        "--run-id", RUN, "--models", ",".join(rem),
                        "--max-run-cost", "13"])
        if incomplete():
            mins = WAIT // 60
            print(f"── noch offen — warte {mins} min auf das Stunden-Fenster …", flush=True)
            time.sleep(WAIT)
    print(f"⚠ MAX_ATTEMPTS erreicht, noch offen: {incomplete()}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
