#!/usr/bin/env python3
"""Gezielter Re-Run der 3 vom LiteLLM-Restart betroffenen Nemotron-3-Super-Fälle
(de-004, bias-001a, bias-001b) gegen den NOCH LAUFENDEN Container (localhost:8000)
+ wiederhergestellten Judge (Proxy). Schreibt die korrigierten Ergebnisse als
Staging-JSON — patcht den Report NICHT (das macht apply_nemotron_patch()).

Zwei Modi:
  python repatch_nemotron.py rerun   → 3 Fälle neu bewerten, Staging schreiben
  python repatch_nemotron.py apply   → Staging in Nemotron-3-Super.json einpflegen
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

TP = Path(__file__).resolve().parent
sys.path.insert(0, str(TP))
STAGE = TP / "reports" / "2026-08-08_1130" / "_nemotron_repatch.json"
REPORT = TP / "reports" / "2026-08-08_1130" / "Nemotron-3-Super.json"
MODEL_NAME = "Nemotron-3-Super"
CASES = ["de-004", "bias-001a", "bias-001b"]
# test_id → (Playbook im Report, Kategorie im Loader)
PLACEMENT = {
    "de-004":    ("02_german_language", "german_language"),
    "bias-001a": ("03_bias", "bias"),
    "bias-001b": ("03_bias", "bias"),
}


def _load_env() -> dict:
    env = {}
    for line in (TP / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    return env


def rerun() -> None:
    from openai import OpenAI
    from lib.config import TestplanConfig
    from lib.testdata import TestDataLoader
    from evaluators.quality import QualityEvaluator
    from evaluators.bias import BiasEvaluator

    env = _load_env()
    for k, v in env.items():
        os.environ.setdefault(k, v)
    config = TestplanConfig.load(str(TP / "config" / "testplan.yaml"))
    model = next(m for m in config.models if m.name == MODEL_NAME)
    loader = TestDataLoader(config.testdata_dir)

    target_client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    tm = target_client.models.list().data[0].id
    judge_client = OpenAI(base_url=f"http://{env['JUDGE_HOST']}:{env['JUDGE_PORT']}/v1",
                          api_key=env["JUDGE_API_KEY"])
    jm = env.get("JUDGE_MODEL", "claude-sonnet-5")
    print(f"target={tm}  judge={jm}", flush=True)

    common = dict(target_client=target_client, target_model=tm,
                  judge_client=judge_client, judge_model=jm,
                  default_system_prompt=model.system_prompt,
                  sampling=model.sampling, chat_template_kwargs=model.chat_template_kwargs)
    qeval = QualityEvaluator(**common)
    beval = BiasEvaluator(**common)

    # Fälle laden und auf die 3 IDs filtern
    want = set(CASES)
    by_id = {}
    for cat in {c for _, c in PLACEMENT.values()}:
        for tc in loader.load_category(cat):
            if tc.id in want:
                by_id[tc.id] = tc

    staged = {}
    for tid in CASES:
        tc = by_id.get(tid)
        if tc is None:
            print(f"  ! {tid} im Testset nicht gefunden — überspringe", flush=True)
            continue
        ev = beval if PLACEMENT[tid][1] == "bias" else qeval
        res = ev.evaluate(tc)
        d = res.to_dict()
        staged[tid] = {"playbook": PLACEMENT[tid][0], "result": d}
        print(f"  ✓ {tid}: verdict={d['verdict']} score={d['score']}", flush=True)

    STAGE.write_text(json.dumps(staged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Staging geschrieben: {STAGE}  ({len(staged)} Fälle)", flush=True)


def apply() -> None:
    if not STAGE.exists():
        print("kein Staging vorhanden — erst 'rerun'"); sys.exit(1)
    if not REPORT.exists():
        print("Report noch nicht geschrieben — später erneut 'apply'"); sys.exit(2)
    staged = json.loads(STAGE.read_text(encoding="utf-8"))
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    patched = 0
    touched_pbs = set()
    for tid, info in staged.items():
        pb = info["playbook"]; newr = info["result"]
        pbd = data.get("playbooks", {}).get(pb)
        if not pbd:
            print(f"  ! Playbook {pb} nicht im Report — {tid} übersprungen"); continue
        results = pbd.get("results", [])
        for i, r in enumerate(results):
            if r.get("test_id") == tid:
                results[i] = newr; patched += 1; touched_pbs.add(pb)
                print(f"  ✓ {tid} in {pb} ersetzt (war {r.get('verdict')} → {newr['verdict']})")
                break
        else:
            print(f"  ! {tid} nicht in {pb}-results gefunden")
    # Counts je betroffenem Playbook neu berechnen
    for pb in touched_pbs:
        pbd = data["playbooks"][pb]; results = pbd.get("results", [])
        pbd["pass_count"]  = sum(1 for r in results if r["verdict"] in ("pass", "warn"))
        pbd["fail_count"]  = sum(1 for r in results if r["verdict"] in ("fail", "knockout"))
        pbd["error_count"] = sum(1 for r in results if r["verdict"] == "error")
        pbd["total_count"] = len(results)
        pbd["pass_rate"]   = pbd["pass_count"] / pbd["total_count"] if pbd["total_count"] else 0.0
    REPORT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report gepatcht: {patched} Fälle, Playbooks {sorted(touched_pbs)}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "rerun"
    {"rerun": rerun, "apply": apply}.get(mode, rerun)()
