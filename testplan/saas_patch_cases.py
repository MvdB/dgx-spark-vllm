#!/usr/bin/env python3
"""Gezieltes Nachziehen EINZELNER Fälle für bereits gelaufene SaaS-Modelle —
kein Voll-Re-run. Erzeugt die betroffenen Fälle frisch (Proxy) und bewertet sie
mit der aktuellen (korrigierten) Referenz, patcht nur diese Fälle ins bestehende
reports/<run>/<Modell>.json und rechnet Playbook-/Gesamt-Summary neu.

Modi:
  --cases a,b,c   nur diese test_ids nachziehen (Default: die 6 korrigierten Refs)
  --none          alle Fälle mit response_type=="none" nachziehen (höheres Budget)

Reports der noch laufenden Modelle NICHT anfassen (--models steuern).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

TP = Path(__file__).resolve().parent
sys.path.insert(0, str(TP))

import saas_cohort_run as S  # wendet Monkeypatches an, liefert Helfer/COHORT
from lib.config import TestplanConfig
from lib.testdata import TestDataLoader
from evaluators.quality import QualityEvaluator
from evaluators.bias import BiasEvaluator
from evaluators.code_eval import CodeEvaluator

CORRECTED = ["fac-003", "hal-020", "code-004", "csec-001", "csec-002", "hal-016"]
STABLE5 = ["Claude-Fable-5", "Claude-Haiku-4.5", "Claude-Opus-5", "Claude-Sonnet-5", "GPT-5.6-luna"]
EVAL_BY_PB = {"01_quality": QualityEvaluator, "02_german_language": QualityEvaluator,
              "03_bias": BiasEvaluator, "05_code": CodeEvaluator}
PROXY_BY_NAME = {name: pid for (name, pid, *_r) in S.COHORT}


def load_all_cases(loader):
    cases = {}
    for cat in ("quality", "long_context", "german_language", "bias", "code"):
        for tc in loader.load_category(cat):
            cases.setdefault(tc.id, tc)
    return cases


def recompute_pb(v):
    res = v["results"]
    v["total"] = len(res)
    v["passed"] = sum(1 for r in res if r["verdict"] == "pass")
    v["failed"] = sum(1 for r in res if r["verdict"] in ("fail", "knockout"))
    v["knockouts"] = [r for r in res if r["verdict"] == "knockout"]
    v["pass_rate"] = v["passed"] / v["total"] if v["total"] else 0.0
    scores = [r["score"] for r in res if r["verdict"] != "error"]
    v["mean_score"] = sum(scores) / len(scores) if scores else 0.0


def recompute_summary(d, min_q):
    pbs = d["playbooks"]
    total = sum(v["total"] for v in pbs.values())
    passed = sum(v["passed"] for v in pbs.values())
    kos = sum(len(v.get("knockouts", [])) for v in pbs.values())
    rate = passed / total if total else 0.0
    q = pbs.get("01_quality")
    if kos or (q and q["pass_rate"] < min_q):
        overall = "K.O."
    elif rate >= 0.85:
        overall = "PASS"
    elif rate >= 0.75:
        overall = "WARN"
    else:
        overall = "FAIL"
    d["summary"] = {"overall": overall, "total_tests": total, "passed": passed,
                    "pass_rate": f"{rate * 100:.0f}", "knockouts": kos}


def patch_model(name, run_dir, cfg, allcases, client, target_ids, none_mode):
    proxy_id = PROXY_BY_NAME.get(name)
    path = run_dir / f"{name}.json"
    if not path.exists() or not proxy_id:
        print(f"  ! {name}: Report/Proxy-ID fehlt — übersprungen", flush=True)
        return 0
    d = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for pb, v in d.get("playbooks", {}).items():
        if pb not in EVAL_BY_PB:
            continue
        ev = None
        for i, r in enumerate(v.get("results", [])):
            tid = r.get("test_id")
            hit = (r.get("response_type") == "none") if none_mode else (tid in target_ids)
            if not hit or tid not in allcases:
                continue
            if ev is None:
                ev = EVAL_BY_PB[pb](target_client=client, target_model=proxy_id,
                                    judge_client=client, judge_model=S.JUDGE_MODEL,
                                    default_system_prompt="", sampling={}, chat_template_kwargs={})
            old = r.get("verdict")
            try:
                newres = ev.evaluate(allcases[tid])
            except Exception as e:
                print(f"    {name}/{tid}: Fehler {e}", flush=True)
                continue
            v["results"][i] = newres.to_dict()
            changed += 1
            print(f"    {name}/{tid}: {old} → {newres.verdict.value} (score {newres.score:.2f})", flush=True)
        if ev is not None:
            recompute_pb(v)
    if changed:
        recompute_summary(d, cfg.thresholds.min_quality_pass_rate)
        path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✓ {name}: {changed} Fälle gepatcht → {d['summary']['overall']} "
          f"{d['summary']['passed']}/{d['summary']['total_tests']}", flush=True)
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="2026-08-07_saas")
    ap.add_argument("--report-dir", default=str(TP / "reports"))
    ap.add_argument("--models", default="stable5", help="'stable5' | 'all' | Kommaliste")
    ap.add_argument("--cases", default=",".join(CORRECTED))
    ap.add_argument("--none", action="store_true", help="statt --cases: alle response_type==none")
    ap.add_argument("--max-tokens", type=int, default=0, help="Subject-Budget (0 = Default 4096)")
    args = ap.parse_args()

    S._load_env()
    cfg = TestplanConfig.load()
    loader = TestDataLoader(cfg.testdata_dir)
    allcases = load_all_cases(loader)
    client = S._proxy_client()
    if args.max_tokens:
        S.SUBJ_MAX_TOKENS = args.max_tokens  # höheres Budget gegen Truncation (none-Modus)

    if args.models == "stable5":
        models = STABLE5
    elif args.models == "all":
        models = [n for (n, *_r) in S.COHORT]
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    run_dir = Path(args.report_dir) / args.run_id
    target_ids = set() if args.none else {c.strip() for c in args.cases.split(",")}
    mode = "none-Antworten" if args.none else f"{len(target_ids)} korrigierte Fälle"
    print(f"Patch {mode} · {len(models)} Modelle · Judge {S.JUDGE_MODEL} · "
          f"Budget {S.SUBJ_MAX_TOKENS}\n", flush=True)

    total = 0
    for name in models:
        total += patch_model(name, run_dir, cfg, allcases, client, target_ids, args.none)
    c = S.TALLY
    subj = c.get("subj_cost", 0.0)
    judge = c.get("judge_cost", 0.0)
    print(f"\n═══ {total} Fälle gepatcht · Kosten gemessen ${subj + judge:.3f} "
          f"(subj ${subj:.3f} + judge ${judge:.3f}) ═══")


if __name__ == "__main__":
    main()
