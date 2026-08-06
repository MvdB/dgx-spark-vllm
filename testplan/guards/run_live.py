#!/usr/bin/env python3
"""End-to-End: echter GuardEvaluator gegen ein laufendes Guard-Modell auf :8000.

Umgeht Orchestrator/Judge/SSH — laedt die echten Guardrails-Testdaten und den
echten Evaluator, damit Adapter + Metriken gegen das reale Modell laufen.
Aufruf:  run_live.py <protocol> [threshold]
"""
import sys
from pathlib import Path

_TESTPLAN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TESTPLAN))
from openai import OpenAI                          # noqa: E402
from lib.testdata import TestDataLoader            # noqa: E402
from evaluators.guard import GuardEvaluator        # noqa: E402
from evaluators.base import Verdict                # noqa: E402

proto = sys.argv[1] if len(sys.argv) > 1 else "shieldstral"
threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
model = client.models.list().data[0].id
print(f"Modell: {model}  Protokoll: {proto}  Schwelle: {threshold}\n")

cases = TestDataLoader(_TESTPLAN / "testdata").load_category("guardrails")
ev = GuardEvaluator(target_client=client, target_model=model, guard_protocol=proto,
                    threshold=threshold, reasoning_effort="low")
results = ev.evaluate_batch(cases)

agg = next(r for r in results if r.evaluator == "guard.aggregate")
kos = [r for r in results if r.verdict == Verdict.KNOCKOUT]
graded = [r for r in results if r.evaluator == "guard"]

# Fehlklassifikationen zeigen
print("Fehlklassifikationen:")
for r in graded:
    o = r.metadata.get("outcome")
    if o in ("FP", "FN", "ERROR"):
        tag = "Trap" if r.metadata.get("trap") else r.metadata.get("subcategory")
        print(f"  {o}  {r.test_id:14s} [{tag}] {r.reasoning[:60]}")
print()
print("AGGREGAT:", agg.reasoning)
for k in kos:
    print("K.O.:", k.reasoning)
m = agg.metadata
print(f"\nJSON-Kern: recall={m['recall']:.3f} fpr={m['fpr']:.3f} "
      f"trap_fpr={m['trap_fpr']:.3f} f1={m['f1']:.3f} acc={m['accuracy']:.3f} "
      f"lat_ø={m['latency_ms_mean']:.0f}ms")
